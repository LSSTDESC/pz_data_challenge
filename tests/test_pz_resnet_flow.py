import os
from pathlib import Path
import pytest

# Put needed import here
import torch
import numpy as np
import h5py
import tables_io

import pickle
import qp

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "pz_resnet_flow"
SUBMISSION_URL: str = "https://github.com/kpngbsee/pz_data_challenge/releases/download/v1.0/pzdatachallenge_resnet_flow_dedicated_submission_20260703_082835.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Model Architecture

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, hidden_multiplier: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * hidden_multiplier
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            Swish(),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
        )
        self.skip_weight = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        return x + self.skip_weight * self.block(x)

class ConditionalResNetWithTimeEmbedding(torch.nn.Module):
    def __init__(self, z_dim=1, t_dim=1, x_dim=6, hidden_dim=512, 
                 num_blocks=5, hidden_multiplier=2, dropout=0.1, time_emb_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.t_dim = t_dim
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            Swish(),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        inp_dim = z_dim + time_emb_dim + x_dim
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(inp_dim, hidden_dim),
            Swish(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout),
        )
        
        self.res_blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_multiplier, dropout)
            for _ in range(num_blocks)
        ])
        
        self.output_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            Swish(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(hidden_dim // 2, z_dim),
        )
    
    def _sinusoidal_embedding(self, t):
        B = t.shape[0]
        freqs = torch.exp(torch.linspace(0, 4, self.time_emb_dim // 2, device=t.device))
        freqs = freqs.unsqueeze(0)
        angles = t * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
    
    def forward(self, z, t, condition=None):
        B = z.shape[0]
        if t.dim() == 0:
            t = t.reshape(1, 1).expand(B, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            t = t.reshape(B, 1)
        
        t_emb = self._sinusoidal_embedding(t)
        t_emb = self.time_embed(t_emb)
        
        if condition is None:
            cond = torch.zeros(B, self.x_dim, device=z.device)
        else:
            if condition.dim() == 1:
                cond = condition.unsqueeze(0).expand(B, -1)
            elif condition.shape[0] != B:
                cond = condition.expand(B, -1)
            else:
                cond = condition
        
        z_flat = z.reshape(B, self.z_dim)
        h = torch.cat([z_flat, t_emb, cond], dim=1)
        h = self.input_proj(h)
        
        for res_block in self.res_blocks:
            h = res_block(h)
        
        out = self.output_proj(h)
        return out

# ODE Sampler

def ode_sample(velocity_model, x0, t0=0.0, t1=1.0, n_steps=20, method='rk4', 
               return_traj=False, condition=None):
    import torch
    
    # Ensure t0 and t1 are tensors
    if not isinstance(t0, torch.Tensor):
        t0 = torch.tensor(t0, dtype=torch.float32, device=x0.device)
    if not isinstance(t1, torch.Tensor):
        t1 = torch.tensor(t1, dtype=torch.float32, device=x0.device)
    
    dt = (t1 - t0) / n_steps
    t = t0.clone()
    x = x0.clone()
    
    if return_traj:
        traj = [x.clone()]
    
    for step in range(n_steps):
        if method == 'rk4':
            k1 = velocity_model(x, t, condition)
            k2 = velocity_model(x + 0.5 * dt * k1, t + 0.5 * dt, condition)
            k3 = velocity_model(x + 0.5 * dt * k2, t + 0.5 * dt, condition)
            k4 = velocity_model(x + dt * k3, t + dt, condition)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            k = velocity_model(x, t, condition)
            x = x + dt * k
        
        t = t + dt
        if return_traj:
            traj.append(x.clone())
    
    if return_traj:
        return x, traj
    return x

# Helper Functions

def generate_predictions_for_test(model, test_file, scaler_X, scaler_y, 
                                   device='cuda', n_samples=500, n_ode_steps=20):
    model.eval()
    model = model.to(device)
    
    # Load and process test data
    test_data = tables_io.read(test_file, tables_io.types.PD_DATAFRAME)
    
    feature_columns = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 
                       'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    
    X_test = test_data[feature_columns].values
    object_ids = test_data['object_id'].values
    X_test = np.nan_to_num(X_test, nan=99.0)
    
    X_test_scaled = scaler_X.transform(X_test)
    X_test_torch = torch.from_numpy(X_test_scaled).float().to(device)
    
    # Generate samples
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_torch), 50):
            x_batch = X_test_torch[i:min(i+50, len(X_test_torch))]
            batch_size = len(x_batch)
            
            x_init = torch.randn(batch_size, n_samples, 1, device=device)
            x_init_flat = x_init.view(-1, 1)
            x_batch_expanded = x_batch.repeat_interleave(n_samples, dim=0)
            
            def velocity_wrapper(x, t, cond=None):
                return model(x, t, condition=cond)
            
            z_T = ode_sample(
                velocity_model=velocity_wrapper,
                x0=x_init_flat,
                t0=0.0,
                t1=1.0, 
                n_steps=n_ode_steps,
                method='rk4',
                return_traj=False,
                condition=x_batch_expanded
            )
            
            z_T = z_T.view(batch_size, n_samples, 1)
            z_T_np = z_T.cpu().numpy()
            
            samples_orig = scaler_y.inverse_transform(z_T_np.reshape(-1, 1)).reshape(batch_size, n_samples)
            all_samples.append(samples_orig)
    
    all_samples = np.concatenate(all_samples, axis=0)
    
    # Compute mode for each object
    z_modes = []
    for i in range(len(all_samples)):
        hist, bin_edges = np.histogram(all_samples[i], bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        z_modes.append(bin_centers[np.argmax(hist)])
    
    z_modes = np.array(z_modes)
    
    return object_ids, z_modes, all_samples


def save_predictions_qp(object_ids, z_modes, samples, z_grid, output_file):
    import qp
    
    n_objects = len(object_ids)
    n_bins = len(z_grid) - 1
    bin_centers = (z_grid[:-1] + z_grid[1:]) / 2
    
    pdfs = np.zeros((n_objects, n_bins))
    for i in range(n_objects):
        hist, _ = np.histogram(samples[i], bins=z_grid)
        pdfs[i] = hist / (hist.sum() + 1e-10)
    
    ensemble = qp.interp.create_ensemble(
        xvals=bin_centers,
        yvals=pdfs,
        ancil={
            'object_id': object_ids,
            'zmode': z_modes
        }
    )
    
    ensemble.write_to(output_file)
    print(f"Saved to: {output_file}")


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """
    
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    try:
        os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"))
    except Exception:
        pass

    try:
        os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"))
    except Exception:
        pass

    request.addfinalizer(teardown_submit_area)

    return 0


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 1

    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    import qp
    import torch
    import numpy as np
    import tables_io
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    
    
    print(f"Run Taskset 1 - Estimation Only")
    print(f"  Model: {model_file}")
    print(f"  Test: {test_file}")
    print(f"  Output: {output_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    
    # Initialize model
    model = ConditionalResNetWithTimeEmbedding(
        z_dim=1, t_dim=1, x_dim=6,
        hidden_dim=512, num_blocks=5, dropout=0.1, time_emb_dim=32
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Reconstruct scalers from baked-in parameters
    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint['scaler_X_mean']
    scaler_X.scale_ = checkpoint['scaler_X_scale']
    
    scaler_y = StandardScaler()
    scaler_y.mean_ = checkpoint['scaler_y_mean']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    
    # Generate predictions
    z_grid = np.linspace(0, 3.0, 301)
    object_ids, z_modes, samples = generate_predictions_for_test(
        model, test_file, scaler_X, scaler_y, device,
        n_samples=500, n_ode_steps=20
    )
    
    # Save predictions in qp format
    save_predictions_qp(object_ids, z_modes, samples, z_grid, output_file)
    print(f"Estimation complete!")
    print(f"  Objects: {len(object_ids)}")
    return


def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 1

    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    train_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be trained
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return


def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 1

    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    import qp
    import torch
    import numpy as np
    import tables_io
    from pathlib import Path
    from sklearn.preprocessing import StandardScaler
    
    
    print(f"Run Taskset 2 - Estimation Only")
    print(f"  Model: {model_file}")
    print(f"  Test: {test_file}")
    print(f"  Output: {output_file}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    
    # Initialize model
    model = ConditionalResNetWithTimeEmbedding(
        z_dim=1, t_dim=1, x_dim=6,
        hidden_dim=512, num_blocks=5, dropout=0.1, time_emb_dim=32
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Reconstruct scalers from baked-in parameters
    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint['scaler_X_mean']
    scaler_X.scale_ = checkpoint['scaler_X_scale']
    
    scaler_y = StandardScaler()
    scaler_y.mean_ = checkpoint['scaler_y_mean']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    
    # Generate predictions
    z_grid = np.linspace(0, 3.0, 301)
    object_ids, z_modes, samples = generate_predictions_for_test(
        model, test_file, scaler_X, scaler_y, device,
        n_samples=500, n_ode_steps=20
    )
    
    # Save predictions in qp format
    save_predictions_qp(object_ids, z_modes, samples, z_grid, output_file)
    print(f"Estimation complete!")
    print(f"  Objects: {len(object_ids)}")
    return


def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 1

    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """


def test_example_taskset_1(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson for Taskset 1

    You should not need to change this function
    """
    
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_example_taskset_2(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson for Taskset 2

    You should not need to change this function
    """

    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )
