import os
from pathlib import Path
import pytest
import h5py
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # Enable CuDNN auto-tuner for hardware-specific optimizations
    torch.backends.cudnn.benchmark = True
except ImportError:
   pass

import qp  
from scipy.stats import norm

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils


SUBMISSION_NAME: str = "Cin_zs"
SUBMISSION_URL: str = "https://www.dropbox.com/scl/fi/fw39q1z4pxk5d6ffz22ur/Cin_zs_final_submission.tar.gz?rlkey=sqkkh21jrhm2m4epfzdesh4da&st=9hg4nobp&dl=1" 

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    try:
        os.makedirs(os.path.join(SUBMIT_DIR, "output"), exist_ok=True)
    except FileExistsError:
        pass
    request.addfinalizer(teardown_submit_area)
    return 0

# =========================================================================
# CORE MDN PIPELINE 
# =========================================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, data):
        self.mean = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        self.std[self.std == 0] = 1.0  
    def transform(self, data):
        return (data - self.mean) / self.std

class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_features, num_components=10, hidden_sizes=(256,)*7, dropout_rate=0.15):
        super().__init__()
        self.num_components = num_components
        layers = []
        dim = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            dim = h
        self.trunk = nn.Sequential(*layers)
        self.pi_head = nn.Linear(dim, num_components)
        self.mu_head = nn.Linear(dim, num_components)
        self.sigma_head = nn.Linear(dim, num_components)
        self.reg_head = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        pi = torch.softmax(self.pi_head(h), dim=1)
        mu = self.mu_head(h)
        
        # PREVENTS OVERFLOW: Clamp the exponent to a max of 10.0
        sigma_raw = torch.clamp(self.sigma_head(h), max=10.0) 
        sigma = torch.exp(sigma_raw) + 0.001  
        
        z_reg = self.reg_head(h)
        return pi, mu, sigma, z_reg

def mdn_loss_fn(pi, mu, sigma, target, weights=None):
    target = target.expand_as(mu)
    log_scale = torch.log(sigma)
    var = sigma ** 2
    log_norm = -0.5 * np.log(2.0 * np.pi) - log_scale - 0.5 * ((target - mu) ** 2) / var
    log_prob = torch.log(pi + 1e-8) + log_norm
    loss = -torch.logsumexp(log_prob, dim=1)
    if weights is not None:
        loss = loss * weights
    return loss.mean()

def optimize_temperature(pi, mu, sigma, z_true, temp_range=np.linspace(0.4, 1.5, 50)):
    best_t = 1.0
    best_rmse_pit = float('inf')
    N, K = pi.shape
    
    for T in temp_range:
        sigma_adj = sigma * T
        pit_adj = np.zeros(N)
        for k in range(K):
            pit_adj += pi[:, k] * norm.cdf(z_true, loc=mu[:, k], scale=sigma_adj[:, k])
        pit_hist, _ = np.histogram(pit_adj, bins=100, range=(0, 1), density=True)
        rmse_pit = np.sqrt(np.mean((pit_hist - 1.0)**2))
        
        if rmse_pit < best_rmse_pit:
            best_rmse_pit = rmse_pit
            best_t = T
    return best_t

def load_and_engineer_features(file_path, has_redshift=True):
    band_cols = [f"mag_{b}" for b in ["u_lsst", "g_lsst", "r_lsst", "i_lsst", "z_lsst", "y_lsst", "Y_roman", "J_roman", "H_roman"]]
    err_cols = [f"{name}_err" for name in band_cols]
    
    with h5py.File(file_path, 'r') as hdf:
        x = np.stack([hdf[b][()] for b in band_cols], axis=1)
        mag_errors = np.stack([hdf[e][()] for e in err_cols], axis=1)
        z = hdf["redshift"][()] if has_redshift and "redshift" in hdf else None

    bad_mask = ~np.isfinite(x) | ~np.isfinite(mag_errors) | (mag_errors <= 0)
    x[bad_mask] = 30.0 
    mag_errors[bad_mask] = 10.0

    u_g, g_r, r_i, i_z, z_y = x[:,0]-x[:,1], x[:,1]-x[:,2], x[:,2]-x[:,3], x[:,3]-x[:,4], x[:,4]-x[:,5]
    colors = np.stack([u_g, g_r, r_i, i_z, z_y], axis=1)

    eps = 1e-8 
    J_H = x[:, 7] - x[:, 8]
    slope_ratio_1 = J_H / (u_g + eps)
    slope_ratio_2 = i_z / (g_r + eps)
    mag_color_cross = x[:, 2] * u_g
    
    err_J_H = np.sqrt(mag_errors[:, 7]**2 + mag_errors[:, 8]**2)
    weighted_JH_color = J_H * (1.0 / (err_J_H + eps))
    
    err_z_y = np.sqrt(mag_errors[:, 4]**2 + mag_errors[:, 5]**2)
    weighted_zy_color = z_y * (1.0 / (err_z_y + eps))

    optical_break_proxy = (g_r) * (r_i)

    adv_features = np.stack([slope_ratio_1, slope_ratio_2, mag_color_cross, weighted_JH_color, weighted_zy_color, optical_break_proxy], axis=1)
    return np.concatenate([x, mag_errors, colors, adv_features], axis=1), z

# =========================================================================
# TASK SET 2 - DATA AUGMENTATION TOOLS
# =========================================================================

def generate_error_profile(file_path, mag_col="mag_i_lsst", bins=np.arange(18, 26.5, 0.5)):
    BANDS = ["u_lsst", "g_lsst", "r_lsst", "i_lsst", "z_lsst", "y_lsst", "Y_roman", "J_roman", "H_roman"]
    with h5py.File(file_path, 'r') as hdf:
        ref_mag = hdf[mag_col][()]
        valid_mask = np.isfinite(ref_mag) & (ref_mag > 15) & (ref_mag < 30)
        df = pd.DataFrame({"ref_mag": ref_mag[valid_mask]})
        for band in BANDS:
            err_name = f"mag_{band}_err"
            if err_name in hdf:
                df[err_name] = hdf[err_name][()][valid_mask]
    df['mag_bin'] = pd.cut(df['ref_mag'], bins=bins)
    profile = df.groupby('mag_bin', observed=False).median().drop(columns=['ref_mag'])
    profile.index = [f"{inter.left + (inter.right - inter.left)/2:.2f}" for inter in profile.index]
    return profile

def empirical_augmentation(X_np, z_np, profile_df, target_mag_range=(24.0, 26.5), num_copies=1):
    bin_centers = profile_df.index.values.astype(float)
    aug_X, aug_z = [], []
    for _ in range(num_copies):
        X_new, z_new = X_np.copy(), z_np.copy()
        current_i_mag = X_new[:, 3] 
        target_i_mag = np.random.uniform(target_mag_range[0], target_mag_range[1], size=len(X_new))
        delta_mag = np.clip(target_i_mag - current_i_mag, 0, None)
        X_new[:, 0:9] += delta_mag[:, None]
        
        bin_indices = np.abs(X_new[:, 3, None] - bin_centers).argmin(axis=1)
        for b in range(9):
            band_name = ["u_lsst", "g_lsst", "r_lsst", "i_lsst", "z_lsst", "y_lsst", "Y_roman", "J_roman", "H_roman"][b]
            err_col_name = f"mag_{band_name}_err"
            if err_col_name in profile_df.columns:
                new_errors = profile_df[err_col_name].values[bin_indices]
                X_new[:, 9 + b] = new_errors
                noise = np.random.normal(loc=0.0, scale=new_errors)
                X_new[:, b] += noise
                
        u_g, g_r = X_new[:,0]-X_new[:,1], X_new[:,1]-X_new[:,2]
        r_i, i_z = X_new[:,2]-X_new[:,3], X_new[:,3]-X_new[:,4]
        z_y = X_new[:,4]-X_new[:,5]
        X_new[:, 18:23] = np.stack([u_g, g_r, r_i, i_z, z_y], axis=1)
        eps = 1e-8
        J_H = X_new[:, 7] - X_new[:, 8]
        X_new[:, 23] = J_H / (u_g + eps) 
        X_new[:, 24] = i_z / (g_r + eps) 
        X_new[:, 25] = X_new[:, 2] * u_g 
        err_J_H = np.sqrt(X_new[:, 16]**2 + X_new[:, 17]**2)
        X_new[:, 26] = J_H / (err_J_H + eps) 
        err_z_y = np.sqrt(X_new[:, 13]**2 + X_new[:, 14]**2)
        X_new[:, 27] = z_y / (err_z_y + eps) 
        X_new[:, 28] = (g_r) * (r_i) 
        
        bad_mask = ~np.isfinite(X_new)
        X_new[bad_mask] = 0.0
        aug_X.append(X_new)
        aug_z.append(z_new)
    return np.vstack(aug_X), np.concatenate(aug_z)


# =========================================================================
# TASK SET 1 EXECUTORS
# =========================================================================

def run_taskset_1_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_file_str = str(test_file).lower()
    
    model_name = "pz_challenge_taskset_1_flagship_pz_model_1yr.pt"
    if "flagship" in test_file_str and "10yr" in test_file_str: model_name = "pz_challenge_taskset_1_flagship_pz_model_10yr.pt"
    elif "cardinal" in test_file_str and "1yr" in test_file_str: model_name = "pz_challenge_taskset_1_cardinal_pz_model_1yr.pt"
    elif "cardinal" in test_file_str and "10yr" in test_file_str: model_name = "pz_challenge_taskset_1_cardinal_pz_model_10yr.pt"
    
    X_test_np, _ = load_and_engineer_features(test_file, has_redshift=False)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    
    scaler_dict = torch.load(os.path.join(SUBMIT_DIR, f"scaler_{model_name}"), map_location=device, weights_only=False)
    X_scaled = (X_test_t - scaler_dict['mean']) / scaler_dict['std']
    X_scaled = torch.nan_to_num(X_scaled, nan=0.0).clamp(-20.0, 20.0)
    best_t = scaler_dict.get('best_t', 1.0) 
    
    base_model = MixtureDensityNetwork(in_features=29, num_components=10).to(device)
    
    try:
        raw_state_dict = torch.load(os.path.join(SUBMIT_DIR, model_name), map_location=device)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
        base_model.load_state_dict(clean_state_dict)
    except FileNotFoundError:
        pass 
        
    try:
        model = torch.compile(base_model)
    except Exception:
        model = base_model
    
    model.eval()
    with torch.no_grad():
        pi, mu, sigma, z_reg = model(X_scaled) 
        
    sigma_calibrated = sigma.cpu().numpy() * best_t

    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {
        "object_id": obj_ids,
        "zmode": z_reg.cpu().numpy().flatten()
    }

    qp_ensemble = qp.Ensemble(
        qp.mixmod, 
        data=dict(weights=pi.cpu().numpy(), means=mu.cpu().numpy(), stds=sigma_calibrated),
        ancil=ancil_data
    )
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    qp_ensemble.write_to(str(output_file))


def run_taskset_1_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train_np, z_train_np = load_and_engineer_features(train_file, has_redshift=True)
    
    val_size = int(0.2 * len(X_train_np))
    indices = torch.randperm(len(X_train_np)).numpy()
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train_t = torch.tensor(X_train_np[train_idx], dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_train_np[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_train_np[val_idx], dtype=torch.float32, device=device)
    z_val_t = torch.tensor(z_train_np[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)

    scaler = StandardScaler()
    scaler.fit(X_train_t)
    X_train_scaled = scaler.transform(X_train_t)
    X_val_scaled = scaler.transform(X_val_t)

    base_model = MixtureDensityNetwork(in_features=29, num_components=10).to(device)
    try:
        model = torch.compile(base_model)
    except Exception:
        model = base_model
        
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    
    batch_size = 4096
    best_val = float('inf')
    best_state = None
    patience = 40
    epochs_no_improve = 0
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(X_train_scaled.shape[0], device=device)
        for i in range(0, X_train_scaled.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad(set_to_none=True)
            pi, mu, sigma, z_reg = model(X_train_scaled[idx])
            loss = mdn_loss_fn(pi, mu, sigma, z_train_t[idx]) + 50.0 * nn.functional.mse_loss(z_reg, z_train_t[idx])
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pi_v, mu_v, sigma_v, z_reg_v = model(X_val_scaled)
            val_loss = mdn_loss_fn(pi_v, mu_v, sigma_v, z_val_t) + 50.0 * nn.functional.mse_loss(z_reg_v, z_val_t)
            
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in best_state.items()}
        base_model.load_state_dict(clean_state_dict)

    base_model.eval()
    with torch.no_grad():
        pi_v, mu_v, sigma_v, _ = base_model(X_val_scaled)
    best_t = optimize_temperature(pi_v.cpu().numpy(), mu_v.cpu().numpy(), sigma_v.cpu().numpy(), z_train_np[val_idx])

    X_test_np, _ = load_and_engineer_features(test_file, has_redshift=False)
    X_test_scaled = scaler.transform(torch.tensor(X_test_np, dtype=torch.float32, device=device))
    X_test_scaled = torch.nan_to_num(X_test_scaled, nan=0.0).clamp(-20.0, 20.0)
    
    with torch.no_grad():
        pi_test, mu_test, sigma_test, z_reg_test = base_model(X_test_scaled)
        
    sigma_test_calibrated = sigma_test.cpu().numpy() * best_t
        
    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {
        "object_id": obj_ids,
        "zmode": z_reg_test.cpu().numpy().flatten()
    }

    qp_ensemble = qp.Ensemble(
        qp.mixmod, 
        data=dict(weights=pi_test.cpu().numpy(), means=mu_test.cpu().numpy(), stds=sigma_test_calibrated),
        ancil=ancil_data 
    )
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    qp_ensemble.write_to(str(output_file))


# =========================================================================
# TASK SET 2 EXECUTORS
# =========================================================================

def run_taskset_2_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_file_str = str(test_file).lower()
    
    model_name = "pz_challenge_taskset_2_flagship_pz_model_1yr.pt"
    if "flagship" in test_file_str and "10yr" in test_file_str: model_name = "pz_challenge_taskset_2_flagship_pz_model_10yr.pt"
    elif "cardinal" in test_file_str and "1yr" in test_file_str: model_name = "pz_challenge_taskset_2_cardinal_pz_model_1yr.pt"
    elif "cardinal" in test_file_str and "10yr" in test_file_str: model_name = "pz_challenge_taskset_2_cardinal_pz_model_10yr.pt"
    
    X_test_np, _ = load_and_engineer_features(test_file, has_redshift=False)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    
    scaler_dict = torch.load(os.path.join(SUBMIT_DIR, f"scaler_{model_name}"), map_location=device, weights_only=False)
    X_scaled = (X_test_t - scaler_dict['mean']) / scaler_dict['std']
    X_scaled = torch.nan_to_num(X_scaled, nan=0.0).clamp(-20.0, 20.0)
    best_t = scaler_dict.get('best_t', 1.0) 
    
    base_model = MixtureDensityNetwork(in_features=29, num_components=10).to(device)
    
    try:
        raw_state_dict = torch.load(os.path.join(SUBMIT_DIR, model_name), map_location=device)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
        base_model.load_state_dict(clean_state_dict)
    except FileNotFoundError:
        pass 
        
    try:
        model = torch.compile(base_model)
    except Exception:
        model = base_model
    
    model.eval()
    with torch.no_grad():
        pi, mu, sigma, z_reg = model(X_scaled) 
        
    sigma_calibrated = sigma.cpu().numpy() * best_t

    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {
        "object_id": obj_ids,
        "zmode": z_reg.cpu().numpy().flatten()
    }

    qp_ensemble = qp.Ensemble(
        qp.mixmod, 
        data=dict(weights=pi.cpu().numpy(), means=mu.cpu().numpy(), stds=sigma_calibrated),
        ancil=ancil_data
    )
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    qp_ensemble.write_to(str(output_file))


def run_taskset_2_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train_np, z_train_np = load_and_engineer_features(train_file, has_redshift=True)
    X_test_np, _ = load_and_engineer_features(test_file, has_redshift=False)
    
    # --- Holy Grail Augmentation ---
    profile_df = generate_error_profile(test_file)
    X_aug, z_aug = empirical_augmentation(X_train_np, z_train_np, profile_df, target_mag_range=(24.0, 26.5))
    
    X_final_np = np.vstack([X_train_np, X_aug])
    z_final_np = np.concatenate([z_train_np, z_aug])
    
    # --- Density Weights ---
    i_mag_test, i_mag_final = X_test_np[:, 3], X_final_np[:, 3]
    bins = np.linspace(16, 28, 40)
    test_hist, _ = np.histogram(i_mag_test, bins=bins, density=True)
    final_hist, _ = np.histogram(i_mag_final, bins=bins, density=True)
    
    weight_ratio = np.clip(test_hist / (final_hist + 1e-6), 0.1, 50.0) 
    final_bins = np.clip(np.digitize(i_mag_final, bins) - 1, 0, len(weight_ratio)-1)
    sample_weights_np = weight_ratio[final_bins]
    sample_weights_np = sample_weights_np / np.mean(sample_weights_np)
    
    val_size = int(0.2 * len(X_final_np))
    indices = torch.randperm(len(X_final_np)).numpy()
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_train_t = torch.tensor(X_final_np[train_idx], dtype=torch.float32, device=device)
    z_train_t = torch.tensor(z_final_np[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    w_train_t = torch.tensor(sample_weights_np[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    
    X_val_t = torch.tensor(X_final_np[val_idx], dtype=torch.float32, device=device)
    z_val_t = torch.tensor(z_final_np[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    w_val_t = torch.tensor(sample_weights_np[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)

    scaler = StandardScaler()
    scaler.fit(X_train_t)
    X_train_scaled = scaler.transform(X_train_t)
    X_val_scaled = scaler.transform(X_val_t)
    X_val_scaled = torch.nan_to_num(X_val_scaled, nan=0.0).clamp(-20.0, 20.0)

    base_model = MixtureDensityNetwork(in_features=29, num_components=10).to(device)
    try:
        model = torch.compile(base_model)
    except Exception:
        model = base_model
        
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    
    batch_size = 4096
    best_val = float('inf')
    best_state = None
    patience = 40
    epochs_no_improve = 0
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(X_train_scaled.shape[0], device=device)
        for i in range(0, X_train_scaled.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            
            # Re-sanitize batches during training
            X_batch = torch.nan_to_num(X_train_scaled[idx], nan=0.0).clamp(-20.0, 20.0)
            
            optimizer.zero_grad(set_to_none=True)
            pi, mu, sigma, z_reg = model(X_batch)
            
            mse_loss = nn.functional.mse_loss(z_reg, z_train_t[idx], reduction='none')
            mse_loss = (mse_loss * w_train_t[idx]).mean()
            
            loss = mdn_loss_fn(pi, mu, sigma, z_train_t[idx], w_train_t[idx]) + 50.0 * mse_loss
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pi_v, mu_v, sigma_v, z_reg_v = model(X_val_scaled)
            val_mse = nn.functional.mse_loss(z_reg_v, z_val_t, reduction='none')
            val_mse = (val_mse * w_val_t).mean()
            val_loss = mdn_loss_fn(pi_v, mu_v, sigma_v, z_val_t, w_val_t) + 50.0 * val_mse
            
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in best_state.items()}
        base_model.load_state_dict(clean_state_dict)

    base_model.eval()
    with torch.no_grad():
        pi_v, mu_v, sigma_v, _ = base_model(X_val_scaled)
    best_t = optimize_temperature(pi_v.cpu().numpy(), mu_v.cpu().numpy(), sigma_v.cpu().numpy(), z_final_np[val_idx])

    X_test_scaled = scaler.transform(torch.tensor(X_test_np, dtype=torch.float32, device=device))
    X_test_scaled = torch.nan_to_num(X_test_scaled, nan=0.0).clamp(-20.0, 20.0)
    
    with torch.no_grad():
        pi_test, mu_test, sigma_test, z_reg_test = base_model(X_test_scaled)
        
    sigma_test_calibrated = sigma_test.cpu().numpy() * best_t
        
    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {
        "object_id": obj_ids,
        "zmode": z_reg_test.cpu().numpy().flatten()
    }

    qp_ensemble = qp.Ensemble(
        qp.mixmod, 
        data=dict(weights=pi_test.cpu().numpy(), means=mu_test.cpu().numpy(), stds=sigma_test_calibrated),
        ancil=ancil_data 
    )
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    qp_ensemble.write_to(str(output_file))
    
# =========================================================================
# TEST EXECUTORS
# =========================================================================

def test_example_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_1_estimation_only, run_taskset_1_training_and_estimation)

def test_example_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_2_estimation_only, run_taskset_2_training_and_estimation)
