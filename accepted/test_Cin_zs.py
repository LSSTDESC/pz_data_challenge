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
    torch.backends.cudnn.benchmark = True
except ImportError:
   pass

import qp  
from scipy.stats import norm

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4
from pz_data_challenge import submit_utils


SUBMISSION_NAME: str = "Cin_zs"
SUBMISSION_URL: str = "https://www.dropbox.com/scl/fi/oto6cj716ecqwaoyf9slg/Cin_zs_final_submission_all_tasks.tar?rlkey=45pcsg19md0fdibiwwiru3at0&st=alx2t2z2&dl=1" 

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
        sigma_raw = torch.clamp(self.sigma_head(h), max=10.0) 
        sigma = torch.exp(sigma_raw) + 0.001  
        z_reg = self.reg_head(h)
        return pi, mu, sigma, z_reg

def mdn_loss_fn(pi, mu, sigma, target, weights=None, target_confidence=None):
    target = target.expand_as(mu)
    log_scale = torch.log(sigma)
    var = sigma ** 2
    log_norm = -0.5 * np.log(2.0 * np.pi) - log_scale - 0.5 * ((target - mu) ** 2) / var
    log_prob = torch.log(pi + 1e-8) + log_norm
    loss = -torch.logsumexp(log_prob, dim=1)
    if weights is not None:
        loss = loss * weights
    if target_confidence is not None:
        loss = loss * target_confidence
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

def compute_pz_median(pi, mu, sigma, z_grid_resolution=2000, z_max=4.0):
    N, K = pi.shape
    z_grid = np.linspace(0, z_max, z_grid_resolution)
    z_median = np.zeros(N)
    for i in range(N):
        cdf = np.zeros_like(z_grid)
        for k in range(K):
            dist = norm(loc=mu[i, k], scale=sigma[i, k])
            cdf += pi[i, k] * dist.cdf(z_grid)
        idx = np.searchsorted(cdf, 0.5)
        if idx > 0 and idx < z_grid_resolution:
            x0, x1 = cdf[idx-1], cdf[idx]
            y0, y1 = z_grid[idx-1], z_grid[idx]
            z_median[i] = y0 + (0.5 - x0) * (y1 - y0) / (x1 - x0)
        else:
            z_median[i] = z_grid[np.clip(idx, 0, z_grid_resolution-1)]
    return z_median

# =========================================================================
# DATA LOADING & FEATURE ENGINEERING
# =========================================================================

def load_raw_data(file_path, is_train, fuse_cosmos=False):
    band_cols = [f"mag_{b}" for b in ["u_lsst", "g_lsst", "r_lsst", "i_lsst", "z_lsst", "y_lsst", "Y_roman", "J_roman", "H_roman"]]
    err_cols = [f"{name}_err" for name in band_cols]
    
    with h5py.File(file_path, 'r') as hdf:
        mags = np.stack([hdf[b][()] for b in band_cols], axis=1)
        errs = np.stack([hdf[e][()] for e in err_cols], axis=1)
        if not is_train:
            return mags, errs, None, None
            
        z_spec = hdf["redshift"][()] 
        z_narrow = hdf["redshift_manyband"][()] if "redshift_manyband" in hdf else np.full_like(z_spec, np.nan)

    bad_mask = ~np.isfinite(mags) | ~np.isfinite(errs) | (errs <= 0)
    mags[bad_mask] = 30.0 
    errs[bad_mask] = 10.0

    z_combined = np.copy(z_spec)
    conf = np.ones_like(z_spec) 
    
    if fuse_cosmos:
        use_narrow_mask = ~np.isfinite(z_spec) & np.isfinite(z_narrow)
        z_combined[use_narrow_mask] = z_narrow[use_narrow_mask]
        conf[use_narrow_mask] = 0.3 

    valid_mask = np.isfinite(z_combined)
    return mags[valid_mask], errs[valid_mask], z_combined[valid_mask], conf[valid_mask]

def engineer_29_features(mags, mag_errors):
    u_g, g_r, r_i, i_z, z_y = mags[:,0]-mags[:,1], mags[:,1]-mags[:,2], mags[:,2]-mags[:,3], mags[:,3]-mags[:,4], mags[:,4]-mags[:,5]
    colors = np.stack([u_g, g_r, r_i, i_z, z_y], axis=1)
    
    eps = 1e-8 
    J_H = mags[:, 7] - mags[:, 8]
    slope_ratio_1 = J_H / (u_g + eps)
    slope_ratio_2 = i_z / (g_r + eps)
    mag_color_cross = mags[:, 2] * u_g
    err_J_H = np.sqrt(mag_errors[:, 7]**2 + mag_errors[:, 8]**2)
    weighted_JH_color = J_H * (1.0 / (err_J_H + eps))
    err_z_y = np.sqrt(mag_errors[:, 4]**2 + mag_errors[:, 5]**2)
    weighted_zy_color = z_y * (1.0 / (err_z_y + eps))
    optical_break_proxy = (g_r) * (r_i)

    adv_features = np.stack([slope_ratio_1, slope_ratio_2, mag_color_cross, weighted_JH_color, weighted_zy_color, optical_break_proxy], axis=1)
    X = np.concatenate([mags, mag_errors, colors, adv_features], axis=1)
    bad_mask = ~np.isfinite(X)
    X[bad_mask] = 0.0
    return X

def engineer_45_features(mags, mag_errors):
    u_g, g_r, r_i, i_z, z_y = mags[:,0]-mags[:,1], mags[:,1]-mags[:,2], mags[:,2]-mags[:,3], mags[:,3]-mags[:,4], mags[:,4]-mags[:,5]
    colors = np.stack([u_g, g_r, r_i, i_z, z_y], axis=1)

    flux = 10.0 ** (-0.4 * (mags - 25.0))
    
    d1, d2 = u_g - g_r, g_r - r_i
    d3, d4 = r_i - i_z, i_z - z_y
    color_derivatives = np.stack([d1, d2, d3, d4], axis=1)
    total_roughness = np.abs(d1) + np.abs(d2) + np.abs(d3) + np.abs(d4)

    J_H = mags[:, 7] - mags[:, 8]
    Y_J = mags[:, 6] - mags[:, 7]
    opt_nir_contradiction_1 = u_g - J_H
    opt_nir_contradiction_2 = g_r - Y_J

    eps = 1e-8 
    slope_ratio_1 = J_H / (u_g + eps)
    slope_ratio_2 = i_z / (g_r + eps)
    mag_color_cross = mags[:, 2] * u_g
    err_J_H = np.sqrt(mag_errors[:, 7]**2 + mag_errors[:, 8]**2)
    weighted_JH_color = J_H * (1.0 / (err_J_H + eps))
    err_z_y = np.sqrt(mag_errors[:, 4]**2 + mag_errors[:, 5]**2)
    weighted_zy_color = z_y * (1.0 / (err_z_y + eps))
    optical_break_proxy = (g_r) * (r_i)

    adv_features = np.stack([
        slope_ratio_1, slope_ratio_2, mag_color_cross, 
        weighted_JH_color, weighted_zy_color, optical_break_proxy,
        total_roughness, opt_nir_contradiction_1, opt_nir_contradiction_2
    ], axis=1)
    
    X = np.concatenate([mags, mag_errors, colors, flux, color_derivatives, adv_features], axis=1)
    bad_mask = ~np.isfinite(X)
    X[bad_mask] = 0.0
    return X

# =========================================================================
# AUGMENTATION ENGINES
# =========================================================================

def generate_error_profile(mags_test, errs_test, mag_col_idx=3, bins=np.arange(18, 26.5, 0.5)):
    ref_mag = mags_test[:, mag_col_idx]
    valid_mask = np.isfinite(ref_mag) & (ref_mag > 15) & (ref_mag < 30)
    
    df = pd.DataFrame({"ref_mag": ref_mag[valid_mask]})
    for b in range(9):
        df[f"err_{b}"] = errs_test[valid_mask, b]
            
    df['mag_bin'] = pd.cut(df['ref_mag'], bins=bins)
    profile = df.groupby('mag_bin', observed=False).median().drop(columns=['ref_mag'])
    profile.index = [f"{inter.left + (inter.right - inter.left)/2:.2f}" for inter in profile.index]
    return profile

def empirical_augmentation(mags_np, errs_np, z_np, conf_np, profile_df, target_mag_range=(24.0, 25.5), num_copies=1):
    bin_centers = profile_df.index.values.astype(float)
    aug_mags, aug_errs, aug_z, aug_conf = [], [], [], []
    for _ in range(num_copies):
        mags_new, errs_new = mags_np.copy(), np.zeros_like(errs_np)
        current_i_mag = mags_new[:, 3] 
        target_i_mag = np.random.uniform(target_mag_range[0], target_mag_range[1], size=len(mags_new))
        delta_mag = np.clip(target_i_mag - current_i_mag, 0, None)
        mags_new += delta_mag[:, None]
        
        bin_indices = np.abs(mags_new[:, 3, None] - bin_centers).argmin(axis=1)
        for b in range(9):
            if f"err_{b}" in profile_df.columns:
                new_errors = profile_df[f"err_{b}"].values[bin_indices]
                errs_new[:, b] = new_errors
                mags_new[:, b] += np.random.normal(loc=0.0, scale=new_errors)
                
        aug_mags.append(mags_new)
        aug_errs.append(errs_new)
        aug_z.append(z_np.copy())
        aug_conf.append(conf_np.copy())
    return np.vstack(aug_mags), np.vstack(aug_errs), np.concatenate(aug_z), np.concatenate(aug_conf)

def synthetic_blending(mags_np, errs_np, z_np, conf_np, num_blends=15000):
    idx_A = np.random.choice(len(mags_np), num_blends)
    idx_B = np.random.choice(len(mags_np), num_blends)
    
    mags_A, errs_A = mags_np[idx_A], errs_np[idx_A]
    mags_B, errs_B = mags_np[idx_B], errs_np[idx_B]
    
    flux_A = 10.0 ** (-0.4 * (mags_A - 25.0))
    flux_B = 10.0 ** (-0.4 * (mags_B - 25.0))
    
    flux_blend = flux_A + flux_B
    mags_blend = 25.0 - 2.5 * np.log10(flux_blend)
    errs_blend = np.sqrt(errs_A**2 + errs_B**2)
    
    mags_dual = np.vstack([mags_blend, mags_blend])
    errs_dual = np.vstack([errs_blend, errs_blend])
    z_dual = np.concatenate([z_np[idx_A], z_np[idx_B]])
    conf_dual = np.concatenate([conf_np[idx_A], conf_np[idx_B]]) * 0.4
    
    return mags_dual, errs_dual, z_dual, conf_dual

# =========================================================================
# UNIVERSAL TEST EXECUTOR HELPER
# =========================================================================

def execute_estimation(model_file: str | Path, test_file: str | Path, output_file: str | Path, taskset: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_file_str = str(test_file).lower()
    
    in_feats = 45 if taskset == 4 else 29
    
    sim = "flagship" if "flagship" in test_file_str else "cardinal"
    depth = "10yr" if "10yr" in test_file_str else "1yr"
    model_name = f"pz_challenge_taskset_{taskset}_{sim}_pz_model_{depth}.pt"
    
    m_test, e_test, _, _ = load_raw_data(test_file, is_train=False, fuse_cosmos=False)
    X_test_np = engineer_45_features(m_test, e_test) if taskset == 4 else engineer_29_features(m_test, e_test)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    
    scaler_dict = torch.load(os.path.join(SUBMIT_DIR, f"scaler_{model_name}"), map_location=device, weights_only=False)
    X_scaled = (X_test_t - scaler_dict['mean']) / scaler_dict['std']
    X_scaled = torch.nan_to_num(X_scaled, nan=0.0).clamp(-20.0, 20.0)
    best_t = scaler_dict.get('best_t', 1.0) 
    
    base_model = MixtureDensityNetwork(in_features=in_feats, num_components=10).to(device)
    try:
        raw_state_dict = torch.load(os.path.join(SUBMIT_DIR, model_name), map_location=device)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}
        base_model.load_state_dict(clean_state_dict)
    except FileNotFoundError:
        pass 
        
    model = torch.compile(base_model) if hasattr(torch, "compile") else base_model
    
    model.eval()
    with torch.no_grad():
        pi, mu, sigma, _ = model(X_scaled) 
        
    pi_test_np = pi.cpu().numpy()
    mu_test_np = mu.cpu().numpy()
    sigma_calibrated = sigma.cpu().numpy() * best_t
    z_median = compute_pz_median(pi_test_np, mu_test_np, sigma_calibrated)

    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {"object_id": obj_ids, "zmode": z_median}
    qp_ensemble = qp.Ensemble(qp.mixmod, data=dict(weights=pi_test_np, means=mu_test_np, stds=sigma_calibrated), ancil=ancil_data)
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    qp_ensemble.write_to(str(output_file))

def execute_training(train_file: str | Path, test_file: str | Path, output_file: str | Path, taskset: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fuse_cosmos = taskset in [3, 4]
    in_feats = 45 if taskset == 4 else 29
    
    m_train, e_train, z_train, c_train = load_raw_data(train_file, is_train=True, fuse_cosmos=fuse_cosmos)
    m_test, e_test, _, _ = load_raw_data(test_file, is_train=False, fuse_cosmos=False)

    mags_all, errs_all, z_all, conf_all = m_train, e_train, z_train, c_train

    if taskset > 1:
        profile_df = generate_error_profile(m_test, e_test)
        spec_mask = (c_train == 1.0)
        m_aug, e_aug, z_aug, c_aug = empirical_augmentation(
            m_train[spec_mask], e_train[spec_mask], z_train[spec_mask], c_train[spec_mask], profile_df, target_mag_range=(24.0, 25.5)
        )
        mags_all = np.vstack([mags_all, m_aug])
        errs_all = np.vstack([errs_all, e_aug])
        z_all = np.concatenate([z_all, z_aug])
        conf_all = np.concatenate([conf_all, c_aug])
        
        if taskset == 4:
            m_blend, e_blend, z_blend, c_blend = synthetic_blending(m_train, e_train, z_train, c_train)
            mags_all = np.vstack([mags_all, m_blend])
            errs_all = np.vstack([errs_all, e_blend])
            z_all = np.concatenate([z_all, z_blend])
            conf_all = np.concatenate([conf_all, c_blend])
            
        i_test, i_train = m_test[:, 3], mags_all[:, 3]
        bins = np.linspace(16, 28, 40)
        test_hist, _ = np.histogram(i_test, bins=bins, density=True)
        train_hist, _ = np.histogram(i_train, bins=bins, density=True)
        weight_ratio = np.clip(test_hist / (train_hist + 1e-6), 0.1, 50.0) 
        train_bins = np.clip(np.digitize(i_train, bins) - 1, 0, len(weight_ratio)-1)
        weights_all = weight_ratio[train_bins]
        weights_all = weights_all / np.mean(weights_all)
    else:
        weights_all = np.ones(len(z_all))

    X_train_np = engineer_45_features(mags_all, errs_all) if taskset == 4 else engineer_29_features(mags_all, errs_all)

    val_size = int(0.2 * len(X_train_np))
    indices = torch.randperm(len(X_train_np)).numpy()
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    scaler = StandardScaler()
    scaler.fit(torch.tensor(X_train_np[train_idx], dtype=torch.float32, device=device))
    
    X_train_scaled = scaler.transform(torch.tensor(X_train_np[train_idx], dtype=torch.float32, device=device))
    z_train_t = torch.tensor(z_all[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    w_train_t = torch.tensor(weights_all[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    c_train_t = torch.tensor(conf_all[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    
    X_val_scaled = scaler.transform(torch.tensor(X_train_np[val_idx], dtype=torch.float32, device=device))
    z_val_t = torch.tensor(z_all[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    w_val_t = torch.tensor(weights_all[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    c_val_t = torch.tensor(conf_all[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)

    X_val_scaled = torch.nan_to_num(X_val_scaled, nan=0.0).clamp(-20.0, 20.0)

    base_model = MixtureDensityNetwork(in_features=in_feats, num_components=10).to(device)
    model = torch.compile(base_model) if hasattr(torch, "compile") else base_model
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    
    batch_size = 4096
    best_val, best_state = float('inf'), None
    patience, epochs_no_improve = 40, 0
    
    for epoch in range(300):
        model.train()
        perm = torch.randperm(X_train_scaled.shape[0], device=device)
        for i in range(0, X_train_scaled.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            X_batch = torch.nan_to_num(X_train_scaled[idx], nan=0.0).clamp(-20.0, 20.0)
            
            optimizer.zero_grad(set_to_none=True)
            pi, mu, sigma, z_reg = model(X_batch)
            
            mse_loss = nn.functional.mse_loss(z_reg, z_train_t[idx], reduction='none')
            mse_loss = (mse_loss * w_train_t[idx] * c_train_t[idx]).mean()
            
            loss = mdn_loss_fn(pi, mu, sigma, z_train_t[idx], w_train_t[idx], c_train_t[idx]) + 50.0 * mse_loss
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pi_v, mu_v, sigma_v, z_reg_v = model(X_val_scaled)
            val_mse = nn.functional.mse_loss(z_reg_v, z_val_t, reduction='none')
            val_mse = (val_mse * w_val_t * c_val_t).mean()
            val_loss = mdn_loss_fn(pi_v, mu_v, sigma_v, z_val_t, w_val_t, c_val_t) + 50.0 * val_mse
            
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
        
    eval_mask = conf_all[val_idx] == 1.0
    best_t = optimize_temperature(
        pi_v.cpu().numpy()[eval_mask], 
        mu_v.cpu().numpy()[eval_mask], 
        sigma_v.cpu().numpy()[eval_mask], 
        z_all[val_idx][eval_mask]
    )

    X_test_np = engineer_45_features(m_test, e_test) if taskset == 4 else engineer_29_features(m_test, e_test)
    X_test_scaled = scaler.transform(torch.tensor(X_test_np, dtype=torch.float32, device=device))
    X_test_scaled = torch.nan_to_num(X_test_scaled, nan=0.0).clamp(-20.0, 20.0)
    
    with torch.no_grad():
        pi_test, mu_test, sigma_test, _ = base_model(X_test_scaled)
        
    pi_test_np = pi_test.cpu().numpy()
    mu_test_np = mu_test.cpu().numpy()
    sigma_test_calibrated = sigma_test.cpu().numpy() * best_t
    z_median_test = compute_pz_median(pi_test_np, mu_test_np, sigma_test_calibrated)
        
    with h5py.File(test_file, 'r') as f:
        obj_ids = f["object_id"][()]
        
    ancil_data = {"object_id": obj_ids, "zmode": z_median_test}
    qp_ensemble = qp.Ensemble(qp.mixmod, data=dict(weights=pi_test_np, means=mu_test_np, stds=sigma_test_calibrated), ancil=ancil_data)
    
    output_dir = os.path.dirname(str(output_file))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    qp_ensemble.write_to(str(output_file))


# =========================================================================
# TASK SET WRAPPERS
# =========================================================================

def run_taskset_1_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_estimation(model_file, test_file, output_file, taskset=1)

def run_taskset_1_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_training(train_file, test_file, output_file, taskset=1)

def run_taskset_2_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_estimation(model_file, test_file, output_file, taskset=2)

def run_taskset_2_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_training(train_file, test_file, output_file, taskset=2)

def run_taskset_3_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_estimation(model_file, test_file, output_file, taskset=3)

def run_taskset_3_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_training(train_file, test_file, output_file, taskset=3)

def run_taskset_4_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_estimation(model_file, test_file, output_file, taskset=4)

def run_taskset_4_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    execute_training(train_file, test_file, output_file, taskset=4)


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

def test_example_taskset_3(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_3(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_3_estimation_only, run_taskset_3_training_and_estimation)

def test_example_taskset_4(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_4(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_4_estimation_only, run_taskset_4_training_and_estimation)