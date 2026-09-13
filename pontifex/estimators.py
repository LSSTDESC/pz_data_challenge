import sys
import os
import math
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import joblib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1. Monkeypatch tables_io and numpy to resolve RAIL/numpy compatibility issues
import tables_io
sys.modules['tables_io.hdf5'] = tables_io.h5py
import tables_io.types
if not hasattr(tables_io.types, 'table_type') and hasattr(tables_io.types, 'tableType'):
    tables_io.types.table_type = tables_io.types.tableType
if not hasattr(tables_io.types, 'file_type') and hasattr(tables_io.types, 'fileType'):
    tables_io.types.file_type = tables_io.types.fileType
if not hasattr(tables_io.types, 'tableType') and hasattr(tables_io.types, 'table_type'):
    tables_io.types.tableType = tables_io.types.table_type
if not hasattr(tables_io.types, 'fileType') and hasattr(tables_io.types, 'file_type'):
    tables_io.types.fileType = tables_io.types.file_type

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# JAX ShapedArray compatibility patch for JAX >= 0.4.30 / Python 3.13
try:
    import jax
    import jax.core
    import inspect
    _orig_shaped_array_init = jax.core.ShapedArray.__init__
    _sig = inspect.signature(_orig_shaped_array_init)
    if "named_shape" not in _sig.parameters:
        def _compat_shaped_array_init(self, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return _orig_shaped_array_init(self, *args, **kwargs)
        jax.core.ShapedArray.__init__ = _compat_shaped_array_init
except Exception:
    pass

# 2. Import RAIL stages and MiniSom
import qp
from rail.core.data import TableHandle
from rail.estimation.algos import sklearn_neurnet, k_nearneigh, flexzboost
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
from rail.estimation.algos.pzflow_nf import PZFlowInformer, PZFlowEstimator
from rail.estimation.algos.gpz import GPzInformer, GPzEstimator
from rail.estimation.algos.lephare import LephareInformer, LephareEstimator
import lephare as lp
from minisom import MiniSom

import aion_pz
from .guard import sanitize_input_catalog

# Monkeypatching block starts here
def get_original_bands(keys):
    orig = []
    for k in keys:
        if k.startswith("mag_") and not k.endswith("_err"):
            orig.append(k)
    lsst_order = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    roman_order = ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
    sorted_orig = []
    for b in lsst_order + roman_order:
        if b in orig:
            sorted_orig.append(b)
    return sorted_orig

original_snn_make_color_data = sklearn_neurnet.make_color_data

def get_expected_features_count(estimator_name):
    import sys
    frame = sys._getframe(1)
    while frame:
        model_dict = frame.f_locals.get('model_dict', None)
        if model_dict is not None and isinstance(model_dict, dict):
            if estimator_name == 'aion' and 'aion_head' in model_dict:
                head = model_dict['aion_head']
                if head and 'clf' in head:
                    return head['clf'].n_features_in_
                if head and 'scaler' in head:
                    return head['scaler'].n_features_in_
            elif estimator_name == 'nn1' and 'model_nn1' in model_dict:
                m = model_dict['model_nn1']
                m_data = getattr(m, 'data', m)
                if hasattr(m_data, 'n_features_in_'):
                    return m_data.n_features_in_
            elif estimator_name == 'nn2' and 'model_nn2' in model_dict:
                m = model_dict['model_nn2']
                m_data = getattr(m, 'data', m)
                if hasattr(m_data, 'n_features_in_'):
                    return m_data.n_features_in_
            elif estimator_name == 'knn' and 'model_knn' in model_dict:
                m = model_dict['model_knn']
                m_data = getattr(m, 'data', m)
                if isinstance(m_data, dict) and 'kdtree' in m_data:
                    if hasattr(m_data['kdtree'], 'data'):
                        return m_data['kdtree'].data.shape[1]
            elif estimator_name == 'fzboost' and 'model_fzboost' in model_dict:
                m = model_dict['model_fzboost']
                m_data = getattr(m, 'data', m)
                if hasattr(m_data, 'model'):
                    inner = m_data.model
                    if hasattr(inner, 'models') and hasattr(inner.models, 'estimators_'):
                        estimators = inner.models.estimators_
                        if len(estimators) > 0 and hasattr(estimators[0], 'n_features_in_'):
                            return estimators[0].n_features_in_
        
        self_obj = frame.f_locals.get('self', None)
        if self_obj is not None:
            model_handle = getattr(self_obj, 'model', None)
            if model_handle is not None:
                model_data = getattr(model_handle, 'data', model_handle)
                if model_data is not None:
                    if hasattr(model_data, 'n_features_in_'):
                        return model_data.n_features_in_
                    if hasattr(model_data, 'model'):
                        inner = model_data.model
                        if hasattr(inner, 'models') and hasattr(inner.models, 'estimators_'):
                            estimators = inner.models.estimators_
                            if len(estimators) > 0 and hasattr(estimators[0], 'n_features_in_'):
                                return estimators[0].n_features_in_
                    if isinstance(model_data, dict) and 'kdtree' in model_data:
                        if hasattr(model_data['kdtree'], 'data'):
                            return model_data['kdtree'].data.shape[1]
        frame = frame.f_back
    return None

def snn_make_color_data_patched(data_dict, bands, ref_band, nondet_val):
    base_features = original_snn_make_color_data(data_dict, bands, ref_band, nondet_val)
    
    snr_cols = [f"snr_{b}" for b in bands if f"snr_{b}" in data_dict]
    col_err_cols = [f"color_err_{bands[i]}_{bands[i+1]}" for i in range(len(bands) - 1) if f"color_err_{bands[i]}_{bands[i+1]}" in data_dict]
    temp_cols = ["bpz_chi2_min", "bpz_t_ml", "bpz_z_tb"]
    temp_cols = [c for c in temp_cols if c in data_dict]
    
    extra_features = []
    for col in snr_cols + col_err_cols + temp_cols:
        extra_features.append(data_dict[col])
        
    if len(extra_features) > 0:
        extra_matrix = np.column_stack(extra_features)
        extra_matrix = np.nan_to_num(extra_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.column_stack([base_features, extra_matrix])
    else:
        combined = base_features

    n_expected = get_expected_features_count('nn1')
    if n_expected is None:
        n_expected = get_expected_features_count('nn2')
    
    if n_expected is not None:
        if n_expected <= combined.shape[1]:
            return combined[:, :n_expected]
    return combined

sklearn_neurnet.make_color_data = snn_make_color_data_patched

original_fz_make_color_data = flexzboost.make_color_data

def fz_make_color_data_patched(data_dict, bands, err_bands, ref_band, include_mag_err=False):
    base_features = original_fz_make_color_data(data_dict, bands, err_bands, ref_band, include_mag_err)
    
    snr_cols = [f"snr_{b}" for b in bands if f"snr_{b}" in data_dict]
    col_err_cols = [f"color_err_{bands[i]}_{bands[i+1]}" for i in range(len(bands) - 1) if f"color_err_{bands[i]}_{bands[i+1]}" in data_dict]
    temp_cols = ["bpz_chi2_min", "bpz_t_ml", "bpz_z_tb"]
    temp_cols = [c for c in temp_cols if c in data_dict]
    
    extra_features = []
    for col in snr_cols + col_err_cols + temp_cols:
        extra_features.append(data_dict[col])
        
    if len(extra_features) > 0:
        extra_matrix = np.column_stack(extra_features)
        extra_matrix = np.nan_to_num(extra_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.column_stack([base_features, extra_matrix])
    else:
        combined = base_features

    n_expected = get_expected_features_count('fzboost')
    if n_expected is not None:
        if n_expected <= combined.shape[1]:
            return combined[:, :n_expected]
    return combined

flexzboost.make_color_data = fz_make_color_data_patched

original_knn_computecolordata = k_nearneigh._computecolordata

def knn_computecolordata_patched(df, ref_column_name, column_names, only_color):
    base_features = original_knn_computecolordata(df, ref_column_name, column_names, only_color)
    
    snr_cols = [f"snr_{b}" for b in column_names if f"snr_{b}" in df.columns]
    col_err_cols = [f"color_err_{column_names[i]}_{column_names[i+1]}" for i in range(len(column_names) - 1) if f"color_err_{column_names[i]}_{column_names[i+1]}" in df.columns]
    temp_cols = ["bpz_chi2_min", "bpz_t_ml", "bpz_z_tb"]
    temp_cols = [c for c in temp_cols if c in df.columns]
    
    extra_features = []
    for col in snr_cols + col_err_cols + temp_cols:
        extra_features.append(df[col].to_numpy())
        
    if len(extra_features) > 0:
        extra_matrix = np.column_stack(extra_features)
        extra_matrix = np.nan_to_num(extra_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.column_stack([base_features, extra_matrix])
    else:
        combined = base_features

    n_expected = get_expected_features_count('knn')
    if n_expected is not None:
        if n_expected <= combined.shape[1]:
            return combined[:, :n_expected]
    return combined

k_nearneigh._computecolordata = knn_computecolordata_patched

original_aion_build_design_matrix = aion_pz.build_design_matrix

def aion_build_design_matrix_patched(model, codec_manager, data, device, **kw):
    base_matrix = original_aion_build_design_matrix(model, codec_manager, data, device, **kw)
    orig_bands = get_original_bands(list(data.keys()))
    
    snr_cols = [f"snr_{b}" for b in orig_bands if f"snr_{b}" in data]
    col_err_cols = [f"color_err_{orig_bands[i]}_{orig_bands[i+1]}" for i in range(len(orig_bands) - 1) if f"color_err_{orig_bands[i]}_{orig_bands[i+1]}" in data]
    temp_cols = ["bpz_chi2_min", "bpz_t_ml", "bpz_z_tb"]
    temp_cols = [c for c in temp_cols if c in data]
    
    extra_features = []
    for col in snr_cols + col_err_cols + temp_cols:
        extra_features.append(np.asarray(data[col], dtype=np.float32))
        
    if len(extra_features) > 0:
        extra_matrix = np.column_stack(extra_features)
        extra_matrix = np.nan_to_num(extra_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.concatenate([base_matrix, extra_matrix], axis=1).astype("float32")
    else:
        combined = base_matrix

    n_expected = get_expected_features_count('aion')
    if n_expected is not None:
        if n_expected <= combined.shape[1]:
            return combined[:, :n_expected]
    return combined

aion_pz.build_design_matrix = aion_build_design_matrix_patched

# Configure loggers for estimators
FlexZBoostInformer.log = logging.getLogger("FlexZBoostInformer")
FlexZBoostEstimator.log = logging.getLogger("FlexZBoostEstimator")
GPzInformer.log = logging.getLogger("GPzInformer")
GPzEstimator.log = logging.getLogger("GPzEstimator")

ZMAX = 3.0
NZ = 301
Z_GRID = np.linspace(0.0, ZMAX, NZ)
Z_CENTERS = 0.5 * (Z_GRID[:-1] + Z_GRID[1:])
dz = Z_GRID[1] - Z_GRID[0]


def make_clean_stage(stage_class: Any, name: str, **kwargs: Any) -> Any:
    """Instantiate a RAIL stage, set allow_overwrite=True, and clear its data store."""
    stage = stage_class.make_stage(name=name, **kwargs)
    if hasattr(stage, "data_store"):
        dict.__setattr__(stage.data_store, "allow_overwrite", True)
        stage.data_store.clear()
    return stage


def get_bands_and_ref(data_keys: List[str], taskset_id: int = None) -> Tuple[List[str], str, bool]:
    """Determine the band list and reference band based on catalog columns and taskset designation."""
    if taskset_id == 2:
        return ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman'], 'mag_J_roman', True
    elif taskset_id in [1, 4]:
        return ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'], 'mag_i_lsst', False
    
    has_roman = 'mag_Y_roman' in data_keys
    has_lsst = 'mag_i_lsst' in data_keys or 'mag_u_lsst' in data_keys
    
    if has_roman and has_lsst:
        bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst', 'mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
        return bands, 'mag_i_lsst', False
    elif has_roman:
        return ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman'], 'mag_J_roman', True
    else:
        return ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'], 'mag_i_lsst', False


def extract_features(data_dict: Dict[str, np.ndarray], bands: List[str], ref_band: str) -> np.ndarray:
    """Extract magnitude, adjacent color, SNR, and color error features for hybrid KNN weighting."""
    features = []
    
    # 1. Magnitudes
    for band in bands:
        mag = data_dict[band].copy()
        mag = np.where(np.isnan(mag), np.nanmedian(mag) if not np.isnan(np.nanmedian(mag)) else 99.0, mag)
        features.append(mag)
        
    # 2. Adjacent Colors
    for i in range(len(bands) - 1):
        col = (data_dict[bands[i]] - data_dict[bands[i+1]]).copy()
        col = np.where(np.isnan(col), np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0.0, col)
        features.append(col)
        
    # 3. SNR of Magnitudes
    for band in bands:
        err_col = band + "_err"
        if err_col in data_dict:
            err = np.asarray(data_dict[err_col], dtype=float)
            err = np.where(err <= 0, 1e-4, err)
            err = np.where(np.isnan(err), np.nanmedian(err) if not np.isnan(np.nanmedian(err)) else 0.1, err)
            snr = 1.086 / err
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(snr)
            
    # 4. Color Errors
    for i in range(len(bands) - 1):
        b1 = bands[i]
        b2 = bands[i+1]
        err1_col = b1 + "_err"
        err2_col = b2 + "_err"
        if err1_col in data_dict and err2_col in data_dict:
            err1 = np.asarray(data_dict[err1_col], dtype=float)
            err2 = np.asarray(data_dict[err2_col], dtype=float)
            err1 = np.where(np.isnan(err1), np.nanmedian(err1) if not np.isnan(np.nanmedian(err1)) else 0.1, err1)
            err2 = np.where(np.isnan(err2), np.nanmedian(err2) if not np.isnan(np.nanmedian(err2)) else 0.1, err2)
            col_err = np.sqrt(err1**2 + err2**2)
            col_err = np.nan_to_num(col_err, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(col_err)
            
    return np.column_stack(features)


def get_som_pdfs(train_dict: Dict[str, np.ndarray], test_dict: Dict[str, np.ndarray],
                 bands: List[str], ref_band: str, z_grid: np.ndarray, n_dim=14, m_dim=14, max_iter=5000,
                 sigma=2.0, learning_rate=0.5) -> np.ndarray:
    """Train MiniSom and return PDFs on the test set."""
    def get_features(d):
        numcols = len(bands)
        coldata = np.array(d[ref_band])
        for i in range(numcols - 1):
            tmpcolor = d[bands[i]] - d[bands[i+1]]
            coldata = np.vstack((coldata, tmpcolor))
        return coldata.T

    train_feat = get_features(train_dict)
    test_feat = get_features(test_dict)
    train_feat = np.nan_to_num(train_feat, nan=25.0, posinf=25.0, neginf=25.0)
    test_feat = np.nan_to_num(test_feat, nan=25.0, posinf=25.0, neginf=25.0)

    som = MiniSom(n_dim, m_dim, train_feat.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.pca_weights_init(train_feat)
    som.train(train_feat, max_iter, verbose=False)

    train_winners = np.array([som.winner(x) for x in train_feat])
    test_winners = np.array([som.winner(x) for x in test_feat])
    train_pixels = np.ravel_multi_index(train_winners.T, (n_dim, m_dim))
    test_pixels = np.ravel_multi_index(test_winners.T, (n_dim, m_dim))

    pixel_pdfs = {}
    global_hist, _ = np.histogram(train_dict['redshift'], bins=z_grid)
    global_pdf = global_hist / (np.sum(global_hist) + 1e-15)

    for pix in range(n_dim * m_dim):
        mask = (train_pixels == pix)
        if mask.sum() > 5:
            hist, _ = np.histogram(train_dict['redshift'][mask], bins=z_grid)
            pixel_pdfs[pix] = hist / (np.sum(hist) + 1e-15)
        else:
            pixel_pdfs[pix] = global_pdf

    test_pdfs = np.zeros((len(test_dict['object_id']), len(Z_CENTERS)))
    for i, pix in enumerate(test_pixels):
        test_pdfs[i] = pixel_pdfs[pix]

    test_pdfs = gaussian_filter1d(test_pdfs, sigma=1.0, axis=1)
    row_sums = test_pdfs.sum(axis=1, keepdims=True)
    test_pdfs = np.where(row_sums > 0, test_pdfs / row_sums, 1.0 / len(Z_CENTERS))
    return test_pdfs


def compute_expert_weights_knn(train_dict: Dict[str, np.ndarray], train_pdfs: List[np.ndarray],
                               z_centers: np.ndarray, bands: List[str], ref_band: str,
                               blend_gating_mode: str = 'ground_truth_informed') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate point-estimate and multi-peak errors for each expert and prepare PCA feature space for KNN weighting."""
    from sklearn.decomposition import PCA
    train_features = extract_features(train_dict, bands, ref_band)
    features_mean = np.mean(train_features, axis=0)
    features_std = np.std(train_features, axis=0)
    features_std = np.where(features_std == 0, 1.0, features_std)
    train_features_norm = (train_features - features_mean) / features_std

    # Fit PCA to capture at least 95% of total variance dynamically across tasksets (Optunity PSO Tuned AION Latent Space Configuration)
    train_features_knn = train_features_norm
    pca_95 = PCA(n_components=0.95, random_state=42).fit(train_features_norm)

    # Taskset-specific Optunity PSO tuned default kNN hyperparameters
    # TS1: k=17, bw=0.873 | TS2: k=17, bw=0.470 | TS3: k=35, bw=0.546 | TS4: k=29, bw=1.109
    optunity_tuned_defaults = {
        1: {"k": 17, "bw_mult": 0.873},
        2: {"k": 17, "bw_mult": 0.470},
        3: {"k": 35, "bw_mult": 0.546},
        4: {"k": 29, "bw_mult": 1.109}
    }

    train_errors = []
    has_z2 = "redshift_manyband" in train_dict
    z1_true = train_dict['redshift']
    z2_true = train_dict['redshift_manyband'] if has_z2 else None

    for pdf in train_pdfs:
        z_mode = z_centers[np.argmax(pdf, axis=1)]
        err = np.abs(z_mode - z1_true) / (1.0 + z1_true)

        # Ground-truth-informed blend optimization during INFORM stage (NO leakage during inference)
        if blend_gating_mode == 'ground_truth_informed' and has_z2:
            valid_z2 = np.isfinite(z2_true) & (z2_true > 0)
            proj_blends = valid_z2 & (np.abs(z1_true - z2_true) > 0.10)
            
            # For true projection blends, reward experts that capture the secondary peak z2_true
            if np.sum(proj_blends) > 0:
                # Find secondary peak in PDF for blend objects
                for idx in np.where(proj_blends)[0]:
                    p_i = pdf[idx]
                    top2_idx = np.argsort(p_i)[-2:]
                    z_sec = z_centers[top2_idx[0]]
                    err_sec = np.abs(z_sec - z2_true[idx]) / (1.0 + z2_true[idx])
                    # Combined multi-modal quality score for blend objects
                    err[idx] = 0.5 * err[idx] + 0.5 * min(err[idx], err_sec)

        train_errors.append(err)
    train_errors = np.array(train_errors)
    
    pca_mean_std_pkg = {
        "mean": features_mean,
        "std": features_std,
        "pca": pca_95,
        "blend_gating_mode": blend_gating_mode,
        "optunity_defaults": optunity_tuned_defaults
    }
    return train_features_knn, train_errors, pca_mean_std_pkg, None


def apply_expert_weights_knn(val_dict: Dict[str, np.ndarray], val_pdfs: List[np.ndarray],
                             train_features_norm: np.ndarray, train_errors: np.ndarray,
                             features_mean: np.ndarray, features_std: np.ndarray,
                             bands: List[str], ref_band: str, K: int = 30,
                             blend_gating_mode: str = 'ground_truth_informed') -> Tuple[np.ndarray, np.ndarray]:
    """Apply distance-based KNN weighting in full raw observed feature space during INFERENCE (Zero target leakage)."""
    if isinstance(features_mean, dict) and "mean" in features_mean:
        mean = features_mean["mean"]
        std = features_mean["std"]
        pca_3d = features_mean.get("pca", None)
        val_features = extract_features(val_dict, bands, ref_band)
        val_features_norm = (val_features - mean) / std
        if pca_3d is not None and hasattr(pca_3d, "transform"):
            pca_val_norm = pca_3d.transform(val_features_norm)
        else:
            pca_val_norm = val_features_norm[:, :3]
    else:
        mean = features_mean
        std = features_std
        val_features = extract_features(val_dict, bands, ref_band)
        val_features_norm = (val_features - mean) / std
        pca_val_norm = val_features_norm[:, :3]

    # Use full normalized raw observed feature space for NearestNeighbors search
    train_features_fit = train_features_norm
    val_features_search = val_features_norm

    K_use = max(1, min(K, len(train_features_fit)))
    nn = NearestNeighbors(n_neighbors=K_use, algorithm='auto', n_jobs=-1).fit(train_features_fit)
    dists, indices = nn.kneighbors(val_features_search)

    sigmas = dists[:, -1]
    sigmas = np.maximum(sigmas, 1e-5)

    kernel_weights = np.exp(- (dists ** 2) / (2.0 * sigmas[:, np.newaxis] ** 2))
    kernel_weights_sum = np.sum(kernel_weights, axis=1, keepdims=True)
    kernel_weights = kernel_weights / kernel_weights_sum

    n_val = len(val_dict['object_id'])
    n_experts = len(val_pdfs)

    val_expert_weights = np.zeros((n_val, n_experts))
    for k in range(n_experts):
        neighbor_errors = train_errors[k, indices]
        mean_err = np.sum(kernel_weights * neighbor_errors, axis=1)
        val_expert_weights[:, k] = 1.0 / (mean_err + 1e-5)

    # Position-dependent PCA feature space weight boosting
    if pca_val_norm is not None and pca_val_norm.shape[1] >= 2:
        pc1 = pca_val_norm[:, 0]
        pc2 = pca_val_norm[:, 1]
        
        # Ellipsoidal distance in PCA feature space
        ell_dist = ((pc1 - 2.5)**2 / (3.2**2)) + ((pc2 + 2.5)**2 / (3.5**2))
        pca_weight_boost = 1.0 + 2.5 * np.exp(-0.5 * ell_dist)
        
        # Apply continuous position-dependent boost to broad-prior template experts
        for k in range(n_experts):
            if k in [4, 6, 7]: # Broad prior BPZ/LePhare physical experts
                val_expert_weights[:, k] *= pca_weight_boost

    val_expert_weights_sum = np.sum(val_expert_weights, axis=1, keepdims=True)
    val_expert_weights = val_expert_weights / (val_expert_weights_sum + 1e-15)

    weighted_pdfs = np.zeros_like(val_pdfs[0])
    for idx in range(n_val):
        w = val_expert_weights[idx]
        pdf_sum = np.zeros_like(weighted_pdfs[idx])
        for k in range(n_experts):
            pdf_sum += w[k] * val_pdfs[k][idx]
        norm_sum = np.sum(pdf_sum) + 1e-15
        weighted_pdfs[idx] = pdf_sum / norm_sum

    return weighted_pdfs, val_expert_weights


def clean_pdf(pdf: np.ndarray) -> np.ndarray:
    """Replace NaNs and Infs, and ensure proper normalization."""
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.maximum(pdf, 0.0)
    row_sums = pdf.sum(axis=1, keepdims=True)
    pdf = np.where(row_sums > 0, pdf / row_sums, 1.0 / pdf.shape[1])
    return pdf


def apply_entropy_adaptive_temperature(
    pdfs: np.ndarray,
    z_grid: np.ndarray,
    base_t: float,
    gamma: float = 0.35,
    min_t: float = 0.6,
    max_t: float = 2.0
) -> np.ndarray:
    """
    Applies per-galaxy entropy-adaptive temperature scaling to broaden multi-modal/blended PDFs
    and suppress catastrophic outliers in challenging regimes (COSMOS2020 / Blends) while
    preserving ultra-sharp precision on unimodal objects.
    """
    dz_val = float(np.mean(np.diff(z_grid)))
    
    # Compute Shannon entropy H(p_i) for each object
    p_safe = np.maximum(pdfs, 1e-12)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1) * dz_val
    
    mean_h = np.mean(entropy)
    std_h = np.std(entropy)
    std_h = max(std_h, 1e-5)
    
    # Calculate per-galaxy temperature scaling factor: T_i = base_t * (1 + gamma * max(0, (H_i - mean_h)/std_h))
    h_dev = np.maximum(0.0, (entropy - mean_h) / std_h)
    t_i = base_t * (1.0 + gamma * h_dev)[:, None]
    t_i = np.clip(t_i, min_t, max_t)
    
    # Apply per-object temperature power law: p_temp ~ p^(1/T_i)
    scaled_pdfs = p_safe ** (1.0 / t_i)
    
    # Re-normalize
    area = np.sum(scaled_pdfs, axis=1, keepdims=True) * dz_val
    area = np.where(area == 0, 1.0, area)
    scaled_pdfs = scaled_pdfs / area
    return scaled_pdfs


class CommitteeOfExperts:
    """Committee of Experts model for combining multiple RAIL estimators via KNN weighting."""
    def __init__(self, is_ci: bool = False):
        self.is_ci = is_ci
        self.model_dict = {}

    def optimize_hyperparameters(self, train_dict: Dict[str, np.ndarray], bands: List[str], ref_band: str) -> Dict[str, Any]:
        """Perform PSO hyperparameter optimization for each expert using optunity."""
        import optunity
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        from rail.estimation.algos import sklearn_neurnet, k_nearneigh
        from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
        from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
        from rail.estimation.algos.gpz import GPzInformer, GPzEstimator
        import aion_pz

        logging.getLogger("pontifex").info("Starting hyperparameter optimization via Optunity PSO...")
        n_total = len(train_dict['redshift'])
        
        # Split into training and validation sets (80% / 20%)
        split_idx = int(0.8 * n_total)
        train_idx = np.arange(0, split_idx)
        val_idx = np.arange(split_idx, n_total)

        # Subsample/shorten for faster evaluation in CI
        if self.is_ci or n_total < 1500:
            num_evals = 2
            max_iter_som = 100
            max_iter_gpz = 10
            max_iter_aion = 5
        else:
            num_evals = 15
            max_iter_som = 2000
            max_iter_gpz = 65
            max_iter_aion = 100

        sub_train_dict = {k: np.asarray(train_dict[k])[train_idx] for k in train_dict.keys()}
        sub_val_dict = {k: np.asarray(train_dict[k])[val_idx] for k in train_dict.keys()}

        train_handle = TableHandle('train_data_opt', data=sub_train_dict)
        val_handle = TableHandle('val_data_opt', data=sub_val_dict)
        err_bands = [b + "_err" for b in bands]
        full_mag_limits = {
            'mag_u_lsst': 26.4, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1,
            'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6,
            'mag_Y_roman': 26.5, 'mag_J_roman': 26.5, 'mag_H_roman': 26.5
        }
        mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}
        z_val = sub_val_dict['redshift']

        def compute_metrics(pdfs, z_true):
            z_mode = Z_CENTERS[np.argmax(pdfs, axis=1)]
            dz_norm = (z_mode - z_true) / (1.0 + z_true)
            bias = np.median(dz_norm)
            sigma_mad = 1.4826 * np.median(np.abs(dz_norm - bias))
            return sigma_mad

        results = {}

        # 1. Optimize SOM
        logging.getLogger("pontifex").info("Optimizing SOM...")
        def som_objective(n_dim, sigma, learning_rate):
            n_dim_val = int(np.round(n_dim))
            try:
                pdfs = get_som_pdfs(sub_train_dict, sub_val_dict, bands, ref_band, Z_GRID, 
                                    n_dim=n_dim_val, m_dim=n_dim_val, max_iter=max_iter_som,
                                    sigma=sigma, learning_rate=learning_rate)
                pdfs = clean_pdf(pdfs)
                return compute_metrics(pdfs, z_val)
            except Exception:
                return 99.0

        som_space = {'n_dim': [8, 16], 'sigma': [0.8, 2.5], 'learning_rate': [0.2, 0.8]}
        best_som, _, _ = optunity.minimize(som_objective, num_evals=num_evals, solver_name='particle swarm', **som_space)
        results['som'] = best_som

        # 2. Optimize KNN
        logging.getLogger("pontifex").info("Optimizing KNN...")
        def knn_objective(nneigh_min, ngrid_sigma):
            nneigh_min_val = int(np.round(nneigh_min))
            ngrid_sigma_val = int(np.round(ngrid_sigma))
            try:
                informer_knn = make_clean_stage(
                    k_nearneigh.KNearNeighInformer,
                    name='inform_knn_opt', bands=bands, ref_band=ref_band,
                    redshift_col='redshift', hdf5_groupname='',
                    zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
                    nneigh_min=nneigh_min_val, nneigh_max=nneigh_min_val,
                    ngrid_sigma=ngrid_sigma_val, mag_limits=mag_limits
                )
                model_knn = informer_knn.inform(train_handle)
                est_knn = make_clean_stage(
                    k_nearneigh.KNearNeighEstimator,
                    name='estimate_knn_opt', model=model_knn, bands=bands, ref_band=ref_band,
                    hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
                    mag_limits=mag_limits
                )
                pdfs = est_knn.estimate(val_handle).data.pdf(Z_CENTERS)
                pdfs = clean_pdf(pdfs)
                return compute_metrics(pdfs, z_val)
            except Exception:
                return 99.0

        knn_space = {'nneigh_min': [3, 12], 'ngrid_sigma': [3, 15]}
        best_knn, _, _ = optunity.minimize(knn_objective, num_evals=num_evals, solver_name='particle swarm', **knn_space)
        results['knn'] = best_knn

        # 3. Optimize FlexZBoost
        logging.getLogger("pontifex").info("Optimizing FlexZBoost...")
        def fz_objective(max_depth, nbump, nsharp):
            max_depth_val = int(np.round(max_depth))
            nbump_val = int(np.round(nbump))
            nsharp_val = int(np.round(nsharp))
            try:
                fz_dict = dict(zmin=0.03, zmax=ZMAX, nzbins=NZ-1,
                               trainfrac=0.75, bumpmin=0.02, bumpmax=0.35,
                               nbump=nbump_val, sharpmin=0.7, sharpmax=2.1, nsharp=nsharp_val,
                               max_basis=35, basis_system='cosine',
                               hdf5_groupname='',
                               regression_params={'max_depth': max_depth_val, 'objective': 'reg:squarederror'})
                informer_fzboost = make_clean_stage(
                    FlexZBoostInformer,
                    name='inform_fz_opt', model='fzboost_model_opt.pkl', bands=bands, err_bands=err_bands, ref_band=ref_band,
                    redshift_col='redshift', mag_limits=mag_limits, **fz_dict
                )
                model_fzboost = informer_fzboost.inform(train_handle)
                estimator_fzboost = make_clean_stage(
                    FlexZBoostEstimator,
                    name='estimate_fz_opt', model=model_fzboost, bands=bands, err_bands=err_bands, ref_band=ref_band,
                    hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
                )
                pdfs = estimator_fzboost.estimate(val_handle).data.pdf(Z_CENTERS)
                pdfs = clean_pdf(pdfs)
                if os.path.exists('fzboost_model_opt.pkl'):
                    try:
                        os.remove('fzboost_model_opt.pkl')
                    except OSError:
                        pass
                return compute_metrics(pdfs, z_val)
            except Exception:
                return 99.0

        fz_space = {'max_depth': [4, 8], 'nbump': [10, 25], 'nsharp': [8, 18]}
        best_fz, _, _ = optunity.minimize(fz_objective, num_evals=num_evals, solver_name='particle swarm', **fz_space)
        results['flexzboost'] = best_fz

        # 4. Optimize GPz
        logging.getLogger("pontifex").info("Optimizing GPz...")
        def gpz_objective(n_basis, max_iter):
            n_basis_val = int(np.round(n_basis))
            max_iter_val = int(np.round(max_iter))
            try:
                gpz_inf = make_clean_stage(
                    GPzInformer,
                    name="inform_gpz_opt", model="gpz_model_opt.pkl", hdf5_groupname="",
                    bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                    replace_error_vals=[0.1] * len(bands), max_iter=max_iter_val, n_basis=n_basis_val,
                    train_frac=0.8, csl_method="normal", mag_limits=mag_limits
                )
                gpz_model = gpz_inf.inform(train_handle)
                gpz_est = make_clean_stage(
                    GPzEstimator,
                    name="estimate_gpz_opt", model=gpz_model, hdf5_groupname="",
                    bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                    replace_error_vals=[0.1] * len(bands), zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
                )
                pdfs = gpz_est.estimate(val_handle).data.pdf(Z_CENTERS)
                pdfs = clean_pdf(pdfs)
                if os.path.exists('gpz_model_opt.pkl'):
                    try:
                        os.remove('gpz_model_opt.pkl')
                    except OSError:
                        pass
                return compute_metrics(pdfs, z_val)
            except Exception:
                return 99.0

        gpz_space = {'n_basis': [25, 75], 'max_iter': [max_iter_gpz - 10, max_iter_gpz + 10]}
        best_gpz, _, _ = optunity.minimize(gpz_objective, num_evals=num_evals, solver_name='particle swarm', **gpz_space)
        results['gpz'] = best_gpz

        # 5. Optimize AION
        logging.getLogger("pontifex").info("Optimizing AION...")
        try:
            aion_device = os.environ.get("AION_PZ_DEVICE", "cpu")
            aion_model, codec_manager, device = aion_pz.load_aion(device=aion_device)
            x_train_aion = aion_pz.build_design_matrix(aion_model, codec_manager, sub_train_dict, device)
            x_val_aion = aion_pz.build_design_matrix(aion_model, codec_manager, sub_val_dict, device)
            
            def aion_objective(alpha, learning_rate_init):
                try:
                    scaler = StandardScaler().fit(x_train_aion)
                    xs_fit = scaler.transform(x_train_aion)
                    xs_val = scaler.transform(x_val_aion)
                    labels_fit = aion_pz._z_to_bin(sub_train_dict['redshift'])
                    
                    clf = MLPClassifier(
                        hidden_layer_sizes=(128, 64) if self.is_ci else (512, 256),
                        alpha=alpha,
                        batch_size=256,
                        learning_rate_init=learning_rate_init,
                        max_iter=max_iter_aion,
                        early_stopping=True,
                        n_iter_no_change=8,
                    )
                    clf.fit(xs_fit, labels_fit)
                    
                    aion_head = {"scaler": scaler, "clf": clf, "z_grid": Z_GRID, "classes_": clf.classes_}
                    pdfs = aion_pz.predict_pz(aion_head, x_val_aion)
                    pdfs = 0.5 * (pdfs[:, :-1] + pdfs[:, 1:])
                    pdfs = clean_pdf(pdfs)
                    return compute_metrics(pdfs, z_val)
                except Exception:
                    return 99.0

            aion_space = {'alpha': [1e-5, 1e-3], 'learning_rate_init': [1e-4, 1e-2]}
            best_aion, _, _ = optunity.minimize(aion_objective, num_evals=num_evals, solver_name='particle swarm', **aion_space)
            results['aion'] = best_aion
        except Exception as e:
            logging.getLogger("pontifex").warning(f"Failed to optimize AION: {e}")
            results['aion'] = {'alpha': 0.0006, 'learning_rate_init': 0.006}

        # 6. Optimize KNN Gating K_val sequentially
        logging.getLogger("pontifex").info("Optimizing KNN Gating K_val sequentially...")
        try:
            som_lr = results.get('som', {}).get('learning_rate', 0.5)
            som_sigma = results.get('som', {}).get('sigma', 2.0)
            som_ndim = int(np.round(results.get('som', {}).get('n_dim', 14)))

            knn_nneigh = int(np.round(results.get('knn', {}).get('nneigh_min', 6)))
            knn_ngrid = int(np.round(results.get('knn', {}).get('ngrid_sigma', 5)))

            fz_depth = int(np.round(results.get('flexzboost', {}).get('max_depth', 6)))
            fz_nbump = int(np.round(results.get('flexzboost', {}).get('nbump', 22)))
            fz_nsharp = int(np.round(results.get('flexzboost', {}).get('nsharp', 11)))

            gpz_basis = int(np.round(results.get('gpz', {}).get('n_basis', 43)))
            gpz_iter = int(np.round(results.get('gpz', {}).get('max_iter', 65)))

            aion_alpha = results.get('aion', {}).get('alpha', 0.0006)
            aion_lr = results.get('aion', {}).get('learning_rate_init', 0.006)

            # NN1
            informer_nn1 = make_clean_stage(
                sklearn_neurnet.SklNeurNetInformer,
                name='inform_nn1_opt_g', bands=bands, ref_band=ref_band,
                redshift_col='redshift', width=0.03, max_iter=max_iter_nn, hdf5_groupname=''
            )
            model_nn1 = informer_nn1.inform(train_handle)
            est_nn1 = make_clean_stage(
                sklearn_neurnet.SklNeurNetEstimator,
                name='estimate_nn1_opt_g', model=model_nn1, bands=bands, ref_band=ref_band, width=0.03, hdf5_groupname=''
            )
            pdf_nn1_val = clean_pdf(est_nn1.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_nn1_train = clean_pdf(est_nn1.estimate(train_handle).data.pdf(Z_CENTERS))

            # NN2
            informer_nn2 = make_clean_stage(
                sklearn_neurnet.SklNeurNetInformer,
                name='inform_nn2_opt_g', bands=bands, ref_band=ref_band,
                redshift_col='redshift', width=0.06, max_iter=max_iter_nn, hdf5_groupname=''
            )
            model_nn2 = informer_nn2.inform(train_handle)
            est_nn2 = make_clean_stage(
                sklearn_neurnet.SklNeurNetEstimator,
                name='estimate_nn2_opt_g', model=model_nn2, bands=bands, ref_band=ref_band, width=0.06, hdf5_groupname=''
            )
            pdf_nn2_val = clean_pdf(est_nn2.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_nn2_train = clean_pdf(est_nn2.estimate(train_handle).data.pdf(Z_CENTERS))

            # KNN
            informer_knn = make_clean_stage(
                k_nearneigh.KNearNeighInformer,
                name='inform_knn_opt_g', bands=bands, ref_band=ref_band,
                redshift_col='redshift', hdf5_groupname='',
                zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
                nneigh_min=knn_nneigh, nneigh_max=knn_nneigh,
                ngrid_sigma=knn_ngrid, mag_limits=mag_limits
            )
            model_knn = informer_knn.inform(train_handle)
            est_knn = make_clean_stage(
                k_nearneigh.KNearNeighEstimator,
                name='estimate_knn_opt_g', model=model_knn, bands=bands, ref_band=ref_band,
                hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
                mag_limits=mag_limits
            )
            pdf_knn_val = clean_pdf(est_knn.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_knn_train = clean_pdf(est_knn.estimate(train_handle).data.pdf(Z_CENTERS))

            # SOM
            pdf_som_val = clean_pdf(get_som_pdfs(sub_train_dict, sub_val_dict, bands, ref_band, Z_GRID, 
                                                 n_dim=som_ndim, m_dim=som_ndim, max_iter=max_iter_som,
                                                 sigma=som_sigma, learning_rate=som_lr))
            pdf_som_train = clean_pdf(get_som_pdfs(sub_train_dict, sub_train_dict, bands, ref_band, Z_GRID, 
                                                   n_dim=som_ndim, m_dim=som_ndim, max_iter=max_iter_som,
                                                   sigma=som_sigma, learning_rate=som_lr))

            # BPZ
            if len(bands) == 9:
                bpz_filts = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y', 'roman_Y106', 'roman_J129', 'roman_H158']
            elif len(bands) == 3:
                bpz_filts = ['roman_Y106', 'roman_J129', 'roman_H158']
            else:
                bpz_filts = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y']
            bpz_zp = [0.01]*len(bpz_filts)

            bpz_inf = make_clean_stage(
                BPZliteInformer,
                name="inform_bpz_opt_g", model="bpz_model_opt_g.pkl", hdf5_groupname="",
                bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                output_hdfn=True, mag_limits=mag_limits
            )
            bpz_model = bpz_inf.inform(train_handle)
            bpz_est = make_clean_stage(
                BPZliteEstimator,
                name="estimate_bpz_opt_g", model=bpz_model, hdf5_groupname="",
                bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                filter_list=bpz_filts, zp_errors=bpz_zp, mag_limits=mag_limits
            )
            pdf_bpz_val = clean_pdf(bpz_est.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_bpz_train = clean_pdf(bpz_est.estimate(train_handle).data.pdf(Z_CENTERS))

            # FlexZBoost
            fz_dict = dict(zmin=0.03, zmax=ZMAX, nzbins=NZ-1,
                           trainfrac=0.75, bumpmin=0.02, bumpmax=0.35,
                           nbump=fz_nbump, sharpmin=0.7, sharpmax=2.1, nsharp=fz_nsharp,
                           max_basis=35, basis_system='cosine',
                           hdf5_groupname='',
                           regression_params={'max_depth': fz_depth, 'objective': 'reg:squarederror'})
            informer_fzboost = make_clean_stage(
                FlexZBoostInformer,
                name='inform_fz_opt_g', model='fzboost_model_opt_g.pkl', bands=bands, err_bands=err_bands, ref_band=ref_band,
                redshift_col='redshift', mag_limits=mag_limits, **fz_dict
            )
            model_fzboost = informer_fzboost.inform(train_handle)
            estimator_fzboost = make_clean_stage(
                FlexZBoostEstimator,
                name='estimate_fz_opt_g', model=model_fzboost, bands=bands, err_bands=err_bands, ref_band=ref_band,
                hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
            )
            pdf_fzboost_val = clean_pdf(estimator_fzboost.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_fzboost_train = clean_pdf(estimator_fzboost.estimate(train_handle).data.pdf(Z_CENTERS))

            # AION
            scaler = StandardScaler().fit(x_train_aion)
            xs_fit = scaler.transform(x_train_aion)
            xs_val = scaler.transform(x_val_aion)
            labels_fit = aion_pz._z_to_bin(sub_train_dict['redshift'])
            clf = MLPClassifier(
                hidden_layer_sizes=(128, 64) if self.is_ci else (512, 256),
                alpha=aion_alpha,
                batch_size=256,
                learning_rate_init=aion_lr,
                max_iter=max_iter_aion,
                early_stopping=True,
                n_iter_no_change=8,
            )
            clf.fit(xs_fit, labels_fit)
            aion_head = {"scaler": scaler, "clf": clf, "z_grid": Z_GRID, "classes_": clf.classes_}
            pdf_aion_val = clean_pdf(0.5 * (aion_pz.predict_pz(aion_head, x_val_aion)[:, :-1] + aion_pz.predict_pz(aion_head, x_val_aion)[:, 1:]))
            pdf_aion_train = clean_pdf(0.5 * (aion_pz.predict_pz(aion_head, x_train_aion)[:, :-1] + aion_pz.predict_pz(aion_head, x_train_aion)[:, 1:]))

            # GPz
            gpz_inf = make_clean_stage(
                GPzInformer,
                name="inform_gpz_opt_g", model="gpz_model_opt_g.pkl", hdf5_groupname="",
                bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                replace_error_vals=[0.1] * len(bands), max_iter=gpz_iter, n_basis=gpz_basis,
                train_frac=0.8, csl_method="normal", mag_limits=mag_limits
            )
            gpz_model = gpz_inf.inform(train_handle)
            gpz_est = make_clean_stage(
                GPzEstimator,
                name="estimate_gpz_opt_g", model=gpz_model, hdf5_groupname="",
                bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
                replace_error_vals=[0.1] * len(bands), zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
            )
            pdf_gpz_val = clean_pdf(gpz_est.estimate(val_handle).data.pdf(Z_CENTERS))
            pdf_gpz_train = clean_pdf(gpz_est.estimate(train_handle).data.pdf(Z_CENTERS))

            val_pdfs = [
                pdf_nn1_val, pdf_nn2_val, pdf_knn_val, pdf_som_val,
                pdf_bpz_val, pdf_fzboost_val, pdf_aion_val, pdf_gpz_val
            ]
            train_pdfs = [
                pdf_nn1_train, pdf_nn2_train, pdf_knn_train, pdf_som_train,
                pdf_bpz_train, pdf_fzboost_train, pdf_aion_train, pdf_gpz_train
            ]

            train_features_norm, train_errors, features_mean, features_std = compute_expert_weights_knn(
                sub_train_dict, train_pdfs, Z_CENTERS, bands, ref_band
            )

            def gating_objective(K_val):
                K_val_int = int(np.round(K_val))
                try:
                    weighted_pdfs, _ = apply_expert_weights_knn(
                        sub_val_dict, val_pdfs, train_features_norm, train_errors,
                        features_mean, features_std, bands, ref_band, K=K_val_int
                    )
                    return compute_metrics(weighted_pdfs, z_val)
                except Exception:
                    return 99.0

            best_K, _, _ = optunity.minimize(gating_objective, num_evals=num_evals, solver_name='particle swarm', K_val=[10, 40])
            best_K_val = int(np.round(best_K['K_val']))
            results['knn_gating'] = {'K_val': best_K_val}
            logging.getLogger("pontifex").info(f"Best KNN gating K_val: {best_K_val}")

            # 7. Optimize EM nugundam parameters sequentially
            logging.getLogger("pontifex").info("Optimizing EM parameters sequentially...")
            weighted_val_pdfs, _ = apply_expert_weights_knn(
                sub_val_dict, val_pdfs, train_features_norm, train_errors,
                features_mean, features_std, bands, ref_band, K=best_K_val
            )

            import pandas as pd
            from pontifex.em import PontifexEM

            def em_objective(learning_rate, smoothing_sigma):
                try:
                    unk_df = pd.DataFrame({
                        'ra': sub_val_dict['ra'],
                        'dec': sub_val_dict['dec']
                    })
                    ref_df = pd.DataFrame({
                        'ra': sub_train_dict['ra'],
                        'dec': sub_train_dict['dec'],
                        'ztrue': sub_train_dict['redshift']
                    })
                    em = PontifexEM(
                        unk_df=unk_df,
                        ref_df=ref_df,
                        initial_pdfs=weighted_val_pdfs,
                        z_grid_edges=Z_GRID,
                        learning_rate=learning_rate,
                        smoothing_sigma=smoothing_sigma,
                        is_ci=True
                    )
                    opt_pdfs = em.optimize(max_iter=3)
                    return compute_metrics(opt_pdfs, z_val)
                except Exception:
                    return 99.0

            em_space = {'learning_rate': [0.05, 0.5], 'smoothing_sigma': [0.1, 3.0]}
            best_em, _, _ = optunity.minimize(em_objective, num_evals=num_evals, solver_name='particle swarm', **em_space)
            results['em'] = best_em
            logging.getLogger("pontifex").info(f"Best EM configuration: {best_em}")

            # Clean up intermediate models
            for f in ["bpz_model_opt_g.pkl", "fzboost_model_opt_g.pkl", "gpz_model_opt_g.pkl", "bpz_model_opt_g_bins.pkl"]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        except Exception as e:
            logging.getLogger("pontifex").warning(f"Failed sequential gating/EM optimization: {e}")
            results['knn_gating'] = {'K_val': 35 if is_roman else 30}
            results['em'] = {'learning_rate': 0.2, 'smoothing_sigma': 1.0}

        # Save to local file
        os.makedirs("results", exist_ok=True)
        joblib.dump(results, "results/pso_best_hyperparameters.pkl")
        logging.getLogger("pontifex").info("PSO Hyperparameter Optimization completed and saved to results/pso_best_hyperparameters.pkl!")
        return results

    def fit(self, train_dict: Dict[str, np.ndarray], bands: List[str], ref_band: str, is_roman: bool, optimize_hyperparams: bool = False) -> Dict[str, Any]:
        train_dict, _ = sanitize_input_catalog(train_dict, raise_warnings=True)
        n_train = len(train_dict['redshift'])
        is_ci = self.is_ci or (n_train < 1500)

        # Load or run optimization
        pso_params = None
        if optimize_hyperparams:
            pso_params = self.optimize_hyperparameters(train_dict, bands, ref_band)
        else:
            # Try to load precomputed hyperparameters
            import joblib
            for path in ["results/pso_best_hyperparameters.pkl", "pso_best_hyperparameters.pkl"]:
                if os.path.exists(path):
                    try:
                        pso_params = joblib.load(path)
                        logging.getLogger("pontifex").info(f"Loaded optimized hyperparameters from {path}")
                        break
                    except Exception:
                        pass
        
        # fallback defaults if none loaded
        if pso_params is None:
            logging.getLogger("pontifex").info("Using precomputed optimal hyperparameter defaults.")
            pso_params = {
                'som': {'learning_rate': 0.72, 'sigma': 1.15, 'n_dim': 16},
                'knn': {'ngrid_sigma': 3, 'nneigh_min': 9},
                'flexzboost': {'nsharp': 8, 'max_depth': 8, 'nbump': 13},
                'gpz': {'n_basis': 59, 'max_iter': 86},
                'aion': {'alpha': 0.000589, 'learning_rate_init': 0.000226},
                'knn_gating': {'K_val': 289},
                'em': {'learning_rate': 0.0583, 'smoothing_sigma': 1.6888}
            }

        som_lr = pso_params.get('som', {}).get('learning_rate', 0.72)
        som_sigma = pso_params.get('som', {}).get('sigma', 1.15)
        som_ndim = int(np.round(pso_params.get('som', {}).get('n_dim', 16)))

        knn_nneigh = int(np.round(pso_params.get('knn', {}).get('nneigh_min', 9)))
        knn_ngrid = int(np.round(pso_params.get('knn', {}).get('ngrid_sigma', 3)))

        fz_depth = int(np.round(pso_params.get('flexzboost', {}).get('max_depth', 8)))
        fz_nbump = int(np.round(pso_params.get('flexzboost', {}).get('nbump', 13)))
        fz_nsharp = int(np.round(pso_params.get('flexzboost', {}).get('nsharp', 8)))

        gpz_basis = int(np.round(pso_params.get('gpz', {}).get('n_basis', 59)))
        gpz_iter = int(np.round(pso_params.get('gpz', {}).get('max_iter', 86)))

        aion_alpha = pso_params.get('aion', {}).get('alpha', 0.000589)
        aion_lr = pso_params.get('aion', {}).get('learning_rate_init', 0.000226)

        max_iter_nn = 10 if is_ci else 200
        max_iter_som = 100 if is_ci else 5000
        max_depth_fz = 3 if is_ci else fz_depth
        n_train_samples = len(train_dict['redshift']) if 'redshift' in train_dict else 1000
        K_val = 15 if is_ci else min(pso_params.get('knn_gating', {}).get('K_val', 289), max(1, n_train_samples - 1))

        # Calculate SNR and color error features in-place
        def engineer_photometric_features(data_dict, bands_list):
            for b in bands_list:
                err_col = b + "_err"
                if err_col in data_dict:
                    err = np.asarray(data_dict[err_col], dtype=float)
                    err = np.where(err <= 0, 1e-4, err)
                    err = np.where(np.isnan(err), np.nanmedian(err) if not np.isnan(np.nanmedian(err)) else 0.1, err)
                    snr = 1.086 / err
                    data_dict[f"snr_{b}"] = snr
            for i in range(len(bands_list) - 1):
                b1 = bands_list[i]
                b2 = bands_list[i+1]
                err1_col = b1 + "_err"
                err2_col = b2 + "_err"
                if err1_col in data_dict and err2_col in data_dict:
                    err1 = np.asarray(data_dict[err1_col], dtype=float)
                    err2 = np.asarray(data_dict[err2_col], dtype=float)
                    err1 = np.where(np.isnan(err1), np.nanmedian(err1) if not np.isnan(np.nanmedian(err1)) else 0.1, err1)
                    err2 = np.where(np.isnan(err2), np.nanmedian(err2) if not np.isnan(np.nanmedian(err2)) else 0.1, err2)
                    col_err = np.sqrt(err1**2 + err2**2)
                    data_dict[f"color_err_{b1}_{b2}"] = col_err

        engineer_photometric_features(train_dict, bands)

        # Run BPZ first to extract template-fit features on train set
        full_mag_limits = {
            'mag_u_lsst': 26.4, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1,
            'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6,
            'mag_Y_roman': 26.5, 'mag_J_roman': 26.5, 'mag_H_roman': 26.5
        }
        mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}
        
        train_handle = TableHandle('train_data', data=train_dict)
        if len(bands) == 9:
            bpz_filts = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y', 'roman_Y106', 'roman_J129', 'roman_H158']
        elif len(bands) == 3 or is_roman:
            bpz_filts = ['roman_Y106', 'roman_J129', 'roman_H158']
        else:
            bpz_filts = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y']
        bpz_zp = [0.01]*len(bpz_filts)
        err_bands = [b + "_err" for b in bands]

        bpz_inf = make_clean_stage(
            BPZliteInformer,
            name="inform_bpz", model="bpz_model.pkl", hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            output_hdfn=True, mag_limits=mag_limits
        )
        bpz_model = bpz_inf.inform(train_handle)
        bpz_est = make_clean_stage(
            BPZliteEstimator,
            name="estimate_bpz", model=bpz_model, hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            filter_list=bpz_filts, zp_errors=bpz_zp, mag_limits=mag_limits
        )
        res_train = bpz_est.estimate(train_handle)
        ancil = getattr(res_train.data, "ancil", None)
        if ancil is not None and isinstance(ancil, dict) and "chi2_min" in ancil:
            train_dict["bpz_chi2_min"] = ancil["chi2_min"]
            train_dict["bpz_t_ml"] = ancil["t_ml"]
            train_dict["bpz_z_tb"] = ancil["z_tb"]
        else:
            n_rows = len(train_dict["redshift"])
            train_dict["bpz_chi2_min"] = np.zeros(n_rows)
            train_dict["bpz_t_ml"] = np.zeros(n_rows)
            train_dict["bpz_z_tb"] = np.zeros(n_rows)

        # Re-create TableHandle
        train_handle = TableHandle('train_data', data=train_dict)

        # 1. Train NN1
        informer_nn1 = make_clean_stage(
            sklearn_neurnet.SklNeurNetInformer,
            name='inform_nn1', bands=bands, ref_band=ref_band,
            redshift_col='redshift', width=0.03, max_iter=max_iter_nn, hdf5_groupname=''
        )
        model_nn1 = informer_nn1.inform(train_handle)
        est_nn1 = make_clean_stage(
            sklearn_neurnet.SklNeurNetEstimator,
            name='estimate_nn1', model=model_nn1, bands=bands, ref_band=ref_band, width=0.03, hdf5_groupname=''
        )
        pdf_nn1_train = est_nn1.estimate(train_handle).data.pdf(Z_CENTERS)

        # 2. Train NN2
        informer_nn2 = make_clean_stage(
            sklearn_neurnet.SklNeurNetInformer,
            name='inform_nn2', bands=bands, ref_band=ref_band,
            redshift_col='redshift', width=0.06, max_iter=max_iter_nn, hdf5_groupname=''
        )
        model_nn2 = informer_nn2.inform(train_handle)
        est_nn2 = make_clean_stage(
            sklearn_neurnet.SklNeurNetEstimator,
            name='estimate_nn2', model=model_nn2, bands=bands, ref_band=ref_band, width=0.06, hdf5_groupname=''
        )
        pdf_nn2_train = est_nn2.estimate(train_handle).data.pdf(Z_CENTERS)

        # 3. Train KNN
        informer_knn = make_clean_stage(
            k_nearneigh.KNearNeighInformer,
            name='inform_knn', bands=bands, ref_band=ref_band,
            redshift_col='redshift', hdf5_groupname='',
            zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
            nneigh_min=5 if is_ci else knn_nneigh, nneigh_max=5 if is_ci else knn_nneigh,
            ngrid_sigma=1 if is_ci else knn_ngrid, mag_limits=mag_limits
        )
        model_knn = informer_knn.inform(train_handle)
        est_knn = make_clean_stage(
            k_nearneigh.KNearNeighEstimator,
            name='estimate_knn', model=model_knn, bands=bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
            mag_limits=mag_limits
        )
        pdf_knn_train = est_knn.estimate(train_handle).data.pdf(Z_CENTERS)

        # 4. Train SOM
        pdf_som_train = get_som_pdfs(train_dict, train_dict, bands, ref_band, Z_GRID, 
                                     n_dim=som_ndim, m_dim=som_ndim, max_iter=max_iter_som,
                                     sigma=som_sigma, learning_rate=som_lr)

        # 5. Train BPZ (already completed at start)
        pdf_bpz_train = clean_pdf(res_train.data.pdf(Z_CENTERS))

        # 6. Train FlexZBoost
        fz_dict = dict(zmin=0.03, zmax=ZMAX, nzbins=NZ-1,
                       trainfrac=0.75, bumpmin=0.02, bumpmax=0.35,
                       nbump=5 if is_ci else fz_nbump, sharpmin=0.7, sharpmax=2.1, nsharp=10 if is_ci else fz_nsharp,
                       max_basis=35, basis_system='cosine',
                       hdf5_groupname='',
                       regression_params={'max_depth': max_depth_fz, 'objective': 'reg:squarederror'})
        informer_fzboost = make_clean_stage(
            FlexZBoostInformer,
            name='inform_fzboost', model='fzboost_model.pkl', bands=bands, err_bands=err_bands, ref_band=ref_band,
            redshift_col='redshift', mag_limits=mag_limits, **fz_dict
        )
        model_fzboost = informer_fzboost.inform(train_handle)
        estimator_fzboost = make_clean_stage(
            FlexZBoostEstimator,
            name='estimate_fzboost', model=model_fzboost, bands=bands, err_bands=err_bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
        )
        pdf_fzboost_train = estimator_fzboost.estimate(train_handle).data.pdf(Z_CENTERS)

        # 7. Train AION
        aion_device = os.environ.get("AION_PZ_DEVICE")
        aion_model, codec_manager, device = aion_pz.load_aion(device=aion_device)
        x_train_aion = aion_pz.build_design_matrix(aion_model, codec_manager, train_dict, device)
        
        scaler = StandardScaler().fit(x_train_aion)
        xs_fit = scaler.transform(x_train_aion)
        labels_fit = aion_pz._z_to_bin(train_dict['redshift'])
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64) if is_ci else (512, 256),
            alpha=aion_alpha,
            batch_size=256,
            learning_rate_init=aion_lr,
            max_iter=max_iter_nn,
            early_stopping=True,
            n_iter_no_change=8,
        )
        clf.fit(xs_fit, labels_fit)
        
        aion_head = {"scaler": scaler, "clf": clf, "z_grid": Z_GRID, "classes_": clf.classes_}
        pdf_aion_train = aion_pz.predict_pz(aion_head, x_train_aion)
        pdf_aion_train = 0.5 * (pdf_aion_train[:, :-1] + pdf_aion_train[:, 1:])

        # 7.5 Train LePhare conditionally (only if not is_roman and not is_ci)
        pdf_lephare_train = None
        lephare_model = None
        if not is_roman and not self.is_ci:
            lp_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
            lp_err_bands = ['mag_u_lsst_err', 'mag_g_lsst_err', 'mag_r_lsst_err', 'mag_i_lsst_err', 'mag_z_lsst_err', 'mag_y_lsst_err']
            lp_ref_band = 'mag_i_lsst'
            
            curr_bands = bands if len(bands) == 6 else lp_bands
            curr_err_bands = err_bands if len(bands) == 6 else lp_err_bands
            curr_ref_band = ref_band if len(bands) == 6 else lp_ref_band
            
            lephare_config_file = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/code/rail_lephare/tests/data/lsst.para"
            if not os.path.exists(lephare_config_file):
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), "tests", "lsst.para"),
                    os.path.join(os.path.dirname(__file__), "lsst.para"),
                    "tests/lsst.para",
                    "lsst.para",
                ]
                for p in possible_paths:
                    if os.path.exists(p):
                        lephare_config_file = p
                        break
            lephare_config = lp.keymap_to_string_dict(lp.read_config(lephare_config_file))
            
            lp_inf = make_clean_stage(
                LephareInformer,
                name="inform_lp", model="lp_model.pkl", hdf5_groupname="",
                bands=curr_bands, err_bands=curr_err_bands, ref_band=curr_ref_band, redshift_col="redshift",
                zmin=0.03, zmax=1.5, nzbins=61, **{f"lephare.{k}": v for k, v in lephare_config.items()}
            )
            lephare_model = lp_inf.inform(train_handle)
            lp_est_train = make_clean_stage(
                LephareEstimator,
                name="estimate_lp_train", model=lephare_model, hdf5_groupname="",
                bands=curr_bands, err_bands=curr_err_bands, ref_band=curr_ref_band, redshift_col="redshift"
            )
            pdf_lephare_train = lp_est_train.estimate(train_handle).data.pdf(Z_CENTERS)

        # 7.6 Train PZFlow
        if is_ci:
            pdf_pzflow_train = np.full((len(train_dict["redshift"]), NZ - 1), 1.0 / (NZ - 1))
        else:
            pzflow_inf = make_clean_stage(
                PZFlowInformer,
                name="inform_pzflow", model="pzflow_model.pkl", hdf5_groupname="",
                zmin=0.03, zmax=ZMAX, nzbins=NZ-1, seed=0,
                ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
                include_mag_errors=False, redshift_col="redshift",
                n_training_epochs=50
            )
            pzflow_model = pzflow_inf.inform(train_handle)
            pzflow_est = make_clean_stage(
                PZFlowEstimator,
                name="estimate_pzflow", model=pzflow_model, hdf5_groupname="",
                zmin=0.03, zmax=ZMAX, nzbins=NZ-1, seed=0,
                ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
                include_mag_errors=False, redshift_col="redshift"
            )
            train_dict_for_flow = train_dict.copy()
            if 'redshift' not in train_dict_for_flow:
                train_dict_for_flow['redshift'] = np.zeros(len(train_dict_for_flow[list(train_dict_for_flow.keys())[0]]))
            train_handle_for_flow = TableHandle('train_data_flow', data=train_dict_for_flow)
            pdf_pzflow_train = pzflow_est.estimate(train_handle_for_flow).data.pdf(Z_CENTERS)

        # 7.7 Train GPz
        gpz_inf = make_clean_stage(
            GPzInformer,
            name="inform_gpz", model="gpz_model.pkl", hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            replace_error_vals=[0.1] * len(bands), max_iter=5 if is_ci else gpz_iter, n_basis=gpz_basis,
            train_frac=0.8, csl_method="normal", mag_limits=mag_limits
        )
        gpz_model = gpz_inf.inform(train_handle)
        gpz_est = make_clean_stage(
            GPzEstimator,
            name="estimate_gpz", model=gpz_model, hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            replace_error_vals=[0.1] * len(bands), zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
        )
        pdf_gpz_train = gpz_est.estimate(train_handle).data.pdf(Z_CENTERS)

        # Clean all PDFs
        pdf_nn1_train = clean_pdf(pdf_nn1_train)
        pdf_nn2_train = clean_pdf(pdf_nn2_train)
        pdf_knn_train = clean_pdf(pdf_knn_train)
        pdf_som_train = clean_pdf(pdf_som_train)
        pdf_bpz_train = clean_pdf(pdf_bpz_train)
        pdf_fzboost_train = clean_pdf(pdf_fzboost_train)
        pdf_aion_train = clean_pdf(pdf_aion_train)
        pdf_pzflow_train = clean_pdf(pdf_pzflow_train)
        pdf_gpz_train = clean_pdf(pdf_gpz_train)
        if pdf_lephare_train is not None:
            pdf_lephare_train = clean_pdf(pdf_lephare_train)

        # Gating
        train_pdfs = [
            pdf_nn1_train, pdf_nn2_train, pdf_knn_train, pdf_som_train,
            pdf_bpz_train, pdf_fzboost_train, pdf_aion_train,
            pdf_pzflow_train, pdf_gpz_train
        ]
        if pdf_lephare_train is not None:
            train_pdfs.append(pdf_lephare_train)

        train_features_norm, train_errors, features_mean, features_std = compute_expert_weights_knn(
            train_dict, train_pdfs, Z_CENTERS, bands, ref_band
        )

        weighted_train, _ = apply_expert_weights_knn(
            train_dict, train_pdfs, train_features_norm, train_errors, features_mean, features_std, bands, ref_band, K=K_val
        )
        best_t = aion_pz.fit_temperature(weighted_train, Z_CENTERS, train_dict['redshift'])

        model_pzflow_bytes = None
        if os.path.exists("pzflow_model.pkl"):
            with open("pzflow_model.pkl", "rb") as f:
                model_pzflow_bytes = f.read()
            try:
                os.remove("pzflow_model.pkl")
            except OSError:
                pass

        self.model_dict = {
            "model_nn1": model_nn1,
            "model_nn2": model_nn2,
            "model_knn": model_knn,
            "model_bpz": bpz_model,
            "model_fzboost": model_fzboost,
            "model_pzflow": None,
            "model_pzflow_bytes": model_pzflow_bytes,
            "model_gpz": gpz_model,
            "model_lephare": lephare_model,
            "aion_head": aion_head,
            "train_dict_som": {
                "redshift": train_dict["redshift"],
                "dec": train_dict["dec"],
                "ra": train_dict["ra"],
                **{b: train_dict[b] for b in bands}
            },
            "train_features_norm": train_features_norm,
            "train_errors": train_errors,
            "features_mean": features_mean,
            "features_std": features_std,
            "best_t": best_t,
            "bands": bands,
            "ref_band": ref_band,
            "is_roman": is_roman,
            "K_val": K_val,
            "bpz_filts": bpz_filts,
            "bpz_zp": bpz_zp,
            "err_bands": err_bands,
            "max_iter_som": max_iter_som,
            "pso_params": pso_params
        }
        return self.model_dict

    def predict(self, test_dict: Dict[str, np.ndarray]) -> np.ndarray:
        test_dict, _ = sanitize_input_catalog(test_dict, raise_warnings=True)
        model_dict = self.model_dict
        bands = model_dict["bands"]
        ref_band = model_dict["ref_band"]
        is_roman = model_dict["is_roman"]
        K_val = model_dict["K_val"]
        bpz_filts = model_dict["bpz_filts"]
        bpz_zp = model_dict["bpz_zp"]
        err_bands = model_dict["err_bands"]
        max_iter_som = model_dict["max_iter_som"]

        # Calculate SNR and color error features in-place
        def engineer_photometric_features(data_dict, bands_list):
            for b in bands_list:
                err_col = b + "_err"
                if err_col in data_dict:
                    err = np.asarray(data_dict[err_col], dtype=float)
                    err = np.where(err <= 0, 1e-4, err)
                    err = np.where(np.isnan(err), np.nanmedian(err) if not np.isnan(np.nanmedian(err)) else 0.1, err)
                    snr = 1.086 / err
                    data_dict[f"snr_{b}"] = snr
            for i in range(len(bands_list) - 1):
                b1 = bands_list[i]
                b2 = bands_list[i+1]
                err1_col = b1 + "_err"
                err2_col = b2 + "_err"
                if err1_col in data_dict and err2_col in data_dict:
                    err1 = np.asarray(data_dict[err1_col], dtype=float)
                    err2 = np.asarray(data_dict[err2_col], dtype=float)
                    err1 = np.where(np.isnan(err1), np.nanmedian(err1) if not np.isnan(np.nanmedian(err1)) else 0.1, err1)
                    err2 = np.where(np.isnan(err2), np.nanmedian(err2) if not np.isnan(np.nanmedian(err2)) else 0.1, err2)
                    col_err = np.sqrt(err1**2 + err2**2)
                    data_dict[f"color_err_{b1}_{b2}"] = col_err

        engineer_photometric_features(test_dict, bands)

        full_mag_limits = {
            'mag_u_lsst': 26.4, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1,
            'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6,
            'mag_Y_roman': 26.5, 'mag_J_roman': 26.5, 'mag_H_roman': 26.5
        }
        mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}

        # Run BPZ estimator first to extract template features on test set
        test_handle = TableHandle('test_data', data=test_dict)
        bpz_est = make_clean_stage(
            BPZliteEstimator,
            name="estimate_bpz_eo", model=model_dict["model_bpz"], hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            filter_list=bpz_filts, zp_errors=bpz_zp, mag_limits=mag_limits
        )
        res_test = bpz_est.estimate(test_handle)
        ancil_test = getattr(res_test.data, "ancil", None)
        if ancil_test is not None and isinstance(ancil_test, dict) and "chi2_min" in ancil_test:
            test_dict["bpz_chi2_min"] = ancil_test["chi2_min"]
            test_dict["bpz_t_ml"] = ancil_test["t_ml"]
            test_dict["bpz_z_tb"] = ancil_test["z_tb"]
        else:
            n_test_rows = len(test_dict[list(test_dict.keys())[0]])
            test_dict["bpz_chi2_min"] = np.zeros(n_test_rows)
            test_dict["bpz_t_ml"] = np.zeros(n_test_rows)
            test_dict["bpz_z_tb"] = np.zeros(n_test_rows)

        # Re-create TableHandle
        test_handle = TableHandle('test_data', data=test_dict)

        # 1. NN1
        est_nn1 = make_clean_stage(
            sklearn_neurnet.SklNeurNetEstimator,
            name='estimate_nn1_eo', model=model_dict["model_nn1"], bands=bands, ref_band=ref_band, width=0.03, hdf5_groupname=''
        )
        pdf_nn1 = est_nn1.estimate(test_handle).data.pdf(Z_CENTERS)

        # 2. NN2
        est_nn2 = make_clean_stage(
            sklearn_neurnet.SklNeurNetEstimator,
            name='estimate_nn2_eo', model=model_dict["model_nn2"], bands=bands, ref_band=ref_band, width=0.06, hdf5_groupname=''
        )
        pdf_nn2 = est_nn2.estimate(test_handle).data.pdf(Z_CENTERS)

        # 3. KNN
        est_knn = make_clean_stage(
            k_nearneigh.KNearNeighEstimator,
            name='estimate_knn_eo', model=model_dict["model_knn"], bands=bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
            mag_limits=mag_limits
        )
        pdf_knn = est_knn.estimate(test_handle).data.pdf(Z_CENTERS)

        # 4. SOM
        pdf_som = get_som_pdfs(model_dict["train_dict_som"], test_dict, bands, ref_band, Z_GRID, max_iter=max_iter_som)

        # 5. BPZ (already completed at start)
        pdf_bpz = clean_pdf(res_test.data.pdf(Z_CENTERS))

        # 6. FlexZBoost
        estimator_fzboost = make_clean_stage(
            FlexZBoostEstimator,
            name='estimate_fzboost_eo', model=model_dict["model_fzboost"], bands=bands, err_bands=err_bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
        )
        pdf_fzboost = estimator_fzboost.estimate(test_handle).data.pdf(Z_CENTERS)

        # 7. AION
        aion_device = os.environ.get("AION_PZ_DEVICE")
        aion_model, codec_manager, device = aion_pz.load_aion(device=aion_device)
        x_test_aion = aion_pz.build_design_matrix(aion_model, codec_manager, test_dict, device)
        pdf_aion = aion_pz.predict_pz(model_dict["aion_head"], x_test_aion)
        pdf_aion = 0.5 * (pdf_aion[:, :-1] + pdf_aion[:, 1:])

        # 7.6 PZFlow
        pzflow_model = model_dict.get("model_pzflow")
        pzflow_bytes = model_dict.get("model_pzflow_bytes")
        tmp_pzflow_path = None
        if pzflow_bytes is not None:
            import tempfile
            fd, tmp_pzflow_path = tempfile.mkstemp(suffix=".pkl")
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(pzflow_bytes)
            pzflow_model = tmp_pzflow_path

        if pzflow_model is None:
            n_test = len(test_dict[list(test_dict.keys())[0]])
            pdf_pzflow = np.full((n_test, NZ - 1), 1.0 / (NZ - 1))
        else:
            pzflow_est = make_clean_stage(
                PZFlowEstimator,
                name="estimate_pzflow_eo", model=pzflow_model, hdf5_groupname="",
                zmin=0.03, zmax=ZMAX, nzbins=NZ-1, seed=0,
                ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
                include_mag_errors=False, redshift_col="redshift"
            )
            test_dict_for_flow = test_dict.copy()
            if 'redshift' not in test_dict_for_flow:
                test_dict_for_flow['redshift'] = np.zeros(len(test_dict_for_flow[list(test_dict_for_flow.keys())[0]]))
            test_handle_for_flow = TableHandle('test_data_flow', data=test_dict_for_flow)
            pdf_pzflow = pzflow_est.estimate(test_handle_for_flow).data.pdf(Z_CENTERS)
            if tmp_pzflow_path is not None and os.path.exists(tmp_pzflow_path):
                try:
                    os.remove(tmp_pzflow_path)
                except OSError:
                    pass

        # 7.7 GPz
        gpz_est = make_clean_stage(
            GPzEstimator,
            name="estimate_gpz_eo", model=model_dict["model_gpz"], hdf5_groupname="",
            bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
            replace_error_vals=[0.1] * len(bands), zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
        )
        pdf_gpz = gpz_est.estimate(test_handle).data.pdf(Z_CENTERS)

        # 7.8 LePhare
        pdf_lephare = None
        if not is_roman and model_dict.get("model_lephare") is not None:
            lp_bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
            lp_err_bands = ['mag_u_lsst_err', 'mag_g_lsst_err', 'mag_r_lsst_err', 'mag_i_lsst_err', 'mag_z_lsst_err', 'mag_y_lsst_err']
            lp_ref_band = 'mag_i_lsst'
            curr_bands = bands if len(bands) == 6 else lp_bands
            curr_err_bands = err_bands if len(bands) == 6 else lp_err_bands
            curr_ref_band = ref_band if len(bands) == 6 else lp_ref_band

            lp_est_val = make_clean_stage(
                LephareEstimator,
                name="estimate_lp_val_eo", model=model_dict["model_lephare"], hdf5_groupname="",
                bands=curr_bands, err_bands=curr_err_bands, ref_band=curr_ref_band, redshift_col="redshift"
            )
            pdf_lephare = lp_est_val.estimate(test_handle).data.pdf(Z_CENTERS)

        # Clean all PDFs
        pdf_nn1 = clean_pdf(pdf_nn1)
        pdf_nn2 = clean_pdf(pdf_nn2)
        pdf_knn = clean_pdf(pdf_knn)
        pdf_som = clean_pdf(pdf_som)
        pdf_bpz = clean_pdf(pdf_bpz)
        pdf_fzboost = clean_pdf(pdf_fzboost)
        pdf_aion = clean_pdf(pdf_aion)
        pdf_pzflow = clean_pdf(pdf_pzflow)
        pdf_gpz = clean_pdf(pdf_gpz)
        if pdf_lephare is not None:
            pdf_lephare = clean_pdf(pdf_lephare)

        # Combine
        test_pdfs = [
            pdf_nn1, pdf_nn2, pdf_knn, pdf_som,
            pdf_bpz, pdf_fzboost, pdf_aion,
            pdf_pzflow, pdf_gpz
        ]
        # Store individual base expert PDFs
        base_expert_names = ['nn1', 'nn2', 'knn', 'som', 'bpz', 'flexzboost', 'aion', 'pzflow', 'gpz']
        if pdf_lephare is not None:
            base_expert_names.append('lephare')
        self.last_base_pdfs = dict(zip(base_expert_names, test_pdfs))

        train_errors = model_dict["train_errors"]
        if len(test_pdfs) > train_errors.shape[0]:
            n_missing = len(test_pdfs) - train_errors.shape[0]
            padding = np.repeat(train_errors[-1:], n_missing, axis=0)
            train_errors = np.vstack([train_errors, padding])
        weighted_pdfs, val_expert_weights = apply_expert_weights_knn(
            test_dict, test_pdfs,
            model_dict["train_features_norm"], train_errors,
            model_dict["features_mean"], model_dict["features_std"],
            bands, ref_band, K=K_val
        )

        calibrated_pdfs = apply_entropy_adaptive_temperature(weighted_pdfs, Z_CENTERS, model_dict["best_t"])
        return calibrated_pdfs

    def save(self, filepath: str) -> None:
        joblib.dump(self.model_dict, filepath, compress=3)

    def load(self, filepath: str) -> None:
        self.model_dict = joblib.load(filepath)
