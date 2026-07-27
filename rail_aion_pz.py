"""RAIL + AION Committee of Experts photometric redshift estimator.

Combines SklNeurNet (NN1, NN2), KNearNeigh (KNN), MiniSom (SOM), BPZlite (BPZ),
FlexZBoost, and AION-1 (frozen embeddings + MLP head) via local KNN-based
prior weights.

Integrates a monkeypatching layer for tables_io/numpy compatibility and
automatic speed scaling for CI mock runs.
"""

import sys
import os
import math
from pathlib import Path
from typing import Any
import joblib
import numpy as np

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
    _orig_shaped_array_new = jax.core.ShapedArray.__new__
    def _compat_shaped_array_new(cls, *args, **kwargs):
        kwargs.pop("named_shape", None)
        return _orig_shaped_array_new(cls, *args, **kwargs)
    jax.core.ShapedArray.__new__ = _compat_shaped_array_new
except Exception:
    pass

# 2. Import RAIL and other dependencies
import qp
from rail.core.data import TableHandle
from rail.estimation.algos import sklearn_neurnet, k_nearneigh
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
from rail.estimation.algos.pzflow_nf import PZFlowInformer, PZFlowEstimator
from rail.estimation.algos.gpz import GPzInformer, GPzEstimator
from rail.estimation.algos.lephare import LephareInformer, LephareEstimator
import lephare as lp
import logging
FlexZBoostInformer.log = logging.getLogger("FlexZBoostInformer")
FlexZBoostEstimator.log = logging.getLogger("FlexZBoostEstimator")
GPzInformer.log = logging.getLogger("GPzInformer")
GPzEstimator.log = logging.getLogger("GPzEstimator")
from minisom import MiniSom
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

import aion_pz

# Grid setup
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


def get_bands_and_ref(data_keys: list[str]) -> tuple[list[str], str, bool]:
    """Determine the band list and reference band based on catalog columns."""
    has_roman = 'mag_Y_roman' in data_keys
    has_lsst = 'mag_i_lsst' in data_keys or 'mag_u_lsst' in data_keys
    
    if has_roman and has_lsst:
        # Combined
        bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst', 'mag_Y_roman', 'mag_J_roman', 'mag_H_roman']
        return bands, 'mag_i_lsst', False
    elif has_roman:
        # Roman-only
        return ['mag_Y_roman', 'mag_J_roman', 'mag_H_roman'], 'mag_J_roman', True
    else:
        # Rubin-only
        return ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'], 'mag_i_lsst', False


def extract_features(data_dict: dict[str, np.ndarray], bands: list[str], ref_band: str) -> np.ndarray:
    """Extract magnitude and adjacent color features for KNN weighting."""
    features = []
    for band in bands:
        mag = data_dict[band].copy()
        mag = np.where(np.isnan(mag), np.nanmedian(mag) if not np.isnan(np.nanmedian(mag)) else 99.0, mag)
        features.append(mag)
    for i in range(len(bands) - 1):
        col = (data_dict[bands[i]] - data_dict[bands[i+1]]).copy()
        col = np.where(np.isnan(col), np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0.0, col)
        features.append(col)
    return np.column_stack(features)


def get_som_pdfs(train_dict: dict[str, np.ndarray], test_dict: dict[str, np.ndarray],
                 bands: list[str], ref_band: str, z_grid: np.ndarray, n_dim=15, m_dim=15, max_iter=5000) -> np.ndarray:
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

    som = MiniSom(n_dim, m_dim, train_feat.shape[1], sigma=1.5, learning_rate=0.5, random_seed=42)
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


def compute_expert_weights_knn(train_dict: dict[str, np.ndarray], train_pdfs: list[np.ndarray],
                               z_centers: np.ndarray, bands: list[str], ref_band: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate point-estimate errors for each expert and prepare features for weighting."""
    train_features = extract_features(train_dict, bands, ref_band)
    features_mean = np.mean(train_features, axis=0)
    features_std = np.std(train_features, axis=0)
    features_std = np.where(features_std == 0, 1.0, features_std)
    train_features_norm = (train_features - features_mean) / features_std

    train_errors = []
    for pdf in train_pdfs:
        z_mode = z_centers[np.argmax(pdf, axis=1)]
        err = np.abs(z_mode - train_dict['redshift']) / (1.0 + train_dict['redshift'])
        train_errors.append(err)
    train_errors = np.array(train_errors) # Shape: (n_experts, n_train)
    return train_features_norm, train_errors, features_mean, features_std


def apply_expert_weights_knn(val_dict: dict[str, np.ndarray], val_pdfs: list[np.ndarray],
                             train_features_norm: np.ndarray, train_errors: np.ndarray,
                             features_mean: np.ndarray, features_std: np.ndarray,
                             bands: list[str], ref_band: str, K: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Apply distance-based KNN weighting to combine expert PDFs."""
    val_features = extract_features(val_dict, bands, ref_band)
    val_features_norm = (val_features - features_mean) / features_std

    # Adjust K if catalog size is too small
    K = min(K, len(train_features_norm))

    nn = NearestNeighbors(n_neighbors=K, algorithm='auto', n_jobs=-1).fit(train_features_norm)
    dists, indices = nn.kneighbors(val_features_norm)

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

    val_expert_weights_sum = np.sum(val_expert_weights, axis=1, keepdims=True)
    val_expert_weights = val_expert_weights / (val_expert_weights_sum + 1e-15)

    weighted_pdfs = np.zeros_like(val_pdfs[0])
    for idx in range(n_val):
        w = val_expert_weights[idx]
        pdf_sum = np.zeros_like(weighted_pdfs[idx])
        for k in range(n_experts):
            pdf_sum += w[k] * val_pdfs[k][idx]
        weighted_pdfs[idx] = pdf_sum / (np.sum(pdf_sum) + 1e-15)

    return weighted_pdfs, val_expert_weights


def clean_pdf(pdf: np.ndarray) -> np.ndarray:
    """Replace NaNs and Infs, and ensure proper normalization."""
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    pdf = np.maximum(pdf, 0.0)
    row_sums = pdf.sum(axis=1, keepdims=True)
    pdf = np.where(row_sums > 0, pdf / row_sums, 1.0 / pdf.shape[1])
    return pdf


def train_and_estimate(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    save_model_to: str | Path | None = None,
    seed: int = 42,
) -> None:
    """Train the committee experts on train_file, combine via KNN weights, and predict on test_file."""
    train_dict = aion_pz.load_catalog(train_file)
    test_dict = aion_pz.load_catalog(test_file)

    # Subsample training labels if there are missing redshifts
    z_true = np.asarray(train_dict[aion_pz.REDSHIFT_COL], dtype="float64")
    if aion_pz.MANYBAND_COL in train_dict:
        z_many = np.asarray(train_dict[aion_pz.MANYBAND_COL], dtype="float64")
        fill = ~np.isfinite(z_true) & np.isfinite(z_many)
        z_true[fill] = z_many[fill]
    good = np.isfinite(z_true)
    train_dict = {k: v[good] for k, v in train_dict.items()}
    train_dict['redshift'] = z_true[good]

    bands, ref_band, is_roman = get_bands_and_ref(list(train_dict.keys()))

    # Determine run mode (CI vs Production)
    n_train = len(train_dict['redshift'])
    is_ci = n_train < 1500

    max_iter_nn = 10 if is_ci else 200
    max_iter_som = 100 if is_ci else 5000
    max_depth_fz = 3 if is_ci else 8
    K_val = 15 if is_roman else (250 if not is_ci else 15)

    train_handle = TableHandle('train_data', data=train_dict)
    test_handle = TableHandle('test_data', data=test_dict)

    # 1. Train NN1 (width=0.03)
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
    pdf_nn1 = est_nn1.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_nn1_train = est_nn1.estimate(train_handle).data.pdf(Z_CENTERS)

    # 2. Train NN2 (width=0.06)
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
    pdf_nn2 = est_nn2.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_nn2_train = est_nn2.estimate(train_handle).data.pdf(Z_CENTERS)

    # 3. Train KNN
    full_mag_limits = {
        'mag_u_lsst': 26.4,
        'mag_g_lsst': 27.8,
        'mag_r_lsst': 27.1,
        'mag_i_lsst': 26.7,
        'mag_z_lsst': 25.8,
        'mag_y_lsst': 24.6,
        'mag_Y_roman': 26.5,
        'mag_J_roman': 26.5,
        'mag_H_roman': 26.5
    }
    mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}

    informer_knn = make_clean_stage(
        k_nearneigh.KNearNeighInformer,
        name='inform_knn', bands=bands, ref_band=ref_band,
        redshift_col='redshift', hdf5_groupname='',
        zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
        nneigh_min=5 if is_ci else 3, nneigh_max=5 if is_ci else 5,
        ngrid_sigma=1 if is_ci else 10, mag_limits=mag_limits
    )
    model_knn = informer_knn.inform(train_handle)
    est_knn = make_clean_stage(
        k_nearneigh.KNearNeighEstimator,
        name='estimate_knn', model=model_knn, bands=bands, ref_band=ref_band,
        hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
        mag_limits=mag_limits
    )
    pdf_knn = est_knn.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_knn_train = est_knn.estimate(train_handle).data.pdf(Z_CENTERS)

    # 4. Train SOM
    pdf_som = get_som_pdfs(train_dict, test_dict, bands, ref_band, Z_GRID, max_iter=max_iter_som)
    pdf_som_train = get_som_pdfs(train_dict, train_dict, bands, ref_band, Z_GRID, max_iter=max_iter_som)

    # 5. Train BPZ
    if 'mag_Y_roman' in train_dict and ('mag_i_lsst' in train_dict or 'mag_u_lsst' in train_dict):
        bpz_filts = ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z', 'DC2LSST_y', 'roman_Y106', 'roman_J129', 'roman_H158']
    elif 'mag_Y_roman' in train_dict:
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
    pdf_bpz = bpz_est.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_bpz_train = bpz_est.estimate(train_handle).data.pdf(Z_CENTERS)

    # 6. Train FlexZBoost
    fz_dict = dict(zmin=0.03, zmax=ZMAX, nzbins=NZ-1,
                   trainfrac=0.75, bumpmin=0.02, bumpmax=0.35,
                   nbump=5 if is_ci else 20, sharpmin=0.7, sharpmax=2.1, nsharp=10 if is_ci else 15,
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
    pdf_fzboost = estimator_fzboost.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_fzboost_train = estimator_fzboost.estimate(train_handle).data.pdf(Z_CENTERS)

    # 7. Train AION
    aion_device = os.environ.get("AION_PZ_DEVICE")
    aion_model, codec_manager, device = aion_pz.load_aion(device=aion_device)
    x_train_aion = aion_pz.build_design_matrix(aion_model, codec_manager, train_dict, device)
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler().fit(x_train_aion)
    xs_fit = scaler.transform(x_train_aion)
    labels_fit = aion_pz._z_to_bin(train_dict['redshift'])
    
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64) if is_ci else (512, 256),
        alpha=1e-4,
        batch_size='auto' if is_ci else 256,
        learning_rate_init=1e-3,
        max_iter=max_iter_nn,
        early_stopping=True if len(xs_fit) >= 100 else False,
        n_iter_no_change=8,
    )
    clf.fit(xs_fit, labels_fit)
    
    aion_head = {"scaler": scaler, "clf": clf, "z_grid": Z_GRID, "classes_": clf.classes_}
    pdf_aion_train = aion_pz.predict_pz(aion_head, x_train_aion)
    pdf_aion_train = 0.5 * (pdf_aion_train[:, :-1] + pdf_aion_train[:, 1:])
    
    x_test_aion = aion_pz.build_design_matrix(aion_model, codec_manager, test_dict, device)
    pdf_aion = aion_pz.predict_pz(aion_head, x_test_aion)
    pdf_aion = 0.5 * (pdf_aion[:, :-1] + pdf_aion[:, 1:])

    # 7.5 Train LePhare conditionally (only if not is_roman)
    pdf_lephare = None
    pdf_lephare_train = None
    lephare_model = None
    if not is_roman:
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
        
        lp_est_val = make_clean_stage(
            LephareEstimator,
            name="estimate_lp_val", model=lephare_model, hdf5_groupname="",
            bands=curr_bands, err_bands=curr_err_bands, ref_band=curr_ref_band, redshift_col="redshift"
        )
        pdf_lephare = lp_est_val.estimate(test_handle).data.pdf(Z_CENTERS)
        
        lp_est_train = make_clean_stage(
            LephareEstimator,
            name="estimate_lp_train", model=lephare_model, hdf5_groupname="",
            bands=curr_bands, err_bands=curr_err_bands, ref_band=curr_ref_band, redshift_col="redshift"
        )
        pdf_lephare_train = lp_est_train.estimate(train_handle).data.pdf(Z_CENTERS)

    # 7.6 Train PZFlow
    pzflow_inf = make_clean_stage(
        PZFlowInformer,
        name="inform_pzflow", model="pzflow_model.pkl", hdf5_groupname="",
        zmin=0.03, zmax=ZMAX, nzbins=NZ-1, seed=0,
        ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
        include_mag_errors=False, redshift_col="redshift",
        n_training_epochs=5 if is_ci else 50
    )
    pzflow_model = pzflow_inf.inform(train_handle)
    pzflow_est = make_clean_stage(
        PZFlowEstimator,
        name="estimate_pzflow", model=pzflow_model, hdf5_groupname="",
        zmin=0.03, zmax=ZMAX, nzbins=NZ-1, seed=0,
        ref_band=ref_band, column_names=bands, mag_limits=mag_limits,
        include_mag_errors=False, redshift_col="redshift"
    )
    
    test_dict_for_flow = test_dict.copy()
    if 'redshift' not in test_dict_for_flow:
        test_dict_for_flow['redshift'] = np.zeros(len(test_dict_for_flow[list(test_dict_for_flow.keys())[0]]))
    test_handle_for_flow = TableHandle('test_data_flow', data=test_dict_for_flow)
    pdf_pzflow = pzflow_est.estimate(test_handle_for_flow).data.pdf(Z_CENTERS)
    
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
        replace_error_vals=[0.1] * len(bands), max_iter=20 if is_ci else 150, n_basis=60,
        train_frac=0.8, csl_method="normal", mag_limits=mag_limits
    )
    gpz_model = gpz_inf.inform(train_handle)
    gpz_est = make_clean_stage(
        GPzEstimator,
        name="estimate_gpz", model=gpz_model, hdf5_groupname="",
        bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
        replace_error_vals=[0.1] * len(bands), zmin=0.03, zmax=ZMAX, nzbins=NZ-1, mag_limits=mag_limits
    )
    pdf_gpz = gpz_est.estimate(test_handle).data.pdf(Z_CENTERS)
    pdf_gpz_train = gpz_est.estimate(train_handle).data.pdf(Z_CENTERS)

    # 7.8 Clean all PDFs
    pdf_nn1 = clean_pdf(pdf_nn1)
    pdf_nn1_train = clean_pdf(pdf_nn1_train)
    pdf_nn2 = clean_pdf(pdf_nn2)
    pdf_nn2_train = clean_pdf(pdf_nn2_train)
    pdf_knn = clean_pdf(pdf_knn)
    pdf_knn_train = clean_pdf(pdf_knn_train)
    pdf_som = clean_pdf(pdf_som)
    pdf_som_train = clean_pdf(pdf_som_train)
    pdf_bpz = clean_pdf(pdf_bpz)
    pdf_bpz_train = clean_pdf(pdf_bpz_train)
    pdf_fzboost = clean_pdf(pdf_fzboost)
    pdf_fzboost_train = clean_pdf(pdf_fzboost_train)
    pdf_aion = clean_pdf(pdf_aion)
    pdf_aion_train = clean_pdf(pdf_aion_train)
    pdf_pzflow = clean_pdf(pdf_pzflow)
    pdf_pzflow_train = clean_pdf(pdf_pzflow_train)
    pdf_gpz = clean_pdf(pdf_gpz)
    pdf_gpz_train = clean_pdf(pdf_gpz_train)
    if pdf_lephare is not None:
        pdf_lephare = clean_pdf(pdf_lephare)
        pdf_lephare_train = clean_pdf(pdf_lephare_train)

    # 8. Combine experts using KNN weights
    train_pdfs = [
        pdf_nn1_train, pdf_nn2_train, pdf_knn_train, pdf_som_train,
        pdf_bpz_train, pdf_fzboost_train, pdf_aion_train,
        pdf_pzflow_train, pdf_gpz_train
    ]
    test_pdfs = [
        pdf_nn1, pdf_nn2, pdf_knn, pdf_som,
        pdf_bpz, pdf_fzboost, pdf_aion,
        pdf_pzflow, pdf_gpz
    ]
    if pdf_lephare is not None:
        train_pdfs.append(pdf_lephare_train)
        test_pdfs.append(pdf_lephare)
    
    train_features_norm, train_errors, features_mean, features_std = compute_expert_weights_knn(
        train_dict, train_pdfs, Z_CENTERS, bands, ref_band
    )
    
    weighted_pdfs, val_expert_weights = apply_expert_weights_knn(
        test_dict, test_pdfs, train_features_norm, train_errors, features_mean, features_std, bands, ref_band, K=K_val
    )

    # 9. Apply temperature scaling to combined PDF
    weighted_train, _ = apply_expert_weights_knn(
        train_dict, train_pdfs, train_features_norm, train_errors, features_mean, features_std, bands, ref_band, K=K_val
    )
    best_t = aion_pz.fit_temperature(weighted_train, Z_CENTERS, train_dict['redshift'])
    calibrated_pdfs = aion_pz.apply_temperature(weighted_pdfs, Z_CENTERS, best_t)

    # Interpolate from Z_CENTERS to Z_GRID for writing
    from scipy.interpolate import interp1d
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 10. Write outputs
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)

    # Save models
    if save_model_to is not None:
        model_pzflow_bytes = None
        if os.path.exists("pzflow_model.pkl"):
            with open("pzflow_model.pkl", "rb") as f:
                model_pzflow_bytes = f.read()
            try:
                os.remove("pzflow_model.pkl")
            except OSError:
                pass

        model_dict = {
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
            "max_iter_som": max_iter_som
        }
        joblib.dump(model_dict, str(save_model_to), compress=3)


def estimate_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Run committee prediction using saved model weights."""
    model_dict = joblib.load(str(model_file))
    test_dict = aion_pz.load_catalog(test_file)

    bands = model_dict["bands"]
    ref_band = model_dict["ref_band"]
    is_roman = model_dict["is_roman"]
    K_val = model_dict["K_val"]
    bpz_filts = model_dict["bpz_filts"]
    bpz_zp = model_dict["bpz_zp"]
    err_bands = model_dict["err_bands"]
    max_iter_som = model_dict["max_iter_som"]

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
    full_mag_limits = {
        'mag_u_lsst': 26.4,
        'mag_g_lsst': 27.8,
        'mag_r_lsst': 27.1,
        'mag_i_lsst': 26.7,
        'mag_z_lsst': 25.8,
        'mag_y_lsst': 24.6,
        'mag_Y_roman': 26.5,
        'mag_J_roman': 26.5,
        'mag_H_roman': 26.5
    }
    mag_limits = {k: v for k, v in full_mag_limits.items() if k in bands}
    est_knn = make_clean_stage(
        k_nearneigh.KNearNeighEstimator,
        name='estimate_knn_eo', model=model_dict["model_knn"], bands=bands, ref_band=ref_band,
        hdf5_groupname='', zmin=0.03, zmax=ZMAX, nzbins=NZ-1, nondetect_val=np.nan,
        mag_limits=mag_limits
    )
    pdf_knn = est_knn.estimate(test_handle).data.pdf(Z_CENTERS)

    # 4. SOM
    pdf_som = get_som_pdfs(model_dict["train_dict_som"], test_dict, bands, ref_band, Z_GRID, max_iter=max_iter_som)

    # 5. BPZ
    bpz_est = make_clean_stage(
        BPZliteEstimator,
        name="estimate_bpz_eo", model=model_dict["model_bpz"], hdf5_groupname="",
        bands=bands, err_bands=err_bands, ref_band=ref_band, redshift_col="redshift",
        filter_list=bpz_filts, zp_errors=bpz_zp, mag_limits=mag_limits
    )
    pdf_bpz = bpz_est.estimate(test_handle).data.pdf(Z_CENTERS)

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

    # 7.8 LePhare conditionally (only if not is_roman)
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

    # 7.9 Clean all PDFs
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

    # 8. Combine experts using saved KNN weights
    test_pdfs = [
        pdf_nn1, pdf_nn2, pdf_knn, pdf_som,
        pdf_bpz, pdf_fzboost, pdf_aion,
        pdf_pzflow, pdf_gpz
    ]
    if pdf_lephare is not None:
        test_pdfs.append(pdf_lephare)

    train_errors = model_dict["train_errors"]
    if len(test_pdfs) > train_errors.shape[0]:
        n_missing = len(test_pdfs) - train_errors.shape[0]
        padding = np.repeat(train_errors[-1:], n_missing, axis=0)
        train_errors = np.vstack([train_errors, padding])
    elif len(test_pdfs) < train_errors.shape[0]:
        train_errors = train_errors[:len(test_pdfs)]

    weighted_pdfs, val_expert_weights = apply_expert_weights_knn(
        test_dict, test_pdfs,
        model_dict["train_features_norm"], train_errors,
        model_dict["features_mean"], model_dict["features_std"],
        bands, ref_band, K=K_val
    )

    # 9. Apply temperature scaling
    calibrated_pdfs = aion_pz.apply_temperature(weighted_pdfs, Z_CENTERS, model_dict["best_t"])

    # Interpolate from Z_CENTERS to Z_GRID for writing
    from scipy.interpolate import interp1d
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 10. Write outputs
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)
