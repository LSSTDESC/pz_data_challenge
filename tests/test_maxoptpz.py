#mariano.dominguez@unc.edu.ar july 2026
import sys
import tables_io
tables_io.hdf5 = tables_io.h5py
sys.modules['tables_io.hdf5'] = tables_io.h5py
import logging
from rail.core.stage import RailStage
RailStage.log = logging.getLogger('rail')
import tables_io.types
if not hasattr(tables_io.types, 'table_type'):
    tables_io.types.table_type = tables_io.types.tableType
if not hasattr(tables_io.types, 'file_type'):
    tables_io.types.file_type = tables_io.types.fileType

import os
import pickle
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import pytest
import qp
from pathlib import Path
from rail.core.data import TableHandle
from rail.estimation.algos import sklearn_neurnet

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

SUBMISSION_NAME: str = "maxoptpz"
SUBMISSION_URL: str = "https://github.com/mardom/pz_data_challenge/releases/download/v1.0.0/maxoptpz.tgz"
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

bands = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
ref_band = 'mag_i_lsst'
z_grid = np.linspace(0.03, 1.5, 61)
z_centers = 0.5 * (z_grid[:-1] + z_grid[1:])


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
        os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"))
    except Exception:
        pass

    try:
        os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"))
    except Exception:
        pass

    request.addfinalizer(teardown_submit_area)
    return 0


from rail.estimation.algos import sklearn_neurnet, k_nearneigh
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.estimation.algos.flexzboost import FlexZBoostInformer, FlexZBoostEstimator
from minisom import MiniSom
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors
from rail.utils import catalog_utils
catalog_utils.load_yaml('tests/catalogs.yaml')
catalog_utils.apply('cardinal_roman_rubin')

# Monkey-patch flexcode to force n_jobs = 1 (avoid Loky multiprocessing context-switching thrashing)
import flexcode.regression_models
old_init = flexcode.regression_models.XGBoost.__init__
def new_init(self, max_basis, params, *args, **kwargs):
    kwargs['n_jobs'] = 1
    old_init(self, max_basis, params, *args, **kwargs)
    self.n_jobs = 1
    if self.models is not None:
        self.models.n_jobs = 1
flexcode.regression_models.XGBoost.__init__ = new_init


def get_filter_list(bands):
    filters = []
    for b in bands:
        if 'lsst' in b:
            short_name = b.split('_')[1]
            filters.append(f"DC2LSST_{short_name}")
        elif 'roman' in b:
            short_name = b.split('_')[1]
            if short_name == 'Y':
                filters.append("roman_Y106")
            elif short_name == 'J':
                filters.append("roman_J129")
            elif short_name == 'H':
                filters.append("roman_H158")
            else:
                filters.append(f"roman_{short_name}")
        else:
            filters.append(b)
    return filters



def get_features(d, bands, ref_band):
    numcols = len(bands)
    coldata = np.array(d[ref_band])
    for i in range(numcols - 1):
        tmpcolor = d[bands[i]] - d[bands[i+1]]
        coldata = np.vstack((coldata, tmpcolor))
    return coldata.T

def extract_features(data_dict, bands, ref_band):
    features = []
    for band in bands:
        mag = data_dict[band].copy()
        mag = np.where(np.isnan(mag), np.nanmedian(mag) if not np.isnan(np.nanmedian(mag)) else 99.0, mag)
        features.append(mag)
    for i in range(len(bands) - 1):
        col = (data_dict[bands[i]] - data_dict[bands[i+1]]).copy()
        col = np.where(np.isnan(col), np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0.0, col)
        features.append(col)
    features = np.column_stack(features)
    return features

def clean_pdf(pdfs):
    pdfs = np.nan_to_num(pdfs, nan=1.0/len(z_centers))
    row_sums = pdfs.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, pdfs / row_sums, 1.0 / len(z_centers))

class CustomSOM:
    def __init__(self, n_dim=10, m_dim=10):
        self.n_dim = n_dim
        self.m_dim = m_dim
        self.som = None
        self.pixel_pdfs = None
        self.global_pdf = None
        
    def fit(self, train_feat, redshift):
        self.som = MiniSom(self.n_dim, self.m_dim, train_feat.shape[1], sigma=1.5, learning_rate=0.5, random_seed=42)
        self.som.pca_weights_init(train_feat)
        self.som.train(train_feat, min(5000, len(train_feat)), verbose=False)
        
        train_winners = np.array([self.som.winner(x) for x in train_feat])
        train_pixels = np.ravel_multi_index(train_winners.T, (self.n_dim, self.m_dim))
        
        self.pixel_pdfs = {}
        global_hist, _ = np.histogram(redshift, bins=z_grid)
        self.global_pdf = global_hist / (np.sum(global_hist) + 1e-15)
        
        for pix in range(self.n_dim * self.m_dim):
            mask = (train_pixels == pix)
            if mask.sum() > 2:
                hist, _ = np.histogram(redshift[mask], bins=z_grid)
                self.pixel_pdfs[pix] = hist / (np.sum(hist) + 1e-15)
            else:
                self.pixel_pdfs[pix] = self.global_pdf
                
    def predict(self, test_feat):
        test_winners = np.array([self.som.winner(x) for x in test_feat])
        test_pixels = np.ravel_multi_index(test_winners.T, (self.n_dim, self.m_dim))
        
        test_pdfs = np.zeros((len(test_feat), len(z_centers)))
        for i, pix in enumerate(test_pixels):
            test_pdfs[i] = self.pixel_pdfs[pix]
            
        test_pdfs = gaussian_filter1d(test_pdfs, sigma=1.0, axis=1)
        return clean_pdf(test_pdfs)

def compute_expert_weights_knn(train_dict, train_pdfs, z_centers, bands, ref_band):
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
    train_errors = np.array(train_errors)
    
    return train_features_norm, train_errors, features_mean, features_std

def apply_expert_weights_knn_pred(data_dict, expert_pdfs, train_features_norm, train_errors, features_mean, features_std, bands, ref_band, K=50):
    val_features = extract_features(data_dict, bands, ref_band)
    val_features_norm = (val_features - features_mean) / features_std
    
    K_actual = min(K, len(train_features_norm))
    nn = NearestNeighbors(n_neighbors=K_actual, algorithm='auto', n_jobs=-1).fit(train_features_norm)
    dists, indices = nn.kneighbors(val_features_norm)
    
    sigmas = dists[:, -1]
    sigmas = np.maximum(sigmas, 1e-5)
    kernel_weights = np.exp(- (dists ** 2) / (2.0 * sigmas[:, np.newaxis] ** 2))
    kernel_weights_sum = np.sum(kernel_weights, axis=1, keepdims=True)
    kernel_weights = kernel_weights / kernel_weights_sum
    
    n_data = len(data_dict['object_id'])
    n_experts = len(expert_pdfs)
    
    val_expert_weights = np.zeros((n_data, n_experts))
    for k in range(n_experts):
        neighbor_errors = train_errors[k, indices]
        mean_err = np.sum(kernel_weights * neighbor_errors, axis=1)
        val_expert_weights[:, k] = 1.0 / (mean_err + 1e-5)
        
    val_expert_weights_sum = np.sum(val_expert_weights, axis=1, keepdims=True)
    val_expert_weights = val_expert_weights / (val_expert_weights_sum + 1e-15)
    
    weighted_pdfs = np.zeros_like(expert_pdfs[0])
    for idx in range(n_data):
        w = val_expert_weights[idx]
        pdf_sum = np.zeros_like(weighted_pdfs[idx])
        for k in range(n_experts):
            pdf_sum += w[k] * expert_pdfs[k][idx]
        weighted_pdfs[idx] = pdf_sum / (np.sum(pdf_sum) + 1e-15)
        
    return clean_pdf(weighted_pdfs)

def train_experts(train_handle, bands, ref_band, z_grid, stage_name):
    # Train NN1
    informer_nn1 = sklearn_neurnet.SklNeurNetInformer.make_stage(
        name=f'inform_nn1_{stage_name}', bands=bands, ref_band=ref_band,
        redshift_col='redshift', width=0.03, max_iter=50, hdf5_groupname=''
    )
    model_nn1 = informer_nn1.inform(train_handle)
    
    # Train NN2
    informer_nn2 = sklearn_neurnet.SklNeurNetInformer.make_stage(
        name=f'inform_nn2_{stage_name}', bands=bands, ref_band=ref_band,
        redshift_col='redshift', width=0.06, max_iter=50, hdf5_groupname=''
    )
    model_nn2 = informer_nn2.inform(train_handle)
    
    # Train KNN
    mag_limits = {b: 28.0 for b in bands}
    informer_knn = k_nearneigh.KNearNeighInformer.make_stage(
        name=f'inform_knn_{stage_name}', bands=bands, ref_band=ref_band,
        redshift_col='redshift', hdf5_groupname='',
        zmin=0.03, zmax=1.5, nzbins=61, nondetect_val=np.nan,
        mag_limits=mag_limits, nneigh_min=3, nneigh_max=5
    )
    model_knn = informer_knn.inform(train_handle)
    
    # Train SOM
    train_dict = train_handle.data
    train_feat = get_features(train_dict, bands, ref_band)
    som_model = CustomSOM()
    som_model.fit(train_feat, train_dict['redshift'])
    
    # Train BPZ
    err_bands = [f"{b}_err" for b in bands]
    filter_list = get_filter_list(bands)
    informer_bpz = BPZliteInformer.make_stage(
        name=f'inform_bpz_{stage_name}', bands=bands, err_bands=err_bands, filter_list=filter_list, ref_band=ref_band,
        redshift_col='redshift', hdf5_groupname='', zmin=0.03, zmax=1.5, nzbins=61
    )
    model_bpz = informer_bpz.inform(train_handle)
    
    # For FlexZBoost specifically, downsample training set to 3000 to keep it very fast
    n_fz = len(train_handle.data['redshift'])
    if n_fz > 3000:
        rng_fz = np.random.default_rng(42)
        idx_fz = rng_fz.choice(n_fz, size=3000, replace=False)
        fz_train_data = {k: v[idx_fz] for k, v in train_handle.data.items()}
        fz_train_handle = TableHandle(f'fz_train_{stage_name}', data=fz_train_data)
    else:
        fz_train_handle = train_handle

    fz_dict = dict(zmin=0.03, zmax=1.5, nzbins=61,
                   trainfrac=1.0, bumpmin=0.02, bumpmax=0.35,
                   nbump=1, sharpmin=0.7, sharpmax=2.1, nsharp=1,
                   max_basis=15, basis_system='cosine',
                   hdf5_groupname='',
                   regression_params={'max_depth': 5, 'n_estimators': 20, 'n_jobs': 1, 'objective': 'reg:squarederror'})
    informer_fzboost = FlexZBoostInformer.make_stage(
        name=f'inform_fzboost_{stage_name}', bands=bands, err_bands=err_bands, ref_band=ref_band,
        redshift_col='redshift', **fz_dict
    )
    model_fzboost = informer_fzboost.inform(fz_train_handle)
    
    return {
        'NN1': model_nn1,
        'NN2': model_nn2,
        'KNN': model_knn,
        'SOM': som_model,
        'BPZ': model_bpz,
        'FZB': model_fzboost
    }

def predict_expert(model_name, model_obj, data_handle, bands, ref_band, stage_name):
    if model_name == 'SOM':
        feat = get_features(data_handle.data, bands, ref_band)
        return model_obj.predict(feat)
    elif model_name == 'NN1':
        est = sklearn_neurnet.SklNeurNetEstimator.make_stage(
            name=f'est_nn1_{stage_name}', model=model_obj, bands=bands, ref_band=ref_band, width=0.03, hdf5_groupname=''
        )
        return clean_pdf(est.estimate(data_handle).data.pdf(z_centers))
    elif model_name == 'NN2':
        est = sklearn_neurnet.SklNeurNetEstimator.make_stage(
            name=f'est_nn2_{stage_name}', model=model_obj, bands=bands, ref_band=ref_band, width=0.06, hdf5_groupname=''
        )
        return clean_pdf(est.estimate(data_handle).data.pdf(z_centers))
    elif model_name == 'KNN':
        mag_limits = {b: 28.0 for b in bands}
        est = k_nearneigh.KNearNeighEstimator.make_stage(
            name=f'est_knn_{stage_name}', model=model_obj, bands=bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=1.5, nzbins=61, nondetect_val=np.nan,
            mag_limits=mag_limits
        )
        return clean_pdf(est.estimate(data_handle).data.pdf(z_centers))
    elif model_name == 'BPZ':
        err_bands = [f"{b}_err" for b in bands]
        filter_list = get_filter_list(bands)
        zp_errors = [0.1] * len(bands)
        est = BPZliteEstimator.make_stage(
            name=f'est_bpz_{stage_name}', model=model_obj, bands=bands, err_bands=err_bands,
            filter_list=filter_list, zp_errors=zp_errors, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=1.5, nzbins=61
        )
        return clean_pdf(est.estimate(data_handle).data.pdf(z_centers))
    elif model_name == 'FZB':
        err_bands = [f"{b}_err" for b in bands]
        est = FlexZBoostEstimator.make_stage(
            name=f'est_fzb_{stage_name}', model=model_obj, bands=bands, err_bands=err_bands, ref_band=ref_band,
            hdf5_groupname='', zmin=0.03, zmax=1.5, nzbins=61
        )
        return clean_pdf(est.estimate(data_handle).data.pdf(z_centers))

def _run_estimation_common(model_file: str | Path, test_file: str | Path, output_file: str | Path, stage_name: str) -> None:
    import sys
    sys.modules['__main__'].CustomSOM = CustomSOM
    with open(model_file, 'rb') as f:
        moe_model = pickle.load(f)
        
    test_data = tables_io.read(test_file)
    for b in bands:
        test_data[b] = np.nan_to_num(test_data[b], nan=99.0)
        
    test_handle = TableHandle('test_data', data=test_data)
    
    # 1. Generate PDFs for each expert
    expert_pdfs = []
    for name, model_obj in moe_model['models'].items():
        pdf = predict_expert(name, model_obj, test_handle, bands, ref_band, f"{stage_name}_pred_{name}")
        expert_pdfs.append(pdf)
        
    # 2. Apply weights to compute the final weighted PDF
    pdfs = apply_expert_weights_knn_pred(
        test_data, expert_pdfs, moe_model['train_features_norm'], moe_model['train_errors'],
        moe_model['features_mean'], moe_model['features_std'], bands, ref_band, K=moe_model['K']
    )
    
    z_mode = z_centers[np.argmax(pdfs, axis=1)]
    ancil = {'zmode': z_mode, 'object_id': test_data['object_id']}
    ensemble = qp.Ensemble(qp.interp, data=dict(xvals=z_centers, yvals=pdfs), ancil=ancil)
    ensemble.write_to(output_file)

def _run_training_and_estimation_common(train_file: str | Path, test_file: str | Path, output_file: str | Path, stage_name: str) -> None:
    train_data = tables_io.read(train_file)
    test_data = tables_io.read(test_file)
    
    for b in bands:
        train_data[b] = np.nan_to_num(train_data[b], nan=99.0)
        test_data[b] = np.nan_to_num(test_data[b], nan=99.0)
        
    mask_valid_z = ~np.isnan(train_data['redshift'])
    train_data = {k: v[mask_valid_z] for k, v in train_data.items()}
    
    # Downsample training set if it is very large to speed up tests
    n_train = len(train_data['redshift'])
    if n_train > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_train, size=10000, replace=False)
        train_data = {k: v[idx] for k, v in train_data.items()}
        
    train_handle = TableHandle('train_data', data=train_data)
    test_handle = TableHandle('test_data', data=test_data)
    
    # 1. Train all estimators
    models = train_experts(train_handle, bands, ref_band, z_grid, stage_name)
    
    # 2. Predict on training data to compute local errors
    train_pdfs = []
    for name, model_obj in models.items():
        pdf = predict_expert(name, model_obj, train_handle, bands, ref_band, f"{stage_name}_train_eval_{name}")
        train_pdfs.append(pdf)
        
    # 3. Compute expert weights KNN setup
    train_features_norm, train_errors, features_mean, features_std = compute_expert_weights_knn(
        train_data, train_pdfs, z_centers, bands, ref_band
    )
    
    moe_model = {
        'models': models,
        'train_features_norm': train_features_norm,
        'train_errors': train_errors,
        'features_mean': features_mean,
        'features_std': features_std,
        'K': 50
    }
    
    # 4. Generate PDFs for test set
    expert_pdfs = []
    for name, model_obj in models.items():
        pdf = predict_expert(name, model_obj, test_handle, bands, ref_band, f"{stage_name}_pred_test_{name}")
        expert_pdfs.append(pdf)
        
    # 5. Apply weights
    pdfs = apply_expert_weights_knn_pred(
        test_data, expert_pdfs, train_features_norm, train_errors,
        features_mean, features_std, bands, ref_band, K=50
    )
    
    z_mode = z_centers[np.argmax(pdfs, axis=1)]
    ancil = {'zmode': z_mode, 'object_id': test_data['object_id']}
    ensemble = qp.Ensemble(qp.interp, data=dict(xvals=z_centers, yvals=pdfs), ancil=ancil)
    ensemble.write_to(output_file)



# Taskset 1
def run_taskset_1_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_estimation_common(model_file, test_file, output_file, 't1_est_only')

def run_taskset_1_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_training_and_estimation_common(train_file, test_file, output_file, 't1_train_est')


# Taskset 2
def run_taskset_2_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_estimation_common(model_file, test_file, output_file, 't2_est_only')

def run_taskset_2_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_training_and_estimation_common(train_file, test_file, output_file, 't2_train_est')


# Taskset 3
def run_taskset_3_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_estimation_common(model_file, test_file, output_file, 't3_est_only')

def run_taskset_3_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_training_and_estimation_common(train_file, test_file, output_file, 't3_train_est')


# Taskset 4
def run_taskset_4_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_estimation_common(model_file, test_file, output_file, 't4_est_only')

def run_taskset_4_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_training_and_estimation_common(train_file, test_file, output_file, 't4_train_est')


# Tests
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
