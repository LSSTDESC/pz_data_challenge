import os
import logging
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import aion_pz
from .estimators import CommitteeOfExperts, get_bands_and_ref, Z_CENTERS, Z_GRID, extract_features
from .em import PontifexEM
from .guard import sanitize_input_catalog

logger = logging.getLogger(__name__)

def train_and_estimate(
    train_file: Union[str, Path],
    test_file: Union[str, Path],
    output_file: Union[str, Path],
    save_model_to: Union[str, Path, None] = None,
    seed: int = 42,
    optimize_hyperparams: bool = False,
) -> None:
    """
    Train the committee of experts, perform footprint-corrected EM calibration
    via Nugundam, and predict on the test catalog.
    """
    logger.info(f"Loading catalogs: train={train_file}, test={test_file}")
    train_dict = aion_pz.load_catalog(train_file)
    test_dict = aion_pz.load_catalog(test_file)

    # Sanitize and protect input catalogs from NaNs, Infs, and unphysical features
    train_dict, _ = sanitize_input_catalog(train_dict, raise_warnings=True)
    test_dict, _ = sanitize_input_catalog(test_dict, raise_warnings=True)

    # Resolve true redshifts
    z_true = np.asarray(train_dict[aion_pz.REDSHIFT_COL], dtype="float64")
    if aion_pz.MANYBAND_COL in train_dict:
        z_many = np.asarray(train_dict[aion_pz.MANYBAND_COL], dtype="float64")
        fill = ~np.isfinite(z_true) & np.isfinite(z_many)
        z_true[fill] = z_many[fill]
    good = np.isfinite(z_true)
    train_dict = {k: v[good] for k, v in train_dict.items()}
    train_dict['redshift'] = z_true[good]

    bands, ref_band, is_roman = get_bands_and_ref(list(train_dict.keys()))

    # Determine CI vs Production mode
    n_train = len(train_dict['redshift'])
    is_ci = (n_train < 1500) or ("PZDC_CI_MAX_TRAIN" in os.environ)

    # 1. Fit committee of experts
    logger.info("Fitting Committee of Experts...")
    committee = CommitteeOfExperts(is_ci=is_ci)
    committee.fit(train_dict, bands, ref_band, is_roman, optimize_hyperparams=optimize_hyperparams)

    # 2. Predict initial PDFs on Z_CENTERS
    logger.info("Predicting initial PDFs...")
    initial_pdfs = committee.predict(test_dict)

    # 3. Perform Nugundam EM Calibration selectively for blended & PCA-outlier sources
    logger.info("Identifying blended & outlier candidate sources via color variance and PCA feature subvolume...")
    colors = []
    for i in range(len(bands) - 1):
        if bands[i] in test_dict and bands[i+1] in test_dict:
            c = np.asarray(test_dict[bands[i]], dtype=float) - np.asarray(test_dict[bands[i+1]], dtype=float)
            colors.append(np.nan_to_num(c, nan=0.0))
    if len(colors) > 0:
        colors_mat = np.column_stack(colors)
        color_var = np.var(colors_mat, axis=1)
        med_var = np.median(color_var)
        blend_score = color_var / (med_var + 1e-15)
        is_blend = blend_score > 4.5
    else:
        is_blend = np.zeros(len(initial_pdfs), dtype=bool)

    # Calculate 3D PCA Outlier Subvolume Mask using smooth curved ellipsoidal contour
    feat_test = extract_features(test_dict, bands, ref_band)
    if hasattr(committee, "scaler") and committee.scaler is not None:
        feat_norm = committee.scaler.transform(feat_test)
    else:
        scaler = StandardScaler().fit(feat_test)
        feat_norm = scaler.transform(feat_test)

    if hasattr(committee, "pca") and committee.pca is not None:
        pca_space = committee.pca.transform(feat_norm)
    else:
        pca = PCA(n_components=3).fit(feat_norm)
        pca_space = pca.transform(feat_norm)

    # Ellipsoidal Outlier Region centered at [PC1=2.5, PC2=-2.5] enclosing the concentration of green/red outliers
    ell_dist = ((pca_space[:, 0] - 2.5)**2 / (3.2**2)) + ((pca_space[:, 1] + 2.5)**2 / (3.5**2))
    is_pca_outlier_vol = ell_dist <= 1.0
    is_em_triggered = is_blend | is_pca_outlier_vol

    n_triggered = int(np.sum(is_em_triggered))
    calibrated_pdfs = initial_pdfs.copy()

    ref_df = pd.DataFrame({
        'ra': train_dict['ra'],
        'dec': train_dict['dec'],
        'ztrue': train_dict['redshift']
    })
    unk_df = pd.DataFrame({
        'ra': test_dict['ra'],
        'dec': test_dict['dec']
    })

    if n_triggered > 0:
        logger.info(f"Running Nugundam EM Calibration selectively on {n_triggered}/{len(initial_pdfs)} candidate sources (Blends: {int(np.sum(is_blend))}, PCA Volume: {int(np.sum(is_pca_outlier_vol))})...")
        em_max_iter = 2 if is_ci else 5
        pso_params = committee.model_dict.get("pso_params", {})
        em_params = pso_params.get("em", {})
        lr = em_params.get("learning_rate", 0.2)
        smooth = em_params.get("smoothing_sigma", 1.0)
        
        unk_trig_df = unk_df.iloc[is_em_triggered].reset_index(drop=True)
        initial_trig_pdfs = initial_pdfs[is_em_triggered]

        em = PontifexEM(
            unk_df=unk_trig_df,
            ref_df=ref_df,
            initial_pdfs=initial_trig_pdfs,
            z_grid_edges=Z_GRID,
            use_bias_correction=True,
            floor_val=1e-5,
            learning_rate=lr,
            smoothing_sigma=smooth,
            is_ci=is_ci
        )
        calibrated_trig_pdfs = em.optimize(max_iter=em_max_iter)
        calibrated_pdfs[is_em_triggered] = calibrated_trig_pdfs
    else:
        logger.info("No blended or outlier sources detected above threshold. Keeping unimodal calibrated PDFs.")

    # 4. Interpolate and normalize onto Z_GRID
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 5. Write outputs and save model weights
    logger.info(f"Writing final calibrated PDFs to {output_file}")
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)

    if save_model_to is not None:
        logger.info(f"Saving trained model weights to {save_model_to}")
        committee.save(str(save_model_to))


def estimate_only(
    model_file: Union[str, Path],
    test_file: Union[str, Path],
    output_file: Union[str, Path],
) -> None:
    """
    Run committee prediction and Nugundam EM calibration selectively on blended and outlier sources using saved model weights.
    """
    logger.info(f"Loading model weights from {model_file}...")
    committee = CommitteeOfExperts()
    committee.load(str(model_file))
    
    logger.info(f"Loading test catalog: {test_file}")
    test_dict = aion_pz.load_catalog(test_file)
    test_dict, _ = sanitize_input_catalog(test_dict, raise_warnings=True)

    # 1. Predict initial PDFs on Z_CENTERS
    logger.info("Predicting initial PDFs...")
    initial_pdfs = committee.predict(test_dict)

    # 2. Identify blended and PCA outlier candidate sources
    bands = committee.model_dict["bands"]
    ref_band = committee.model_dict.get("ref_band", "i_lsst")
    colors = []
    for i in range(len(bands) - 1):
        if bands[i] in test_dict and bands[i+1] in test_dict:
            c = np.asarray(test_dict[bands[i]], dtype=float) - np.asarray(test_dict[bands[i+1]], dtype=float)
            colors.append(np.nan_to_num(c, nan=0.0))
    if len(colors) > 0:
        colors_mat = np.column_stack(colors)
        color_var = np.var(colors_mat, axis=1)
        med_var = np.median(color_var)
        blend_score = color_var / (med_var + 1e-15)
        is_blend = blend_score > 4.5
    else:
        is_blend = np.zeros(len(initial_pdfs), dtype=bool)

    feat_test = extract_features(test_dict, bands, ref_band)
    if hasattr(committee, "scaler") and committee.scaler is not None:
        feat_norm = committee.scaler.transform(feat_test)
    else:
        scaler = StandardScaler().fit(feat_test)
        feat_norm = scaler.transform(feat_test)

    if hasattr(committee, "pca") and committee.pca is not None:
        pca_space = committee.pca.transform(feat_norm)
    else:
        pca = PCA(n_components=3).fit(feat_norm)
        pca_space = pca.transform(feat_norm)

    # Ellipsoidal Outlier Region centered at [PC1=2.5, PC2=-2.5] enclosing the concentration of green/red outliers
    ell_dist = ((pca_space[:, 0] - 2.5)**2 / (3.2**2)) + ((pca_space[:, 1] + 2.5)**2 / (3.5**2))
    is_pca_outlier_vol = ell_dist <= 1.0
    is_em_triggered = is_blend | is_pca_outlier_vol

    n_triggered = int(np.sum(is_em_triggered))
    calibrated_pdfs = initial_pdfs.copy()

    # 3. Run Nugundam EM Calibration selectively using stored training coordinates as reference
    train_som = committee.model_dict["train_dict_som"]
    ref_df = pd.DataFrame({
        'ra': train_som['ra'],
        'dec': train_som['dec'],
        'ztrue': train_som['redshift']
    })
    unk_df = pd.DataFrame({
        'ra': test_dict['ra'],
        'dec': test_dict['dec']
    })

    if n_triggered > 0:
        logger.info(f"Running Nugundam EM Calibration selectively on {n_triggered}/{len(initial_pdfs)} candidate sources (Blends: {int(np.sum(is_blend))}, PCA Volume: {int(np.sum(is_pca_outlier_vol))})...")
        n_test = len(test_dict[list(test_dict.keys())[0]])
        is_ci = (n_test < 1500) or ("PZDC_CI_MAX_TRAIN" in os.environ)
        em_max_iter = 2 if is_ci else 5
        pso_params = committee.model_dict.get("pso_params", {})
        em_params = pso_params.get("em", {})
        lr = em_params.get("learning_rate", 0.2)
        smooth = em_params.get("smoothing_sigma", 1.0)

        unk_trig_df = unk_df.iloc[is_em_triggered].reset_index(drop=True)
        initial_trig_pdfs = initial_pdfs[is_em_triggered]

        em = PontifexEM(
            unk_df=unk_trig_df,
            ref_df=ref_df,
            initial_pdfs=initial_trig_pdfs,
            z_grid_edges=Z_GRID,
            use_bias_correction=True,
            floor_val=1e-5,
            learning_rate=lr,
            smoothing_sigma=smooth,
            is_ci=is_ci
        )
        calibrated_trig_pdfs = em.optimize(max_iter=em_max_iter)
        calibrated_pdfs[is_em_triggered] = calibrated_trig_pdfs
    else:
        logger.info("No blended or outlier sources detected above threshold. Keeping unimodal calibrated PDFs.")

    # 4. Interpolate and normalize onto Z_GRID
    f_interp = interp1d(Z_CENTERS, calibrated_pdfs, axis=1, kind='linear', fill_value='extrapolate')
    calibrated_pdfs_301 = f_interp(Z_GRID)
    calibrated_pdfs_301 = aion_pz._renorm(calibrated_pdfs_301, Z_GRID)

    # 5. Write outputs
    logger.info(f"Writing final calibrated PDFs to {output_file}")
    aion_pz.write_qp(calibrated_pdfs_301, test_dict[aion_pz.OBJECT_ID_COL], output_file, z_grid=Z_GRID)

