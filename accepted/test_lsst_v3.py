import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import qp
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils


SUBMISSION_NAME: str = "lsst_v3"
SUBMISSION_URL: str = "https://github.com/e06243046/pz_data_challenge_submission/releases/download/submission-lsst_v3_v1/lsst_v3_submission.tgz"

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = os.environ.get("PZDC_PUBLIC_AREA", "tests/public")

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
ROMAN_BANDS = ["Y", "J", "H"]
ALL_BANDS = LSST_BANDS + ROMAN_BANDS
COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("z", "y"), ("y", "Y"), ("Y", "J"), ("J", "H"),
    ("u", "r"), ("u", "i"), ("g", "i"), ("g", "z"),
    ("r", "z"), ("r", "y"), ("i", "Y"), ("i", "J"),
    ("i", "H"), ("z", "Y"), ("z", "J"), ("z", "H"),
    ("y", "H"), ("Y", "H"),
]
CURVATURE_SPECS = [
    ("sed_curvature_u_g_r", "u_g", "g_r"),
    ("sed_curvature_g_r_i", "g_r", "r_i"),
    ("sed_curvature_r_i_z", "r_i", "i_z"),
    ("sed_curvature_i_z_Y", "i_z", "z_Y"),
    ("sed_curvature_z_Y_J", "z_Y", "Y_J"),
    ("sed_curvature_Y_J_H", "Y_J", "J_H"),
    ("sed_curvature_g_i_H", "g_i", "i_H"),
    ("sed_curvature_r_z_H", "r_z", "z_H"),
    ("sed_curvature_i_Y_H", "i_Y", "Y_H"),
]


def _mag_key(band: str) -> str:
    return f"mag_{band}_lsst" if band in LSST_BANDS else f"mag_{band}_roman"


def _err_key(band: str) -> str:
    return f"mag_{band}_lsst_err" if band in LSST_BANDS else f"mag_{band}_roman_err"


def _build_base_features(path: str | Path, require_redshift: bool = False):
    with h5py.File(path, "r") as f:
        object_id = f["object_id"][:].astype(int)
        mags = {}
        errs = {}
        finite_mag = {}
        nan_flags = {}
        for band in ALL_BANDS:
            mag = f[_mag_key(band)][:].astype(float)
            err = f[_err_key(band)][:].astype(float)
            missing = np.isnan(mag)
            finite_mag[band] = ~missing
            nan_flags[f"nan_{band}"] = missing.astype(float)
            mag[missing] = 99.0
            err[np.isnan(err)] = 99.0
            mags[_mag_key(band)] = mag
            errs[_err_key(band)] = err

        colors = {}
        color_missing = {}
        color_errs = {}
        for b1, b2 in COLOR_PAIRS:
            pair = f"{b1}_{b2}"
            valid = finite_mag[b1] & finite_mag[b2]
            color = np.zeros(len(object_id), dtype=float)
            raw_color = mags[_mag_key(b1)] - mags[_mag_key(b2)]
            color[valid] = np.clip(raw_color[valid], -10.0, 10.0)
            colors[f"color_{pair}"] = color
            color_missing[f"color_missing_{pair}"] = (~valid).astype(float)
            color_err = np.sqrt(errs[_err_key(b1)] ** 2 + errs[_err_key(b2)] ** 2)
            color_err[~valid] = 99.0
            color_errs[f"color_err_{pair}"] = np.clip(color_err, 0.0, 99.0)

        curvature_features = {}
        curvature_missing = {}
        curvature_errs = {}
        for name, left_pair, right_pair in CURVATURE_SPECS:
            left = colors[f"color_{left_pair}"]
            right = colors[f"color_{right_pair}"]
            missing = np.maximum(
                color_missing[f"color_missing_{left_pair}"],
                color_missing[f"color_missing_{right_pair}"],
            )
            valid = missing == 0
            curvature = np.zeros(len(object_id), dtype=float)
            curvature[valid] = np.clip(left[valid] - right[valid], -20.0, 20.0)
            curvature_features[name] = curvature
            curvature_missing[f"{name}_missing"] = missing
            coeffs = {}
            for band, coeff in [
                (left_pair.split("_")[0], 1.0),
                (left_pair.split("_")[1], -1.0),
                (right_pair.split("_")[0], -1.0),
                (right_pair.split("_")[1], 1.0),
            ]:
                coeffs[band] = coeffs.get(band, 0.0) + coeff
            variance = np.zeros(len(object_id), dtype=float)
            for band, coeff in coeffs.items():
                variance += (coeff * errs[_err_key(band)]) ** 2
            curv_err = np.sqrt(variance)
            curv_err[~valid] = 99.0
            curvature_errs[f"{name}_err"] = np.clip(curv_err, 0.0, 99.0)

        detected_matrix = np.column_stack([finite_mag[band] for band in ALL_BANDS])
        mag_matrix = np.column_stack([mags[_mag_key(band)] for band in ALL_BANDS])
        masked_mags = np.where(detected_matrix, mag_matrix, np.nan)
        with np.errstate(all="ignore"):
            detection_features = {
                "n_detected_lsst": np.sum([finite_mag[b] for b in LSST_BANDS], axis=0).astype(float),
                "n_detected_roman": np.sum([finite_mag[b] for b in ROMAN_BANDS], axis=0).astype(float),
                "n_detected_total": np.sum([finite_mag[b] for b in ALL_BANDS], axis=0).astype(float),
                "mean_mag_detected": np.nan_to_num(np.nanmean(masked_mags, axis=1), nan=99.0),
                "brightest_detected_mag": np.nan_to_num(np.nanmin(masked_mags, axis=1), nan=99.0),
                "faintest_detected_mag": np.nan_to_num(np.nanmax(masked_mags, axis=1), nan=99.0),
                "optical_to_nir_color_i_H": colors["color_i_H"],
            }

        feature_names = []
        feature_arrays = []
        for band in ALL_BANDS:
            err = errs[_err_key(band)]
            for fname, arr in [
                (_mag_key(band), mags[_mag_key(band)]),
                (_err_key(band), err),
                (f"nan_{band}", nan_flags[f"nan_{band}"]),
                (f"snr_{band}", np.clip(1.0857 / np.maximum(err, 1e-6), 0.0, 1000.0)),
                (f"logerr_{band}", np.log1p(np.clip(err, 0.0, 99.0))),
            ]:
                feature_names.append(fname)
                feature_arrays.append(arr)
        for b1, b2 in COLOR_PAIRS:
            pair = f"{b1}_{b2}"
            for fname, arr in [
                (f"color_{pair}", colors[f"color_{pair}"]),
                (f"color_missing_{pair}", color_missing[f"color_missing_{pair}"]),
                (f"color_err_{pair}", color_errs[f"color_err_{pair}"]),
            ]:
                feature_names.append(fname)
                feature_arrays.append(arr)
        for name, _, _ in CURVATURE_SPECS:
            for fname, arr in [
                (name, curvature_features[name]),
                (f"{name}_missing", curvature_missing[f"{name}_missing"]),
                (f"{name}_err", curvature_errs[f"{name}_err"]),
            ]:
                feature_names.append(fname)
                feature_arrays.append(arr)
        for fname, arr in detection_features.items():
            feature_names.append(fname)
            feature_arrays.append(arr)

        redshift = f["redshift"][:] if require_redshift else None

    return np.column_stack(feature_arrays), redshift, object_id, np.array(feature_names, dtype=str)


def _neutral_prior_block(n_objects: int, zgrid: np.ndarray):
    prior = np.ones((n_objects, len(zgrid)), dtype=float) / (zgrid[-1] - zgrid[0])
    summary = np.column_stack([
        np.full(n_objects, 1.5),
        np.full(n_objects, 1.5),
        np.full(n_objects, np.sqrt(0.75)),
        np.full(n_objects, 0.48),
        np.full(n_objects, 1.5),
        np.full(n_objects, 2.52),
        np.full(n_objects, 1.0 / 3.0),
        np.full(n_objects, np.log(3.0)),
    ])
    return np.hstack((prior, summary))


def _extract_pz_vals(predict_output, n_samples: int, n_grid: int):
    if isinstance(predict_output, tuple):
        for item in predict_output:
            arr = np.asarray(item)
            if arr.shape == (n_samples, n_grid):
                return arr
            if arr.shape == (n_grid, n_samples):
                return arr.T
    arr = np.asarray(predict_output)
    return arr.T if arr.shape == (n_grid, n_samples) else arr


def _apply_calibration(pz_vals, z_mode, mag_val, model_bundle):
    zgrid = model_bundle["zgrid"]
    pz_calib = np.copy(pz_vals)
    for i in range(5):
        for j in range(5):
            mask = (
                (z_mode >= model_bundle["z_bins_edges"][i])
                & (z_mode < model_bundle["z_bins_edges"][i + 1])
                & (mag_val >= model_bundle["mag_bins_edges"][j])
                & (mag_val < model_bundle["mag_bins_edges"][j + 1])
            )
            if np.any(mask):
                tempered = pz_calib[mask] ** (1.0 / model_bundle["T_map"][i, j])
                norms = np.trapz(tempered, zgrid, axis=1)
                norms[norms == 0.0] = 1.0
                pz_calib[mask] = tempered / norms[:, None]
    return pz_calib


def _write_interp_ensemble(output_file: str | Path, zgrid, pz_vals, object_id):
    pz_vals = np.clip(np.asarray(pz_vals), 0.0, None)
    norms = np.trapz(pz_vals, zgrid, axis=1)
    norms[norms == 0.0] = 1.0
    pz_vals = pz_vals / norms[:, None]
    zmode = zgrid[np.argmax(pz_vals, axis=1)]
    ensemble = qp.interp.create_ensemble(zgrid, pz_vals)
    ensemble.set_ancil({"object_id": object_id.astype(int), "zmode": zmode})
    ensemble.write_to(output_file)


def _run_lsst_v3_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    with open(model_file, "rb") as handle:
        model_bundle = pickle.load(handle)
    X_base, _, object_id, _ = _build_base_features(test_file)
    X_scaled = model_bundle["scaler"].transform(X_base)
    zgrid = model_bundle["zgrid"]
    X = np.hstack((X_scaled, _neutral_prior_block(len(X_scaled), zgrid)))
    pz_vals = _extract_pz_vals(model_bundle["model"].predict(X, n_grid=len(zgrid)), len(X), len(zgrid))
    pz_vals = np.clip(pz_vals, 0.0, None)
    norms = np.trapz(pz_vals, zgrid, axis=1)
    norms[norms == 0.0] = 1.0
    pz_vals = pz_vals / norms[:, None]
    X_raw = model_bundle["scaler"].inverse_transform(X_scaled)
    mag_val = X_raw[:, int(model_bundle.get("i_idx", 15))]
    z_mode = zgrid[np.argmax(pz_vals, axis=1)]
    pz_calib = _apply_calibration(pz_vals, z_mode, mag_val, model_bundle)
    _write_interp_ensemble(output_file, zgrid, pz_calib, object_id)


def _run_baseline_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    X_train_base, y_train, _, _ = _build_base_features(train_file, require_redshift=True)
    X_test_base, _, object_id, feature_names = _build_base_features(test_file)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_base)
    X_test = scaler.transform(X_test_base)
    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    z_pred = np.clip(model.predict(X_test), 0.0, 3.0)
    train_pred = model.predict(X_train)
    sigma = max(float(np.std(y_train - train_pred)), 0.02)
    i_idx = list(feature_names).index("mag_i_lsst")
    mag_i = X_test_base[:, i_idx]
    zgrid = np.linspace(0.0, 3.0, 301)
    scale = sigma * np.clip(1.0 + 0.08 * (mag_i - np.nanmedian(mag_i)), 0.7, 1.8)
    pz_vals = np.exp(-0.5 * ((zgrid[None, :] - z_pred[:, None]) / scale[:, None]) ** 2)
    _write_interp_ensemble(output_file, zgrid, pz_vals, object_id)


def run_taskset_1_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_lsst_v3_estimation_only(model_file, test_file, output_file)


def run_taskset_2_estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_lsst_v3_estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_baseline_training_and_estimation(train_file, test_file, output_file)


def run_taskset_2_training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    _run_baseline_training_and_estimation(train_file, test_file, output_file)


@pytest.fixture(name="setup_public_area", scope="module")
def setup_public_area() -> int:
    if os.path.exists(PUBLIC_AREA):
        return 0
    submit_utils.download_and_extract_tar(
        "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz",
        "tests",
    )
    return 0


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"), exist_ok=True)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"), exist_ok=True)
    request.addfinalizer(lambda: None)
    return 0


def test_lsst_v3_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_lsst_v3_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )
