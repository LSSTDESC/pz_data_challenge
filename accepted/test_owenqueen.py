#mariano.dominguez@unc.edu.ar july 2026
import os
import pickle
from pathlib import Path

import numpy as np
import qp
import tables_io
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1

from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the submission data files
# and needed model files
SUBMISSION_NAME: str = "owenqueen"
SUBMISSION_URL: str = "https://github.com/owencqueen/pz_data_challenge/releases/download/submit-owenqueen/submit_owenqueen.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# ---------------------------------------------------------------------------
# Photo-z model: quantile regression on magnitudes + colors.
#
# HistGradientBoostingRegressor handles NaN (non-detections) natively, so we
# can feed magnitudes/colors directly without imputation. We fit one regressor
# per target quantile of redshift; the per-object predicted quantiles define a
# full p(z) via qp's quantile parameterization. The median quantile is used as
# the `zmode` point estimate.
# ---------------------------------------------------------------------------

# Bands ordered by effective wavelength (blue -> red): LSST ugrizy + Roman YJH.
_BAND_ORDER = ["u", "g", "r", "i", "z", "y", "Y", "J", "H"]
_LSST_BANDS = {"u", "g", "r", "i", "z", "y"}

# CDF levels at which we predict redshift quantiles.
_QUANTILES = np.array(
    [0.02, 0.05, 0.10, 0.16, 0.25, 0.40, 0.50, 0.60, 0.75, 0.84, 0.90, 0.95, 0.98]
)


def _band_col(b: str) -> str:
    return f"mag_{b}_lsst" if b in _LSST_BANDS else f"mag_{b}_roman"


def _load_df(path: str | Path):
    return tables_io.convert(tables_io.read(str(path)), tables_io.types.PD_DATAFRAME)


def _build_features(df) -> tuple[np.ndarray, list[str]]:
    """Magnitudes + adjacent colors for whatever bands are present."""
    present = [b for b in _BAND_ORDER if _band_col(b) in df.columns]
    mags = {b: df[_band_col(b)].to_numpy(dtype=float) for b in present}

    feats: list[np.ndarray] = []
    names: list[str] = []
    for b in present:
        feats.append(mags[b])
        names.append(f"mag_{b}")
    for b1, b2 in zip(present[:-1], present[1:]):
        feats.append(mags[b1] - mags[b2])
        names.append(f"col_{b1}_{b2}")

    return np.column_stack(feats), names


def _train_model(train_file: str | Path) -> dict:
    """Fit one quantile regressor per CDF level; return a picklable dict."""
    df = _load_df(train_file)
    X, feat_names = _build_features(df)
    y = df["redshift"].to_numpy(dtype=float)

    regressors = []
    for q in _QUANTILES:
        reg = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(q),
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=63,
            min_samples_leaf=60,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
        reg.fit(X, y)
        regressors.append(reg)

    return {
        "quantiles": _QUANTILES,
        "regressors": regressors,
        "feat_names": feat_names,
        "z_max": float(y.max()),
    }


def _estimate(model: dict, test_file: str | Path, output_file: str | Path) -> None:
    """Predict per-object redshift quantiles and write a qp.quant ensemble."""
    df = _load_df(test_file)
    X, _ = _build_features(df)

    quants = model["quantiles"]
    # (n_obj, n_quant) matrix of predicted z at each CDF level.
    preds = np.column_stack([reg.predict(X) for reg in model["regressors"]])
    # Enforce monotonic, non-negative quantiles (fix any quantile crossing).
    preds = np.clip(preds, 0.0, None)
    preds = np.maximum.accumulate(preds, axis=1)

    median_idx = int(np.argmin(np.abs(quants - 0.5)))
    zmode = preds[:, median_idx]

    ens = qp.Ensemble(qp.quant, data=dict(quants=quants, locs=preds))
    ens.set_ancil(
        {
            "object_id": df["object_id"].astype("int64").to_numpy(),
            "zmode": zmode,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(str(output_file))), exist_ok=True)
    ens.write_to(str(output_file))


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


# ---------------------------------------------------------------------------
# Subtask 2: estimation only -- load a pre-trained model and estimate p(z).
# ---------------------------------------------------------------------------
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
    """
    with open(model_file, "rb") as fin:
        model = pickle.load(fin)
    _estimate(model, test_file, output_file)


# ---------------------------------------------------------------------------
# Subtask 3: train a model on train_file, then estimate p(z) on test_file.
# ---------------------------------------------------------------------------
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
    """
    model = _train_model(train_file)
    _estimate(model, test_file, output_file)


def test_example_taskset_1(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test function to validate a submission for Taskset 1

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
