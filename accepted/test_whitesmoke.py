"""RAIL+AION photo-z submission for the LSST-DESC PZ data challenge.

Implements the eight required entry points (estimation-only + train-and-estimate
for task sets 1-4) and delegates the actual work to the top-level
``rail_aion_pz`` module.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the top-level rail_aion_pz module importable regardless of pytest's rootdir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rail_aion_pz  # noqa: E402

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils  # noqa: F401

SUBMISSION_NAME: str = "whitesmoke"
# Once models/estimates are pre-trained and uploaded to GitHub releases,
# specify the URL here to download them. For now, empty runs training on the fly.
SUBMISSION_URL: str = ""

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Set AION_PZ_DEVICE=cpu to force CPU; otherwise CUDA is auto-detected.
_DEVICE = os.environ.get("AION_PZ_DEVICE")


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """Download the submission data if a URL is set, and prepare directory structure."""
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            os.makedirs(SUBMIT_DIR, exist_ok=True)
        else:
            submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN") and SUBMISSION_URL:
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    for sub in ("outputs_2", "outputs_3"):
        os.makedirs(os.path.join(SUBMIT_DIR, sub), exist_ok=True)

    request.addfinalizer(teardown_submit_area)
    return 0


CI_MAX_TRAIN: int = int(os.environ.get("PZDC_CI_MAX_TRAIN", "0"))


def _maybe_subsample_train(train_file: str) -> str:
    """Return train_file unchanged unless PZDC_CI_MAX_TRAIN>0, in which case write a
    stratified-by-nothing random subsample to a temp hdf5 (keeps CI train fast)."""
    if CI_MAX_TRAIN <= 0:
        return train_file
    import tempfile
    import tables_io
    d = tables_io.read(train_file)
    keys = list(d.keys())
    n = len(d[keys[0]])
    if n <= CI_MAX_TRAIN:
        return train_file
    idx = np.sort(np.random.default_rng(0).choice(n, CI_MAX_TRAIN, replace=False))
    sub = {k: np.asarray(d[k])[idx] for k in keys}
    stem = os.path.join(tempfile.mkdtemp(), "ci_train")
    tables_io.write(sub, stem, "hdf5")
    return stem + ".hdf5"


# ---------------------------------------------------------------------------
# Task-set entry points.
# ---------------------------------------------------------------------------

def _estimation_only(model_file, test_file, output_file) -> None:
    rail_aion_pz.estimate_only(model_file, test_file, output_file)


def _training_and_estimation(train_file, test_file, output_file) -> None:
    train_file = _maybe_subsample_train(str(train_file))
    rail_aion_pz.train_and_estimate(train_file, test_file, output_file)



# task set 1
def run_taskset_1_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 2
def run_taskset_2_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_2_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 3
def run_taskset_3_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 4
def run_taskset_4_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_rail_aion_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_rail_aion_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )


def test_rail_aion_taskset_3(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_3_estimation_only,
        run_taskset_3_training_and_estimation,
    )


def test_rail_aion_taskset_4(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
