"""AION photo-z submission for the LSST-DESC PZ data challenge.

Implements the six required entry points (estimation-only + train-and-estimate
for task sets 1-4 are wired to the same AION pipeline) and delegates the actual
work to the top-level ``aion_pz`` module.

Approach: frozen AION-1 embeddings (LSST grizy mapped onto AION's HSC magnitude
modalities) concatenated with raw LSST u + Roman photometry, fed to a small MLP
classifier over a redshift grid -> per-object p(z) written as a qp ensemble.
"""

import os
import sys
from pathlib import Path

import pytest

# Make the top-level aion_pz module importable regardless of pytest's rootdir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import aion_pz  # noqa: E402

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils  # noqa: F401

# SUBMISSION_URL points to the tarball of pre-made estimates + models, extracted
# into submissions/aion/ by the test harness.
SUBMISSION_NAME: str = "aion"
SUBMISSION_URL: str = (
    "https://github.com/ymuza/pz_challenge/releases/download/v0.2/aion_submission.tgz"
)

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Set AION_PZ_DEVICE=cpu to force CPU; otherwise CUDA is auto-detected.
_DEVICE = os.environ.get("AION_PZ_DEVICE")


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """Download the submission data (pre-trained models) if a URL is set."""
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            # No pre-trained tarball yet: create the directory so the
            # train-and-estimation path (subtask 3) can still run.
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


# ---------------------------------------------------------------------------
# Task-set entry points.  All four task sets share the same estimator; task
# sets 3/4 simply carry extra columns (Roman bands) which aion_pz picks up
# automatically when present.
# ---------------------------------------------------------------------------

def _estimation_only(model_file, test_file, output_file) -> None:
    aion_pz.estimate_only(model_file, test_file, output_file, device=_DEVICE)


def _training_and_estimation(train_file, test_file, output_file) -> None:
    aion_pz.train_and_estimate(train_file, test_file, output_file, device=_DEVICE)


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
# Validation tests (structure mirrors the challenge template).
# ---------------------------------------------------------------------------

def test_aion_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_aion_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )


def test_aion_taskset_3(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_3_estimation_only,
        run_taskset_3_training_and_estimation,
    )


def test_aion_taskset_4(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None if not SUBMISSION_URL else run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
