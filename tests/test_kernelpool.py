"""CI test for the `kernelpool` task-set-3/4 submission (LSST-DESC PZ Data Challenge).

kernelpool = a pool of FlexZBoost (two settings) + GPz with per-i-band-bin weights and a per-bin calibration,
both fitted by the likelihood of the many-band labels of a COSMOS hold-out seen through the organizers'
label-noise kernel; raw-pool point estimate. Method: pz_method_writeup.md in the kernelpool repository.

The method lives in the pip-installable `kernelpool` package (pinned in requirements_kernelpool.txt); this file
is the thin entry point the harness discovers. Task sets 1 and 2 are not part of this submission.

Pre-made estimates + model bundles are hosted at SUBMISSION_URL and unpacked into submissions/kernelpool/.
The train+estimate path trains three estimators per combination (~15 min each on 4 cores at full size);
PZDC_CI_MAX_TRAIN=<n> subsamples the training file for CI (validity smoke test) -- the shipped full-scale
models and estimates carry the real performance. PZDC_CI_MAX_TRAIN=0 (default) trains on the full data.
"""
import os
from pathlib import Path

import pytest

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

from kernelpool import estimate_only, train_and_estimate

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "kernelpool"
SUBMISSION_URL: str = "https://github.com/ThomasReiprich/pz_kernelpool/releases/download/v1.0.0/kernelpool_submission.tgz"   # GitHub release .tgz of the pre-made estimates + model bundles

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = os.environ.get("PZDC_PUBLIC_AREA", "tests/public")


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """

    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL or SUBMISSION_URL.startswith("__"):
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


def _estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    """Apply a shipped model bundle (three member models, pool weights, calibration) to test_file."""
    estimate_only(model_file, test_file, output_file)


def _training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    """Train the three members, fit pool weights and calibration through the kernel, estimate test_file."""
    train_and_estimate(train_file, test_file, output_file)


def run_taskset_1_estimation_only(model_file, test_file, output_file) -> None:
    raise NotImplementedError("task set 1 is not part of the kernelpool submission")


def run_taskset_1_training_and_estimation(train_file, test_file, output_file) -> None:
    raise NotImplementedError("task set 1 is not part of the kernelpool submission")


def run_taskset_2_estimation_only(model_file, test_file, output_file) -> None:
    raise NotImplementedError("task set 2 is not part of the kernelpool submission")


def run_taskset_2_training_and_estimation(train_file, test_file, output_file) -> None:
    raise NotImplementedError("task set 2 is not part of the kernelpool submission")


def run_taskset_3_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


def run_taskset_4_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


@pytest.mark.skip(reason="Task sets 1 and 2 are not part of this submission; only the task set 3 and 4 run functions are provided.")
def test_example_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_1_estimation_only, run_taskset_1_training_and_estimation)


@pytest.mark.skip(reason="Task sets 1 and 2 are not part of this submission; only the task set 3 and 4 run functions are provided.")
def test_example_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(PUBLIC_AREA, SUBMISSION_NAME, run_taskset_2_estimation_only, run_taskset_2_training_and_estimation)


def test_example_taskset_3(setup_public_area: int, setup_submit_area: int) -> None:
    """
    Test fuction to validate a submisson for Taskset 3

    You should not need to change this function
    """
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_3_estimation_only,
        run_taskset_3_training_and_estimation,
    )


def test_example_taskset_4(setup_public_area: int, setup_submit_area: int) -> None:
    """
    Test fuction to validate a submisson for Taskset 4

    You should not need to change this function
    """
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
