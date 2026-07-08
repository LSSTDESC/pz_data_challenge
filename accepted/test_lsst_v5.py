import os

import pytest

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils

SUBMISSION_NAME: str = "lsst_v5"
SUBMISSION_URL: str = "https://github.com/e06243046/pz_data_challenge_submission/releases/download/lsst_v5_submission/lsst_v5_estimates_only.tgz"

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

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
            raise ValueError(
                f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set"
            )
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"), exist_ok=True)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"), exist_ok=True)
    request.addfinalizer(lambda: None)
    return 0


def test_lsst_v5_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    # Subtask 1 only: validate pre-made estimate files.
    # Subtasks 2 and 3 require TPU hardware for training and are not submitted.
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None,
        None,
    )


def test_lsst_v5_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        None,
        None,
    )
