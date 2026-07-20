import os
from pathlib import Path
import pytest

# Put needed import here
from inference_submission import run_estimation
from train_submission import train_submission

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "cosom"
SUBMISSION_URL: str = "https://github.com/AlvaroCATA/files_pzchallenge/releases/download/files/models_and_predictions.tar.xz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


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


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Run inference using a pretrained Co-SOM model.
    """


    run_estimation(

        model_file=model_file,

        test_file=test_file,

        output_file=output_file

    )


def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Train Co-SOM model and run inference.
    """

    model_file = (
        f"{SUBMIT_DIR}/cosom_model.pkl"
    )


    # ======================================================
    # Train model
    # ======================================================

    train_submission(

        train_file=train_file,

        output_model=model_file

    )


    # ======================================================
    # Run inference
    # ======================================================

    run_estimation(

        model_file=model_file,

        test_file=test_file,

        output_file=output_file

    )


def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Run inference using a pretrained Co-SOM model for Task 2.
    """

    run_estimation(

        model_file=model_file,

        test_file=test_file,

        output_file=output_file

    )
    
def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Train Co-SOM model for Task Set 2 and run inference.

    Parameters
    ----------
    train_file : str or Path
        Task 2 training HDF5 file.

    test_file : str or Path
        Task 2 test HDF5 file.

    output_file : str or Path
        Output qp file.
    """


    model_file = (
        f"{SUBMIT_DIR}/task2_cosom_model.pkl"
    )


    # ======================================================
    # Train Task 2 Co-SOM model
    # ======================================================

    train_submission(

        train_file=train_file,

        output_model=model_file

    )


    # ======================================================
    # Run Task 2 inference
    # ======================================================

    run_estimation(

        model_file=model_file,

        test_file=test_file,

        output_file=output_file

    )





def test_example_taskset_1(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson for Taskset 1

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


def test_example_taskset_2(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test fuction to validate a submisson for Taskset 2

    You should not need to change this function
    """

    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )
