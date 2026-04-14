import os
from pathlib import Path
import pytest

# Put needed import here
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.core.data import TableHandle
from rail.utils import catalog_utils


# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "bpz_31temps"
SUBMISSION_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/submit_bpz_31temps.tgz"

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
    User supplied function to run estimation for task set 1

    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    test_data = TableHandle("test", path=test_file)
    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        hdf5_groupname="",
        model=model_file,
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_file
    pz_out.write()


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

    Parameters
    ----------
    train_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be trained
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    train_data = TableHandle("train", path=train_file)
    test_data = TableHandle("test", path=test_file)

    informer = BPZliteInformer.make_stage(
        name="inform",
        output_HDFN=True,
    )
    model = informer.inform(train_data)

    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_2_estimation_only(
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

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    test_data = TableHandle("test", path=test_file)
    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        model=model_file,
        hdf5_groupname="",
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 1

    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    train_data = TableHandle("train", path=train_file)
    test_data = TableHandle("test", path=test_file)

    informer = BPZliteInformer.make_stage(
        name="inform",
        output_hdf5=True,
    )
    model = informer.inform(train_data)

    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_file
    pz_out.write()

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
