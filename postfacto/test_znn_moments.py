import os
from pathlib import Path
import pytest

# Put needed import here
import numpy as np
from rail.utils.path_utils import find_rail_file

import znn

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "znn_moments"
SUBMISSION_URL: str = "https://github.com/JaimeRZP/znn/releases/download/v26.11/znn_moments.tgz"

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


# znn_moments: a pair of znn networks after the moment networks of Jeffrey & Wandelt (2020). F predicts
# log(1 + z) and starts from the pop-cosmos pre-trained weights shipped with znn; G, trained on F's residuals with
# F fixed, predicts their scale. The p(z) is a Student-t in log(1 + z) around F's prediction with G's scale,
# truncated to 0 <= z <= 3. The submitted models (subtask 2) were trained for up to 100 epochs on 80% of each
# training set. Subtask 3 trains the same model, with F from the pre-trained weights, for TRAIN_EPOCHS epochs only
# (about 2 min per network per epoch on a full training set on CPU), so that all 16 configurations fit in the CI
# job's 6 h limit.
TRAIN_EPOCHS: int = 1
SEED: int = 42

# band -> (catalogue magnitude column, filter curve shipped with RAIL), ordered by wavelength; the columns map to
# these filters as in tests/catalogs.yaml ("cardinal_roman_rubin"). The model files store their own copy of these
# settings and of the curves; this is used by subtask 3.
FILTER_DIR = find_rail_file("examples_data/estimation_data/data/FILTER")
BANDS = {
    "u": ("mag_u_lsst", f"{FILTER_DIR}/DC2LSST_u.res"),
    "g": ("mag_g_lsst", f"{FILTER_DIR}/DC2LSST_g.res"),
    "r": ("mag_r_lsst", f"{FILTER_DIR}/DC2LSST_r.res"),
    "i": ("mag_i_lsst", f"{FILTER_DIR}/DC2LSST_i.res"),
    "z": ("mag_z_lsst", f"{FILTER_DIR}/DC2LSST_z.res"),
    "y": ("mag_y_lsst", f"{FILTER_DIR}/DC2LSST_y.res"),
    "Y": ("mag_Y_roman", f"{FILTER_DIR}/roman_Y106.res"),
    "J": ("mag_J_roman", f"{FILTER_DIR}/roman_J129.res"),
    "H": ("mag_H_roman", f"{FILTER_DIR}/roman_H158.res"),
}
REF_BAND = "i"


def estimate(model, features: dict, test_file: str | Path, output_file: str | Path) -> None:
    """Write the p(z) of a trained model on test_file, in qp format, with zmode and object_id in its ancil."""
    test = znn.read_catalog(test_file)
    pz = znn.predict(model, test, **features, ids=test["object_id"].to_numpy())
    pz.write_to(str(output_file))


def estimation_only(model_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    """Subtask 2, the same for every task set: estimate with a saved model."""
    model, features = znn.load_model_file(model_file)
    estimate(model, features, test_file, output_file)


def training_and_estimation(train_file: str | Path, test_file: str | Path, output_file: str | Path) -> None:
    """
    Subtask 3, the same for every task set: train a model, then estimate.

    The training labels are the spectroscopic `redshift`, or `redshift_manyband` for the objects without one (the
    COSMOS-only objects of task sets 3 and 4).
    """
    train = znn.read_catalog(train_file)
    z = train["redshift"].to_numpy()
    if "redshift_manyband" in train:
        z = np.where(np.isnan(z), train["redshift_manyband"].to_numpy(), z)
    keep = np.isfinite(z)

    config = znn.DEFAULT_FEATURE_CONFIG
    model, _ = znn.train_model(
        znn.build_model, train[keep], BANDS, REF_BAND, config, Y=z[keep],
        EPOCHS=TRAIN_EPOCHS, random_state=SEED, pretraining="popcosmos",
    )
    estimate(model, {"bands": BANDS, "ref_band": REF_BAND, "config": config}, test_file, output_file)


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
    estimation_only(model_file, test_file, output_file)


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
    training_and_estimation(train_file, test_file, output_file)


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
    estimation_only(model_file, test_file, output_file)


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
    training_and_estimation(train_file, test_file, output_file)


def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 3

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
    estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 3

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
    training_and_estimation(train_file, test_file, output_file)


def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 4

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
    estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 4

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
    training_and_estimation(train_file, test_file, output_file)


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

    
def test_example_taskset_3(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
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


def test_example_taskset_4(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
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
