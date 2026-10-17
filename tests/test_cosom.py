"""
Co-SOM submission validation for the PZ Data Challenge.

This file provides the required submission functions for Tasksets 3 and 4.
"""

import os
import sys
from pathlib import Path

import pytest


# -------------------------------------------------------------------------
# Make local files in tests/ importable
# -------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


# -------------------------------------------------------------------------
# Submission configuration
# -------------------------------------------------------------------------

SUBMISSION_NAME = "cosom"

SUBMISSION_URL = (
    "https://github.com/AlvaroCATA/files_pzchallenge/raw/refs/heads/main/pz_challenge_taskset_34.tar.xz"
)

SUBMIT_DIR = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA = "tests/public"


# -------------------------------------------------------------------------
# Co-SOM imports
# -------------------------------------------------------------------------

from cosom_inference import (
    load_submission_model,
    find_best_bmu,
    predict_objects,
    save_qp_submission,
)

from preprocessing import preprocess_test_hdf5

from train_submission import train_submission


# -------------------------------------------------------------------------
# Official challenge runners
# -------------------------------------------------------------------------

from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4


# -------------------------------------------------------------------------
# Public area fixture
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Submission area fixture
# -------------------------------------------------------------------------

@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request):
    """
    Download and extract the public submission archive if necessary.
    """

    if not os.path.exists(SUBMIT_DIR):

        if not SUBMISSION_URL:
            raise ValueError(
                "SUBMISSION_URL must be provided when the submission "
                "directory does not exist."
            )

        from pz_data_challenge import submit_utils

        submit_utils.download_and_extract_tar(
            SUBMISSION_URL,
            SUBMIT_DIR,
        )

    # Directories used by the official taskset runners
    os.makedirs(
        os.path.join(SUBMIT_DIR, "outputs_2"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(SUBMIT_DIR, "outputs_3"),
        exist_ok=True,
    )

    def teardown_submit_area():
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    request.addfinalizer(teardown_submit_area)

    return 0


# -------------------------------------------------------------------------
# Generic Co-SOM inference
# -------------------------------------------------------------------------

def _run_estimation(
    model_file,
    test_file,
    output_file,
):
    """
    Run Co-SOM inference using a pre-trained model.
    """

    model = load_submission_model(model_file)

    X_test, object_id = preprocess_test_hdf5(
        test_file,
        model["metadata"]["preprocess_params"],
    )

    matches = find_best_bmu(
        X_test,
        model["regions"],
        k_candidates=10,
    )

    predictions = predict_objects(
        matches,
        model,
    )

    save_qp_submission(
        pdf=predictions["pdf"],
        z_edges=model["metadata"]["z_edges_global"],
        object_id=object_id,
        filename=output_file,
    )


# -------------------------------------------------------------------------
# Generic Co-SOM training + inference
# -------------------------------------------------------------------------

def _run_training_and_estimation(
    train_file,
    test_file,
    output_file,
):
    """
    Train Co-SOM from the supplied training data and run inference.
    """

    output_path = Path(output_file)

    temp_model = output_path.with_name(
        f".{output_path.stem}_cosom_model.pkl"
    )

    try:

        train_submission(
            train_file=train_file,
            output_model=temp_model,
            seed=42,
        )

        _run_estimation(
            temp_model,
            test_file,
            output_file,
        )

    finally:

        try:
            temp_model.unlink()
        except FileNotFoundError:
            pass


# -------------------------------------------------------------------------
# Taskset 3
# -------------------------------------------------------------------------

def run_taskset_3_estimation_only(
    model_file,
    test_file,
    output_file,
):
    """
    Taskset 3: estimation using a pre-trained Co-SOM model.
    """

    _run_estimation(
        model_file,
        test_file,
        output_file,
    )


def run_taskset_3_training_and_estimation(
    train_file,
    test_file,
    output_file,
):
    """
    Taskset 3: train Co-SOM and estimate photo-z.
    """

    _run_training_and_estimation(
        train_file,
        test_file,
        output_file,
    )


# -------------------------------------------------------------------------
# Taskset 4
# -------------------------------------------------------------------------

def run_taskset_4_estimation_only(
    model_file,
    test_file,
    output_file,
):
    """
    Taskset 4: estimation using a pre-trained Co-SOM model.
    """

    _run_estimation(
        model_file,
        test_file,
        output_file,
    )


def run_taskset_4_training_and_estimation(
    train_file,
    test_file,
    output_file,
):
    """
    Taskset 4: train Co-SOM and estimate photo-z.
    """

    _run_training_and_estimation(
        train_file,
        test_file,
        output_file,
    )


# -------------------------------------------------------------------------
# Pytest: Taskset 3
# -------------------------------------------------------------------------

def test_example_taskset_3(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test function to validate the submission for Taskset 3.
    """

    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_3_estimation_only,
        run_taskset_3_training_and_estimation,
    )


# -------------------------------------------------------------------------
# Pytest: Taskset 4
# -------------------------------------------------------------------------

def test_example_taskset_4(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Test function to validate the submission for Taskset 4.
    """

    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
