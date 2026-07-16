import math
import os
from pathlib import Path

import numpy as np
import pytest
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

from pz_data_challenge import submit_utils
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "tq_flexzboost"
SUBMISSION_URL: str = ""

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Catalog tag mapping the Rubin (ugrizy) + Roman (YJH) bands to columns.
CATALOG_TAG = "cardinal_roman_rubin"

# FlexZBoost configuration shared by the informer and estimator.
_ZMIN = 0.0
_ZMAX = 6.0
_NZ = 601
_ZGRID = np.linspace(_ZMIN, _ZMAX, _NZ)

# Point estimates to attach to the output: the PDF mode and the risk-minimizing
# "best" estimate.  These names are the tokens understood by RAIL's
# PointEstimationMixin, which writes them to ancil under the same names.
_POINT_ESTIMATES = ["zmode", "zbest"]


def _attach_ancil(estimator: FlexZBoostEstimator, pz_out, test_data: TableHandle) -> None:
    """Attach the point estimates and object_id that the validator requires.

    FlexZBoost's ``_process_chunk`` builds its ancil by hand and only recognises
    the tokens ``mode``/``mean``/``median``; it never calls the
    PointEstimationMixin, so ``calculated_point_estimates=["zmode", "zbest"]``
    alone would leave the ancil empty.  Invoke the mixin explicitly on the
    finished ensemble to populate ``zmode`` and ``zbest``.
    """
    ensemble = estimator.calculate_point_estimates(pz_out.data, grid=_ZGRID)
    object_id = dict(object_id=np.asarray(test_data()["object_id"]))
    if ensemble.ancil is None:  # pragma: no cover
        ensemble.set_ancil(object_id)
    else:
        ensemble.add_to_ancil(object_id)


def _clean_training_file(train_file: str | Path) -> str:
    """Drop rows with NaN redshift, which FlexZBoost training cannot use."""
    path = str(train_file)
    data = tables_io.read(path)
    bad_mask = np.isnan(data["redshift"])
    if not bad_mask.any():
        return path
    cleaned_path = path.replace(".hdf5", "_cleaned.hdf5")
    cleaned_data = {key: val[~bad_mask] for key, val in data.items()}
    tables_io.write(cleaned_data, cleaned_path)
    return cleaned_path


def _make_fzb_informer() -> FlexZBoostInformer:
    # A single bump threshold and a single sharpening value, so the informer
    # skips the grid search and trains on that one combination.
    return FlexZBoostInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
        max_basis=50,
        nbump=1,
        bumpmin=0.02,
        bumpmax=0.02,
        nsharp=1,
        sharpmin=1.2,
        sharpmax=1.2,
    )


def _make_fzb_estimator(model) -> FlexZBoostEstimator:
    # qp_representation defaults to "interp" (qp.interp); use the default.
    return FlexZBoostEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        output_mode="return",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
        calculated_point_estimates=_POINT_ESTIMATES,
    )


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """

    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(
                f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set"
            )
        submit_utils.download_and_extract_tar(SUBMISSION_URL, "submissions")

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

    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply(CATALOG_TAG)

    return 0


def run_taskset_x_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Load a pre-trained FlexZBoost model and estimate p(z) for the test set."""
    test_data = TableHandle("test", path=str(test_file))
    estimator = _make_fzb_estimator(str(model_file))
    pz_out = estimator.estimate(test_data)
    _attach_ancil(estimator, pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_x_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Train a FlexZBoost model on the training set and estimate p(z) for the test set."""
    cleaned_train_file = _clean_training_file(train_file)
    train_data = TableHandle("train", path=cleaned_train_file)
    test_data = TableHandle("test", path=str(test_file))
    informer = _make_fzb_informer()
    model = informer.inform(train_data)
    estimator = _make_fzb_estimator(model)
    pz_out = estimator.estimate(test_data)
    _attach_ancil(estimator, pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_training_and_estimation(train_file, test_file, output_file)


def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_training_and_estimation(train_file, test_file, output_file)


def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_training_and_estimation(train_file, test_file, output_file)


def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_training_and_estimation(train_file, test_file, output_file)


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
