import os
from pathlib import Path
import pytest

import numpy as np

from rail.core.data import TableHandle
from rail.estimation.algos.lephare import (
    LephareInformer,
    LephareEstimator,
    lsst_default_config,
)
from rail.utils import catalog_utils

import lephare as lp

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2

from pz_data_challenge import submit_utils

# We seem to need more for LePHARE at the current address
submit_utils._DOWNLOAD_TIMEOUT = 300

SUBMISSION_NAME: str = "lephare"
SUBMISSION_URL: str = "https://www.raphaelshirley.co.uk/data/submit_lephare.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# LePHARE specific globals
flux_cols = [f"mag_{b}_lsst" for b in "ugrizy"]
flux_cols += [f"mag_{b}_roman" for b in "YJH"]
flux_err_cols = [f"mag_{b}_lsst_err" for b in "ugrizy"]
flux_err_cols += [f"mag_{b}_roman_err" for b in "YJH"]

config = lsst_default_config.copy()
updates = {
    "MAG_REF": "2",
    "ERR_SCALE": "0.02",
    "FILTER_CALIB": "0",
    "FILTER_LIST": lsst_default_config["FILTER_LIST"]
    + ",roman/Roman_WFI.F106.dat,roman/Roman_WFI.F129.dat,roman/Roman_WFI.F158.dat",
    "GLB_CONTEXT": "0",
    "MABS_CONTEXT": "0",
    # Remove AGN for speed
    "ZPHOTLIB": "LSST_STAR_MAG,LSST_GAL_MAG",
    "Z_STEP": "0.02,0.,3.",
    "EM_DISPERSION": "0.5,1.,1.5",
    "ERR_FACTOR": "1.",
}
config.update(updates)
params = {f"lephare.{k}": v for k, v in updates.items()}
params["gal.MOD_EXTINC"] = ("16,24,24,31,24,31,24,31",)


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    if not os.path.exists(SUBMIT_DIR):
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

    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply("cardinal_roman_rubin")

    lp.data_retrieval.get_auxiliary_data(
        keymap=config, additional_files=["examples/output.para"]
    )

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
    estimator = LephareEstimator.make_stage(
        name="estimate_lephare",
        model=model_file,
        output_mode="return",
        run_dir=SUBMIT_DIR,
        bands=flux_cols,
        err_bands=flux_err_cols,
        hdf5_groupname="",
        lephare_config_from_model=False,
        **params,
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

    informer = LephareInformer.make_stage(
        name="inform_lephare",
        nondetect_val=np.nan,
        model="lephare.pkl",
        hdf5_groupname="",
        **params,
        bands=flux_cols,
        err_bands=flux_err_cols,
        ref_band="mag_g_lsst",
        zmin=float(config["Z_STEP"].split(",")[1]),
        zmax=float(config["Z_STEP"].split(",")[2]),
        nzbins=1
        + float(config["Z_STEP"].split(",")[2]) / float(config["Z_STEP"].split(",")[0]),
    )
    model = informer.inform(train_data)

    estimator = LephareEstimator.make_stage(
        name="estimate_lephare",
        # nondetect_val=np.nan,
        model="lephare.pkl",
        hdf5_groupname="",
        # aliases=dict(input="test_data", output="lephare_estim"),
        # use_inform_offsets=False,
        bands=flux_cols,
        err_bands=flux_err_cols,
        output_mode="return",
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
    estimator = LephareEstimator.make_stage(
        name="estimate_lephare",
        model=model_file,
        output_mode="return",
        run_dir=SUBMIT_DIR,
        bands=flux_cols,
        err_bands=flux_err_cols,
        hdf5_groupname="",
        lephare_config_from_model=False,
        **params,
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

    informer = LephareInformer.make_stage(
        name="inform_lephare",
        nondetect_val=np.nan,
        model="lephare.pkl",
        hdf5_groupname="",
        **params,
        bands=flux_cols,
        err_bands=flux_err_cols,
        ref_band="mag_g_lsst",
        zmin=float(config["Z_STEP"].split(",")[1]),
        zmax=float(config["Z_STEP"].split(",")[2]),
        nzbins=1
        + float(config["Z_STEP"].split(",")[2]) / float(config["Z_STEP"].split(",")[0]),
    )
    model = informer.inform(train_data)

    estimator = LephareEstimator.make_stage(
        name="estimate_lephare",
        # nondetect_val=np.nan,
        model="lephare.pkl",
        hdf5_groupname="",
        # aliases=dict(input="test_data", output="lephare_estim"),
        # use_inform_offsets=False,
        bands=flux_cols,
        err_bands=flux_err_cols,
        output_mode="return",
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
    Test fuction to validate a submisson for Taskset 1

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
