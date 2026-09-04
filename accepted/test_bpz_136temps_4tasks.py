import os
from pathlib import Path
import pytest

from rail.core.data import TableHandle
from rail.estimation.algos.bpz_lite import BPZliteInformer, BPZliteEstimator
from rail.utils import catalog_utils
from rail.utils.path_utils import RAILDIR

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils
SUBMISSION_NAME: str = "bpz_136temps_4tasks"
SUBMISSION_URL: str = "https://portal.nersc.gov/cfs/lsst/schmidt9/submit_bpz_136temps_4tasks.tgz"

mag_limits_10yr = {
    "mag_u_lsst": 27.79,
    "mag_g_lsst": 29.04,
    "mag_r_lsst": 29.06,
    "mag_i_lsst": 28.62,
    "mag_z_lsst": 27.98,
    "mag_y_lsst": 27.05,
    "mag_Y_roman": 26.4,
    "mag_J_roman": 26.4,
    "mag_H_roman": 26.4,
    "mag_F_roman": 26.4,
}
mag_limits_4yr = {
    'mag_u_lsst': 27.2926,
    'mag_g_lsst': 28.5426,
    'mag_r_lsst': 28.5626,
    'mag_i_lsst': 28.1226,
    'mag_z_lsst': 27.4826,
    'mag_y_lsst': 26.5526,
    'mag_Y_roman': 26.4,
    'mag_J_roman': 26.4,
    'mag_H_roman': 26.4,
    'mag_F_roman': 26.4,
}
mag_limits_1yr = {
    'mag_u_lsst': 26.54,
    'mag_g_lsst': 27.79,
    'mag_r_lsst': 27.81,
    'mag_i_lsst': 27.37,
    'mag_z_lsst': 26.73,
    'mag_y_lsst': 25.8,
    'mag_Y_roman': 26.4,
    'mag_J_roman': 26.4,
    'mag_H_roman': 26.4,
    'mag_F_roman': 26.4,
}

filterlist = [
    "DC2LSST_u",
    "DC2LSST_g",
    "DC2LSST_r",
    "DC2LSST_i",
    "DC2LSST_z",
    "DC2LSST_y",
    "roman_Y106",
    "roman_J129",
    "roman_H158"
    ]


lsstbands = ['u','g','r','i','z','y']
romanbands = ['Y', 'J', 'H']
errbands = []
bands = []
for band in lsstbands:
    bands.append(f"mag_{band}_lsst")
    errbands.append(f"mag_{band}_lsst_err")
for band in romanbands:
    bands.append(f"mag_{band}_roman")
    errbands.append(f"mag_{band}_roman_err")
    
# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"
TEMPLATE_FILE: str = "bpz_136seds.tar"
TEMPLATE_FULLPATH = os.path.join(
    RAILDIR,
    "rail/examples_data/estimation_data/data/SED/",
    TEMPLATE_FILE
)
TEMPLATE_PARTIALPATH = os.path.join(
    RAILDIR,
    "rail/examples_data/estimation_data/data/SED/")

@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:

    if not os.path.exists(TEMPLATE_FULLPATH):
        print("grabbing 136 SED tar file...")
        print(f"untarring files in {TEMPLATE_PARTIALPATH}")
        os.system(
                f"curl -o {TEMPLATE_FULLPATH} https://portal.nersc.gov/cfs/lsst/schmidt9/bpz_136seds.tar"
            )
        os.system(f"tar -xvf {TEMPLATE_FULLPATH} -C {TEMPLATE_PARTIALPATH}")
    
    if not os.path.exists(SUBMIT_DIR):
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            # os.system(f"\\rm -rf {SUBMIT_DIR}")
            print("remove teardown!")
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

    return 0


def run_taskset_1_estimation_only(mfile, testfile, outfile):
    run_taskset_x_estimation_only(mfile, testfile, outfile)

def run_taskset_1_training_and_estimation(mfile, testfile, outfile):
    run_taskset_x_training_and_estimation(mfile, testfile, outfile)

def run_taskset_2_estimation_only(mfile, testfile, outfile):
    run_taskset_x_estimation_only(mfile, testfile, outfile)

def run_taskset_2_training_and_estimation(mfile, testfile, outfile):
    run_taskset_x_training_and_estimation(mfile, testfile, outfile)

def run_taskset_3_estimation_only(mfile, testfile, outfile):
    run_taskset_x_estimation_only(mfile, testfile, outfile)

def run_taskset_3_training_and_estimation(mfile, testfile, outfile):
    run_taskset_x_training_and_estimation(mfile, testfile, outfile)

def run_taskset_4_estimation_only(mfile, testfile, outfile):
    run_taskset_x_estimation_only(mfile, testfile, outfile)

def run_taskset_4_training_and_estimation(mfile, testfile, outfile):
    run_taskset_x_training_and_estimation(mfile, testfile, outfile)


def run_taskset_x_estimation_only(
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
    print(f"debug: running for output file {output_file}")
    
    test_data = TableHandle("test", path=test_file)

    if "flagship" in output_file:
        kern = 0.12
        madauflag = "yes"
    elif "cardinal" in output_file:
        kern = 0.04
        madauflag = "no"
    else:
        kern = 0.0
        madauflag = "no"

    if "10yr" in output_file:
        maglim = mag_limits_10yr
    elif "4yr" in output_file:
        maglim = mag_limits_4yr
    elif "1yr" in output_file:
        maglim = mag_limits_1yr
    else:
        maglim = mag_limits_10yr
        print("Not in the expected cases, just set to 10yr")
    
    informer = BPZliteInformer.make_stage(
        name="inform",
        output_HDFN=True,
        hdf5_groupname="",
        nt_array=[10,54,72],
    )
    tmpmodel = informer.inform(test_data)


    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        hdf5_groupname="",
        model=tmpmodel,
        spectra_file="flagship_136seds.list",
        bands=bands,
        err_bands=errbands,
        filter_list=filterlist,
        override_file_offsets=True,
        zp_offsets=[0.,0.,0.,0.,0.,0.,0.,0.,0.],
        zp_errors=[0.01, 0.01, 0.01,0.01, 0.01, 0.01,0.01, 0.01, 0.01],
        mag_limits=maglim,
        gauss_kernel=kern,
        madau_flag=madauflag,
    )
    pz_out = estimator.estimate(test_data)
    pz_out.data.ancil["object_id"] = test_data()["object_id"].astype(int)
    pz_out.path = output_file
    pz_out.write()

def run_taskset_x_training_and_estimation(
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

    if "flagship" in output_file:
        kern = 0.12
        madauflag = "yes"
    elif "cardinal" in output_file:
        kern = 0.04
        madauflag = "no"
    else:
        kern = 0.0
        madauflag = "no"

    if "10yr" in output_file:
        maglim = mag_limits_10yr
    elif "4yr" in output_file:
        maglim = mag_limits_4yr
    elif "1yr" in output_file:
        maglim = mag_limits_1yr
    else:
        maglim = mag_limits_10yr
        print("Not in the expected cases, just set to 10yr")
    
    informer = BPZliteInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        output_HDFN=True,
        nt_array=[10,54,72],
    )
    model = informer.inform(train_data)

    estimator = BPZliteEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        spectra_file="flagship_136seds.list",
        bands=bands,
        err_bands=errbands,
        filter_list=filterlist,
        override_file_offsets=True,
        zp_offsets=[0.,0.,0.,0.,0.,0.,0.,0.,0.],
        zp_errors=[0.01, 0.01, 0.01,0.01, 0.01, 0.01,0.01, 0.01, 0.01],
        mag_limits=maglim,
        gauss_kernel=kern,
        madau_flag=madauflag,
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
