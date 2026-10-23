import os
from pathlib import Path
import numpy as np
import pytest
import tables_io

from rail.core.data import TableHandle
from rail.estimation.algos.tpz_lite import TPZliteInformer, TPZliteEstimator
from rail.utils import catalog_utils
from rail.utils.path_utils import RAILDIR

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
# from pz_data_challenge.taskset_3 import run_taskset_3
# from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils
SUBMISSION_NAME: str = "tpz_colors_curvature"
#SUBMISSION_URL: str = "https://portal.nersc.gov/cfs/lsst/schmidt9/submit_tpz_colors_curvature.tgz"
SUBMISSION_URL: str = "https://portal.nersc.gov/cfs/lsst/schmidt9/submit_tpz_colors_curvature.tgz"

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

feature_limits_10yr = {
    "mag_i_lsst": 28.62,
    'mag_u_lsstmag_g_lsst': 0.0,
    'mag_g_lsstmag_r_lsst': 0.0,
    'mag_r_lsstmag_i_lsst': 0.0,
    'mag_i_lsstmag_z_lsst': 0.0,
    'mag_z_lsstmag_y_lsst': 0.0,
    'mag_y_lsstmag_Y_roman': 0.0,
    'mag_Y_romanmag_J_roman': 0.0,
    'mag_J_romanmag_H_roman': 0.0,
    'curve_mag_u_lsstmag_g_lsstmag_r_lsst': 0.0,
    'curve_mag_g_lsstmag_r_lsstmag_i_lsst': 0.0,
    'curve_mag_r_lsstmag_i_lsstmag_z_lsst': 0.0,
    'curve_mag_i_lsstmag_z_lsstmag_y_lsst': 0.0,
    'curve_mag_z_lsstmag_y_lsstmag_Y_roman': 0.0,
    'curve_mag_y_lsstmag_Y_romanmag_J_roman': 0.0,
    'curve_mag_Y_romanmag_J_romanmag_H_roman': 0.0,    
}

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

def prepare_data(infile):
    xdata = tables_io.read(infile, tType=3)
    for band, errb in zip(bands, errbands):
        mask = ~(np.logical_and(np.isfinite(xdata[band]), np.isfinite(xdata[errb])))
        xdata.loc[mask, band] = mag_limits_10yr[band]
        xdata.loc[mask, errb] = 1.25

    allbands = []
    allerrs = []
    features = ['mag_i_lsst']
    featureerrs = ['mag_i_lsst_err']
    
    for band in lsstbands:
        allbands.append(f"mag_{band}_lsst")
        allerrs.append(f"mag_{band}_lsst_err")
    for band in romanbands:
        allbands.append(f"mag_{band}_roman")
        allerrs.append(f"mag_{band}_roman_err")
    nbands = len(allbands)
    for ii in range(nbands - 1):
        featurename = f"{allbands[ii]}{allbands[ii+1]}"
        xdata[featurename] = xdata[allbands[ii]] - xdata[allbands[ii+1]]
        features.append(featurename)
        featureerrname = f"{allbands[ii]}{allbands[ii+1]}_err"
        featureerr = np.sqrt((xdata[allerrs[ii]]**2) + (xdata[allerrs[ii+1]]**2))
        xdata[featureerrname] = featureerr
        featureerrs.append(featureerrname)

    for ii in range(nbands - 2):
        featurename = f"curve_{allbands[ii]}{allbands[ii+1]}{allbands[ii+2]}"
        xdata[f"curve_{allbands[ii]}{allbands[ii+1]}{allbands[ii+2]}"] = xdata[allbands[ii]] - 2.0 * xdata[allbands[ii+1]] + xdata[allbands[ii+2]]
        features.append(featurename)
        featureerrname = featurename + "_err"
        featurerr = np.sqrt((xdata[allerrs[ii]]**2) + 2. * (xdata[allerrs[ii+1]]**2) + (xdata[allerrs[ii+2]]**2))
        xdata[featureerrname] = featureerr
        featureerrs.append(featureerrname)
        basename = infile[:-5]
        outname = basename + "_transform.hdf5"
        tables_io.write(xdata, outname)
        #data = tables_io.convert(xdata, tables_io.types.NUMPY_DICT)
        return outname, features, featureerrs

        
@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:

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
    
    test_datax, features, featureerrs = prepare_data(test_file)
    #TableHandle("test", path=test_file)
    test_data = TableHandle("test", path=test_datax)
    
    estimator = TPZliteEstimator.make_stage(
        name="estimate",
        hdf5_groupname="",
        model=model_file,
        bands=features,
        err_bands=featureerrs,
        nondetect_val=np.nan,
        redshift_col="redshift",
        mag_limits=feature_limits_10yr,
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
    # train_data = TableHandle("train", path=train_file)
    # test_data = TableHandle("test", path=test_file)

    test_datax, features, featureerrs = prepare_data(test_file)
    train_datax, tfeatures, tfeatureerrs = prepare_data(train_file)

    test_data =	TableHandle("test", path=test_datax)
    train_data = TableHandle("train", path=train_datax)
    
    informer = TPZliteInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        bands=features,
        err_bands=featureerrs,
        nondetect_val=np.nan,
        mag_limits=feature_limits_10yr,
        seed=1295,
        redshift_col="redshift",
        n_random=10,
        n_trees=10,
        min_leaf=5,
        n_att=3,
    )
    model = informer.inform(train_data)

    estimator = TPZliteEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        bands=features,
        redshift_col="redshift",
        err_bands=featureerrs,
        nondetect_val=np.nan,
        mag_limits=feature_limits_10yr,
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
