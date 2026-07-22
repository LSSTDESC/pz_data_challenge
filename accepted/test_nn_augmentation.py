import os
from pathlib import Path
import pytest

# Put needed import here
import tables_io
import numpy as np
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils
from scipy.spatial import cKDTree

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "nn_augmentation"
SUBMISSION_URL: str = "https://portal.nersc.gov/cfs/lsst/PZ/submit_nn_augmentation.tgz"

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

def make_informer_task1_task2(case_name) -> FlexZBoostInformer:
    return FlexZBoostInformer.make_stage(
        name=case_name,
        hdf5_groupname="",
        zmax=3.0,
        nzbins=301,
        nondetect_val=np.nan,
        trainfrac = 0.75,
        bumpmin = 0.02,
        bumpmax = 0.35,
        sharpmin = 0.7,
        sharpmax = 2.1,
        nsharp = 15,
        max_basis = 35,
        regression_params = {'max_depth': 8, 'objective': 'reg:squarederror'},
        mag_limits={'mag_u_lsst': 26.5, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1, 'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6},
        bands=['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'],
        err_bands=['mag_u_lsst_err', 'mag_g_lsst_err', 'mag_r_lsst_err', 'mag_i_lsst_err', 'mag_z_lsst_err', 'mag_y_lsst_err'],
        ref_band = 'mag_i_lsst',
        redshift_col = 'redshift')

def make_estimator(model) -> FlexZBoostEstimator:
    return FlexZBoostEstimator.make_stage(
        name='estimate',
        model=model,
        chunk_size = 10000,
        hdf5_groupname = '',
        zmin = 0.0,
        zmax = 3.0,
        nzbins = 301,
        id_col = 'object_id',
        calc_summary_stats = False,
        calculated_point_estimates = ['zmode'],
        nondetect_val = np.nan,
        mag_limits={'mag_u_lsst': 26.5, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1, 'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6},
        bands=['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'],
        err_bands=['mag_u_lsst_err', 'mag_g_lsst_err', 'mag_r_lsst_err', 'mag_i_lsst_err', 'mag_z_lsst_err', 'mag_y_lsst_err'],
        ref_band = 'mag_i_lsst')

def make_informer_task3_task4(case_name) -> FlexZBoostInformer:
    return FlexZBoostInformer.make_stage(
        name=case_name,
        hdf5_groupname="",
        zmax=3.0,
        nzbins=301,
        nondetect_val=np.nan,
        trainfrac = 0.75,
        bumpmin = 0.02,
        bumpmax = 0.35,
        sharpmin = 0.7,
        sharpmax = 2.1,
        nsharp = 15,
        max_basis = 35,
        regression_params = {'max_depth': 8, 'objective': 'reg:squarederror'},
        mag_limits={'mag_u_lsst': 26.5, 'mag_g_lsst': 27.8, 'mag_r_lsst': 27.1, 'mag_i_lsst': 26.7, 'mag_z_lsst': 25.8, 'mag_y_lsst': 24.6},
        bands=['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'],
        err_bands=['mag_u_lsst_err', 'mag_g_lsst_err', 'mag_r_lsst_err', 'mag_i_lsst_err', 'mag_z_lsst_err', 'mag_y_lsst_err'],
        ref_band = 'mag_i_lsst',
        redshift_col = 'redshift_combined')

def attach_object_ids(pz_out, test_data: TableHandle) -> None:
    pz_out.data.ancil["object_id"] = np.asarray(test_data()["object_id"])


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

    test_data = TableHandle("test", path=str(test_file))

    estimator = make_estimator(str(model_file))
    pz_out = estimator.estimate(test_data)
    attach_object_ids(pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()

    return


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

    train_df = tables_io.read(train_file, tType='pandasDataFrame')
    test_df = tables_io.read(test_file, tType='pandasDataFrame')

    test_df.replace(np.nan, 99, inplace=True)

    ##write something about getting the sim file here
    if not os.path.exists(os.getcwd()+'/diffsky_augmentation_catalogs'):
        print("grabbing diffsky catalogs")
        os.system(
                "curl -O https://portal.nersc.gov/cfs/lsst/PZ/diffsky_augmentation_catalogs.tar.gz"
            )
        print('untarring')
        os.system(f"mkdir -p {os.getcwd()}/diffsky_augmentation_catalogs && tar -xf diffsky_augmentation_catalogs.tar.gz -C {os.getcwd()}/diffsky_augmentation_catalogs")
    
    if "1yr" in train_file:
        sim_file = 'diffsky_augmentation_catalogs/diffsky/diffsky_yr1_errors.hdf5'
    if "10yr" in train_file:
        sim_file = 'diffsky_augmentation_catalogs/diffsky/diffsky_yr10_errors.hdf5'

    sim_df = tables_io.read(sim_file, tType='pandasDataFrame')
    sim_df.replace(np.inf, 99, inplace=True)
    sim_df.replace(np.nan, 99, inplace=True)

    colorslist = ['ug', 'gr', 'ri', 'iz', 'zy', 'gz']
    for color in colorslist:
        train_df[color] = train_df[f'mag_{color[0]}_lsst'] - train_df[f'mag_{color[1]}_lsst']
        test_df[color] = test_df[f'mag_{color[0]}_lsst'] - test_df[f'mag_{color[1]}_lsst']
        sim_df[color] = sim_df[f'{color[0]}'] - sim_df[f'{color[1]}']

    #Run initial photo-z with flexzboost (unaugmented)
    train_data = TableHandle("train", path=str(train_file))
    test_data = TableHandle("test", path=str(test_file))

    informer = make_informer_task1_task2("unaugmented")
    model = informer.inform(train_data)

    estimator = make_estimator(model)
    pz_out = estimator.estimate(test_data)

    test_df['pz'] = pz_out.data.ancil['zmode'].reshape(1,-1)[0]

    ##Do nearest neighbor matching
    test_df['index'] = test_df.index.values.tolist()

    test_arr = test_df[['ug', 'gr', 'ri', 'iz', 'zy', 'mag_i_lsst', 'pz']].to_numpy()
    sim_arr = sim_df[['ug', 'gr', 'ri', 'iz', 'zy', 'i', 'redshift']].to_numpy()

    tree = cKDTree(sim_arr)
    distances, indices = tree.query(test_arr, k=1)

    masked_sim = sim_df.iloc[indices]

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    combined_train = {}


    for band in bands:
        spec_band = train_data()["mag_"+band+"_lsst"]
        spec_band_err = train_data()["mag_"+band+"_lsst_err"]

        aug_band = masked_sim[band].to_numpy()
        total_band = np.concatenate((spec_band, aug_band))
        combined_train["mag_"+band+"_lsst"] = total_band

        aug_band_err = masked_sim[band+'_err']
        total_err = np.concatenate((spec_band_err, aug_band_err))
        combined_train["mag_"+band+"_lsst_err"] = total_err

    spec_red = train_data()['redshift']
    aug_red = masked_sim['redshift'].to_numpy()

    combined_train['redshift'] = np.concatenate((spec_red, aug_red))

    ##retrain photo-z with augmented training set
    model_name = (train_file.split('public/'))[1].split('_training')[0]+'_pz_model_'+(train_file.split('training_'))[1].split('.hdf5')[0]
    
    augmented_train_data = TableHandle("aug_train", combined_train)
    augmented_informer = make_informer_task1_task2(model_name)
    augmented_model = augmented_informer.inform(augmented_train_data)

    augmented_estimator = make_estimator(augmented_model)
    augmented_pz_out = augmented_estimator.estimate(test_data)
    attach_object_ids(augmented_pz_out, test_data)
    augmented_pz_out.path = output_file
    augmented_pz_out.write()
    return


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
    run_taskset_1_estimation_only(model_file, test_file, output_file)
    return


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

    run_taskset_1_training_and_estimation(train_file, test_file, output_file)
    return


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

    run_taskset_1_estimation_only(model_file, test_file, output_file)
    return


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

    train_df = tables_io.read(train_file, tType='pandasDataFrame')
    test_df = tables_io.read(test_file, tType='pandasDataFrame')

    test_df.replace(np.nan, 99, inplace=True)

    ##write something about getting the sim file here
    if not os.path.exists(os.getcwd()+'/diffsky_augmentation_catalogs'):
        print("grabbing diffsky catalogs")
        os.system(
                "curl -O https://portal.nersc.gov/cfs/lsst/PZ/diffsky_augmentation_catalogs.tar.gz"
            )
        print('untarring')
        os.system(f"mkdir -p {os.getcwd()}/diffsky_augmentation_catalogs && tar -xf diffsky_augmentation_catalogs.tar.gz -C {os.getcwd()}/diffsky_augmentation_catalogs")
    
    if "1yr" in train_file:
        sim_file = 'diffsky_augmentation_catalogs/diffsky/diffsky_yr1_errors.hdf5'
    if "10yr" in train_file:
        sim_file = 'diffsky_augmentation_catalogs/diffsky/diffsky_yr10_errors.hdf5'

    sim_df = tables_io.read(sim_file, tType='pandasDataFrame')
    sim_df.replace(np.inf, 99, inplace=True)
    sim_df.replace(np.nan, 99, inplace=True)

    colorslist = ['ug', 'gr', 'ri', 'iz', 'zy', 'gz']
    for color in colorslist:
        train_df[color] = train_df[f'mag_{color[0]}_lsst'] - train_df[f'mag_{color[1]}_lsst']
        test_df[color] = test_df[f'mag_{color[0]}_lsst'] - test_df[f'mag_{color[1]}_lsst']
        sim_df[color] = sim_df[f'{color[0]}'] - sim_df[f'{color[1]}']

    #Run initial photo-z with flexzboost (unaugmented)
    train_data = TableHandle("train", path=str(train_file))
    test_data = TableHandle("test", path=str(test_file))

    idx = np.argwhere(np.isnan(train_data()['redshift'])).reshape(1, -1)[0]
    redshift_arr = train_data()['redshift']
    redshift_arr[idx] = train_data()['redshift_manyband'][idx]

    train_data()['redshift_combined'] = redshift_arr

    informer = make_informer_task3_task4("unaugmented")
    model = informer.inform(train_data)

    estimator = make_estimator(model)
    pz_out = estimator.estimate(test_data)

    test_df['pz'] = pz_out.data.ancil['zmode'].reshape(1,-1)[0]

    ##Do nearest neighbor matching
    test_df['index'] = test_df.index.values.tolist()

    test_arr = test_df[['ug', 'gr', 'ri', 'iz', 'zy', 'mag_i_lsst', 'pz']].to_numpy()
    sim_arr = sim_df[['ug', 'gr', 'ri', 'iz', 'zy', 'i', 'redshift']].to_numpy()

    tree = cKDTree(sim_arr)
    distances, indices = tree.query(test_arr, k=1)

    masked_sim = sim_df.iloc[indices]

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    combined_train = {}


    for band in bands:
        spec_band = train_data()["mag_"+band+"_lsst"]
        spec_band_err = train_data()["mag_"+band+"_lsst_err"]

        aug_band = masked_sim[band].to_numpy()
        total_band = np.concatenate((spec_band, aug_band))
        combined_train["mag_"+band+"_lsst"] = total_band

        aug_band_err = masked_sim[band+'_err']
        total_err = np.concatenate((spec_band_err, aug_band_err))
        combined_train["mag_"+band+"_lsst_err"] = total_err

    spec_red = train_data()['redshift']
    aug_red = masked_sim['redshift'].to_numpy()

    combined_train['redshift_combined'] = np.concatenate((spec_red, aug_red))

    ##retrain photo-z with augmented training set
    model_name = (train_file.split('public/'))[1].split('_training')[0]+'_pz_model_'+(train_file.split('training_'))[1].split('.hdf5')[0]
    
    augmented_train_data = TableHandle("aug_train", combined_train)
    augmented_informer = make_informer_task3_task4(model_name)
    augmented_model = augmented_informer.inform(augmented_train_data)

    augmented_estimator = make_estimator(augmented_model)
    augmented_pz_out = augmented_estimator.estimate(test_data)
    attach_object_ids(augmented_pz_out, test_data)
    augmented_pz_out.path = output_file
    augmented_pz_out.write()
    return


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

    run_taskset_1_estimation_only(model_file=model_file, test_file=test_file, output_file=output_file)
    return


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

    run_taskset_3_training_and_estimation(train_file, test_file, output_file)
    return


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
