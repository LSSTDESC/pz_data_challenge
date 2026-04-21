import os
from pathlib import Path
import pytest

# Put needed import here
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import tables_io
import qp

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "easy_forest"
SUBMISSION_URL: str = "https://github.com/joselotl/data_share/releases/download/v0.1/easy_forest_output.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    print('estoy pasando')
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """
    
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set")
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)
        print('ase')

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}")
        print('osi')

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


features = [
    'mag_H_roman','mag_J_roman','mag_Y_roman',
    'mag_g_lsst','mag_r_lsst','mag_i_lsst',
    'mag_z_lsst','mag_y_lsst','mag_u_lsst',
]

def train_easy_random_forest(df):
    # ---- Handle NaNs ----
    df[features] = df[features].fillna(30)

    # ---- Split ----
    X = df[features].values
    y = df['redshift'].values

    # ---- Train easy model ----
    forest = RandomForestRegressor(n_estimators=50)
    forest.fit(X, y)
    train_pred = forest.predict(X)
    sigma = np.std(y - train_pred)

    return forest,sigma

def estimate_easy_random_forest(df,model):
    forest,sigma = model

    X = df[features].values
    # ---- Predict ----
    z_pred = forest.predict(X)

    z_err = sigma * np.ones_like(z_pred)

    # ---- Output ----
    out = pd.DataFrame({
        "z": z_pred,
        "z_err": z_err
    })
    return out
    
def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    ) -> None:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    test_data = tables_io.read(test_file)
    test_data = tables_io.convert(test_data, tables_io.types.PD_DATAFRAME)
    preds = estimate_easy_random_forest(test_data,model)
    ens_n = qp.Ensemble(qp.stats.norm, data=dict(loc=np.array(preds['z']), scale=np.array(preds['z_err'])))
    ancil = {'object_id': test_data['object_id'].astype('int64').to_numpy(),
        'z_mode': preds['z'].to_numpy()}
    ens_n.set_ancil(ancil)
    ens_n.write_to(output_file)

def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    model_file: str | Path | None = None,
    # train a model using the "train_file" and make p(z) estimates
    # and write them to "output_file"
    ) -> None:
    test_data = tables_io.read(test_file)
    train_data = tables_io.read(train_file)
    test_data = tables_io.convert(test_data, tables_io.types.PD_DATAFRAME)
    train_data = tables_io.convert(train_data, tables_io.types.PD_DATAFRAME)
   
    model = train_easy_random_forest(train_data)

    if model_file is not None:
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

    preds = estimate_easy_random_forest(test_data,model)
    ens_n = qp.Ensemble(qp.stats.norm, data=dict(loc=np.array(preds['z']), scale=np.array(preds['z_err'])))
    ancil = {'object_id': test_data['object_id'].astype('int64').to_numpy(),
        'z_mode': preds['z'].to_numpy()}
    ens_n.set_ancil(ancil)
    ens_n.write_to(output_file)

run_taskset_2_training_and_estimation = run_taskset_1_training_and_estimation
run_taskset_2_estimation_only = run_taskset_1_estimation_only

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
