import math
import os
from pathlib import Path

import numpy as np
import pytest
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.k_nearneigh import KNearNeighEstimator, KNearNeighInformer
from rail.utils import catalog_utils

from pz_data_challenge import submit_utils
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

SUBMISSION_NAME: str = "rail_knn_4tasks"
SUBMISSION_URL: str = (
    "https://github.com/Lhior/pz_data_challenge/releases/download/"
    "pzdc-rail-knn-4tasks-v1/rail_knn_4tasks_submission.tgz"
)

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

CATALOG_TAG = "cardinal_roman_rubin"
_CHUNK = 150_000
_ZMAX = 3.0
_NZ = 151


def _attach_object_ids(pz_out, test_data: TableHandle) -> None:
    pz_out.data.ancil["object_id"] = np.asarray(test_data()["object_id"])


def _clean_training_file(train_file: str | Path) -> str:
    path = str(train_file)
    data = tables_io.read(path)
    bad_mask = np.isnan(data["redshift"])
    if not bad_mask.any():
        return path
    cleaned_path = path.replace(".hdf5", "_cleaned.hdf5")
    cleaned_data = {key: val[~bad_mask] for key, val in data.items()}
    tables_io.write(cleaned_data, cleaned_path)
    return cleaned_path


def _make_knn_informer() -> KNearNeighInformer:
    return KNearNeighInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        zmax=_ZMAX,
        nzbins=_NZ,
        chunk_size=_CHUNK,
        nondetect_val=math.nan,
        trainfrac=0.2,
        nneigh_min=3,
        nneigh_max=5,
        ngrid_sigma=6,
    )


def _make_knn_estimator(model) -> KNearNeighEstimator:
    return KNearNeighEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        output_mode="return",
        nzbins=_NZ,
        zmax=_ZMAX,
        chunk_size=_CHUNK,
        nondetect_val=math.nan,
    )


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
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
    test_data = TableHandle("test", path=str(test_file))
    estimator = _make_knn_estimator(str(model_file))
    pz_out = estimator.estimate(test_data)
    _attach_object_ids(pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_x_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    cleaned_train_file = _clean_training_file(train_file)
    train_data = TableHandle("train", path=cleaned_train_file)
    test_data = TableHandle("test", path=str(test_file))
    informer = _make_knn_informer()
    model = informer.inform(train_data)
    estimator = _make_knn_estimator(model)
    pz_out = estimator.estimate(test_data)
    _attach_object_ids(pz_out, test_data)
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
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
