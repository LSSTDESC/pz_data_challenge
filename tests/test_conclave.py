"""CI test for the `conclave` TS1 submission (LSST-DESC PZ Data Challenge).

conclave = a committee ensemble of PZFlow + GPz + FlexZBoost combined with convex-QP
optimal weights and a global-PIT recalibration. The method lives in the pip-installable
`conclave` package (github.com/rhw/conclave, pinned in requirements_conclave.txt); this
test file is the thin submission entry point that the upstream harness discovers and runs.

Structure mirrors the accepted reference submissions (e.g. test_rail_knn_test.py): it defines
the four run_taskset_* entry points and calls run_taskset_1/2 over SIMS x SCENARIOS. The two
functions delegate to conclave.submission, which internally applies the estimators' catalog tag
and freezes the QP weights + recalibrator (the blind test has no redshift, so recal is fit on a
held-out slice of the training file). Pre-made estimates + pretrained models are hosted at
SUBMISSION_URL and unpacked into submissions/conclave/.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from rail.core.data import TableHandle
from rail.utils import catalog_utils

from pz_data_challenge import submit_utils
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2

from conclave.submission import (
    run_taskset_1_training_and_estimation as _conclave_train_and_estimate,
    run_taskset_1_estimation_only as _conclave_estimate_only,
)

# CI concession: the full method trains 3 estimators (PZFlow/GPz/FlexZBoost) on 100k x 4
# sim/scenario — hours, over the GitHub-runner budget. For the train+estimate CI path we train
# on a subsample so CI proves the pipeline runs and emits valid p(z); the REAL full-scale-trained
# models + estimates ship in the release tarball (SUBMISSION_URL) and feed the estimation-only
# path. PZDC_CI_MAX_TRAIN=0 (default) trains on the full data (the real submission behaviour).
CI_MAX_TRAIN: int = int(os.environ.get("PZDC_CI_MAX_TRAIN", "0"))

SUBMISSION_NAME: str = "conclave"
# GitHub release .tgz of the pre-made estimates + pretrained models (set before opening the PR).
SUBMISSION_URL: str = "https://github.com/rhw/conclave/releases/download/ts1-v1/conclave_submission.tgz"

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = os.environ.get("PZDC_PUBLIC_AREA", "tests/public")

# Official catalog tag for reading the challenge catalogs. conclave's estimators additionally
# apply their own (equivalent LSST+Roman) tag internally via conclave.bands.apply_band_set.
CATALOG_TAG = "cardinal_roman_rubin"


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(
                f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set"
            )
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN"):
            os.system(f"\\rm -rf {SUBMIT_DIR}/outputs_2 {SUBMIT_DIR}/outputs_3")

    for sub in ("outputs_2", "outputs_3"):
        try:
            os.makedirs(os.path.join(SUBMIT_DIR, sub))
        except Exception:
            pass
    request.addfinalizer(teardown_submit_area)

    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply(CATALOG_TAG)
    return 0


def _maybe_subsample_train(train_file: str) -> str:
    """Return train_file unchanged unless PZDC_CI_MAX_TRAIN>0, in which case write a
    stratified-by-nothing random subsample to a temp hdf5 (keeps CI train fast)."""
    if CI_MAX_TRAIN <= 0:
        return train_file
    import tempfile
    import tables_io
    d = tables_io.read(train_file)
    keys = list(d.keys())
    n = len(d[keys[0]])
    if n <= CI_MAX_TRAIN:
        return train_file
    idx = np.sort(np.random.default_rng(0).choice(n, CI_MAX_TRAIN, replace=False))
    sub = {k: np.asarray(d[k])[idx] for k in keys}
    stem = os.path.join(tempfile.mkdtemp(), "ci_train")
    tables_io.write(sub, stem, "hdf5")
    return stem + ".hdf5"


def run_taskset_1_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _conclave_estimate_only(str(model_file), str(test_file), str(output_file))


def run_taskset_1_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    train_file = _maybe_subsample_train(str(train_file))
    _conclave_train_and_estimate(train_file, str(test_file), str(output_file))


def run_taskset_2_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    run_taskset_1_estimation_only(model_file, test_file, output_file)


def run_taskset_2_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    run_taskset_1_training_and_estimation(train_file, test_file, output_file)


def test_example_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_example_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )
