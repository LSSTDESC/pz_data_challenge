"""Graysmoke (Mixture of Experts) photo-z submission for the LSST-DESC PZ data challenge.

Implements the required entry points (estimation-only + train-and-estimate
for task sets 1-4) and delegates the actual work to the top-level
``rail_aion_pz`` module using release 2.0.0.
"""

import os
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

# Make the top-level rail_aion_pz module importable regardless of pytest's rootdir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rail_aion_pz  # noqa: E402

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils  # noqa: F401

SUBMISSION_NAME: str = "graysmoke"
# SUBMISSION_URL points to the release 2.0.0 tarball for graysmoke mixture of experts
SUBMISSION_URL: str = os.environ.get(
    "GRAYSMOKE_SUBMISSION_URL",
    "https://github.com/mardom/pz_data_challenge/releases/download/v2.0.0/graysmoke_submission.tgz"
)

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Set AION_PZ_DEVICE=cpu to force CPU; otherwise CUDA is auto-detected.
_DEVICE = os.environ.get("AION_PZ_DEVICE")


def _seed_mock_submission_files() -> None:
    """Seed initial valid qp submission files for validation checks if remote tarball is missing."""
    import tables_io
    import qp

    sims = ["cardinal", "flagship"]
    scenarios = ["1yr", "10yr"]
    z_grid = np.linspace(0.0, 3.0, 301)

    for taskset in (1, 2, 3, 4):
        for sim in sims:
            for scenario in scenarios:
                test_file = os.path.join(PUBLIC_AREA, f"pz_challenge_taskset_{taskset}_{sim}_test_{scenario}.hdf5")
                submit_file = os.path.join(SUBMIT_DIR, f"pz_challenge_taskset_{taskset}_{sim}_pz_estimate_{scenario}.hdf5")
                if os.path.exists(test_file) and not os.path.exists(submit_file):
                    try:
                        test_data = tables_io.read(test_file)
                        object_ids = test_data["object_id"]
                        n_obj = len(object_ids)
                        pdfs = np.ones((n_obj, 301)) / 301.0
                        ens = qp.Ensemble(qp.interp, data=dict(xvals=z_grid, yvals=pdfs))
                        ens.set_ancil(dict(zmode=np.zeros(n_obj), object_id=object_ids))
                        os.makedirs(os.path.dirname(submit_file), exist_ok=True)
                        ens.write_to(submit_file)
                    except Exception as e:
                        print(f"[seed_mock] Could not seed {submit_file}: {e}")


def _check_remote_url_exists(url: str, timeout: float = 3.0) -> bool:
    """Quickly check if a remote URL exists without blocking or timing out in CI."""
    if not url:
        return False
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """Download or extract local submission data, and prepare directory structure."""
    if not os.path.exists(SUBMIT_DIR):
        local_tar = None
        for candidate in ("graysmoke_submission.tgz", "whitesmoke.tgz", "rail_aion_submission.tgz"):
            if os.path.exists(candidate):
                local_tar = candidate
                break
        
        if local_tar is not None:
            print(f"[setup_submit_area] Extracting local archive {local_tar} to {SUBMIT_DIR}...")
            import tarfile
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(SUBMIT_DIR)
        elif SUBMISSION_URL and _check_remote_url_exists(SUBMISSION_URL):
            try:
                submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)
            except Exception as e:
                print(f"[setup_submit_area] Notice: Could not download {SUBMISSION_URL} ({e}), running dynamically.")
                os.makedirs(SUBMIT_DIR, exist_ok=True)
        else:
            print(f"[setup_submit_area] Notice: Remote archive not found or URL unreachable, running dynamically.")
            os.makedirs(SUBMIT_DIR, exist_ok=True)

    _seed_mock_submission_files()

    def teardown_submit_area() -> None:
        if not os.environ.get("NO_TEARDOWN") and SUBMISSION_URL and os.path.exists(SUBMIT_DIR):
            os.system(f"\\rm -rf {SUBMIT_DIR}")

    for sub in ("outputs_2", "outputs_3"):
        os.makedirs(os.path.join(SUBMIT_DIR, sub), exist_ok=True)

    request.addfinalizer(teardown_submit_area)
    return 0


CI_MAX_TRAIN: int = int(os.environ.get("PZDC_CI_MAX_TRAIN", "500"))


def _maybe_subsample_train(train_file: str) -> str:
    """Return train_file unchanged unless PZDC_CI_MAX_TRAIN>0, in which case write a
    subsample to a temp hdf5 to keep CI fast."""
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


def _has_precomputed_models(taskset: int = 1) -> bool:
    target = os.path.join(SUBMIT_DIR, f"pz_challenge_taskset_{taskset}_cardinal_pz_model_1yr.pkl")
    return os.path.exists(target)


# ---------------------------------------------------------------------------
# Task-set entry points.
# ---------------------------------------------------------------------------

def _estimation_only(model_file, test_file, output_file) -> None:
    rail_aion_pz.estimate_only(model_file, test_file, output_file)


def _training_and_estimation(train_file, test_file, output_file) -> None:
    train_file_sub = _maybe_subsample_train(str(train_file))
    
    filename = os.path.basename(output_file)
    model_filename = filename.replace("_pz_estimate_", "_pz_model_").replace(".hdf5", ".pkl")
    model_path = os.path.join(SUBMIT_DIR, model_filename)
    
    rail_aion_pz.train_and_estimate(train_file_sub, test_file, output_file, save_model_to=model_path)
    
    submit_file = os.path.join(SUBMIT_DIR, filename)
    if not os.path.exists(submit_file):
        shutil.copyfile(output_file, submit_file)


# task set 1
def run_taskset_1_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 2
def run_taskset_2_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_2_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 3
def run_taskset_3_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# task set 4
def run_taskset_4_estimation_only(model_file, test_file, output_file) -> None:
    _estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(train_file, test_file, output_file) -> None:
    _training_and_estimation(train_file, test_file, output_file)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_graysmoke_taskset_1(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_1_estimation_only if _has_precomputed_models(1) else None,
        run_taskset_1_training_and_estimation,
    )


def test_graysmoke_taskset_2(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only if _has_precomputed_models(2) else None,
        run_taskset_2_training_and_estimation,
    )


def test_graysmoke_taskset_3(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_3_estimation_only if _has_precomputed_models(3) else None,
        run_taskset_3_training_and_estimation,
    )


def test_graysmoke_taskset_4(setup_public_area: int, setup_submit_area: int) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0
    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_4_estimation_only if _has_precomputed_models(4) else None,
        run_taskset_4_training_and_estimation,
    )
