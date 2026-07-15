import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import pytest

import importlib.util

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

TASKSET1_SUBMISSION_PATH = SCRIPTS_DIR / "taskset1_mlp_submission.py"
_spec = importlib.util.spec_from_file_location("taskset1_mlp_submission", TASKSET1_SUBMISSION_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_module)
export_model_predictions_to_qp = _module.export_model_predictions_to_qp

FEATURE_COLS = "mag_u_lsst,mag_g_lsst,mag_r_lsst,mag_i_lsst,mag_z_lsst,mag_y_lsst"
TRAIN_SCRIPT = Path(__file__).resolve().parent / "scripts" / "train_simple_mlp_flow_torch.py"


def _run_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    requested_name = Path(model_file).name
    if requested_name not in MODEL_PATHS:
        raise FileNotFoundError(f"No bundled model mapping for {requested_name}")
    resolved_model_file = MODEL_PATHS[requested_name]
    export_model_predictions_to_qp(
        model_file=resolved_model_file,
        test_file=test_file,
        output_file=output_file,
        posterior_samples=32,
        sample_steps=32,
        z_grid_size=121,
        device="cpu",
    )


def _run_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "mlp_train"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--input", str(train_file),
            "--target-col", "redshift",
            "--feature-cols", FEATURE_COLS,
            "--outdir", str(outdir),
            "--epochs", "5",
            "--batch-size", "1024",
            "--hidden", "256",
            "--depth", "3",
            "--posterior-samples", "32",
            "--sample-steps", "32",
            "--seed", "7",
            "--split-seed", "7",
            "--device", "cpu",
        ]
        subprocess.run(cmd, check=True)

        model_path = outdir / "simple_mlp_flow_model.pt"
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        export_model_predictions_to_qp(
            model_file=model_path,
            test_file=test_file,
            output_file=output_file,
            posterior_samples=32,
            sample_steps=32,
            z_grid_size=121,
            device="cpu",
        )

def _stage_packaged_submission_files() -> None:
    submit_root = Path(SUBMIT_DIR)
    nested_root = submit_root / SUBMISSION_NAME
    subtask1_dir = nested_root / "subtask1"

    if subtask1_dir.exists():
        for src in subtask1_dir.glob("*.hdf5"):
            dst = submit_root / src.name
            shutil.copy2(src, dst)

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "mlp_pretrained_subtask12"
SUBMISSION_URL: str = "https://github.com/RongFangeecs/pz_data_challenge/releases/download/v0.1.0-mlp-pretrained-subtask12/mlp_pretrained_subtask12.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

MODEL_PATHS = {
    "pz_challenge_taskset_1_cardinal_pz_model_1yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts1_cardinal_1yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_1_cardinal_pz_model_10yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts1_cardinal_10yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_1_flagship_pz_model_1yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts1_flagship_1yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_1_flagship_pz_model_10yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts1_flagship_10yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_2_cardinal_pz_model_1yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts2_cardinal_1yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_2_cardinal_pz_model_10yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts2_cardinal_10yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_2_flagship_pz_model_1yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts2_flagship_1yr_simple_mlp_flow_model.pt",
    "pz_challenge_taskset_2_flagship_pz_model_10yr.pkl": Path(SUBMIT_DIR) / SUBMISSION_NAME / "subtask2" / "ts2_flagship_10yr_simple_mlp_flow_model.pt",
}


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
        _stage_packaged_submission_files()

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


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)
def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)
def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)
def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)
def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)
def run_taskset_3_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)
def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)
def run_taskset_4_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)
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

    
@pytest.mark.skip(reason="Current first-pass MLP submission covers tasksets 1 and 2 only.")
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


@pytest.mark.skip(reason="Current first-pass MLP submission covers tasksets 1 and 2 only.")
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
