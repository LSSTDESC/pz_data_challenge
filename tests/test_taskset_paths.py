import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pz_data_challenge.taskset_1 as taskset_1_module
import pz_data_challenge.taskset_2 as taskset_2_module
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2


def _capture_paths(monkeypatch: pytest.MonkeyPatch, module, runner) -> list[str]:
    filenames: list[str] = []

    def record_paths(*paths) -> None:
        filenames.extend(Path(str(path)).name for path in paths)

    def check_pz_submission_file(submit_file, test_file) -> list[int]:
        record_paths(submit_file, test_file)
        return [7]

    def estimation_only(model_file, test_file, output_file) -> None:
        record_paths(model_file, test_file, output_file)

    def training_and_estimation(training_file, test_file, output_file) -> None:
        record_paths(training_file, test_file, output_file)

    monkeypatch.setattr(
        module.submit_utils,
        "check_pz_submission_file",
        check_pz_submission_file,
    )
    monkeypatch.setattr(
        module.submit_utils,
        "pretty_print_manifest_dict",
        lambda manifest_dict: None,
    )
    monkeypatch.setattr(
        module.submit_utils,
        "pretty_print_time_dict",
        lambda manifest_dict: None,
    )
    monkeypatch.setattr(
        module.submit_utils,
        "check_manifest_dict",
        lambda manifest_dict: None,
    )

    runner(
        "public",
        "submission",
        estimation_only,
        training_and_estimation,
    )

    return filenames


def test_taskset_1_builds_only_taskset_1_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filenames = _capture_paths(monkeypatch, taskset_1_module, run_taskset_1)

    assert filenames
    assert all(name.startswith("pz_challenge_taskset_1_") for name in filenames)
    assert all("taskset_2" not in name for name in filenames)


def test_taskset_2_builds_only_taskset_2_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filenames = _capture_paths(monkeypatch, taskset_2_module, run_taskset_2)

    assert filenames
    assert all(name.startswith("pz_challenge_taskset_2_") for name in filenames)
    assert all("taskset_1" not in name for name in filenames)