import os
import time
from typing import Any, Callable

from . import submit_utils

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]


def run_taskset_2(
    public_area: str,
    submission: str,
    run_taskset_2_estimation_only: Callable,
    run_taskset_2_training_and_estimation: Callable,
) -> None:

    submit_dir: str = f"submissions/{submission}"

    manifest_dict: dict[str, Any] = {}

    for sim in SIMS:
        for scenario in SCENARIOS:

            key = f"{sim}_{scenario}"

            submit_file = os.path.join(
                submit_dir, f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5"
            )
            model_file = os.path.join(
                submit_dir, f"pz_challenge_taskset_1_{sim}_pz_model_{scenario}.hdf5"
            )
            training_file = os.path.join(
                public_area, f"pz_challenge_taskset_1_{sim}_training_{scenario}.hdf5"
            )
            test_file = os.path.join(
                public_area, f"pz_challenge_taskset_1_{sim}_test_{scenario}.hdf5"
            )
            output_file_2 = os.path.join(
                submit_dir,
                "outputs_2",
                f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5",
            )
            output_file_3 = os.path.join(
                submit_dir,
                "outputs_3",
                f"pz_challenge_taskset_1_{sim}_pz_estimate_{scenario}.hdf5",
            )

            # Check on the premade submission files
            manifest_dict[f"{key}_1"] = submit_utils.check_pz_submission_file(
                submit_file, test_file
            )

            # Run the estimate only function
            if run_taskset_2_estimation_only is not None:
                time_2_before = time.time()
                run_taskset_2_estimation_only(model_file, test_file, output_file_2)
                time_2 = time.time() - time_2_before
                manifest_dict[f"{key}_time_2"] = time_2

                # Check on the files made by the estimate only function
                manifest_dict[f"{key}_2"] = submit_utils.check_pz_submission_file(
                    output_file_2, test_file
                )

            # Run the train and estimate function
            if run_taskset_2_training_and_estimation is not None:
                time_3_before = time.time()
                run_taskset_2_training_and_estimation(
                    training_file, test_file, output_file_3
                )
                time_3 = time.time() - time_3_before
                manifest_dict[f"{key}_time_3"] = time_3

                # Check on the files made by the estimate only function
                manifest_dict[f"{key}_3"] = submit_utils.check_pz_submission_file(
                    output_file_3, test_file
                )

    submit_utils.pretty_print_manifest_dict(manifest_dict)
    submit_utils.pretty_print_time_dict(manifest_dict)

    submit.check_manifest_dict(manifest_dict)
    
