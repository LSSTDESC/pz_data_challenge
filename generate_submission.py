#!/usr/bin/env python3
"""Script to generate pre-computed predictions and model binaries for the rail+aion submission.

Trains the full Mixture of Experts pipeline on all task sets, simulations, and
scenarios, saving the outputs directly to submissions/rail_aion/ so they are
ready for the test suite and final packaging.
"""

import os
import sys
from pathlib import Path

# Insert repository root to make rail_aion_pz importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rail_aion_pz

def main():
    public_area = "tests/public"
    submit_dir = "submissions/whitesmoke"
    os.makedirs(submit_dir, exist_ok=True)

    sims = ["cardinal", "flagship"]
    scenarios = ["1yr", "10yr"]
    tasksets = [1, 2, 3, 4]

    print("======================================================================")
    print("Generating pre-computed submission files for all Task Sets...")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Device: {os.environ.get('AION_PZ_DEVICE', 'cpu')}")
    print("======================================================================\n")

    for taskset in tasksets:
        for sim in sims:
            for scenario in scenarios:
                print(f"--- Task Set {taskset} | {sim} | {scenario} ---")
                
                train_file = os.path.join(
                    public_area, f"pz_challenge_taskset_{taskset}_{sim}_training_{scenario}.hdf5"
                )
                test_file = os.path.join(
                    public_area, f"pz_challenge_taskset_{taskset}_{sim}_test_{scenario}.hdf5"
                )
                
                output_file = os.path.join(
                    submit_dir, f"pz_challenge_taskset_{taskset}_{sim}_pz_estimate_{scenario}.hdf5"
                )
                model_file = os.path.join(
                    submit_dir, f"pz_challenge_taskset_{taskset}_{sim}_pz_model_{scenario}.pkl"
                )
                
                if not os.path.exists(train_file) or not os.path.exists(test_file):
                    print(f"Warning: Data files not found. Skipping combination.\n")
                    continue
                
                print(f"Training experts and writing to outputs...")
                rail_aion_pz.train_and_estimate(
                    train_file=train_file,
                    test_file=test_file,
                    output_file=output_file,
                    save_model_to=model_file
                )
                print(f"Success! Generated:")
                print(f"  Estimate: {output_file}")
                print(f"  Model:    {model_file}\n")

    print("======================================================================")
    print("All submission files successfully generated!")
    print("======================================================================")

if __name__ == "__main__":
    main()
