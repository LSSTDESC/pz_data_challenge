#!/usr/bin/env python3
"""Build the complete whitesmoke submission files."""
import os
import sys
import time
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rail_aion_pz

PUBLIC_AREA = "tests/public"
OUT_DIR = "submissions/whitesmoke"
TASKSETS = [1, 2, 3, 4]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]

CI_MAX_TRAIN: int = int(os.environ.get("PZDC_CI_MAX_TRAIN", "0"))

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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ["AION_PZ_DEVICE"] = "cpu"  # Keep CPU for predictability

    for ts in TASKSETS:
        for sim in SIMS:
            for scenario in SCENARIOS:
                tag = f"taskset_{ts}_{sim}_{scenario}"
                train_file = f"{PUBLIC_AREA}/pz_challenge_taskset_{ts}_{sim}_training_{scenario}.hdf5"
                test_file = f"{PUBLIC_AREA}/pz_challenge_taskset_{ts}_{sim}_test_{scenario}.hdf5"
                model_file = f"{OUT_DIR}/pz_challenge_taskset_{ts}_{sim}_pz_model_{scenario}.pkl"
                out_file = f"{OUT_DIR}/pz_challenge_taskset_{ts}_{sim}_pz_estimate_{scenario}.hdf5"

                if not (os.path.exists(train_file) and os.path.exists(test_file)):
                    print(f"[skip] {tag}: missing input files")
                    continue

                print(f"[run ] {tag} ...", flush=True)
                t0 = time.time()
                sub_train = _maybe_subsample_train(train_file)
                rail_aion_pz.train_and_estimate(
                    sub_train, test_file, out_file,
                    save_model_to=model_file,
                )
                print(f"[done] {tag} in {time.time() - t0:.1f}s -> {out_file}", flush=True)

    print("ALL DONE")

if __name__ == "__main__":
    main()
