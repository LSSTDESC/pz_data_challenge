import os
import sys
import glob
import time
import shutil
import tarfile
import h5py
import numpy as np

# Ensure path to curated Pontifex package and pz_challenge
REPO_ROOT = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/code/pz_challenge"
PONTIFEX_SRC = "/home/mardom/Rubin-LSST-Research/Photometric-Redshift/code/Pontifex/src"
sys.path.insert(0, PONTIFEX_SRC)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pontifex
from pz_data_challenge.submit_utils import check_pz_submission_file

SUBMIT_DIR = os.path.join(REPO_ROOT, "submissions/expiation")
PUBLIC_AREA = os.path.join(REPO_ROOT, "tests/public")
MODELS_DIR = os.path.join(REPO_ROOT, "submissions/whitesmoke")
BACKUP_REPO = "/media/mardom/Backup/Rubin-LSST-Research/Photometric-Redshift/data/pz_data_challenge"

sims = ["cardinal", "flagship"]
scenarios = ["1yr", "10yr"]
tasksets = [1, 2, 3, 4]

os.makedirs(SUBMIT_DIR, exist_ok=True)
for sub in ("outputs_2", "outputs_3", "pontifex"):
    os.makedirs(os.path.join(SUBMIT_DIR, sub), exist_ok=True)

# Copy embedded curated pontifex source files for self-contained release
for f in ["__init__.py", "em.py", "estimators.py", "guard.py", "pipeline.py"]:
    src_f = os.path.join(PONTIFEX_SRC, "pontifex", f)
    dst_f = os.path.join(SUBMIT_DIR, "pontifex", f)
    if os.path.exists(src_f):
        shutil.copyfile(src_f, dst_f)
    elif os.path.exists(os.path.join(PONTIFEX_SRC, "pontifex/pz", f)):
        shutil.copyfile(os.path.join(PONTIFEX_SRC, "pontifex/pz", f), dst_f)

# Copy stats taskset yaml files
for stat_yaml in glob.glob(os.path.join(MODELS_DIR, "stats_taskset*.yaml")):
    shutil.copyfile(stat_yaml, os.path.join(SUBMIT_DIR, os.path.basename(stat_yaml)))

print("=" * 80)
print("=== Launching Expiation Full Production Generation (320,000 Galaxies) ===")
print("=" * 80)

total_start = time.time()
successful_files = 0
total_galaxies = 0

for t in tasksets:
    for sim in sims:
        for sc in scenarios:
            prefix = f"pz_challenge_taskset_{t}_{sim}"
            test_file = os.path.join(PUBLIC_AREA, f"{prefix}_test_{sc}.hdf5")
            model_src = os.path.join(MODELS_DIR, f"{prefix}_pz_model_{sc}.pkl")
            model_dst = os.path.join(SUBMIT_DIR, f"{prefix}_pz_model_{sc}.pkl")
            submit_file = os.path.join(SUBMIT_DIR, f"{prefix}_pz_estimate_{sc}.hdf5")
            out_3_file = os.path.join(SUBMIT_DIR, "outputs_3", f"{prefix}_pz_estimate_{sc}.hdf5")

            if not os.path.exists(test_file):
                print(f"[ERROR] Test file missing: {test_file}")
                continue
            if not os.path.exists(model_src):
                print(f"[ERROR] Model source missing: {model_src}")
                continue

            # Ensure model is copied to submit directory
            if not os.path.exists(model_dst) or os.path.getsize(model_dst) == 0:
                shutil.copyfile(model_src, model_dst)

            t0 = time.time()
            print(f"\n[RUNNING] Taskset {t} | {sim} | {sc} ...")
            pontifex.estimate_only(model_dst, test_file, submit_file)

            # Copy to outputs_3
            shutil.copyfile(submit_file, out_3_file)

            # Validation check
            flags = check_pz_submission_file(submit_file, test_file)
            assert flags == [1, 2, 3, 4, 5, 6, 7], f"Validation failed for {submit_file} with flags {flags}"

            with h5py.File(submit_file, "r") as hf:
                n_gals = hf["ancil"]["zmode"].shape[0]
                zmode = hf["ancil"]["zmode"][:]
                yvals = hf["data"]["yvals"][:]
                assert not np.any(np.isnan(yvals)), f"NaNs detected in yvals for {submit_file}"

            elapsed = time.time() - t0
            print(f"[PASSED] Taskset {t} {sim} {sc}: {n_gals} galaxies in {elapsed:.1f}s | zmode mean={np.mean(zmode):.4f}, range=[{np.min(zmode):.4f}, {np.max(zmode):.4f}]")
            successful_files += 1
            total_galaxies += n_gals

print("\n" + "=" * 80)
print(f"Summary: Successfully processed {successful_files}/16 files, {total_galaxies}/320,000 galaxies in {time.time()-total_start:.1f}s")
print("=" * 80)

assert total_galaxies == 320000, f"Expected 320,000 galaxies, got {total_galaxies}"

# Build expiation.tgz tarball in repo root
archive_name = os.path.join(REPO_ROOT, "expiation.tgz")
print(f"\n=== Building {archive_name} ===")
with tarfile.open(archive_name, "w:gz") as tar:
    tar.add(SUBMIT_DIR, arcname=".")

print(f"Archive created: {archive_name} ({os.path.getsize(archive_name) / 1e6:.2f} MB)")

# Copy archive and submissions to Backup repository if available
if os.path.exists(BACKUP_REPO):
    backup_submit = os.path.join(BACKUP_REPO, "submissions/expiation")
    os.makedirs(backup_submit, exist_ok=True)
    backup_archive = os.path.join(BACKUP_REPO, "expiation.tgz")
    shutil.copyfile(archive_name, backup_archive)
    print(f"Copied archive to {backup_archive}")

print("\n=== Expiation Release Generation Complete and Verified 100% Compliant ===")
