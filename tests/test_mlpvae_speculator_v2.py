"""mlpvae_speculator_v2 submission for the PZ Data Challenge (Task Sets 1 & 2).

Self-contained on purpose: a submission PR only adds this one file plus
requirements_mlpvae_speculator_v2.txt and the workflow yaml (see PLAN.md's
submission-mechanism notes) -- `submissions_src/` is local dev scratch,
gitignored, and never committed, so nothing here may import from it.

Neither of this model's two dependencies is pip-installable: one repo
(github.com/Klinjin/photoz_mlpvae) carries vendored plain-Python source
(the model class, a third package `photoz_vae` it imports from, and
obs_catalog's feature pipeline) plus small checkpoints; the other
(github.com/Klinjin/speculator) carries ~1.5GB of Speculator weights via
Git LFS. `ensure_model_deps` below clones both on demand into
SUBMIT_DIR/_deps (SUBMIT_DIR is already gitignored, so no new rule
needed) instead of relying on hardcoded local paths, which a CI runner
would not have.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import qp
import torch

from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2

SUBMISSION_NAME: str = "mlpvae_speculator_v2"
SUBMISSION_URL: str = ""

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
# Task Set 2 shares Task Set 1's exact column schema (its distinguishing
# feature is training/test distribution mismatch, not a different format --
# see PLAN.md), so both are set up identically here.
TASKSETS = [1, 2]

# ---------------------------------------------------------------------------
# Model/code dependencies (see module docstring)
# ---------------------------------------------------------------------------
DEPS_DIR = Path(SUBMIT_DIR) / "_deps"
PHOTOZ_MLPVAE_URL = "https://github.com/Klinjin/photoz_mlpvae.git"
SPECULATOR_URL = "https://github.com/Klinjin/speculator.git"
# Below this, a Speculator .npz is still an unsmudged Git LFS pointer (~130 bytes).
_LFS_SMUDGED_MIN_BYTES = 1_000_000


def _run(cmd: list[str], **kwargs) -> None:
    subprocess.run(cmd, check=True, **kwargs)


def _clone_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)])


def ensure_model_deps() -> tuple[str, str, str, str]:
    """Clone dependency repos if needed; return (baseline_ckpt, speculator_dir,
    filter_dir, finetuned_dir). Also inserts the sys.path entries needed to
    import `photoz_mlpvae.*`, `obs_catalog.*`, and `photoz_vae.*` -- all
    three live inside the photoz_mlpvae clone.
    """
    photoz_mlpvae_dir = DEPS_DIR / "photoz_mlpvae"
    speculator_repo_dir = DEPS_DIR / "speculator"

    # Registers the filter.lfs.* smudge/clean hooks in ~/.gitconfig -- without
    # this, `git lfs pull` refuses to check out objects ("Git LFS is not
    # installed for this repository") even though the git-lfs binary exists.
    _run(["git", "lfs", "install", "--skip-repo"])

    _clone_if_missing(PHOTOZ_MLPVAE_URL, photoz_mlpvae_dir)
    _clone_if_missing(SPECULATOR_URL, speculator_repo_dir)

    lfs_marker = speculator_repo_dir / "trained" / "Inoue_IGM" / "30_100" / "pca_basis.npz"
    if not lfs_marker.exists() or lfs_marker.stat().st_size < _LFS_SMUDGED_MIN_BYTES:
        _run(["git", "lfs", "pull"], cwd=str(speculator_repo_dir))

    for p in (str(DEPS_DIR), str(photoz_mlpvae_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    baseline_ckpt = photoz_mlpvae_dir / "trained" / "mlpvae_v2_lsst_gaap1p0" / "best.pt"
    speculator_dir = speculator_repo_dir / "trained" / "Inoue_IGM"
    filter_dir = photoz_mlpvae_dir / "obs_catalog" / "filters"
    finetuned_dir = photoz_mlpvae_dir / "trained" / "mlpvae_speculator_v2_finetuned"

    for p in (baseline_ckpt, speculator_dir, filter_dir, finetuned_dir):
        if not p.exists():
            raise FileNotFoundError(f"Expected submission dependency missing: {p}")

    return str(baseline_ckpt), str(speculator_dir), str(filter_dir), str(finetuned_dir)


BASELINE_CKPT, SPECULATOR_DIR, FILTER_DIR, FINETUNED_DIR = ensure_model_deps()

from obs_catalog.dataloader import build_features  # noqa: E402
from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE  # noqa: E402

# ---------------------------------------------------------------------------
# Adapter: challenge HDF5 schema (mag_{band}_lsst[_err]) -> the model's
# GAAP-named feature columns ({band}_gaap1p0Mag[Err]) -- a thin rename, not
# a reimplementation. build_features's imputation/scaling already handles
# NaN non-detections generically.
# ---------------------------------------------------------------------------
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
TARGET_COL = "redshift"  # obs_catalog.dataloader.TARGET_COL


def load_challenge_hdf5(path: str | Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][()] for k in f.keys()})


def rename_to_gaap_schema(df: pd.DataFrame, is_test: bool) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    missing_bands = []
    for band in LSST_BANDS:
        mag_col = f"mag_{band}_lsst"
        err_col = f"mag_{band}_lsst_err"
        if mag_col not in df.columns or err_col not in df.columns:
            missing_bands.append(band)
            continue
        rename_map[mag_col] = f"{band}_gaap1p0Mag"
        rename_map[err_col] = f"{band}_gaap1p0MagErr"

    if missing_bands:
        raise ValueError(
            f"Challenge file is missing expected LSST columns for bands "
            f"{missing_bands} (looked for mag_{{band}}_lsst / "
            f"mag_{{band}}_lsst_err). Available columns: {list(df.columns)}"
        )

    out = df.rename(columns=rename_map).copy()

    if TARGET_COL not in out.columns:
        if not is_test:
            raise ValueError(
                f"Training file is missing the '{TARGET_COL}' column, "
                f"available columns: {list(out.columns)}"
            )
        out[TARGET_COL] = np.nan

    return out


def load_and_adapt(path: str | Path, is_test: bool) -> pd.DataFrame:
    return rename_to_gaap_schema(load_challenge_hdf5(path), is_test=is_test)


# ---------------------------------------------------------------------------
# Inference + qp output
# ---------------------------------------------------------------------------
Z_GRID = np.linspace(0.0, 6.0, 301, dtype=np.float32)

DEFAULT_FINETUNE_EPOCHS = 60
DEFAULT_FINETUNE_LR = 1e-4
DEFAULT_FINETUNE_BATCH = 256
DEFAULT_FINETUNE_PATIENCE = 15


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gaussian_grid_pdf(z_pred: np.ndarray, sigma_z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """Evaluate N(z_pred, sigma_z) on a shared grid and renormalize per-object."""
    sigma = np.clip(sigma_z, 1e-3, None).astype(np.float64)
    mu = z_pred.astype(np.float64)
    diff = (z_grid[None, :] - mu[:, None]) / sigma[:, None]
    pdf = np.exp(-0.5 * diff * diff) / (np.sqrt(2.0 * np.pi) * sigma[:, None])
    area = np.trapezoid(pdf, z_grid, axis=1)
    area = np.where(area > 0, area, 1.0)
    return (pdf / area[:, None]).astype(np.float32)


def _write_qp_output(
    object_ids: np.ndarray, z_pred: np.ndarray, sigma_z: np.ndarray, output_file: str | Path,
) -> None:
    densities = _gaussian_grid_pdf(z_pred, sigma_z, Z_GRID)
    ensemble = qp.interp.create_ensemble(Z_GRID, densities)
    ensemble.set_ancil({
        "object_id": np.asarray(object_ids),
        "zmode": np.asarray(z_pred, dtype=np.float32),
    })
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.write_to(str(output_path))


def _infer(model: PhotozMLPVAE, X: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        _, z_pred_t, sigma_z_t = model.predict_z(torch.from_numpy(X).to(device), n_samples=1)
    return z_pred_t.cpu().numpy(), sigma_z_t.cpu().numpy()


def _run_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    """Subtask 2: inference with a pre-trained checkpoint.

    `model_file` is populated by setup_submit_area with our combo-specific
    FINE-TUNED checkpoint (precomputed once, see PLAN.md), not the raw
    zero-shot DP1 baseline -- fine-tuning gives a large accuracy
    improvement (sigma_NMAD 0.089->0.030, outlier rate 17%->0.7% on
    cardinal/1yr), so shipping the fine-tuned model is the better
    "pretrained model" to submit here, not a weaker sanity-check baseline.
    """
    device = _device()
    model, scaler, col_medians = PhotozMLPVAE.load(
        model_file, SPECULATOR_DIR, FILTER_DIR, device=device
    )

    test_df = load_and_adapt(test_file, is_test=True)
    X, *_ = build_features(
        test_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False,
    )
    z_pred, sigma_z = _infer(model, X, device)
    _write_qp_output(test_df["object_id"].to_numpy(), z_pred, sigma_z, output_file)


def _run_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    *,
    epochs: int = DEFAULT_FINETUNE_EPOCHS,
    lr: float = DEFAULT_FINETUNE_LR,
    batch_size: int = DEFAULT_FINETUNE_BATCH,
    patience: int = DEFAULT_FINETUNE_PATIENCE,
    val_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """Subtask 3: fine-tune the baseline checkpoint on the framework's own
    training file, live during this call, then infer on the test file.

    Warm-starts from BASELINE_CKPT (same partial-state_dict pattern as
    PhotozMLPVAE.load) rather than training from scratch, since the
    baseline was itself warm-started from a synthetic SED pretrain and a
    from-scratch fit is unlikely to beat that on a single task-set's
    training split. Unlike subtask 2's precomputed fine-tune, this always
    trains fresh -- that is subtask 3's whole contract.
    """
    device = _device()

    train_df = load_and_adapt(train_file, is_test=False)
    test_df = load_and_adapt(test_file, is_test=True)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_df))
    n_val = max(1, int(val_frac * len(train_df)))
    val_df = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    tr_df = train_df.iloc[idx[n_val:]].reset_index(drop=True)

    X_tr, mags_tr, errs_tr, mask_tr, z_tr, scaler, col_medians = build_features(
        tr_df, use_colors=True, use_euclid=False, use_gaap=True, fit=True,
    )
    X_val, mags_val, errs_val, mask_val, z_val, _, _ = build_features(
        val_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False,
    )

    model, _, _ = PhotozMLPVAE.load(BASELINE_CKPT, SPECULATOR_DIR, FILTER_DIR, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _batches(n, bs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            yield order[start:start + bs]

    x_tr_t = torch.from_numpy(X_tr).to(device)
    z_tr_t = torch.from_numpy(z_tr).to(device)
    x_val_t = torch.from_numpy(X_val).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_idx in _batches(len(X_tr), batch_size):
            xb = x_tr_t[batch_idx]
            zb = z_tr_t[batch_idx]
            z_pred_b, log_sigma_z_b, _, _ = model.encoder(xb)
            sigma_b = torch.exp(log_sigma_z_b).clamp(min=1e-3)
            nll = (
                0.5 * ((zb - z_pred_b) / sigma_b) ** 2 + torch.log(sigma_b)
            ).mean()
            opt.zero_grad(set_to_none=True)
            nll.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            z_pred_val, _, _, _ = model.encoder(x_val_t)
            val_loss = torch.mean((z_pred_val.squeeze(-1) - z_val_t) ** 2).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    X_test, *_ = build_features(
        test_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False,
    )
    z_pred, sigma_z = _infer(model, X_test, device)
    _write_qp_output(test_df["object_id"].to_numpy(), z_pred, sigma_z, output_file)


def run_taskset_1_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)


def run_taskset_2_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)


def run_taskset_2_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area() -> int:
    """
    Populate SUBMIT_DIR for Task Sets 1 & 2 without a SUBMISSION_URL tarball.

    This submission's model/code dependencies live in two separate GitHub
    repos (cloned on demand by ensure_model_deps), not a single hosted
    tarball -- so instead of the template's download-and-extract, this
    generates what SUBMIT_DIR needs directly: a model_file per
    taskset/sim/scenario combo, populated from the precomputed fine-tuned
    checkpoint for that exact combo (see _run_estimation_only), and each
    combo's subtask-1 "premade" qp file (taken as a fixed snapshot from
    that same fine-tuned model -- the live subtask 2/3 runs below
    regenerate their own outputs fresh either way).
    """
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"), exist_ok=True)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"), exist_ok=True)

    for taskset in TASKSETS:
        estimation_only = (
            run_taskset_1_estimation_only if taskset == 1 else run_taskset_2_estimation_only
        )
        for sim in SIMS:
            for scenario in SCENARIOS:
                model_file = os.path.join(
                    SUBMIT_DIR,
                    f"pz_challenge_taskset_{taskset}_{sim}_pz_model_{scenario}.pkl",
                )
                if not os.path.exists(model_file):
                    finetuned_src = Path(FINETUNED_DIR) / f"taskset{taskset}_{sim}_{scenario}.pt"
                    Path(model_file).write_bytes(finetuned_src.read_bytes())

                estimate_file = os.path.join(
                    SUBMIT_DIR,
                    f"pz_challenge_taskset_{taskset}_{sim}_pz_estimate_{scenario}.hdf5",
                )
                if not os.path.exists(estimate_file):
                    test_file = os.path.join(
                        PUBLIC_AREA,
                        f"pz_challenge_taskset_{taskset}_{sim}_test_{scenario}.hdf5",
                    )
                    estimation_only(model_file, test_file, estimate_file)

    return 0


def test_mlpvae_speculator_v2_taskset_1(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Validate the mlpvae_speculator_v2 submission for Task Set 1.

    Runs all three subtasks: the premade snapshot (subtask 1), estimation
    from our precomputed fine-tuned checkpoint (subtask 2), and a fresh
    warm-started fine-tune on this combo's own training file (subtask 3).
    """
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_1(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_1_estimation_only,
        run_taskset_1_training_and_estimation,
    )


def test_mlpvae_speculator_v2_taskset_2(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    """
    Validate the mlpvae_speculator_v2 submission for Task Set 2.

    Same model/pipeline as Task Set 1 -- Task Set 2's training files apply
    spectroscopic-selection emulation (median i-mag ~22.6) while test files
    go deeper (~i<25.4, median ~24.3), a training/test distribution
    mismatch Task Set 1 doesn't have. No code path differs; subtask 3's
    fine-tune just has a harder generalization problem on this task set.
    """
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_2(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_2_estimation_only,
        run_taskset_2_training_and_estimation,
    )
