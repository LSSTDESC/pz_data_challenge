"""mlpvae_zsep_v4_speculator submission for the PZ Data Challenge (Task Sets 1 & 2).

Same design as mlpvae_speculator_v2 (see PLAN.md) -- self-contained on
purpose (a submission PR only adds this one file plus
requirements_mlpvae_zsep_v4_speculator.txt and the workflow yaml;
`submissions_src/` is local dev scratch, gitignored, never committed).

Model: mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20, the z-separated
architecture (photoz_mlpvae.model.photoz_mlpvae.PhotozMLPVAE, NOT the older
photoz_mlpvae_old.PhotozMLPVAE mlpvae_speculator_v2 uses) -- z gets its own
supervised latent + MLP head; the 15 SPS params keep a VAE latent
conditioned on z. Same two GitHub-hosted dependencies as mlpvae_speculator_v2
(github.com/Klinjin/photoz_mlpvae for vendored source + checkpoints,
github.com/Klinjin/speculator for the Speculator weights via Git LFS) --
same speculator_dir/filter_dir, same use_colors/use_euclid/use_gaap
settings, so no adapter or dependency-hosting changes were needed, only a
different model module/checkpoint path and a different subtask-3 fine-tune
loop (see below).
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

SUBMISSION_NAME: str = "mlpvae_zsep_v4_speculator"
SUBMISSION_URL: str = ""

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
TASKSETS = [1, 2]

# ---------------------------------------------------------------------------
# Model/code dependencies (see module docstring)
# ---------------------------------------------------------------------------
DEPS_DIR = Path(SUBMIT_DIR) / "_deps"
PHOTOZ_MLPVAE_URL = "https://github.com/Klinjin/photoz_mlpvae.git"
SPECULATOR_URL = "https://github.com/Klinjin/speculator.git"
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
    filter_dir, finetuned_dir)."""
    photoz_mlpvae_dir = DEPS_DIR / "photoz_mlpvae"
    speculator_repo_dir = DEPS_DIR / "speculator"

    _run(["git", "lfs", "install", "--skip-repo"])

    _clone_if_missing(PHOTOZ_MLPVAE_URL, photoz_mlpvae_dir)
    _clone_if_missing(SPECULATOR_URL, speculator_repo_dir)

    lfs_marker = speculator_repo_dir / "trained" / "Inoue_IGM" / "30_100" / "pca_basis.npz"
    if not lfs_marker.exists() or lfs_marker.stat().st_size < _LFS_SMUDGED_MIN_BYTES:
        _run(["git", "lfs", "pull"], cwd=str(speculator_repo_dir))

    for p in (str(DEPS_DIR), str(photoz_mlpvae_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    baseline_ckpt = (
        photoz_mlpvae_dir / "trained" / "mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20" / "best.pt"
    )
    speculator_dir = speculator_repo_dir / "trained" / "Inoue_IGM"
    filter_dir = photoz_mlpvae_dir / "obs_catalog" / "filters"
    finetuned_dir = photoz_mlpvae_dir / "trained" / "mlpvae_zsep_v4_speculator_finetuned"

    for p in (baseline_ckpt, speculator_dir, filter_dir, finetuned_dir):
        if not p.exists():
            raise FileNotFoundError(f"Expected submission dependency missing: {p}")

    return str(baseline_ckpt), str(speculator_dir), str(filter_dir), str(finetuned_dir)


BASELINE_CKPT, SPECULATOR_DIR, FILTER_DIR, FINETUNED_DIR = ensure_model_deps()

from obs_catalog.dataloader import build_features  # noqa: E402
from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE  # noqa: E402

# ---------------------------------------------------------------------------
# Adapter: challenge HDF5 schema -> the model's GAAP-named feature columns
# ---------------------------------------------------------------------------
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
TARGET_COL = "redshift"


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
# Same mix as this checkpoint's own config.yaml (lam_z=20.0, lam_r=0.1,
# sigma_floor=0.3); beta=0.0 (no KL term) -- a short warm-start fine-tune
# doesn't need the original run's beta annealing schedule.
FINETUNE_LAM_Z = 20.0
FINETUNE_LAM_R = 0.1
FINETUNE_SIGMA_FLOOR = 0.3


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
    """PhotozMLPVAE.predict_z returns (z_samples, z_mean, z_std) -- same
    3-tuple shape/semantics in both the old and new (z-separated) model
    classes, so this is unchanged from mlpvae_speculator_v2."""
    model.eval()
    with torch.no_grad():
        _, z_pred_t, sigma_z_t = model.predict_z(
            torch.from_numpy(X).to(device), n_samples=1
        )
    return z_pred_t.cpu().numpy(), sigma_z_t.cpu().numpy()


def _run_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    """Subtask 2: inference with a pre-trained checkpoint.

    `model_file` is populated by setup_submit_area with our combo-specific
    FINE-TUNED checkpoint (precomputed once), not the raw baseline --
    same rationale as mlpvae_speculator_v2 (see PLAN.md): fine-tuning is a
    large, consistent accuracy improvement, so ship the better model.
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

    Unlike mlpvae_speculator_v2's fine-tune loop (which called
    model.encoder(x) directly with a hand-rolled z-only NLL), this uses
    PhotozMLPVAE.loss() -- the z-separated model's encoder returns a
    5-tuple (z_pred, mu_z, log_var_z, mu_15, log_var_15), not the old
    4-tuple (z_pred, log_sigma_z, mu_15, log_var_15), so the old hand-rolled
    loss doesn't apply. .loss() is the model's own native training
    objective, and build_features already returns the mags/errs/mask it
    needs for free.
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
    mags_tr_t = torch.from_numpy(mags_tr).to(device)
    errs_tr_t = torch.from_numpy(errs_tr).to(device)
    mask_tr_t = torch.from_numpy(mask_tr).to(device)
    z_tr_t = torch.from_numpy(z_tr).to(device)

    x_val_t = torch.from_numpy(X_val).to(device)
    mags_val_t = torch.from_numpy(mags_val).to(device)
    errs_val_t = torch.from_numpy(errs_val).to(device)
    mask_val_t = torch.from_numpy(mask_val).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for batch_idx in _batches(len(X_tr), batch_size):
            loss_dict = model.loss(
                x_tr_t[batch_idx], mags_tr_t[batch_idx], errs_tr_t[batch_idx],
                mask_tr_t[batch_idx], z_tr_t[batch_idx],
                lam_z=FINETUNE_LAM_Z, lam_r=FINETUNE_LAM_R, beta=0.0,
                sigma_floor=FINETUNE_SIGMA_FLOOR, use_nll_z=True,
            )
            opt.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(
                x_val_t, mags_val_t, errs_val_t, mask_val_t, z_val_t,
                lam_z=FINETUNE_LAM_Z, lam_r=FINETUNE_LAM_R, beta=0.0,
                sigma_floor=FINETUNE_SIGMA_FLOOR, use_nll_z=True,
            )["z_sup"].item()

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
    """Populate SUBMIT_DIR for Task Sets 1 & 2 -- see mlpvae_speculator_v2's
    fixture (PLAN.md) for the full rationale, identical here."""
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


def test_mlpvae_zsep_v4_speculator_taskset_1(
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


def test_mlpvae_zsep_v4_speculator_taskset_2(
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
