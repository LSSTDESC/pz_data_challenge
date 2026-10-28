
from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import qp
import torch

from pz_data_challenge import submit_utils
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

SUBMISSION_NAME: str = "mlpvae_mdn_dp2_lbg_v17"
SUBMISSION_URL: str = (
    "https://github.com/Klinjin/photoz_mlpvae/raw/refs/heads/main/"
    "pz_challenge_submission/mlpvae_mdn_dp2_lbg_v17_4tasks.tgz"
)

SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
TASKSETS = [1, 2, 3, 4]

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

def ensure_model_deps() -> tuple[str, str, str]:
    t0 = time.time()
    photoz_mlpvae_dir = DEPS_DIR / "photoz_mlpvae"
    speculator_repo_dir = DEPS_DIR / "speculator"

    _run(["git", "lfs", "install", "--skip-repo"])

    _clone_if_missing(PHOTOZ_MLPVAE_URL, photoz_mlpvae_dir)
    print(f"[deps] photoz_mlpvae clone done  elapsed={time.time() - t0:.1f}s", flush=True)
    _clone_if_missing(SPECULATOR_URL, speculator_repo_dir)
    print(f"[deps] speculator clone done  elapsed={time.time() - t0:.1f}s", flush=True)

    lfs_marker = speculator_repo_dir / "trained" / "Inoue_IGM" / "30_100" / "pca_basis.npz"
    if not lfs_marker.exists() or lfs_marker.stat().st_size < _LFS_SMUDGED_MIN_BYTES:
        _run(["git", "lfs", "pull"], cwd=str(speculator_repo_dir))
        print(f"[deps] speculator lfs pull done  elapsed={time.time() - t0:.1f}s", flush=True)

    for p in (str(DEPS_DIR), str(photoz_mlpvae_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    baseline_ckpt = (
        photoz_mlpvae_dir / "trained"
        / "mlpvae_mdn_perpop_dp2_lbg_v17_sever_warmv10_fixedgrad_lsst_gaap1p0"
        / "best.pt"
    )
    speculator_dir = speculator_repo_dir / "trained" / "Inoue_IGM"
    filter_dir = photoz_mlpvae_dir / "obs_catalog" / "filters"

    for p in (baseline_ckpt, speculator_dir, filter_dir):
        if not p.exists():
            raise FileNotFoundError(f"Expected submission dependency missing: {p}")

    return str(baseline_ckpt), str(speculator_dir), str(filter_dir)

BASELINE_CKPT, SPECULATOR_DIR, FILTER_DIR = ensure_model_deps()

from obs_catalog.dataloader import build_features, compute_lbg_mask, compute_z_weights  # noqa: E402
from photoz_mlpvae.model.photoz_mlpvae_mdn_perpop import PhotozMLPVAE  # noqa: E402

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

Z_GRID = np.linspace(0.0, 6.0, 301, dtype=np.float32)

DEFAULT_FINETUNE_EPOCHS = 10
DEFAULT_FINETUNE_LR = 1e-3
DEFAULT_FINETUNE_BATCH = 256
DEFAULT_FINETUNE_PATIENCE = 4
FINETUNE_LAM_Z = 20.0
FINETUNE_LAM_R = 0.1
FINETUNE_LAM_PRIOR = 0.0
FINETUNE_LAM_AUX = 0.01
FINETUNE_SIGMA_FLOOR = 0.3
FINETUNE_WARMUP_EPOCHS = 0
FINETUNE_BETA_MAX = 0.1
FINETUNE_BETA_EPOCHS = 50
FINETUNE_RECON_RAMP = 50
FINETUNE_Z_WEIGHT_CAP = 10.0
FINETUNE_ROUTER_NOISE_INIT = 6.0
FINETUNE_ROUTER_NOISE_ANNEAL_EPOCHS = 200
FINETUNE_SEVER_P = 0.02
FINETUNE_SEVER_MAX_REMOVED_FRAC = 0.15
FINETUNE_SEVER_BATCH = 512


def _sever_outlier_scores(model, x, z, zw, lbg, lam_z):
    model.eval()
    grads = []
    activation = {}

    def _hook(module, inp, out):
        activation["a"] = out

    handle = model.encoder.z_head[1].register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            for start in range(0, len(x), FINETUNE_SEVER_BATCH):
                end = min(start + FINETUNE_SEVER_BATCH, len(x))
                xb, zb, zwb, lbgb = x[start:end], z[start:end], zw[start:end], lbg[start:end]
                model.zero_grad(set_to_none=True)
                losses = model.loss(
                    xb, zb, lam_z=lam_z, lam_r=0.0, beta=0.0, use_nll_z=True,
                    lam_aux=0.0, z_weights=zwb, is_lbg=lbgb, is_interloper=lbgb,
                    return_grad_probe=True,
                )
                (zwb * losses["per_gal"]).sum().backward()
                log_pi_t, mu_mix_t, log_var_mix_t = losses["grad_probe_tensors"]
                delta = torch.cat([log_pi_t.grad, mu_mix_t.grad, log_var_mix_t.grad], dim=1)
                a = activation["a"].detach()
                probe = torch.einsum("bo,bi->boi", delta, a).reshape(delta.shape[0], -1)
                grads.append(probe.detach())
    finally:
        handle.remove()
    model.zero_grad(set_to_none=True)
    G = torch.cat(grads, dim=0)
    Gc = G - G.mean(dim=0, keepdim=True)
    C = Gc.T.double() @ Gc.double()
    _, eigvecs = torch.linalg.eigh(C)
    v = eigvecs[:, -1]
    tau = (Gc.double() @ v) ** 2
    return tau.cpu().numpy()

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _lbg_tensor(df: pd.DataFrame, use_true_z: bool, device: str) -> torch.Tensor:
    mask = compute_lbg_mask(df, mag_type="gaap1p0", use_true_z=use_true_z)
    return torch.from_numpy(mask).to(device)

def _gaussian_grid_pdf(z_pred: np.ndarray, sigma_z: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
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

def _infer(
    model: PhotozMLPVAE, X: np.ndarray, lbg_mask: torch.Tensor, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        _, z_pred_t, sigma_z_t = model.predict_z(
            torch.from_numpy(X).to(device), n_samples=1,
            is_lbg=lbg_mask, is_interloper=lbg_mask,
        )
    return z_pred_t.cpu().numpy(), sigma_z_t.cpu().numpy()

def _run_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    t0 = time.time()
    device = _device()
    model, scaler, col_medians = PhotozMLPVAE.load(
        model_file, SPECULATOR_DIR, FILTER_DIR, device=device
    )

    test_df = load_and_adapt(test_file, is_test=True)
    X, *_ = build_features(
        test_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False, standardize=False,
    )
    lbg_mask = _lbg_tensor(test_df, use_true_z=False, device=device)
    z_pred, sigma_z = _infer(model, X, lbg_mask, device)
    _write_qp_output(test_df["object_id"].to_numpy(), z_pred, sigma_z, output_file)
    print(f"[estimate] {model_file} -> {output_file}  elapsed={time.time() - t0:.1f}s", flush=True)

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
    save_model_to: str | Path | None = None,
) -> None:
    t_start = time.time()
    device = _device()
    print(f"[finetune] {train_file} -> {output_file}  device={device}  "
          f"epochs<={epochs} patience={patience}", flush=True)

    train_df = load_and_adapt(train_file, is_test=False)
    train_df = train_df[train_df[TARGET_COL].notna()].reset_index(drop=True)
    test_df = load_and_adapt(test_file, is_test=True)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(train_df))
    n_val = max(1, int(val_frac * len(train_df)))
    val_df = train_df.iloc[idx[:n_val]].reset_index(drop=True)
    tr_df = train_df.iloc[idx[n_val:]].reset_index(drop=True)

    X_tr, z_tr, scaler, col_medians = build_features(
        tr_df, use_colors=True, use_euclid=False, use_gaap=True,
        fit=True, standardize=False,
    )
    X_val, z_val, _, _ = build_features(
        val_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False, standardize=False,
    )

    # Train/val have real z: is_lbg is the oracle (z_spec-gated, confirmed)
    # mask; is_interloper is color-cut positive AND NOT oracle-confirmed
    # (a false positive) -- two distinct masks, not the same one twice.
    # Mirrors train_dp2_lbg_perpop.py's is_lbg_tr/is_interloper_tr exactly
    # (this checkpoint's realistic_lbg_train=False, interloper_component=True).
    lbg_tr_oracle = compute_lbg_mask(tr_df, mag_type="gaap1p0", use_true_z=True)
    lbg_tr_realistic = compute_lbg_mask(tr_df, mag_type="gaap1p0", use_true_z=False)
    lbg_tr = torch.from_numpy(lbg_tr_oracle).to(device)
    interloper_tr = torch.from_numpy(lbg_tr_realistic & ~lbg_tr_oracle).to(device)

    lbg_val_oracle = compute_lbg_mask(val_df, mag_type="gaap1p0", use_true_z=True)
    lbg_val_realistic = compute_lbg_mask(val_df, mag_type="gaap1p0", use_true_z=False)
    lbg_val = torch.from_numpy(lbg_val_oracle).to(device)
    interloper_val = torch.from_numpy(lbg_val_realistic & ~lbg_val_oracle).to(device)

    zw_tr = compute_z_weights(z_tr)
    zw_tr = np.clip(zw_tr, None, FINETUNE_Z_WEIGHT_CAP)
    zw_tr = (zw_tr / zw_tr.mean()).astype(np.float32)
    zw_val = np.ones_like(z_val, dtype=np.float32)

    print(f"[finetune] X_tr={X_tr.shape} X_val={X_val.shape} "
          f"prep_elapsed={time.time() - t_start:.1f}s", flush=True)

    model, _, _ = PhotozMLPVAE.load(BASELINE_CKPT, SPECULATOR_DIR, FILTER_DIR, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_tr_t = torch.from_numpy(X_tr).to(device)
    z_tr_t = torch.from_numpy(z_tr).to(device)
    zw_tr_t = torch.from_numpy(zw_tr).to(device)

    x_val_t = torch.from_numpy(X_val).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)
    zw_val_t = torch.from_numpy(zw_val).to(device)

    # One Sever round "before training starts" (sever_start_epoch=0 in this
    # checkpoint's own recipe), scored against the warm-started model exactly
    # as loaded. sever_protect_lbg=False in that recipe too, so every row is
    # eligible -- no protected subset to carve out.
    n_train_orig = len(x_tr_t)
    tau = _sever_outlier_scores(model, x_tr_t, z_tr_t, zw_tr_t, lbg_tr, FINETUNE_LAM_Z)
    budget = int(round(FINETUNE_SEVER_MAX_REMOVED_FRAC * n_train_orig))
    n_remove = min(max(1, int(round(FINETUNE_SEVER_P * n_train_orig))), budget)
    keep_mask = np.ones(n_train_orig, dtype=bool)
    keep_mask[np.argsort(tau)[-n_remove:]] = False
    print(f"[finetune] Sever round 0: removed {n_remove}/{n_train_orig} rows", flush=True)
    x_tr_t, z_tr_t, zw_tr_t, lbg_tr, interloper_tr = (
        x_tr_t[keep_mask], z_tr_t[keep_mask], zw_tr_t[keep_mask],
        lbg_tr[keep_mask], interloper_tr[keep_mask],
    )

    def _batches(n, bs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            yield order[start:start + bs]

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        model.encoder.router_noise_std = FINETUNE_ROUTER_NOISE_INIT * max(
            0.0, 1.0 - (epoch - 1) / max(FINETUNE_ROUTER_NOISE_ANNEAL_EPOCHS - 1, 1)
        )
        if epoch <= FINETUNE_RECON_RAMP:
            lam_r_eff = FINETUNE_LAM_R * epoch / FINETUNE_RECON_RAMP
        else:
            lam_r_eff = FINETUNE_LAM_R
        beta = min(FINETUNE_BETA_MAX, FINETUNE_BETA_MAX * epoch / max(FINETUNE_BETA_EPOCHS - 1, 1))

        model.train()
        for batch_idx in _batches(len(x_tr_t), batch_size):
            bi = torch.from_numpy(batch_idx).to(device)
            loss_dict = model.loss(
                x_tr_t[bi], z_tr_t[bi],
                lam_z=FINETUNE_LAM_Z, lam_r=lam_r_eff, beta=beta,
                lam_prior=FINETUNE_LAM_PRIOR, sigma_floor=FINETUNE_SIGMA_FLOOR,
                use_nll_z=True, lam_aux=FINETUNE_LAM_AUX, z_weights=zw_tr_t[bi],
                is_lbg=lbg_tr[bi], is_interloper=interloper_tr[bi],
            )
            opt.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = model.loss(
                x_val_t, z_val_t,
                lam_z=FINETUNE_LAM_Z, lam_r=lam_r_eff, beta=beta,
                lam_prior=FINETUNE_LAM_PRIOR, sigma_floor=FINETUNE_SIGMA_FLOOR,
                use_nll_z=True, lam_aux=FINETUNE_LAM_AUX, z_weights=zw_val_t,
                is_lbg=lbg_val, is_interloper=interloper_val,
            )["z_sup"].item()

        print(f"[finetune] epoch {epoch}  lam_r={lam_r_eff:.4f}  beta={beta:.4f}  "
              f"val_z_sup={val_loss:.4f}  epoch_elapsed={time.time() - t_epoch:.1f}s  "
              f"total_elapsed={time.time() - t_start:.1f}s", flush=True)

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
    if save_model_to is not None:
        model.save(str(save_model_to), scaler, col_medians)

    X_test, *_ = build_features(
        test_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False, standardize=False,
    )
    lbg_test = _lbg_tensor(test_df, use_true_z=False, device=device)
    z_pred, sigma_z = _infer(model, X_test, lbg_test, device)
    _write_qp_output(test_df["object_id"].to_numpy(), z_pred, sigma_z, output_file)
    print(f"[finetune] done  total_elapsed={time.time() - t_start:.1f}s", flush=True)

def run_taskset_1_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)

def run_taskset_2_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)

def run_taskset_3_estimation_only(
    model_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_estimation_only(model_file, test_file, output_file)

def run_taskset_4_estimation_only(
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

def run_taskset_3_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)

def run_taskset_4_training_and_estimation(
    train_file: str | Path, test_file: str | Path, output_file: str | Path,
) -> None:
    _run_training_and_estimation(train_file, test_file, output_file)

@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area() -> int:
    t0 = time.time()
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_2"), exist_ok=True)
    os.makedirs(os.path.join(SUBMIT_DIR, "outputs_3"), exist_ok=True)

    marker = os.path.join(
        SUBMIT_DIR, "pz_challenge_taskset_1_cardinal_pz_estimate_1yr.hdf5"
    )
    if SUBMISSION_URL and not os.path.exists(marker):
        submit_utils.download_and_extract_tar(SUBMISSION_URL, SUBMIT_DIR)
        print(f"[submit_area] tarball downloaded+extracted  elapsed={time.time() - t0:.1f}s",
              flush=True)

    ESTIMATION_ONLY_BY_TASKSET = {
        1: run_taskset_1_estimation_only,
        2: run_taskset_2_estimation_only,
        3: run_taskset_3_estimation_only,
        4: run_taskset_4_estimation_only,
    }

    for taskset in TASKSETS:
        estimation_only = ESTIMATION_ONLY_BY_TASKSET[taskset]
        for sim in SIMS:
            for scenario in SCENARIOS:
                model_file = os.path.join(
                    SUBMIT_DIR,
                    f"pz_challenge_taskset_{taskset}_{sim}_pz_model_{scenario}.pkl",
                )
                if not os.path.exists(model_file):
                    Path(model_file).write_bytes(Path(BASELINE_CKPT).read_bytes())

                estimate_file = os.path.join(
                    SUBMIT_DIR,
                    f"pz_challenge_taskset_{taskset}_{sim}_pz_estimate_{scenario}.hdf5",
                )
                if not os.path.exists(estimate_file):
                    print(f"[submit_area] generating fallback estimate for "
                          f"taskset{taskset}_{sim}_{scenario}  elapsed={time.time() - t0:.1f}s",
                          flush=True)
                    test_file = os.path.join(
                        PUBLIC_AREA,
                        f"pz_challenge_taskset_{taskset}_{sim}_test_{scenario}.hdf5",
                    )
                    estimation_only(model_file, test_file, estimate_file)

    print(f"[submit_area] done  elapsed={time.time() - t0:.1f}s", flush=True)
    return 0

def test_mlpvae_mdn_dp2_lbg_v17_taskset_1(
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

def test_mlpvae_mdn_dp2_lbg_v17_taskset_2(
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

def test_mlpvae_mdn_dp2_lbg_v17_taskset_3(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_3(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_3_estimation_only,
        run_taskset_3_training_and_estimation,
    )

def test_mlpvae_mdn_dp2_lbg_v17_taskset_4(
    setup_public_area: int,
    setup_submit_area: int,
) -> None:
    assert setup_public_area == 0
    assert setup_submit_area == 0

    run_taskset_4(
        PUBLIC_AREA,
        SUBMISSION_NAME,
        run_taskset_4_estimation_only,
        run_taskset_4_training_and_estimation,
    )
