"""AION-based photometric-redshift estimation for the LSST-DESC PZ data challenge.

Approach: "frozen AION embeddings + lightweight head" (the recipe from the
AION-1 paper, arXiv:2510.17960, for downstream tasks).

Why not zero-shot generative p(z)?  AION was trained on Legacy Survey / HSC /
DESI / Gaia, not on LSST.  There is no LSST tokenizer, so we approximate the
LSST grizy magnitudes with AION's HSC magnitude modalities (both are 5-band
grizy) to obtain embeddings, then train a small classifier that maps those
embeddings (plus the raw LSST u + Roman bands, which AION cannot ingest) to a
binned redshift distribution.  The trained head learns to correct the
LSST<->HSC filter mismatch against the challenge's true training redshifts.

The output is a `qp` ensemble with `ancil["object_id"]` and `ancil["zmode"]`,
which is exactly what `pz_data_challenge.submit_utils.check_pz_submission_file`
requires.

Heavy dependencies (torch, aion) are imported lazily so this module can be
imported without a GPU / model present (e.g. during test collection).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data / grid configuration
# ---------------------------------------------------------------------------

# LSST bands, in the order AION's HSC magnitude modalities expect (g,r,i,z,y).
# LSST also has a u band, which AION has no counterpart for -> fed to the head
# as a raw feature instead.
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
HSC_BANDS = ["g", "r", "i", "z", "y"]  # bands passed to AION
ROMAN_BANDS = ["Y", "J", "H"]  # present in the roman_rubin task sets only

MAG_COL = "mag_{band}_lsst"
MAG_ERR_COL = "mag_{band}_lsst_err"
ROMAN_MAG_COL = "mag_{band}_roman"
ROMAN_ERR_COL = "mag_{band}_roman_err"
OBJECT_ID_COL = "object_id"
REDSHIFT_COL = "redshift"  # truth column, present only in training files
# Task sets 3/4: COSMOS-selected training objects have no spectroscopic
# redshift; the challenge provides a many-band photometric redshift instead.
# Where both exist it agrees with spec-z to sigma_MAD~0.014 with <1% outliers,
# and the objects lacking spec-z are precisely the faint ones matching the
# test population, so we use it as a fallback training label.
MANYBAND_COL = "redshift_manyband"

# Redshift grid for the output PDFs.  The challenge sims live below z~3.
ZMAX = 3.0
NZ = 301
Z_GRID = np.linspace(0.0, ZMAX, NZ)

# Sentinel used for non-detections / NaNs before feeding the model.
NONDETECT_FILL = 30.0

DEFAULT_BATCH_SIZE = 512
DEFAULT_NUM_ENCODER_TOKENS = 8  # 5 HSC-mag scalar tokens fit comfortably


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_catalog(path: str | Path) -> dict[str, np.ndarray]:
    """Read an HDF5 challenge catalog into a dict of numpy arrays."""
    import tables_io

    data = tables_io.read(str(path))
    # tables_io may return a nested {group: {col: arr}} for grouped files.
    if data and all(isinstance(v, dict) for v in data.values()):
        # flatten a single top-level group
        (only_group,) = data.values() if len(data) == 1 else (_merge_groups(data),)
        data = only_group
    return {k: np.asarray(v) for k, v in data.items()}


def _merge_groups(grouped: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for group in grouped.values():
        merged.update(group)
    return merged


def _subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def _get_mag(data: dict[str, np.ndarray], col: str) -> np.ndarray | None:
    if col not in data:
        return None
    arr = np.asarray(data[col], dtype="float32")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_extra_features(data: dict[str, np.ndarray]) -> np.ndarray:
    """Raw photometric features handed directly to the head.

    Includes all LSST mags (incl. u), their errors, colours, and any Roman
    bands present.  AION cannot ingest u/Roman, so this preserves that info.
    """
    cols: list[np.ndarray] = []

    lsst_mags: dict[str, np.ndarray] = {}
    for band in LSST_BANDS:
        m = _get_mag(data, MAG_COL.format(band=band))
        if m is not None:
            lsst_mags[band] = m
            cols.append(np.nan_to_num(m, nan=NONDETECT_FILL))
        e = _get_mag(data, MAG_ERR_COL.format(band=band))
        if e is not None:
            cols.append(np.nan_to_num(e, nan=1.0))

    # Adjacent LSST colours (u-g, g-r, r-i, i-z, z-y) — strong z indicators.
    ordered = [b for b in LSST_BANDS if b in lsst_mags]
    for b0, b1 in zip(ordered[:-1], ordered[1:]):
        colour = lsst_mags[b0] - lsst_mags[b1]
        cols.append(np.nan_to_num(colour, nan=0.0))

    # Roman bands, if present.
    for band in ROMAN_BANDS:
        m = _get_mag(data, ROMAN_MAG_COL.format(band=band))
        if m is not None:
            cols.append(np.nan_to_num(m, nan=NONDETECT_FILL))

    if not cols:
        n = len(next(iter(data.values())))
        return np.zeros((n, 0), dtype="float32")
    return np.stack(cols, axis=1).astype("float32")


def _hsc_mag_modalities(data: dict[str, np.ndarray], device: str) -> list[Any]:
    """Wrap LSST g,r,i,z,y magnitudes as AION HSC magnitude modalities."""
    import torch
    from aion.modalities import HSCMagG, HSCMagR, HSCMagI, HSCMagZ, HSCMagY

    cls_by_band = {
        "g": HSCMagG, "r": HSCMagR, "i": HSCMagI, "z": HSCMagZ, "y": HSCMagY,
    }
    mods = []
    for band in HSC_BANDS:
        mag = _get_mag(data, MAG_COL.format(band=band))
        if mag is None:
            continue
        mag = np.nan_to_num(mag, nan=NONDETECT_FILL)
        value = torch.tensor(mag, dtype=torch.float32, device=device)
        mods.append(cls_by_band[band](value=value))
    return mods


# ---------------------------------------------------------------------------
# AION embedding extraction
# ---------------------------------------------------------------------------

def load_aion(model_name: str = "polymathic-ai/aion-base", device: str | None = None):
    """Load the pretrained AION model and its codec manager."""
    import torch
    from aion import AION
    from aion.codecs import CodecManager

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = AION.from_pretrained(model_name).to(device).eval()
    codec_manager = CodecManager(device=device)
    return model, codec_manager, device


def extract_embeddings(
    model,
    codec_manager,
    data: dict[str, np.ndarray],
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_encoder_tokens: int = DEFAULT_NUM_ENCODER_TOKENS,
) -> np.ndarray:
    """Mean-pooled AION embeddings from the (HSC-mapped) LSST photometry."""
    import torch

    n = len(data[OBJECT_ID_COL]) if OBJECT_ID_COL in data else len(build_extra_features(data))
    out: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {k: v[start:end] for k, v in data.items()}
        mods = _hsc_mag_modalities(batch, device)
        tokens = codec_manager.encode(*mods)
        emb = model.encode(tokens, num_encoder_tokens=num_encoder_tokens)
        out.append(emb.mean(axis=1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def build_design_matrix(model, codec_manager, data, device, **kw) -> np.ndarray:
    """Full feature matrix: AION embeddings concatenated with raw photometry."""
    emb = extract_embeddings(model, codec_manager, data, device, **kw)
    extra = build_extra_features(data)
    return np.concatenate([emb, extra], axis=1).astype("float32")


# ---------------------------------------------------------------------------
# Head: binned-redshift classifier -> p(z)
# ---------------------------------------------------------------------------

def _z_to_bin(z: np.ndarray) -> np.ndarray:
    idx = np.digitize(z, Z_GRID) - 1
    return np.clip(idx, 0, NZ - 1)


def redshift_weights(
    z: np.ndarray, alpha: float = 0.5, nbins: int = 30, zmax: float = ZMAX
) -> np.ndarray:
    """Per-object weights that upweight sparsely-populated redshift bins.

    w propto (1 / bin_count)^alpha, normalised to mean 1.  alpha=0 -> uniform
    (no reweighting); alpha=1 -> fully balanced across redshift.  Counters the
    high-z collapse caused by few training galaxies at large z.
    """
    edges = np.linspace(0.0, zmax, nbins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins).astype(float)
    counts[counts == 0] = 1.0
    w = (1.0 / counts[idx]) ** alpha
    return w * (len(z) / w.sum())


def feature_shift_weights(
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    cols: tuple[str, ...] = ("mag_i_lsst", "mag_r_lsst"),
    nbins: int = 30,
    clip: float = 25.0,
) -> np.ndarray:
    """Covariate-shift importance weights w(x) ~ p_test(x) / p_train(x).

    For Task Set 2 the training sample (bright, spec-selected) is not
    representative of the faint test set.  Reweighting training objects toward
    the test *feature* (magnitude) distribution — which needs no test labels —
    upweights the rare faint training galaxies that resemble the test set.
    Density ratio is estimated on a low-dim magnitude histogram.
    """
    cols = tuple(c for c in cols if c in train and c in test)
    if not cols:
        return np.ones(len(train[OBJECT_ID_COL]))
    tr = np.stack([np.asarray(train[c], float) for c in cols], axis=1)
    te = np.stack([np.asarray(test[c], float) for c in cols], axis=1)
    edges = []
    for j in range(tr.shape[1]):
        both = np.concatenate([tr[:, j], te[:, j]])
        both = both[np.isfinite(both)]
        lo, hi = np.percentile(both, [1.0, 99.0])
        edges.append(np.linspace(lo, hi, nbins + 1))

    def binned(a):
        idx = np.zeros(len(a), dtype=int)
        mult = 1
        for j in range(a.shape[1]):
            b = np.clip(np.digitize(a[:, j], edges[j]) - 1, 0, nbins - 1)
            idx += b * mult
            mult *= nbins
        return idx

    tr_idx, te_idx = binned(tr), binned(te)
    ncell = nbins ** tr.shape[1]
    tr_hist = np.bincount(tr_idx, minlength=ncell).astype(float)
    te_hist = np.bincount(te_idx, minlength=ncell).astype(float)
    tr_hist /= tr_hist.sum()
    te_hist /= te_hist.sum()
    ratio = np.where(tr_hist > 0, te_hist / (tr_hist + 1e-12), 0.0)
    w = ratio[tr_idx]
    w = np.clip(w, 0.0, clip)
    bad = ~np.isfinite(tr).all(axis=1)
    w[bad] = w[~bad].mean() if (~bad).any() else 1.0
    s = w.sum()
    return w * (len(w) / s) if s > 0 else np.ones(len(w))


def train_head(
    x: np.ndarray,
    z_true: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a scaler + MLP classifier over redshift bins.

    MLPClassifier has no sample_weight, so weighting is applied by resampling the
    training rows with replacement in proportion to `sample_weight`.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    labels = _z_to_bin(z_true)

    if sample_weight is not None:
        p = sample_weight / sample_weight.sum()
        sel = np.random.default_rng(seed).choice(len(xs), size=len(xs), replace=True, p=p)
        xs_fit, labels_fit = xs[sel], labels[sel]
    else:
        xs_fit, labels_fit = xs, labels

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=8,
    )
    clf.fit(xs_fit, labels_fit)
    return {"scaler": scaler, "clf": clf, "z_grid": Z_GRID, "classes_": clf.classes_}


def predict_pz(head: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Return an [N, NZ] array of p(z) on the fixed Z_GRID."""
    scaler = head["scaler"]
    clf = head["clf"]
    classes = head["classes_"]
    proba = clf.predict_proba(scaler.transform(x))  # [N, n_seen_classes]

    pz = np.zeros((x.shape[0], NZ), dtype="float64")
    pz[:, classes] = proba
    # Normalise as a density over the grid.
    dz = Z_GRID[1] - Z_GRID[0]
    norm = pz.sum(axis=1, keepdims=True) * dz
    norm[norm == 0] = 1.0
    return pz / norm


# ---------------------------------------------------------------------------
# PDF calibration
# ---------------------------------------------------------------------------
#
# The binned classifier tends to produce over-confident (too narrow) PDFs, which
# hurts the PIT / CDELoss metrics even when the point estimate is good.  Two
# post-hoc fixes, both fit on a held-out calibration split:
#   * temperature   — p(z)^(1/T) renormalised; T>1 broadens, T<1 sharpens.
#   * PIT recalibration — remap each CDF through R = empirical CDF of the
#     calibration PIT values, which makes PIT uniform by the probability
#     integral transform.

def pz_cdf(pz: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dz = grid[1] - grid[0]
    cdf = np.cumsum(pz, axis=1) * dz
    cdf /= cdf[:, -1:] + 1e-12
    return cdf


def compute_pit(pz: np.ndarray, grid: np.ndarray, z_true: np.ndarray) -> np.ndarray:
    cdf = pz_cdf(pz, grid)
    return np.array([np.interp(z_true[i], grid, cdf[i]) for i in range(len(z_true))])


def _renorm(pz: np.ndarray, grid: np.ndarray) -> np.ndarray:
    dz = grid[1] - grid[0]
    pz = np.clip(pz, 0.0, None)
    norm = pz.sum(axis=1, keepdims=True) * dz
    norm[norm == 0] = 1.0
    return pz / norm


def apply_temperature(pz: np.ndarray, grid: np.ndarray, temperature: float) -> np.ndarray:
    return _renorm(np.clip(pz, 1e-12, None) ** (1.0 / temperature), grid)


def fit_temperature(
    pz_calib: np.ndarray, grid: np.ndarray, z_true_calib: np.ndarray,
    temps: np.ndarray | None = None,
) -> float:
    """Pick the temperature that minimises PIT-KS on the calibration set."""
    if temps is None:
        temps = np.linspace(0.5, 5.0, 46)
    best_t, best_ks = 1.0, np.inf
    for t in temps:
        ks = pit_ks_stat(apply_temperature(pz_calib, grid, t), grid, z_true_calib)
        if ks < best_ks:
            best_ks, best_t = ks, float(t)
    return best_t


def fit_pit_recalibration(
    pz_calib: np.ndarray, grid: np.ndarray, z_true_calib: np.ndarray
) -> dict[str, np.ndarray]:
    pit = np.clip(compute_pit(pz_calib, grid, z_true_calib), 0.0, 1.0)
    sp = np.sort(pit)
    n = len(sp)
    xp = np.concatenate([[0.0], sp, [1.0]])
    fp = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
    xp, idx = np.unique(xp, return_index=True)
    fp = np.maximum.accumulate(fp[idx])
    return {"xp": xp, "fp": fp}


def apply_pit_recalibration(
    pz: np.ndarray, grid: np.ndarray, recal: dict[str, np.ndarray]
) -> np.ndarray:
    cdf = pz_cdf(pz, grid)
    new_cdf = np.interp(cdf, recal["xp"], recal["fp"])
    pdf = np.gradient(new_cdf, grid, axis=1)
    return _renorm(pdf, grid)


def pit_ks_stat(pz: np.ndarray, grid: np.ndarray, z_true: np.ndarray) -> float:
    """KS statistic of the PIT distribution vs Uniform(0,1). Lower = better."""
    pit = np.sort(compute_pit(pz, grid, z_true))
    n = len(pit)
    emp = np.arange(1, n + 1) / n
    return float(np.max(np.abs(pit - emp)))


# ---------------------------------------------------------------------------
# qp output
# ---------------------------------------------------------------------------

def write_qp(
    pz: np.ndarray,
    object_id: np.ndarray,
    output_file: str | Path,
    z_grid: np.ndarray | None = None,
) -> None:
    """Write an interpolated qp ensemble with object_id + zmode ancil."""
    import qp

    grid = Z_GRID if z_grid is None else np.asarray(z_grid, dtype="float64")
    ens = qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": pz})
    zmode = grid[np.argmax(pz, axis=1)]
    ens.set_ancil({"object_id": np.asarray(object_id).astype(int), "zmode": zmode})
    ens.write_to(str(output_file))


# ---------------------------------------------------------------------------
# Model persistence (for the estimation-only subtask)
# ---------------------------------------------------------------------------

def save_head(head: dict[str, Any], model_file: str | Path) -> None:
    import joblib

    joblib.dump(head, str(model_file))


def load_head(model_file: str | Path) -> dict[str, Any]:
    import joblib

    return joblib.load(str(model_file))


# ---------------------------------------------------------------------------
# Zero-shot baseline: AION's native generative p(z) via target_modality=Z
# ---------------------------------------------------------------------------
#
# AION was pretrained with redshift as one of its modalities, so it can emit a
# p(z) directly from photometry with no training on our part.  We feed the LSST
# grizy magnitudes as HSC-magnitude modalities and ask the model for the Z
# modality; softmax over the decoder logits is a distribution over the redshift
# codebook.  Each logit index maps to a real redshift via the codec's
# quantizer, so we read the grid straight from the model rather than guessing.
#
# Caveat: because AION never saw LSST filters (and no u band), this baseline
# carries a systematic offset the trained head is meant to remove.  It is here
# for comparison, not as the primary submission.

def redshift_bin_centers(codec_manager, device: str) -> np.ndarray:
    """Read the redshift value of every Z codebook bin from the codec."""
    import torch
    from aion.modalities import Z

    codec = codec_manager._load_codec(Z).to(device)
    with torch.no_grad():
        centers = codec.quantizer.codebook.detach().float().cpu().numpy()
    return centers.ravel()


def zero_shot_pz(
    model,
    codec_manager,
    data: dict[str, np.ndarray],
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Native AION generative p(z). Returns (pz [N, nbins], z_centers [nbins])."""
    import torch
    from aion.modalities import Z

    z_centers = redshift_bin_centers(codec_manager, device)
    n = len(data[OBJECT_ID_COL])
    out: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {k: v[start:end] for k, v in data.items()}
        mods = _hsc_mag_modalities(batch, device)
        with torch.no_grad():
            logits = model(codec_manager.encode(*mods), target_modality=Z)["tok_z"]
            probs = torch.softmax(logits.squeeze(1), dim=-1)  # [B, nbins]
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out, axis=0), z_centers


def _resample_to_grid(
    pz: np.ndarray, z_centers: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Resample per-bin probability mass onto a uniform density grid."""
    order = np.argsort(z_centers)
    zc = z_centers[order]
    pz = pz[:, order]
    # per-bin mass -> density (guard against duplicate/zero-width bins)
    widths = np.gradient(zc)
    widths[widths <= 0] = np.median(widths[widths > 0]) if np.any(widths > 0) else 1.0
    density = pz / widths
    out = np.empty((pz.shape[0], len(grid)), dtype="float64")
    for i in range(pz.shape[0]):
        out[i] = np.interp(grid, zc, density[i], left=0.0, right=0.0)
    dz = grid[1] - grid[0]
    norm = out.sum(axis=1, keepdims=True) * dz
    norm[norm == 0] = 1.0
    return out / norm


def zero_shot_estimate(
    test_file: str | Path,
    output_file: str | Path,
    model_name: str = "polymathic-ai/aion-base",
    device: str | None = None,
    resample: bool = True,
) -> None:
    """Zero-shot baseline: write AION's native generative p(z) for `test_file`.

    If `resample`, the output is placed on the fixed Z_GRID (matching the head
    approach and the metrics grid); otherwise it uses AION's native bin centers.
    """
    model, codec_manager, device = load_aion(model_name, device)
    test = load_catalog(test_file)
    pz, z_centers = zero_shot_pz(model, codec_manager, test, device)

    if resample:
        pz_out = _resample_to_grid(pz, z_centers, Z_GRID)
        write_qp(pz_out, test[OBJECT_ID_COL], output_file, z_grid=Z_GRID)
    else:
        order = np.argsort(z_centers)
        write_qp(pz[:, order], test[OBJECT_ID_COL], output_file, z_grid=z_centers[order])


# ---------------------------------------------------------------------------
# Top-level entry points used by the submission test functions
# ---------------------------------------------------------------------------

def _predict_calibrated(head: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """predict_pz, applying the stored PIT recalibration if present."""
    pz = predict_pz(head, x)
    recal = head.get("recal")
    if recal is not None:
        pz = apply_pit_recalibration(pz, Z_GRID, recal)
    return pz


def train_and_estimate(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    model_name: str = "polymathic-ai/aion-base",
    device: str | None = None,
    save_model_to: str | Path | None = None,
    calibrate: bool = True,
    calib_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """Train the head on `train_file`, estimate p(z) for `test_file`.

    When `calibrate`, a `calib_frac` slice of the training set is held out to fit
    PIT recalibration (improves PIT / CDELoss); the fitted map is stored in the
    saved model so the estimation-only path reuses it.
    """
    model, codec_manager, device = load_aion(model_name, device)

    train = load_catalog(train_file)
    z_true = np.asarray(train[REDSHIFT_COL], dtype="float64")
    if MANYBAND_COL in train:
        z_many = np.asarray(train[MANYBAND_COL], dtype="float64")
        fill = ~np.isfinite(z_true) & np.isfinite(z_many)
        z_true[fill] = z_many[fill]
    good = np.isfinite(z_true)
    if not good.all():
        train = {k: v[good] for k, v in train.items()}
        z_true = z_true[good]

    n = len(z_true)
    idx = np.random.default_rng(seed).permutation(n)
    if calibrate and n > 100:
        n_cal = max(1, int(calib_frac * n))
        fit_idx, cal_idx = idx[n_cal:], idx[:n_cal]
    else:
        fit_idx, cal_idx = idx, np.array([], dtype=int)

    x_train = build_design_matrix(model, codec_manager, _subset(train, fit_idx), device)
    head = train_head(x_train, z_true[fit_idx])

    if len(cal_idx) > 0:
        x_cal = build_design_matrix(model, codec_manager, _subset(train, cal_idx), device)
        pz_cal = predict_pz(head, x_cal)
        head["recal"] = fit_pit_recalibration(pz_cal, Z_GRID, z_true[cal_idx])

    if save_model_to is not None:
        save_head(head, save_model_to)

    test = load_catalog(test_file)
    x_test = build_design_matrix(model, codec_manager, test, device)
    pz = _predict_calibrated(head, x_test)
    write_qp(pz, test[OBJECT_ID_COL], output_file)


def estimate_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    model_name: str = "polymathic-ai/aion-base",
    device: str | None = None,
) -> None:
    """Estimate p(z) for `test_file` using a pre-trained head in `model_file`."""
    model, codec_manager, device = load_aion(model_name, device)
    head = load_head(model_file)

    test = load_catalog(test_file)
    x_test = build_design_matrix(model, codec_manager, test, device)
    pz = _predict_calibrated(head, x_test)
    write_qp(pz, test[OBJECT_ID_COL], output_file)
