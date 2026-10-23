import os
from pathlib import Path
import pytest

# Put needed import here
from typing import Any

import numpy as np
import qp
import tables_io

if not hasattr(np, "trapz"):  # numpy>=2: qp/PIT internals still call np.trapz
    np.trapz = np.trapezoid

# These are used by test scripts
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

from pz_data_challenge import submit_utils

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "fzb_dimmingtofaint"
SUBMISSION_URL: str = "https://github.com/alex-strange/pz_data_challenge_fzb_dimmingtofaint/releases/download/v1.0.0/fzb_dimmingtofaint_submission.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


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


# --- Self-contained FlexZBoost feature construction and estimation helpers,
# used by run_taskset_1_estimation_only / run_taskset_1_training_and_estimation
# below. These build the photometric feature matrix and run FlexZBoost
# training and prediction using only pip-installable packages that
# pz_data_challenge already requires, so the submission test runs in the
# challenge CI as a standalone file with no extra dependencies. --------------
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
ROMAN_BANDS = ["Y", "J", "H"]
NONDETECT_PLACEHOLDER = 99.0
ERR_PLACEHOLDER = 1.0
COLOR_CLIP = (-10.0, 10.0)
BREAK_SPANNING_COLORS = [("i", "Y"), ("z", "J"), ("r", "z"), ("z", "Y"), ("Y", "J")]

# FlexZBoost redshift grid: p(z) is evaluated from ZMIN to ZMAX on NZBINS points.
ZMIN, ZMAX, NZBINS = 0.0, 3.0, 301


def _lsst_mag_col(b: str) -> str:
    return f"mag_{b}_lsst"


def _lsst_err_col(b: str) -> str:
    return f"mag_{b}_lsst_err"


def _roman_mag_col(b: str) -> str:
    return f"mag_{b}_roman"


def _roman_err_col(b: str) -> str:
    return f"mag_{b}_roman_err"


def _mag_col_for_band(b: str) -> str:
    return _roman_mag_col(b) if b in ROMAN_BANDS else _lsst_mag_col(b)


def _read_table(hdf5_path: str) -> dict[str, np.ndarray]:
    raw = tables_io.read(hdf5_path, tType="numpyDict")
    return {k: np.asarray(v) for k, v in raw.items()}


def _build_features(
    hdf5_path: str,
    nrows: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Feature matrix from a challenge HDF5 file. Returns
    (feature_matrix, object_id, redshift_or_None)."""
    data = _read_table(hdf5_path)

    n_total = len(data["object_id"])
    idx = np.arange(n_total)
    if nrows is not None and nrows < n_total:
        if rng is None:
            rng = np.random.default_rng()
        idx = np.sort(rng.choice(n_total, size=int(nrows), replace=False))

    object_id = np.asarray(data["object_id"])[idx]
    redshift = np.asarray(data["redshift"])[idx] if "redshift" in data else None

    all_bands = [(_lsst_mag_col(b), _lsst_err_col(b)) for b in LSST_BANDS]
    all_bands += [(_roman_mag_col(b), _roman_err_col(b)) for b in ROMAN_BANDS]

    mags: list[np.ndarray] = []
    errs: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    for mag_col, err_col in all_bands:
        m = np.asarray(data[mag_col], dtype=float)[idx]
        e = np.asarray(data[err_col], dtype=float)[idx]
        detected = np.isfinite(m)
        mags.append(np.where(detected, m, NONDETECT_PLACEHOLDER))
        errs.append(np.where(np.isfinite(e), e, ERR_PLACEHOLDER))
        flags.append(detected.astype(float))

    colors: list[np.ndarray] = []

    def _color(mag_col_a: str, mag_col_b: str) -> np.ndarray:
        a = np.asarray(data[mag_col_a], dtype=float)[idx]
        b = np.asarray(data[mag_col_b], dtype=float)[idx]
        both = np.isfinite(a) & np.isfinite(b)
        c = np.where(both, a - b, 0.0)
        return np.clip(c, COLOR_CLIP[0], COLOR_CLIP[1])

    for i in range(len(LSST_BANDS) - 1):
        colors.append(_color(_lsst_mag_col(LSST_BANDS[i]), _lsst_mag_col(LSST_BANDS[i + 1])))
    for i in range(len(ROMAN_BANDS) - 1):
        colors.append(_color(_roman_mag_col(ROMAN_BANDS[i]), _roman_mag_col(ROMAN_BANDS[i + 1])))
    for ba, bb in BREAK_SPANNING_COLORS:
        colors.append(_color(_mag_col_for_band(ba), _mag_col_for_band(bb)))

    columns = mags + colors + errs + flags
    feature_matrix = np.vstack(columns).T.astype(float)
    feature_matrix = np.nan_to_num(
        feature_matrix, nan=0.0, posinf=COLOR_CLIP[1], neginf=COLOR_CLIP[0]
    )
    return feature_matrix, object_id, redshift


def _encode_for_flexzboost(
    feature_matrix: np.ndarray,
    object_id: np.ndarray,
    redshift: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[str], list[str], str]:
    """Encode a feature matrix into synthetic band columns whose adjacent-band
    differences reconstruct it exactly, so an arbitrary feature matrix can be
    fed through the RAIL FlexZBoost estimator's band-based interface."""
    n_obj, n_feat = feature_matrix.shape
    bands = [f"f{i}" for i in range(n_feat)]
    err_bands = [f"f{i}_err" for i in range(n_feat)]
    ref_band = bands[0]

    data_dict: dict[str, Any] = {}
    prev = feature_matrix[:, 0].copy()
    data_dict[bands[0]] = prev
    for i in range(1, n_feat):
        prev = prev - feature_matrix[:, i]
        data_dict[bands[i]] = prev.copy()
    for eb in err_bands:
        data_dict[eb] = np.full(n_obj, ERR_PLACEHOLDER)

    data_dict["object_id"] = np.asarray(object_id)
    if redshift is not None:
        data_dict["redshift"] = np.asarray(redshift)
    return data_dict, bands, err_bands, ref_band


def _mag_limits(bands: list[str]) -> dict[str, float]:
    return {b: NONDETECT_PLACEHOLDER for b in bands}


def _run_estimator(model, data_dict, bands, err_bands, ref_band) -> qp.Ensemble:
    """Run the RAIL FlexZBoost estimator (model may be a .pkl path or a trained
    model handle) and return the qp.Ensemble."""
    from rail.estimation.algos.flexzboost import FlexZBoostEstimator

    estimator = FlexZBoostEstimator.make_stage(
        name="fzboost_estimate",
        hdf5_groupname="",
        model=model,
        bands=bands,
        err_bands=err_bands,
        ref_band=ref_band,
        mag_limits=_mag_limits(bands),
        nondetect_val=float("nan"),
        include_mag_err=False,
        zmin=ZMIN,
        zmax=ZMAX,
        nzbins=NZBINS,
        qp_representation="interp",
        output_mode="return",
        calculated_point_estimates=["mode"],
    )
    handle = estimator.estimate(data_dict)
    return handle.data if hasattr(handle, "data") else handle


def _zmode_from_ensemble(ens: qp.Ensemble) -> np.ndarray:
    ancil = getattr(ens, "ancil", None)
    if ancil is not None:
        for key in ("mode", "zmode", "z_mode"):
            if key in ancil:
                return np.squeeze(np.asarray(ancil[key]))
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    return zgrid[np.argmax(ens.pdf(zgrid), axis=1)]


def _write_ensemble(ens: qp.Ensemble, object_id: np.ndarray, output_file: str | Path) -> None:
    z_mode = _zmode_from_ensemble(ens)
    ens.set_ancil(dict(object_id=np.asarray(object_id), zmode=np.asarray(z_mode)))
    ens.write_to(str(output_file))


# --- Self-contained Task-2 training-set augmentation helpers, used by
# run_taskset_2_training_and_estimation below. These implement the training
# recipe: photometric re-noising of magnitudes, the sim_faint and cmixup
# augmentations, their stacking into one synthetic batch, and a sky-position
# leak filter -- all using only pip-installable packages, so this file stays
# standalone. One deliberate choice: the LSST/Roman noise models are called
# through the pip ``photerr`` package directly instead of RAIL's
# rail.creation.degraders.photometric_errors wrapper, because that wrapper
# only forwards its parameters to photerr and calls
# ``noiseModel(data, random_state=seed)`` -- same draws, one fewer pip
# dependency. --------------------------------------------------------------

# sim_faint recipe constants: every band of a synthesized row is shifted
# fainter by one per-row offset drawn uniformly from this range, so a spread
# of faint depths is produced.
DIM_MAG_MIN = 0.5
DIM_MAG_MAX = 3.0

# cmixup recipe constants (following C-Mixup, Yao et al. 2022, NeurIPS):
# mixing weight lambda ~ Beta(alpha, alpha); partners come from each anchor's
# k nearest neighbors in redshift.
CMIXUP_ALPHA = 0.5
CMIXUP_K_NEIGHBORS = 20

# Shared seed for the augmentation draws and the FlexZBoost fit, and the
# sky-position leak-check radius (in arcseconds).
TASK2_TRAINING_SEED = 42
LEAK_MATCH_ARCSEC = 1.0

# Columns the augmentation models directly; any other training-file column
# (e.g. spectroscopic-selection flags) passes through unchanged per source
# row, so synthetic rows keep the training file's full schema.
AUGMENT_CORE_COLS = {"object_id", "ra", "dec", "redshift"}
AUGMENT_CORE_COLS |= {c for b in LSST_BANDS for c in (_lsst_mag_col(b), _lsst_err_col(b))}
AUGMENT_CORE_COLS |= {c for b in ROMAN_BANDS for c in (_roman_mag_col(b), _roman_err_col(b))}


def _scenario_from_filename(data_file: str | Path) -> str:
    """LSST depth scenario ("1yr" or "10yr") encoded in a challenge Task-2
    file name; it sets the LSST error model's nYrObs when re-noising."""
    name = Path(data_file).name
    for scenario in ("10yr", "1yr"):
        if scenario in name:
            return scenario
    raise ValueError(f"Cannot read a '1yr'/'10yr' scenario from file name {name!r}")


def _apply_photometric_errors(
    truth: dict[str, np.ndarray], scenario: str, seed: int
) -> dict[str, np.ndarray]:
    """Observe true magnitudes with the LSST + Roman photometric error models
    (photerr; Ivezic et al. 2019). ``truth`` maps
    object_id/ra/dec/redshift plus one true-magnitude array per band letter;
    the result carries challenge-schema observed mags and errors, with NaN
    for non-detections. LSST depth follows ``scenario``; Roman is
    point-and-stare, so its depth is fixed."""
    import pandas as pd
    from photerr import LsstErrorModel, RomanErrorModel

    n_yr_obs = 1.0 if scenario == "1yr" else 10.0

    df = pd.DataFrame(
        {
            "object_id": truth["object_id"],
            "ra": truth["ra"],
            "dec": truth["dec"],
            "redshift": truth["redshift"],
            **{b: truth[b] for b in LSST_BANDS},
            **{b: truth[b] for b in ROMAN_BANDS},
        }
    )
    observed = LsstErrorModel(nYrObs=n_yr_obs, ndFlag=np.nan)(df, random_state=seed)
    observed = RomanErrorModel(ndFlag=np.nan)(observed, random_state=seed)

    rename = {}
    for b in LSST_BANDS:
        rename[b] = _lsst_mag_col(b)
        rename[f"{b}_err"] = _lsst_err_col(b)
    for b in ROMAN_BANDS:
        rename[b] = _roman_mag_col(b)
        rename[f"{b}_err"] = _roman_err_col(b)
    observed = observed.rename(columns=rename)

    return {col: np.asarray(observed[col]) for col in observed.columns}


def _dim_and_renoise(
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    offsets: np.ndarray,
    scenario: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Synthesize faint rows from source rows ``idx``: shift every band of a
    row fainter by its own ``offsets`` entry (the same shift in every band, so
    color is preserved), keep non-detections (NaN) non-detected, then
    re-observe the shifted magnitudes with :func:`_apply_photometric_errors`
    at depth ``scenario``. True redshift/ra/dec are carried through unchanged
    -- dimming light does not change a redshift. ``object_id`` is replaced
    with sequential negative IDs, disjoint from every real, non-negative ID.
    """
    n_draw = len(idx)
    truth: dict[str, np.ndarray] = {
        "object_id": -(np.arange(1, n_draw + 1, dtype=float)),
        "ra": np.asarray(data["ra"], dtype=float)[idx],
        "dec": np.asarray(data["dec"], dtype=float)[idx],
        "redshift": np.asarray(data["redshift"], dtype=float)[idx],
    }
    for b in LSST_BANDS:
        original = np.asarray(data[_lsst_mag_col(b)], dtype=float)[idx]
        truth[b] = np.where(np.isfinite(original), original + offsets, np.nan)
    for b in ROMAN_BANDS:
        original = np.asarray(data[_roman_mag_col(b)], dtype=float)[idx]
        truth[b] = np.where(np.isfinite(original), original + offsets, np.nan)

    observed = _apply_photometric_errors(truth, scenario=scenario, seed=seed)

    for col, values in data.items():
        if col not in AUGMENT_CORE_COLS:
            observed[col] = np.asarray(values)[idx]
    return observed


def _synthesize_sim_faint(
    data: dict[str, np.ndarray], seed: int, scenario: str
) -> dict[str, np.ndarray]:
    """One dimmed-and-renoised synthetic row per real row of ``data``
    (sim_faint at the shipped recipe's frac=1.0). Draws are made in a fixed
    order from a seeded generator, so a given seed reproduces the same
    synthetic rows bit for bit."""
    n_in = len(data["object_id"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_in, size=n_in, replace=False)
    offsets = rng.uniform(DIM_MAG_MIN, DIM_MAG_MAX, size=n_in)
    return _dim_and_renoise(data, idx, offsets, scenario, seed)


def _mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """AB-like mag -> flux at an arbitrary zeropoint of 0; the zeropoint
    cancels because every use below only blends fluxes linearly and converts
    back with the exact inverse, :func:`_flux_to_mag`."""
    return 10.0 ** (-0.4 * np.asarray(mag, dtype=float))


def _flux_to_mag(flux: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`_mag_to_flux`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux)


def _find_mixup_partners(
    redshift: np.ndarray, anchor_idx: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """For each anchor row, pick one partner uniformly from its k nearest
    neighbors in redshift, excluding itself."""
    from scipy.spatial import cKDTree

    n_in = len(redshift)
    k_eff = max(1, min(k, n_in - 1))
    tree = cKDTree(np.asarray(redshift, dtype=float).reshape(-1, 1))
    query_z = np.asarray(redshift, dtype=float)[anchor_idx].reshape(-1, 1)
    _, nbr_idx = tree.query(query_z, k=k_eff + 1)

    # Each anchor's own row is guaranteed present (distance 0 to itself), in
    # exactly one column. Push that "self" column to the end (stable sort
    # keeps the rest in increasing-distance order) and keep the first k_eff.
    self_mask = nbr_idx == anchor_idx[:, None]
    order = np.argsort(self_mask, axis=1, kind="stable")
    candidates = np.take_along_axis(nbr_idx, order, axis=1)[:, :k_eff]

    choice_col = rng.integers(0, k_eff, size=len(anchor_idx))
    return candidates[np.arange(len(anchor_idx)), choice_col]


def _blend_mixup_rows(
    data: dict[str, np.ndarray],
    anchor_idx: np.ndarray,
    partner_idx: np.ndarray,
    lam: np.ndarray,
) -> dict[str, np.ndarray]:
    """Synthesize C-mixup rows, the lambda-blend of anchor and partner.
    Magnitudes blend in flux space (NaN in either source stays NaN); magnitude
    errors propagate through the same linear flux blend assuming independent
    Gaussian noise.
    redshift blends with the SAME lambda as the photometry -- label mixup is
    C-mixup's definition, unlike sim_faint's fixed redshift. ra/dec stay the
    anchor's; object_id is replaced with sequential negative IDs."""
    n_draw = len(anchor_idx)
    redshift = np.asarray(data["redshift"], dtype=float)
    out: dict[str, np.ndarray] = {
        "object_id": -(np.arange(1, n_draw + 1, dtype=float)),
        "ra": np.asarray(data["ra"], dtype=float)[anchor_idx],
        "dec": np.asarray(data["dec"], dtype=float)[anchor_idx],
        "redshift": lam * redshift[anchor_idx] + (1.0 - lam) * redshift[partner_idx],
    }

    ln10_over_2p5 = np.log(10.0) / 2.5
    all_bands = [(_lsst_mag_col(b), _lsst_err_col(b)) for b in LSST_BANDS]
    all_bands += [(_roman_mag_col(b), _roman_err_col(b)) for b in ROMAN_BANDS]
    for mag_col, err_col in all_bands:
        mag_anchor = np.asarray(data[mag_col], dtype=float)[anchor_idx]
        mag_partner = np.asarray(data[mag_col], dtype=float)[partner_idx]
        err_anchor = np.asarray(data[err_col], dtype=float)[anchor_idx]
        err_partner = np.asarray(data[err_col], dtype=float)[partner_idx]

        flux_anchor = _mag_to_flux(mag_anchor)
        flux_partner = _mag_to_flux(mag_partner)
        flux_out = lam * flux_anchor + (1.0 - lam) * flux_partner
        out[mag_col] = _flux_to_mag(flux_out)

        ferr_anchor = flux_anchor * ln10_over_2p5 * err_anchor
        ferr_partner = flux_partner * ln10_over_2p5 * err_partner
        ferr_out = np.sqrt((lam * ferr_anchor) ** 2 + ((1.0 - lam) * ferr_partner) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[err_col] = ferr_out / (flux_out * ln10_over_2p5)

    for col, values in data.items():
        if col not in AUGMENT_CORE_COLS:
            out[col] = np.asarray(values)[anchor_idx]
    return out


def _synthesize_cmixup(data: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """One C-mixup synthetic row per real row of ``data`` (cmixup at the
    shipped recipe's frac=1.0). Draws are made in a fixed order from a seeded
    generator, so a given seed reproduces the same synthetic rows bit for
    bit."""
    n_in = len(data["object_id"])
    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(n_in, size=n_in, replace=False)
    partner_idx = _find_mixup_partners(
        np.asarray(data["redshift"], dtype=float), anchor_idx, CMIXUP_K_NEIGHBORS, rng
    )
    lam = rng.beta(CMIXUP_ALPHA, CMIXUP_ALPHA, size=n_in)
    return _blend_mixup_rows(data, anchor_idx, partner_idx, lam)


def _stack_synthetic_rows(
    data: dict[str, np.ndarray], seed: int, scenario: str
) -> dict[str, np.ndarray]:
    """The stacked sim_faint + cmixup synthetic batch, both methods drawn
    from the SAME real rows. Per-method seeds are split from ``seed`` with
    numpy.random.SeedSequence -- index 0 is sim_faint, 1 is cmixup -- so the
    two methods' draws are decorrelated yet deterministic. Each method numbers
    its own rows -1, -2, ...; the second method's IDs are shifted further
    negative so the stacked IDs stay unique."""
    sim_faint_seed = int(np.random.SeedSequence([seed, 0]).generate_state(1)[0])
    cmixup_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    parts = [
        _synthesize_sim_faint(data, sim_faint_seed, scenario),
        _synthesize_cmixup(data, cmixup_seed),
    ]

    id_offset = 0
    for part in parts:
        part["object_id"] = np.asarray(part["object_id"], dtype=float) - id_offset
        id_offset += len(part["object_id"])
    stacked = {col: np.concatenate([part[col] for part in parts]) for col in parts[0]}

    stacked_ids = stacked["object_id"]
    if np.any(stacked_ids >= 0) or len(np.unique(stacked_ids)) != len(stacked_ids):
        raise ValueError("stacked synthetic object_id values must be negative and unique")
    return stacked


def _drop_sky_position_leaks(
    train_data: dict[str, np.ndarray], test_file: str | Path
) -> dict[str, np.ndarray]:
    """Return ``train_data`` without rows lying within LEAK_MATCH_ARCSEC of
    an official test-file position. WHY: the
    official Task-2 training and test draws share ~0.1% of their galaxies
    (verbatim-identical photometry at 0.0 arcsec separation under different
    object_ids), and training on a copy of a test galaxy means being scored
    on data the model saw. Only the test file's public ra/dec are read."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    test_data = _read_table(str(test_file))
    train_coords = SkyCoord(
        ra=np.asarray(train_data["ra"]) * u.deg, dec=np.asarray(train_data["dec"]) * u.deg
    )
    test_coords = SkyCoord(
        ra=np.asarray(test_data["ra"]) * u.deg, dec=np.asarray(test_data["dec"]) * u.deg
    )
    _, separation, _ = train_coords.match_to_catalog_sky(test_coords)
    is_leak = separation.arcsec <= LEAK_MATCH_ARCSEC
    if not np.any(is_leak):
        return train_data
    return {col: np.asarray(values)[~is_leak] for col, values in train_data.items()}


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Load the FlexZBoost model staged in model_file and predict p(z) on
    test_file, writing a qp Ensemble with object_id + zmode ancillary data
    to output_file.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    X_test, id_test, _ = _build_features(str(test_file))
    test_dict, bands, err_bands, ref = _encode_for_flexzboost(X_test, id_test, redshift=None)
    ens = _run_estimator(str(model_file), test_dict, bands, err_bands, ref)
    _write_ensemble(ens, id_test, output_file)


def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Train a FlexZBoost model on train_file and predict p(z) on test_file,
    writing a qp Ensemble with object_id + zmode ancillary data to
    output_file.

    Parameters
    ----------
    train_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be trained
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    from rail.core.stage import DataStore
    from rail.estimation.algos.flexzboost import FlexZBoostInformer

    DataStore.allow_overwrite = True

    X_train, id_train, z_train = _build_features(str(train_file))
    if z_train is None:
        raise ValueError(f"Training file {train_file} lacks a 'redshift' column")
    train_dict, bands, err_bands, ref = _encode_for_flexzboost(X_train, id_train, z_train)

    X_test, id_test, _ = _build_features(str(test_file))
    test_dict, _, _, _ = _encode_for_flexzboost(X_test, id_test, redshift=None)

    model_path = str(Path(output_file).parent / f"_model_{SUBMISSION_NAME}.pkl")
    informer = FlexZBoostInformer.make_stage(
        name="fzboost_inform",
        hdf5_groupname="",
        model=model_path,
        bands=bands,
        err_bands=err_bands,
        ref_band=ref,
        mag_limits=_mag_limits(bands),
        nondetect_val=float("nan"),
        redshift_col="redshift",
        include_mag_err=False,
        zmin=ZMIN,
        zmax=ZMAX,
        nzbins=NZBINS,
        seed=42,
        max_basis=35,
        basis_system="cosine",
        regression_params={"max_depth": 8, "objective": "reg:squarederror"},
    )
    model_handle = informer.inform(train_dict)
    ens = _run_estimator(model_handle, test_dict, bands, err_bands, ref)
    _write_ensemble(ens, id_test, output_file)


def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Load the FlexZBoost model staged in model_file and predict p(z) on
    test_file, writing a qp Ensemble with object_id + zmode ancillary data
    to output_file.  The shipped Task-2 model consumes the same
    self-contained feature encoding as Task 1, so the estimation path is
    the same.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    X_test, id_test, _ = _build_features(str(test_file))
    test_dict, bands, err_bands, ref = _encode_for_flexzboost(X_test, id_test, redshift=None)
    ens = _run_estimator(str(model_file), test_dict, bands, err_bands, ref)
    _write_ensemble(ens, id_test, output_file)


def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Train the full augmented Task-2 FlexZBoost model -- leak-filtered real
    training rows plus one sim_faint and one cmixup synthetic copy, stacked,
    the same recipe the shipped Task-2 model was built with -- and predict
    p(z) on test_file, writing a qp Ensemble with object_id + zmode ancillary
    data to output_file.

    Parameters
    ----------
    train_file:
        Path to the training file contains the labeled photometric data on
        which the PZ estimation will be trained
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    from rail.core.stage import DataStore
    from rail.estimation.algos.flexzboost import FlexZBoostInformer

    DataStore.allow_overwrite = True

    train_data = _read_table(str(train_file))
    if "redshift" not in train_data:
        raise ValueError(f"Training file {train_file} lacks a 'redshift' column")
    train_data = _drop_sky_position_leaks(train_data, test_file)

    scenario = _scenario_from_filename(train_file)
    synthetic = _stack_synthetic_rows(train_data, TASK2_TRAINING_SEED, scenario)
    augmented = {
        col: np.concatenate([np.asarray(train_data[col]), np.asarray(synthetic[col])])
        for col in train_data
    }

    # _build_features reads from disk, so stage the augmented table as a
    # scratch file beside the output (same transient-artifact convention as
    # Task 1's _model_*.pkl; overwritten on each (sim, scenario) call).
    augmented_path = Path(output_file).parent / f"_augmented_train_{SUBMISSION_NAME}.hdf5"
    augmented_path.unlink(missing_ok=True)
    tables_io.write(augmented, str(augmented_path))

    X_train, id_train, z_train = _build_features(str(augmented_path))
    train_dict, bands, err_bands, ref = _encode_for_flexzboost(X_train, id_train, z_train)

    X_test, id_test, _ = _build_features(str(test_file))
    test_dict, _, _, _ = _encode_for_flexzboost(X_test, id_test, redshift=None)

    model_path = str(Path(output_file).parent / f"_model_task2_{SUBMISSION_NAME}.pkl")
    informer = FlexZBoostInformer.make_stage(
        name="fzboost_inform",
        hdf5_groupname="",
        model=model_path,
        bands=bands,
        err_bands=err_bands,
        ref_band=ref,
        mag_limits=_mag_limits(bands),
        nondetect_val=float("nan"),
        redshift_col="redshift",
        include_mag_err=False,
        zmin=ZMIN,
        zmax=ZMAX,
        nzbins=NZBINS,
        seed=TASK2_TRAINING_SEED,
        max_basis=35,
        basis_system="cosine",
        regression_params={"max_depth": 8, "objective": "reg:squarederror"},
    )
    model_handle = informer.inform(train_dict)
    ens = _run_estimator(model_handle, test_dict, bands, err_bands, ref)
    _write_ensemble(ens, id_test, output_file)


def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 3

    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return


def run_taskset_3_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 3

    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    train_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be trained
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return


def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run estimation for task set 4

    This function should use a model stored in model_file, which
    is downloaded as part of the submission tar file.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    model_file:
        Path to the model.  This should be part of the submission
        tar file.
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return


def run_taskset_4_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    User supplied function to run training and estimation for task set 4

    This function should train a model and use it.

    This function should write output data to output_file in qp
    format.

    Parameters
    ----------
    test_file:
        Path to the test file contains the photometric test data on
        which the PZ estimation will be run
    output_file:
        Path to write the output data to.  The output data should
        be written in qp format.
    """
    return


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

    
@pytest.mark.skip(
    reason="Tasksets 3 and 4 are not part of this submission; only the "
    "taskset 1 and 2 run functions above are provided."
)
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


@pytest.mark.skip(
    reason="Tasksets 3 and 4 are not part of this submission; only the "
    "taskset 1 and 2 run functions above are provided."
)
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
