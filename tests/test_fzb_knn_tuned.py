import os
import pickle
import tempfile
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
SUBMISSION_NAME: str = "fzb_knn_tuned"
SUBMISSION_URL: str = "https://github.com/alex-strange/pz_data_challenge_fzb_knn_tuned/releases/download/v3.0/fzb_knn_tuned_submission.tgz"

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"


@pytest.fixture(name="setup_public_area", scope="module")
def setup_public_area() -> int:
    """
    A pytest fixture to download the public challenge data.

    This file is run directly from its own path rather than copied next to
    a ``tests/conftest.py``, so it supplies this fixture itself instead of
    relying on one.
    """
    if os.path.exists(PUBLIC_AREA):
        return 0
    submit_utils.download_and_extract_tar(
        "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz",
        "tests",
    )
    return 0


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


# --- Self-contained FlexZBoost feature construction, used by
# run_taskset_3_estimation_only and run_taskset_4_estimation_only below. A
# submitted model reads its inputs positionally, so predicting has to hand it
# the SAME columns in the SAME order the fit was handed; this section builds
# them from the challenge file using only pip-installable packages that
# pz_data_challenge already requires, so this file runs in the challenge CI as
# a standalone file with no extra dependencies. Which columns a model was
# fitted on is read per taskset: a taskset-3 model's from
# TASKSET_3_MODEL_INPUTS by its file name, a taskset-4 blend's from its own
# model file, which names the input constructions the model was handed and
# carries the constants any of them was measured with on the model's training
# table, and a plain taskset-4 model is handed the shipped columns.
# ---------------------------------------------------------------------------
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
ROMAN_BANDS = ["Y", "J", "H"]
BANDS_BLUEST_FIRST = LSST_BANDS + ROMAN_BANDS
REFERENCE_BAND = "i"
NONDETECT_PLACEHOLDER = 99.0
ERR_PLACEHOLDER = 1.0
COLOR_CLIP = (-10.0, 10.0)
BREAK_SPANNING_COLORS = [("i", "Y"), ("z", "J"), ("r", "z"), ("z", "Y"), ("Y", "J")]

# FlexZBoost redshift grid: p(z) is evaluated from ZMIN to ZMAX on NZBINS points.
ZMIN, ZMAX, NZBINS = 0.0, 3.0, 301

# The input constructions a model may have been handed, each built below
# exactly as the model was handed it. A model naming none was fitted on the
# twelve shipped colours (five adjacent LSST, two adjacent Roman, five
# break-spanning), the nine magnitudes, their nine errors and nine detection
# flags. On top of those:
#   "reference_band_colors"  adds every colour taken against the i band that
#                            the shipped colours do not already form (five);
#   "all_pairs_colors"       replaces the colour block with a colour for every
#                            one of the 36 pairs of the nine bands;
#   "per_color_errors"       follows each colour with its own error, the two
#                            bands' errors in quadrature (RAIL's own
#                            expression);
#   "asinh_magnitudes"       replaces each magnitude column with the arsinh of
#                            its flux, centred on the median and scaled by the
#                            interquartile range of the model's training table
#                            (Tanaka et al. 2018, Eq. 1), whose constants the
#                            model file carries.
# The two colour constructions cannot both be named.
REFERENCE_BAND_COLORS = "reference_band_colors"
ALL_PAIRS_COLORS = "all_pairs_colors"
PER_COLOR_ERRORS = "per_color_errors"
ASINH_MAGNITUDES = "asinh_magnitudes"
INPUT_CONSTRUCTIONS = (REFERENCE_BAND_COLORS, ALL_PAIRS_COLORS, PER_COLOR_ERRORS, ASINH_MAGNITUDES)

# Which input constructions each taskset-3 model was fitted on. A taskset-3
# model file does not name its own, so the set is recorded here per model,
# keyed by "<sim>_<scenario>", which is what the challenge's file names
# encode. The four staged taskset-3 picks share the shipped inputs, the empty
# set: each blended FlexZBoost member and each plain model was fitted on the
# twelve shipped colours, the construction the fzb_dimmingtofaint submission
# used. A model fitted on another input set is one entry here, naming the
# constructions above.
TASKSET_3_MODEL_INPUTS: dict[str, tuple[str, ...]] = {
    "flagship_1yr": (),
    "cardinal_1yr": (),
    "flagship_10yr": (),
    "cardinal_10yr": (),
}


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


def _band_columns() -> list[tuple[str, str]]:
    """Every band's (magnitude column, error column), LSST then Roman."""
    return [(_lsst_mag_col(b), _lsst_err_col(b)) for b in LSST_BANDS] + [
        (_roman_mag_col(b), _roman_err_col(b)) for b in ROMAN_BANDS
    ]


def _shipped_colors() -> list[tuple[str, str]]:
    """The twelve shipped colour pairs, in their shipped order: the five
    adjacent LSST colours, the two adjacent Roman ones, then the
    break-spanning ones."""
    pairs = [(LSST_BANDS[i], LSST_BANDS[i + 1]) for i in range(len(LSST_BANDS) - 1)]
    pairs += [(ROMAN_BANDS[i], ROMAN_BANDS[i + 1]) for i in range(len(ROMAN_BANDS) - 1)]
    return pairs + list(BREAK_SPANNING_COLORS)


def _all_pairs_colors() -> list[tuple[str, str]]:
    """A colour for every pair of the nine bands, bluer minus redder."""
    return [
        (bluer, redder)
        for index, bluer in enumerate(BANDS_BLUEST_FIRST)
        for redder in BANDS_BLUEST_FIRST[index + 1:]
    ]


def _reference_band_colors() -> list[tuple[str, str]]:
    """Every colour taken against the i band, in band order, bluer minus
    redder."""
    reference_position = BANDS_BLUEST_FIRST.index(REFERENCE_BAND)
    return [
        (band, REFERENCE_BAND) if position < reference_position else (REFERENCE_BAND, band)
        for position, band in enumerate(BANDS_BLUEST_FIRST)
        if band != REFERENCE_BAND
    ]


def _colors(input_set: tuple[str, ...]) -> list[tuple[str, str]]:
    """The colour pairs a model handed ``input_set`` reads, in its order."""
    if ALL_PAIRS_COLORS in input_set:
        return _all_pairs_colors()
    pairs = _shipped_colors()
    if REFERENCE_BAND_COLORS in input_set:
        already = set(pairs)
        pairs += [pair for pair in _reference_band_colors() if pair not in already]
    return pairs


def _check_input_set(input_set: tuple[str, ...], normalisation: dict | None) -> None:
    """Refuse an input set this file cannot build exactly, before any column
    is built, since a model reading its columns positionally could not detect
    a wrong one for itself."""
    unknown = sorted(set(input_set) - set(INPUT_CONSTRUCTIONS))
    if unknown:
        raise ValueError(
            f"the model was handed the input constructions {unknown}, which this "
            f"file does not build; it builds {list(INPUT_CONSTRUCTIONS)}"
        )
    if REFERENCE_BAND_COLORS in input_set and ALL_PAIRS_COLORS in input_set:
        raise ValueError(
            f"{REFERENCE_BAND_COLORS} and {ALL_PAIRS_COLORS} each decide the colour "
            f"block, so no model was handed both"
        )
    if ASINH_MAGNITUDES in input_set and not normalisation:
        raise ValueError(
            f"the model was handed {ASINH_MAGNITUDES} and its file carries no "
            f"constants for it, so its brightness columns cannot be built"
        )


def _asinh_flux(magnitude: np.ndarray, constants: tuple[float, float]) -> np.ndarray:
    """Tanaka et al. 2018, Eq. 1, on one band's filled magnitudes: the arsinh
    of the flux the magnitude stands for, less the training table's median,
    over its interquartile range."""
    median, interquartile_range = constants
    return np.arcsinh((np.power(10.0, -0.4 * magnitude) - median) / interquartile_range)


def _read_table(hdf5_path: str) -> dict[str, np.ndarray]:
    raw = tables_io.read(hdf5_path, tType="numpyDict")
    return {k: np.asarray(v) for k, v in raw.items()}


def _build_features(
    hdf5_path: str,
    input_set: tuple[str, ...] = (),
    normalisation: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Feature matrix from a challenge HDF5 file, under the input set
    ``input_set`` with the constants ``normalisation``.

    Column order, which the model reads positionally: the nine brightness
    columns (each magnitude, or its arsinh flux under asinh_magnitudes), then
    each colour (each followed by its own error under per_color_errors), then
    the nine magnitude errors, then the nine detection flags. A band the survey
    did not see enters at NONDETECT_PLACEHOLDER with a zero detection flag, and
    every colour formed from such a band is 0.0. Colours are formed from the
    magnitudes, never from the arsinh fluxes.
    Returns (feature_matrix, object_id, redshift_or_None).
    """
    _check_input_set(input_set, normalisation)
    data = _read_table(hdf5_path)
    object_id = np.asarray(data["object_id"])
    redshift = np.asarray(data["redshift"]) if "redshift" in data else None

    magnitude: dict[str, np.ndarray] = {}
    error: dict[str, np.ndarray] = {}
    detected: dict[str, np.ndarray] = {}
    for mag_col, err_col in _band_columns():
        m = np.asarray(data[mag_col], dtype=float)
        e = np.asarray(data[err_col], dtype=float)
        seen = np.isfinite(m)
        magnitude[mag_col] = np.where(seen, m, NONDETECT_PLACEHOLDER)
        error[mag_col] = np.where(np.isfinite(e), e, ERR_PLACEHOLDER)
        detected[mag_col] = seen

    mag_cols = [mag_col for mag_col, _err_col in _band_columns()]
    if ASINH_MAGNITUDES in input_set:
        columns: list[np.ndarray] = [
            _asinh_flux(magnitude[mag_col], normalisation[mag_col]) for mag_col in mag_cols
        ]
    else:
        columns = [magnitude[mag_col] for mag_col in mag_cols]

    with_errors = PER_COLOR_ERRORS in input_set
    for band_a, band_b in _colors(input_set):
        bluer = _mag_col_for_band(band_a)
        redder = _mag_col_for_band(band_b)
        difference = magnitude[bluer] - magnitude[redder]
        difference = np.where(detected[bluer] & detected[redder], difference, 0.0)
        columns.append(np.clip(difference, COLOR_CLIP[0], COLOR_CLIP[1]))
        if with_errors:
            # RAIL's own per-colour error, its expression unchanged:
            # `np.sqrt(err_a ** 2 + err_b ** 2)`.
            columns.append(np.sqrt(error[bluer] ** 2 + error[redder] ** 2))

    columns += [error[mag_col] for mag_col in mag_cols]
    columns += [detected[mag_col].astype(float) for mag_col in mag_cols]

    feature_matrix = np.vstack(columns).T.astype(float)
    feature_matrix = np.nan_to_num(
        feature_matrix, nan=0.0, posinf=COLOR_CLIP[1], neginf=COLOR_CLIP[0]
    )
    return feature_matrix, object_id, redshift


def _taskset_3_model_inputs(model_file: str | Path) -> tuple[str, ...]:
    """Which input constructions the taskset-3 model in ``model_file`` was
    fitted on.

    The set is recorded per model in ``TASKSET_3_MODEL_INPUTS`` and read off
    the simulation and the depth the challenge's own file name carries
    (``pz_challenge_taskset_3_<sim>_pz_model_<scenario>.pkl``). A name that
    matches none of the four raises rather than silently predicting with the
    wrong columns, which a model reading its inputs positionally could not
    detect for itself.
    """
    name = Path(str(model_file)).name
    for key in TASKSET_3_MODEL_INPUTS:
        sim, scenario = key.split("_", 1)
        if f"_{sim}_" in name and name.endswith(f"_{scenario}.pkl"):
            return TASKSET_3_MODEL_INPUTS[key]
    raise ValueError(
        f"cannot read a simulation and a depth from the model file name {name!r}; "
        f"the name is what says which inputs a taskset-3 model was fitted on"
    )


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


# --- Which kind of model file a dataset ships --------------------------------
# One package may mix two kinds of model file, per dataset: a blend (below),
# or a plain single FlexZBoost model, the pickled flexcode FlexCodeModel the
# unblended submissions (fzb_taskset3 and fzb_taskset4) ship. A plain model is
# known by its class as pickle records it, module and name, so this file needs
# no import of flexcode beyond what unpickling the file already does. Any
# other file goes to the blend route, whose reader refuses what is not its
# taskset's blend.
PLAIN_MODEL_CLASS = ("flexcode.core", "FlexCodeModel")


def _is_plain_flexzboost_model(model_file: str | Path) -> bool:
    """Whether ``model_file`` holds one pickled FlexCodeModel and nothing else."""
    with open(model_file, "rb") as handle:
        contents = pickle.load(handle)
    model_class = type(contents)
    return (model_class.__module__, model_class.__qualname__) == PLAIN_MODEL_CLASS


# --- A plain single FlexZBoost model: the unblended submissions' own route ---
# The plain FlexZBoost route of fzb_taskset3 and fzb_taskset4, unchanged, so a
# plain model is predicted here exactly as its own submission's test predicts
# it: on the columns its taskset's run function names for it (a taskset-3
# model's from TASKSET_3_MODEL_INPUTS, a taskset-4 model the shipped columns),
# the same encoding and the same RAIL stage, its zmode taken from the
# ensemble's own mode.
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


def _estimate_with_plain_model(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    input_set: tuple[str, ...],
) -> None:
    """The unblended submissions' estimation function, unchanged: load the
    FlexZBoost model in model_file, predict p(z) on test_file from the columns
    ``input_set`` builds, and write a qp Ensemble with object_id + zmode
    ancillary data to output_file."""
    X_test, id_test, _ = _build_features(str(test_file), input_set)
    test_dict, bands, err_bands, ref = _encode_for_flexzboost(X_test, id_test, redshift=None)
    ens = _run_estimator(str(model_file), test_dict, bands, err_bands, ref)
    _write_ensemble(ens, id_test, output_file)


# --- The blend: two fitted members, their curves mixed at fixed weights ------
# A blended dataset's model file holds TWO fitted models, each stored as its own
# model file byte for byte: the FlexZBoost model, predicted on the columns its
# taskset's blend format names for it (the feature construction above), and
# RAIL's k-nearest-neighbour estimator, predicted with the configuration its
# fit was staged with, which the model file carries because the limiting
# magnitudes it fills a missing band with are computed by code this file cannot
# import. Each member's curves are read on the graded grid and normalized to
# unit area, mixed at the member's weight, and the mixture is renormalized; its
# peak is the point estimate. The weights are the stacking weights of Hastie,
# Tibshirani & Friedman, The Elements of Statistical Learning, Sec. 8.8,
# Eq. (8.59), fitted on held-out predictions and applied unchanged to the
# members refitted on the whole training file, as that section's own final
# prediction applies them. -------------------------------------------------

# What each taskset's blend files say they are, checked before one is read.
# Taskset 3's blends are the first format, whose members name no inputs;
# taskset 4's are the second, whose FlexZBoost member names its own.
BLENDED_MODEL_FORMAT: dict[int, str] = {
    3: "pz-challenge p(z) blend of member model files at fixed weights, v1",
    4: (
        "pz-challenge p(z) blend of member model files at fixed weights, "
        "FlexZBoost members carrying their own inputs, v2"
    ),
}

# The keys a FlexZBoost member's entry names its inputs and their constants by.
INPUT_SET_KEY = "input_set"
NORMALISATION_KEY = "normalisation"


def _members_handed_taskset_3_inputs(
    model_file: str | Path, members: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """A taskset-3 blend's members, each handed the input set
    ``TASKSET_3_MODEL_INPUTS`` records for the file, the way taskset 3's own
    test handed that set to every member; the k-nearest-neighbour member reads
    its own configured bands and ignores it. These members carry no constants."""
    input_set = _taskset_3_model_inputs(model_file)
    return [{**member, INPUT_SET_KEY: input_set} for member in members]


def _members_as_stored(
    model_file: str | Path, members: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """A taskset-4 blend's members, whose FlexZBoost member names its own
    inputs and their constants in its entry."""
    return members


# Where each taskset's blend members get the inputs their FlexZBoost member
# was handed.
BLEND_MEMBERS_WITH_INPUTS = {
    3: _members_handed_taskset_3_inputs,
    4: _members_as_stored,
}


def _read_blended_model(model_file: str | Path, taskset: int) -> dict[str, Any]:
    """Taskset ``taskset``'s blend in ``model_file``: its format tag, its
    description and its members, each FlexZBoost member naming the inputs it
    was handed.

    A file that does not say it is that taskset's blend is refused rather than
    guessed at, since a plain single model, or another taskset's blend, handed
    to this reader would otherwise fail somewhere far less clear.
    """
    with open(model_file, "rb") as handle:
        contents = pickle.load(handle)
    blend_format = BLENDED_MODEL_FORMAT[taskset]
    if not isinstance(contents, dict) or contents.get("format") != blend_format:
        raise ValueError(f"{model_file} is not a {blend_format!r} file")
    members = BLEND_MEMBERS_WITH_INPUTS[taskset](model_file, contents["members"])
    return {**contents, "members": members}


def _flexzboost_member(model_path: str, test_file: str, member: dict[str, Any]) -> qp.Ensemble:
    """The FlexZBoost member's p(z) of every test galaxy, predicted on the
    columns its entry names, in the same order, with the same encoding and
    the same RAIL stage its fit's predictions were made with."""
    if INPUT_SET_KEY not in member:
        raise ValueError(
            f"the FlexZBoost member {member.get('label')!r} names no input "
            f"constructions, so which columns it reads cannot be known"
        )
    X_test, id_test, _ = _build_features(
        test_file, tuple(member[INPUT_SET_KEY]), member.get(NORMALISATION_KEY)
    )
    test_dict, bands, err_bands, ref = _encode_for_flexzboost(X_test, id_test, redshift=None)
    return _run_estimator(model_path, test_dict, bands, err_bands, ref)


def _knn_member(model_path: str, test_file: str, member: dict[str, Any]) -> qp.Ensemble:
    """The k-nearest-neighbour member's p(z) of every test galaxy.

    RAIL's own estimate stage, built with the configuration the member's fit
    was staged with (carried in the model file), handed the nine magnitude
    columns and nothing else, in the test file's own row order.
    """
    from rail.core.stage import DataStore
    from rail.estimation.algos.k_nearneigh import KNearNeighEstimator

    config = dict(member["estimator_config"])
    data = _read_table(test_file)
    photometry = {band: data[band] for band in config["bands"]}
    DataStore.allow_overwrite = True
    stage = KNearNeighEstimator.make_stage(
        name="knn_estimate", model=model_path, output_mode="return", **config
    )
    handle = stage.estimate(photometry)
    return handle.data if hasattr(handle, "data") else handle


# How each member of a model file is predicted, by the estimator it names.
MEMBER_PREDICTORS = {
    "flexzboost": _flexzboost_member,
    "k_nearneigh": _knn_member,
}


def _as_stored(ensemble: qp.Ensemble, path: str) -> qp.Ensemble:
    """A member's p(z) as it reads back from a file.

    qp normalizes a parameterization when an ensemble is built, so an ensemble
    read back from disk can differ from the one written in its last bits. The
    staged estimate was mixed from each member's p(z) as stored on disk, and
    passing each member through the same write and read here is what makes this
    file's mixture that one exactly rather than to within rounding.
    """
    ensemble.write_to(path)
    return qp.read(path)


def _unit_area_curves(ensemble: qp.Ensemble, zgrid: np.ndarray) -> np.ndarray:
    """One member's curves on the graded grid, each normalized to unit area by
    qp's own interpolated parameterization, one row per galaxy."""
    curves = np.asarray(ensemble.pdf(zgrid), dtype=float).reshape(-1, zgrid.size)
    normalized = qp.Ensemble(qp.interp, data={"xvals": zgrid, "yvals": curves})
    return np.asarray(normalized.objdata["yvals"], dtype=float).reshape(-1, zgrid.size)


def _mix(
    member_curves: list[np.ndarray],
    weights: list[float],
    object_id: np.ndarray,
    zgrid: np.ndarray,
) -> qp.Ensemble:
    """The members' unit-area curves mixed at ``weights``, renormalized, with
    the galaxy ids and the mixture's own peak as its ancillary data."""
    stacked = np.stack([np.asarray(curves, dtype=float) for curves in member_curves])
    mixed = np.average(stacked, axis=0, weights=np.asarray(weights, dtype=float))
    ids = np.asarray(object_id)
    if mixed.shape[0] != ids.size:
        raise ValueError(
            f"the mixed curves cover {mixed.shape[0]} galaxies and the test file "
            f"holds {ids.size}, so which galaxy each curve belongs to is unknown"
        )
    blended = qp.Ensemble(qp.interp, data={"xvals": zgrid, "yvals": mixed})
    blended.set_ancil({"object_id": ids, "zmode": np.asarray(blended.mode(zgrid))})
    return blended


def _estimate_with_blend(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    taskset: int,
) -> None:
    """Load taskset ``taskset``'s blend staged in model_file, predict p(z) on
    test_file with each of its members, mix their curves at the members'
    weights, and write a qp Ensemble with object_id + zmode ancillary data to
    output_file."""
    blend = _read_blended_model(model_file, taskset)
    zgrid = np.linspace(ZMIN, ZMAX, NZBINS)
    object_id = np.asarray(_read_table(str(test_file))["object_id"])

    member_curves: list[np.ndarray] = []
    weights: list[float] = []
    with tempfile.TemporaryDirectory() as scratch:
        for member in blend["members"]:
            estimator = member["estimator"]
            if estimator not in MEMBER_PREDICTORS:
                raise ValueError(
                    f"{model_file} holds a {estimator!r} member, which this file "
                    f"has no prediction route for"
                )
            member_path = os.path.join(scratch, f"{estimator}.pkl")
            with open(member_path, "wb") as handle:
                handle.write(member["model_file"])
            ensemble = MEMBER_PREDICTORS[estimator](member_path, str(test_file), member)
            stored = _as_stored(ensemble, os.path.join(scratch, f"{estimator}_estimate.hdf5"))
            member_curves.append(_unit_area_curves(stored, zgrid))
            weights.append(float(member["weight"]))

    _mix(member_curves, weights, object_id, zgrid).write_to(str(output_file))


def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Predict p(z) on test_file with the model staged in model_file and write a
    qp Ensemble with object_id + zmode ancillary data to output_file: a plain
    single FlexZBoost model through the fzb_taskset3 test's own route, on the
    inputs TASKSET_3_MODEL_INPUTS records for it, and any other file through
    the blend route, which predicts with each member of a taskset-3 blend,
    mixes their curves at the members' weights, and refuses a file that is not
    one.

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
    if _is_plain_flexzboost_model(model_file):
        _estimate_with_plain_model(
            model_file, test_file, output_file, _taskset_3_model_inputs(model_file)
        )
    else:
        _estimate_with_blend(model_file, test_file, output_file, taskset=3)


def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """
    Predict p(z) on test_file with the model staged in model_file and write a
    qp Ensemble with object_id + zmode ancillary data to output_file: a plain
    single FlexZBoost model through the fzb_taskset4 test's own route, on the
    shipped columns, and any other file through the blend route, which
    predicts with each member of a taskset-4 blend, mixes their curves at the
    members' weights, and refuses a file that is not one.

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
    if _is_plain_flexzboost_model(model_file):
        _estimate_with_plain_model(model_file, test_file, output_file, input_set=())
    else:
        _estimate_with_blend(model_file, test_file, output_file, taskset=4)


# No training-and-estimation function is supplied for taskset 3 or taskset 4,
# and ``run_taskset_3`` and ``run_taskset_4`` are called with None for it
# below. The challenge's own runners take None and simply record no outputs_3
# entry for a submission that gives one (``src/pz_data_challenge/taskset_3.py``
# and ``src/pz_data_challenge/taskset_4.py``).
#
# Each blend is two models: a FlexZBoost model built per training file at
# model settings chosen from held-out readings (on taskset 3 the settings
# picked from taskset 3's own settings searches, on the shipped inputs with no
# data augment; on taskset 4 with any input constructions and data augments a
# selection joined), and RAIL's k-nearest-neighbour estimator, mixed at
# weights fitted on held-out galaxies; each plain model is one such FlexZBoost
# model alone. Reproducing that from the public training file means running
# the label preparation, the augment methods and fresh fits of every model of
# the eight datasets inside this test. Two facts rule it out here. A
# FlexZBoost fit is tens of minutes of CPU on the full training file, so eight
# of them plus the augmentation are more than a CI job. The label preparation
# and the augment methods are implementations verified against their founding
# papers, and a second, inlined copy of them in this file could drift from the
# verified one and would make the submission's own claim about its methods
# unprovable. The fitted models themselves ship in the tar and this file
# predicts with them, which is what ``run_taskset_3_estimation_only`` and
# ``run_taskset_4_estimation_only`` above do on every one of the eight.
run_taskset_3_training_and_estimation = None
run_taskset_4_training_and_estimation = None


@pytest.mark.skip(
    reason="Tasksets 1 and 2 are not part of this submission; only the "
    "taskset 3 and 4 run functions above are provided."
)
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
        None,
        None,
    )


@pytest.mark.skip(
    reason="Tasksets 1 and 2 are not part of this submission; only the "
    "taskset 3 and 4 run functions above are provided."
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
        None,
        None,
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
