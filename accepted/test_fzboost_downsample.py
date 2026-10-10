import math
import os
from pathlib import Path

import numpy as np
import pytest
import tables_io
from rail.core.data import TableHandle
from rail.estimation.algos.flexzboost import FlexZBoostEstimator, FlexZBoostInformer
from rail.utils import catalog_utils

from pz_data_challenge import submit_utils
from pz_data_challenge.taskset_1 import run_taskset_1
from pz_data_challenge.taskset_2 import run_taskset_2
from pz_data_challenge.taskset_3 import run_taskset_3
from pz_data_challenge.taskset_4 import run_taskset_4

# Change these to match the name of the submission
# and a URL to download the sumission data files
# and needed model files
SUBMISSION_NAME: str = "fzboost_downsample"
SUBMISSION_URL: str = (
    "https://portal.nersc.gov/cfs/lsst/tqzhang/submit_fzboost_downsample.tgz"
)

# don't change these
SUBMIT_DIR: str = f"submissions/{SUBMISSION_NAME}"
PUBLIC_AREA: str = "tests/public"

# Catalog tag mapping the Rubin (ugrizy) + Roman (YJH) bands to columns.
CATALOG_TAG = "cardinal_roman_rubin"

# FlexZBoost configuration shared by the informer and estimator.
_ZMIN = 0.0
_ZMAX = 6.0
_NZ = 601
_ZGRID = np.linspace(_ZMIN, _ZMAX, _NZ)

# Point estimates to attach to the output: the PDF mode and the risk-minimizing
# "best" estimate.  These names are the tokens understood by RAIL's
# PointEstimationMixin, which writes them to ancil under the same names.
_POINT_ESTIMATES = ["zmode", "zbest"]

# Magnitude-redshift downsampling.  The reference samples pile labels up at
# bright magnitudes and particular redshifts; training on that imprints the
# selection on the posteriors as an effective prior.  Binning on a
# (magnitude, redshift) grid and capping every cell flattens the distribution
# without discarding the sparsely populated regions.
#
# Cell size follows the DESC dp2 hscfy_pz_prepare pipeline (d_mag = d_z = 0.1).
#
# _DOWNSAMPLE_CAP was chosen by sweeping {10, 15, 20, 25, 50, 75, 100, 150, 200,
# 300} and scoring each against the challenge's own metric tiers
# (pz_data_challenge.scoring.metric_dict) on a held-out COSMOS2020 benchmark --
# see nb/fzboost_downsample/downsample_benchmark.ipynb.  Summed over six
# taskset x (sim, scenario) corners, splitting the tiers into point-estimate
# metrics (mean, std, abs_outlier_rate; 18 max each) and distribution metrics
# (CvM, ks, ksamp, outlier):
#
#     cap      point   calib   total
#     none        35      57      92     <- no downsampling
#     10          33      69     102
#     25          36      64     100
#     50+         35      59      94
#
# 25 is the only setting that improves on *both* axes.  Caps below it buy
# calibration by giving back point-estimate accuracy -- cap 10 scores 2 points
# higher in total but drops the point tier below no-downsampling at all, which
# is the wrong trade for a photo-z estimate.  Caps of 50 and above barely bite
# (the busiest cell holds 310-816 objects, so 300 is nearly a no-op).
#
# The totals alone do not separate 25 from 10: re-running cap 25 at downsampling
# seeds 43 and 44 moves the score by 1 point on a 63-point subset, which rescales
# to about the same 2 points.  The choice rests on the point/calibration split
# above, which is structural, not on the aggregate margin.
_D_MAG = 0.1
_D_Z = 0.1
_DOWNSAMPLE_CAP = 25
_DOWNSAMPLE_SEED = 42


def _attach_ancil(estimator: FlexZBoostEstimator, pz_out, test_data: TableHandle) -> None:
    """Attach the point estimates and object_id that the validator requires.

    FlexZBoost's ``_process_chunk`` builds its ancil by hand and only recognises
    the tokens ``mode``/``mean``/``median``; it never calls the
    PointEstimationMixin, so ``calculated_point_estimates=["zmode", "zbest"]``
    alone would leave the ancil empty.  Invoke the mixin explicitly on the
    finished ensemble to populate ``zmode`` and ``zbest``.
    """
    ensemble = estimator.calculate_point_estimates(pz_out.data, grid=_ZGRID)
    object_id = dict(object_id=np.asarray(test_data()["object_id"]))
    if ensemble.ancil is None:  # pragma: no cover
        ensemble.set_ancil(object_id)
    else:
        ensemble.add_to_ancil(object_id)


def _load_training_data(train_file: str | Path) -> dict[str, np.ndarray]:
    """Read a training table and give every row a usable redshift label.

    Taskset 3 and 4 training files leave ``redshift`` NaN for objects with no
    spectroscopic follow-up and supply a COSMOS2020 narrow-band photometric
    redshift in ``redshift_manyband`` instead.  That is 35-54% of the rows, and
    they are not a random subset: they are the faint end (``mag_i`` median 24.36
    against 21.73 for the spectroscopic sample, matching the test set's 24.35)
    and they carry the only labels above z ~ 2.3 -- the spectroscopic redshifts
    stop there, while the narrow-band ones reach z ~ 6.  Dropping them therefore
    threw away both the magnitude regime the test set lives in and the entire
    upper half of the p(z) grid.

    So fill from the narrow-band column wherever the spectroscopic redshift is
    missing, and drop only what is still unlabelled.  Taskset 1 and 2 files have
    no ``redshift_manyband`` column and no NaN redshifts, so this is a no-op for
    them -- hence the membership guard rather than a bare lookup.

    The table is handed to the informer in memory (``inform()`` accepts a plain
    dict of columns), so nothing is written back beside the input and the public
    data area is left exactly as downloaded.
    """
    data = tables_io.read(str(train_file))
    # copy, so the combined label never mutates the table we were handed
    redshift = np.asarray(data["redshift"], dtype="float64")
    if "redshift_manyband" in data:
        manyband = np.asarray(data["redshift_manyband"], dtype="float64")
        redshift = np.where(np.isnan(redshift), manyband, redshift)
    good = np.isfinite(redshift)
    out = {key: np.asarray(val)[good] for key, val in data.items()}
    out["redshift"] = redshift[good]
    return out


def _bin_cap_mask(
    mag: np.ndarray,
    redshift: np.ndarray,
    cap: int,
    d_mag: float,
    d_z: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean mask keeping at most ``cap`` rows per (magnitude, redshift) cell.

    A direct port of ``MagRedshiftDownsampler`` from the DESC dp2
    ``rail.creation.degraders.pz_prepare`` pipeline, written out in numpy so
    this submission needs no dependency beyond what the base one already has.

    Bins are origin-anchored (``floor(x / d)``), so a cell is
    ``[k*d, (k+1)*d)``.  Cells at or below the cap are kept whole -- the cap is
    a ceiling, never a target, and nothing is ever upsampled.  Rows with a
    non-finite magnitude or redshift cannot be binned, so they are kept and do
    not count against any cell's quota; dropping them here would be a silent
    photometric cut rather than a downsampling.
    """
    keep = np.zeros(len(mag), dtype=bool)
    finite = np.isfinite(mag) & np.isfinite(redshift)
    keep[~finite] = True

    idx = np.where(finite)[0]
    if idx.size == 0:
        return keep

    mag_bin = np.floor(mag[idx] / d_mag).astype(np.int64)
    z_bin = np.floor(redshift[idx] / d_z).astype(np.int64)
    # lexsort's last key is primary, so this groups by (mag_bin, z_bin);
    # it is stable, so within a cell the rows stay in input order.
    order = np.lexsort((z_bin, mag_bin))
    idx, mag_bin, z_bin = idx[order], mag_bin[order], z_bin[order]

    new_cell = np.empty(len(idx), dtype=bool)
    new_cell[0] = True
    new_cell[1:] = (mag_bin[1:] != mag_bin[:-1]) | (z_bin[1:] != z_bin[:-1])
    starts = np.where(new_cell)[0]
    ends = np.append(starts[1:], len(idx))

    for start, end in zip(starts, ends):
        members = idx[start:end]
        if len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        keep[members] = True
    return keep


def _downsample(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Flatten the magnitude-redshift distribution of a training table."""
    keep = _bin_cap_mask(
        np.asarray(data["mag_i_lsst"], dtype="float64"),
        np.asarray(data["redshift"], dtype="float64"),
        _DOWNSAMPLE_CAP,
        _D_MAG,
        _D_Z,
        np.random.default_rng(_DOWNSAMPLE_SEED),
    )
    return {key: val[keep] for key, val in data.items()}


def _make_fzb_informer() -> FlexZBoostInformer:
    """Stock FlexZBoost: every tuning parameter left at its RAIL default.

    This submission is a benchmark, so trainfrac, seed, max_basis and the
    bump/sharpen grid searches are all left as FlexZBoostInformer ships them.
    Only the file layout (hdf5_groupname), the challenge's NaN non-detection
    convention and the output p(z) grid are specified.
    """
    return FlexZBoostInformer.make_stage(
        name="inform",
        hdf5_groupname="",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
    )


def _make_fzb_estimator(model) -> FlexZBoostEstimator:
    # qp_representation defaults to "interp" (qp.interp); use the default.
    return FlexZBoostEstimator.make_stage(
        name="estimate",
        model=model,
        hdf5_groupname="",
        output_mode="return",
        nondetect_val=math.nan,
        zmin=_ZMIN,
        zmax=_ZMAX,
        nzbins=_NZ,
        calculated_point_estimates=_POINT_ESTIMATES,
    )


@pytest.fixture(name="setup_submit_area", scope="module")
def setup_submit_area(request: pytest.FixtureRequest) -> int:
    """
    A pytest fixture to download the submission data

    If all the submission data are in a tar file with the
    proper structure you should not need to change this function.
    """

    if not os.path.exists(SUBMIT_DIR):
        if not SUBMISSION_URL:
            raise ValueError(
                f"SUBMISSION_URL in tests/test_{SUBMISSION_NAME}.py has not been set"
            )
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

    catalog_utils.clear()
    catalog_utils.load_yaml("tests/catalogs.yaml")
    catalog_utils.apply(CATALOG_TAG)

    return 0


def run_taskset_x_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    """Load a pre-trained FlexZBoost model and estimate p(z) for the test set."""
    test_data = TableHandle("test", path=str(test_file))
    estimator = _make_fzb_estimator(str(model_file))
    pz_out = estimator.estimate(test_data)
    _attach_ancil(estimator, pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_x_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
    *,
    downsample: bool,
) -> None:
    """Train a FlexZBoost model on the training set and estimate p(z) for the test set."""
    train_data = _load_training_data(train_file)
    if downsample:
        train_data = _downsample(train_data)
    test_data = TableHandle("test", path=str(test_file))
    informer = _make_fzb_informer()
    model = informer.inform(train_data)
    estimator = _make_fzb_estimator(model)
    pz_out = estimator.estimate(test_data)
    _attach_ancil(estimator, pz_out, test_data)
    pz_out.path = output_file
    pz_out.write()


def run_taskset_1_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_1_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    # Taskset 1 is the representative case -- its training and test sets are
    # drawn from the same distribution (mag_i median 22.10 for both), so there
    # is nothing to flatten and downsampling would only move the training set
    # away from the test set while discarding rows.
    run_taskset_x_training_and_estimation(
        train_file, test_file, output_file, downsample=False
    )


def run_taskset_2_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_2_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    # Non-representative training set: flatten it.
    run_taskset_x_training_and_estimation(
        train_file, test_file, output_file, downsample=True
    )


def run_taskset_3_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_3_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    # Non-representative training set: flatten it.
    run_taskset_x_training_and_estimation(
        train_file, test_file, output_file, downsample=True
    )


def run_taskset_4_estimation_only(
    model_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    run_taskset_x_estimation_only(model_file, test_file, output_file)


def run_taskset_4_training_and_estimation(
    train_file: str | Path,
    test_file: str | Path,
    output_file: str | Path,
) -> None:
    # Non-representative training set: flatten it.
    run_taskset_x_training_and_estimation(
        train_file, test_file, output_file, downsample=True
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
