"""Metric computation and plotting for photometric redshift estimates.

This module provides functions for loading truth/submission data and
generating diagnostic plots (point estimates, PIT, Q-Q) using RAIL
plotting utilities.
"""

import os

from typing import Any
import numpy as np
import tables_io
import qp

from rail.plotting.plot_holder import RailPlotHolder
from rail.plotting import pz_plotters, pz_dist_plotters


def get_truth_and_qp_ensemble(
    datadir: str,
    submission_dir: str,
    taskset: str,
    sim: str,
    scenario: str,
    test_label: str = "training",
    eval_label: str = "pz_evaluation",
) -> dict[str, Any]:
    """Load truth table and qp ensemble for a given task configuration.

    Parameters
    ----------
    datadir : str
        Directory containing truth/test data files.
    submission_dir : str
        Directory containing submitted photo-z estimate files.
    taskset : str
        Taskset identifier (e.g. "taskset_1").
    sim : str
        Simulation name (e.g. "cardinal", "flagship").
    scenario : str
        Scenario identifier (e.g. "1yr", "10yr").
    test_label : str
        Label for the test data file. Default is "training".
    eval_label : str
        Label for the evaluation/submission file. Default is "pz_evaluation".

    Returns
    -------
    dict[str, Any]
        Dictionary with keys "{taskset}_{sim}_{scenario}_test" mapping to the
        truth table and "{taskset}_{sim}_{scenario}_evaluate" mapping to the
        qp ensemble.
    """
    data_dict: dict[str, Any] = {}
    key = f"{taskset}_{sim}_{scenario}"

    test_file = os.path.abspath(
        os.path.join(
            datadir, f"pz_challenge_{taskset}_{sim}_{test_label}_{scenario}.hdf5"
        )
    )
    validate_file = os.path.abspath(
        os.path.join(
            submission_dir, f"pz_challenge_{taskset}_{sim}_{eval_label}_{scenario}.hdf5"
        )
    )

    data_dict[f"{key}_test"] = tables_io.read(test_file)
    data_dict[f"{key}_evaluate"] = qp.read(validate_file)

    return data_dict


def get_z_point(submit_data: qp.Ensemble) -> np.ndarray:
    """Extract point redshift estimate from a qp ensemble.

    Parameters
    ----------
    submit_data : qp.Ensemble
        Submitted photo-z ensemble containing ancillary point estimates.

    Returns
    -------
    np.ndarray
        1-D array of point redshift estimates (zmode or z_mode).
    """
    try:
        return np.squeeze(submit_data.ancil["zmode"])
    except KeyError:
        return np.squeeze(submit_data.ancil["z_mode"])


def point_metrics_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any,
) -> dict[str, RailPlotHolder]:
    """Generate 2D histogram of point estimates vs true redshift.

    Parameters
    ----------
    prefix : str
        Prefix for plot output keys.
    test_data : dict[str, np.ndarray]
        Truth data containing 'redshift' and 'mag_i_lsst' arrays.
    submit_data : qp.Ensemble
        Submitted photo-z ensemble.
    **kwargs : Any
        Additional keyword arguments passed to the plotter.

    Returns
    -------
    dict[str, RailPlotHolder]
        Dictionary of named plot holders.
    """
    point_plotter = pz_plotters.PZPlotterPointEstimateVsTrueHist2D(**kwargs)

    return point_plotter.run(
        prefix,
        truth=test_data["redshift"],
        pointEstimate=get_z_point(submit_data),
        magnitude=test_data["mag_i_lsst"],
    )


def point_v_redshfit_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any,
) -> dict[str, RailPlotHolder]:
    """Generate biweight statistics vs redshift plot.

    Parameters
    ----------
    prefix : str
        Prefix for plot output keys.
    test_data : dict[str, np.ndarray]
        Truth data containing 'redshift' and 'mag_i_lsst' arrays.
    submit_data : qp.Ensemble
        Submitted photo-z ensemble.
    **kwargs : Any
        Additional keyword arguments passed to the plotter.

    Returns
    -------
    dict[str, RailPlotHolder]
        Dictionary of named plot holders.
    """
    point_v_redshfit_plotter = pz_plotters.PZPlotterBiweightStatsVsRedshift(**kwargs)

    return point_v_redshfit_plotter.run(
        prefix,
        truth=test_data["redshift"],
        pointEstimate=get_z_point(submit_data),
        magnitude=test_data["mag_i_lsst"],
    )


def point_v_mag_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any,
) -> dict[str, RailPlotHolder]:
    """Generate biweight statistics vs magnitude plot.

    Parameters
    ----------
    prefix : str
        Prefix for plot output keys.
    test_data : dict[str, np.ndarray]
        Truth data containing 'redshift' and 'mag_i_lsst' arrays.
    submit_data : qp.Ensemble
        Submitted photo-z ensemble.
    **kwargs : Any
        Additional keyword arguments passed to the plotter.

    Returns
    -------
    dict[str, RailPlotHolder]
        Dictionary of named plot holders.
    """
    point_v_mag_plotter = pz_plotters.PZPlotterBiweightStatsVsMag(**kwargs)

    return point_v_mag_plotter.run(
        prefix,
        truth=test_data["redshift"],
        pointEstimate=get_z_point(submit_data),
        magnitude=test_data["mag_i_lsst"],
    )


def plot_pit_prob_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any,
) -> dict[str, RailPlotHolder]:
    """Generate PIT probability histogram plot.

    Parameters
    ----------
    prefix : str
        Prefix for plot output keys.
    test_data : dict[str, np.ndarray]
        Truth data containing 'redshift' array.
    submit_data : qp.Ensemble
        Submitted photo-z ensemble.
    **kwargs : Any
        Additional keyword arguments passed to the plotter.

    Returns
    -------
    dict[str, RailPlotHolder]
        Dictionary of named plot holders, or empty dict on failure.
    """
    pit_prob_plotter = pz_dist_plotters.PZPlotterPITProb(**kwargs)

    try:
        return pit_prob_plotter.run(
            prefix,
            truth=test_data["redshift"],
            pz=submit_data,
        )
    except Exception:
        return {}


def plot_pit_qq_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any,
) -> dict[str, RailPlotHolder]:
    """Generate PIT Q-Q plot.

    Parameters
    ----------
    prefix : str
        Prefix for plot output keys.
    test_data : dict[str, np.ndarray]
        Truth data containing 'redshift' array.
    submit_data : qp.Ensemble
        Submitted photo-z ensemble.
    **kwargs : Any
        Additional keyword arguments passed to the plotter.

    Returns
    -------
    dict[str, RailPlotHolder]
        Dictionary of named plot holders, or empty dict on failure.
    """
    pit_qq_plotter = pz_dist_plotters.PZPlotterPITQQ(**kwargs)

    try:
        return pit_qq_plotter.run(
            prefix,
            truth=test_data["redshift"],
            pz=submit_data,
        )
    except Exception:
        return {}
