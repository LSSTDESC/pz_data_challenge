import os

from typing import Any
import numpy as np
import tables_io
import qp

import matplotlib.pyplot as plt
from rail.plotting.plot_holder import RailPlotHolder
from rail.plotting import pz_plotters, pz_dist_plotters


def get_truth_and_qp_ensemble(
    datadir: str,
    submission_dir: str,
    taskset: str,
    sim: str,
    scenario: str,
) -> dict[str, Any]:

    data_dict: dict[str, Any] = {}
    key = f"{taskset}_{sim}_{scenario}"
    
    test_file = os.path.abspath(os.path.join(datadir, f"pz_challenge_{taskset}_{sim}_training_{scenario}.hdf5"))
    validate_file = os.path.abspath(os.path.join(submission_dir, f"pz_challenge_{taskset}_{sim}_pz_validate_{scenario}.hdf5"))

    truth = tables_io.read(test_file)['redshift']
    qp_ensmble = qp.read(validate_file)

    data_dict[f"{key}_test"] = tables_io.read(test_file)
    data_dict[f"{key}_validate"] = qp.read(validate_file)            

    return data_dict


def point_metrics_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any
) -> dict[str, RailPlotHolder]:

    point_plotter = pz_plotters.PZPlotterPointEstimateVsTrueHist2D(**kwargs)
    
    return point_plotter.run(        
        prefix,
        truth=test_data['redshift'],
        pointEstimate=np.squeeze(submit_data.ancil['zmode']),
        magnitude=test_data['mag_i_lsst'],
    )


def point_v_redshfit_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any
) -> dict[str, RailPlotHolder]:
    
    point_v_redshfit_plotter = pz_plotters.PZPlotterBiweightStatsVsRedshift(**kwargs)

    return point_v_redshfit_plotter.run(        
        prefix,
        truth=test_data['redshift'],
        pointEstimate=np.squeeze(submit_data.ancil['zmode']),
        magnitude=test_data['mag_i_lsst'],
    )


def point_v_mag_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any
) -> dict[str, RailPlotHolder]:

    point_v_mag_plotter = pz_plotters.PZPlotterBiweightStatsVsRedshift(**kwargs)
    
    return point_v_mag_plotter.run(        
        prefix,
        truth=test_data['redshift'],
        pointEstimate=np.squeeze(submit_data.ancil['zmode']),
        magnitude=test_data['mag_i_lsst'],
    )


def plot_pit_prob_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any
) -> dict[str, RailPlotHolder]:

    pit_prob_plotter = pz_dist_plotters.PZPlotterPITProb(**kwargs)
    
    return pit_prob_plotter.run(
        prefix,
        truth=test_data['redshift'],
        pz=submit_data,
    )


def plot_pit_qq_plot(
    prefix: str,
    test_data: dict[str, np.ndarray],
    submit_data: qp.Ensemble,
    **kwargs: Any
) -> dict[str, RailPlotHolder]:

    pit_qq_plotter = pz_dist_plotters.PZPlotterPITQQ(**kwargs)
    
    return pit_qq_plotter.run(
        prefix,
        truth=test_data['redshift'],
        pz=submit_data,
    )
