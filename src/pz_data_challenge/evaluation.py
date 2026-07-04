#mariano.dominguez@unc.edu.ar july 2026
"""
Visualization and analysis utilities for astronomical survey submission results.

This module provides functions for loading, processing, and visualizing performance
metrics from multiple algorithm submissions across different tasksets, scenarios,
and simulations. It includes functionality for creating strip plots, Q-Q plots,
and performance comparison visualizations.
"""

import os
from typing import Any
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tables_io

Y_LABEL_STRINGS = [
    "taskset_1_1yr_cardinal",
    "taskset_1_1yr_flagship",
    "taskset_1_10yr_cardinal",
    "taskset_1_10yr_flagship",
    "taskset_2_1yr_cardinal",
    "taskset_2_1yr_flagship",
    "taskset_2_10yr_cardinal",
    "taskset_2_10yr_flagship",
]


def get_submissions(accept_dir: str = "../accepted") -> list[str]:
    """
    Retrieve list of submission names from test files in the accepted directory.

    Scans the specified directory for Python test files matching the pattern
    'test_*.py' and extracts submission identifiers.

    Parameters
    ----------
    accept_dir
        Path to directory containing accepted submission test files.
        Default is "../accepted".

    Returns
    -------
    submissions
        List of submission identifier strings extracted from filenames.

    Examples
    --------
    >>> submissions = get_submissions("./accepted")
    >>> print(submissions)
    ['baseline', 'improved_algo', 'fast_estimator']
    """
    test_string = f"{accept_dir}/test_"
    submissions = [
        f.replace(test_string, "").replace(".py", "")
        for f in glob.glob(f"{test_string}*.py")
    ]
    return submissions


def build_summary_data_dict(
    results_dir: str,
    submissions: list[str] | None = None,
    summary_type: str = "point",
) -> dict[str, Any]:
    """
    Load summary data files for all submissions into a dictionary.

    Reads Parquet files containing summary statistics for each submission
    and organizes them into a dictionary keyed by submission name.

    Parameters
    ----------
    results_dir
        Directory path containing summary result files.
    submissions
        List of submission identifiers to load. If None, automatically
        discovers submissions using get_submissions().
    summary_type
        Type of summary data to load (e.g., "point", "pit-qq", "timing").
        Default is "point".

    Returns
    -------
    data_dict
        Dictionary mapping submission names to their corresponding data tables.

    Examples
    --------
    >>> data = build_summary_data_dict("./results", summary_type="point")
    >>> print(data.keys())
    dict_keys(['baseline', 'improved_algo'])
    """
    if submissions is None:
        submissions = get_submissions()

    data_dict = {}
    for sub_ in submissions:
        submission_file = os.path.join(
            results_dir, f"summary_{sub_}_{summary_type}.parq"
        )
        try:
            data_dict[sub_] = tables_io.read(submission_file)
        except Exception as exc:
            print(f"Skipping {sub_}_{summary_type} because {exc}")
            pass
    return data_dict


def get_metric_summary_dict(
    data_dict: dict[str, Any],
    submissions: list[str],
    metric: str,
) -> dict[str, tuple]:
    """
    Extract specific metric values across all submissions.

    Filters data for task 1 and computes run identifiers based on taskset,
    scenario, and simulation parameters.

    Parameters
    ----------
    data_dict
        Dictionary of submission data tables.
    submissions
        List of submission identifiers to process.
    metric
        Name of the metric column to extract.

    Returns
    -------
    out_dict
        Dictionary mapping submission names to tuples of (metric_values, run_ids).
        Run IDs are computed as: 4*(taskset-1) + 2*(scenario-1) + (sim-1).

    Examples
    --------
    >>> metric_dict = get_metric_summary_dict(data, subs, 'bias')
    >>> values, runs = metric_dict['baseline']
    """
    out_dict: dict[str, tuple] = {}
    for sub_ in submissions:
        try:
            task1_mask = data_dict[sub_]["task"] == 1
        except KeyError:
            continue
        run_ = (
            4 * (data_dict[sub_]["taskset"] - 1)
            + 2 * (data_dict[sub_]["scenario"] - 1)
            + (data_dict[sub_]["sim"] - 1)
        )
        out_dict[sub_] = (data_dict[sub_][metric][task1_mask], run_[task1_mask])
    return out_dict


def get_metric_summary_dict_multi(
    data_dict: dict[str, Any],
    submissions: list[str],
    metric: str,
) -> dict[str, tuple]:
    """
    Extract specific metric values across all tasks for all submissions.

    Computes comprehensive run identifiers incorporating taskset, scenario,
    simulation, and task parameters.

    Parameters
    ----------
    data_dict
        Dictionary of submission data tables.
    submissions
        List of submission identifiers to process.
    metric
        Name of the metric column to extract.

    Returns
    -------
    out_dict
        Dictionary mapping submission names to tuples of (metric_values, run_ids).
        Run IDs are computed as: 16*(taskset-1) + 8*(scenario-1) + 4*(sim-1) + (task-1).

    Examples
    --------
    >>> metric_dict = get_metric_summary_dict_multi(data, subs, 'accuracy')
    >>> values, runs = metric_dict['improved_algo']
    """
    out_dict: dict[str, tuple] = {}
    for sub_ in submissions:
        try:
            run_ = (
                16 * (data_dict[sub_]["taskset"] - 1)
                + 8 * (data_dict[sub_]["scenario"] - 1)
                + 4 * (data_dict[sub_]["sim"] - 1)
                + (data_dict[sub_]["task"] - 1)
            )
        except KeyError:
            continue
        out_dict[sub_] = (data_dict[sub_][metric], run_)
    return out_dict


def make_algo_estimate_time_strip_plot(
    data: dict[str, Any],
    submissions: list[str],
    metric_label: str = "Estimation time [ms/object]",
    metric_limits: list[float] = [1e-2, 1e3],
    metric_ranges: list[list[float]] = [[1e-2, 1], [1e-2, 5], [1e-2, 20]],
) -> Figure:
    """
    Create a strip plot showing estimation time per object using task 2 timing.

    Generates a horizontal strip plot with submissions on the y-axis and
    estimation times on a logarithmic x-axis. Shows mean and standard deviation
    across measurements.

    Parameters
    ----------
    data
        Dictionary of submission data tables.
    submissions
        List of submission identifiers to plot.
    metric_label
        Label for the x-axis. Default is "Estimation time [ms/object]".
    metric_limits
        X-axis limits as [min, max]. Default is [1e-2, 1e3].
    metric_ranges
        List of [min, max] ranges to highlight with gray shading.
        Default is [[1e-2, 1], [1e-2, 5], [1e-2, 20]].

    Returns
    -------
    fig
        Matplotlib Figure object containing the strip plot.

    Notes
    -----
    Times are normalized by dividing by 20k (objects per measurement).
    """
    fig = plt.figure()

    n_sub = len(submissions)
    y_min = -0.5
    y_max = n_sub - 0.5
    for i_sub, sub_ in enumerate(submissions):
        try:
            task2_mask = data[sub_]["task"] == 2
            times = data[sub_]["time"][task2_mask] / 20
        except KeyError:
            continue
        mean_task_2 = np.mean(times)
        std_task_2 = np.std(times)
        _ = plt.errorbar(
            mean_task_2, i_sub, xerr=std_task_2, label=sub_, ls="", marker="."
        )

    _ = plt.yticks(np.linspace(0, n_sub - 1, n_sub), submissions)
    _ = plt.xlabel(metric_label)
    _ = plt.xlim(metric_limits)
    _ = plt.ylim(y_min, y_max)
    _ = plt.legend()

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            metric_range, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.1
        )
    _ = plt.xscale("log")

    plt.tight_layout()
    return fig


def make_algo_inform_time_strip_plot(
    data: dict[str, Any],
    submissions: list[str],
    metric_label: str = "Inform time [s]",
    metric_limits: list[float] = [10, 1e4],
    metric_ranges: list[list[float]] = [[1, 60], [1, 300], [1, 1800]],
) -> Figure:
    """
    Create a strip plot showing inform time for submissions.

    Calculates inform time as the difference between task 3 and task 2 execution
    times, with a minimum floor of 10 seconds.

    Parameters
    ----------
    data
        Dictionary of submission data tables.
    submissions
        List of submission identifiers to plot.
    metric_label
        Label for the x-axis. Default is "Inform time [s]".
    metric_limits
        X-axis limits as [min, max]. Default is [10, 1e4].
    metric_ranges
        List of [min, max] ranges to highlight with gray shading.
        Default is [[1, 60], [1, 300], [1, 1800]].

    Returns
    -------
    fig
        Matplotlib Figure object containing the strip plot.

    Notes
    -----
    Error bars are floored at 10 seconds to ensure visibility.
    X-axis uses logarithmic scale.
    """
    fig = plt.figure()

    n_sub = len(submissions)
    y_min = -0.5
    y_max = n_sub - 0.5

    for i_sub, sub_ in enumerate(submissions):
        try:
            task2_mask = data[sub_]["task"] == 2
            task3_mask = data[sub_]["task"] == 3
        except KeyError:
            continue

        mean_task_2 = np.mean(data[sub_]["time"][task2_mask])
        mean_task_3 = np.mean(data[sub_]["time"][task3_mask])

        std_task_3 = np.std(data[sub_]["time"][task3_mask])

        inform_time = max(mean_task_3 - mean_task_2, 10)

        _ = plt.errorbar(
            inform_time, i_sub, xerr=max(std_task_3, 10), label=sub_, ls="", marker="."
        )

    _ = plt.yticks(np.linspace(0, n_sub - 1, n_sub), submissions)
    _ = plt.xlabel("Inform time [s]")
    _ = plt.ylim(y_min, y_max)
    _ = plt.xlim(metric_limits)
    _ = plt.legend()

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            metric_range, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.1
        )
    _ = plt.xscale("log")

    plt.tight_layout()
    return fig


def make_strip_plot(
    data: dict[str, Any],
    metric_label: str,
    metric_limits: list[float],
    metric_ranges: list[list[float]],
    y_label_strings: list[str] = Y_LABEL_STRINGS,
) -> Figure:
    """
    Create a generic strip plot for comparing metrics across configurations.

    Displays scatter points for each submission across different taskset/scenario
    combinations with highlighted metric ranges.

    Parameters
    ----------
    data
        Dictionary mapping submission names to (values, run_ids) tuples.
    metric_label
        Label for the x-axis describing the metric.
    metric_limits
        X-axis limits as [min, max].
    metric_ranges
        List of [min, max] ranges to highlight with gray shading.
    y_label_strings
        Labels for y-axis ticks corresponding to different configurations.
        Default is Y_LABEL_STRINGS.

    Returns
    -------
    fig
        Matplotlib Figure object containing the strip plot.

    Examples
    --------
    >>> fig = make_strip_plot(metric_data, "Bias", [-0.1, 0.1],
    ...                       [[-0.02, 0.02], [-0.05, 0.05]])
    """
    fig = plt.figure()

    n_y_labels = len(y_label_strings)
    y_min = -0.5
    y_max = n_y_labels - 0.5

    for key, val in data.items():
        plt.scatter(val[0], val[1], label=key)

    _ = plt.yticks(np.linspace(0, n_y_labels - 1, n_y_labels), y_label_strings)
    _ = plt.xlabel(metric_label)
    _ = plt.ylim(y_min, y_max)
    _ = plt.xlim(metric_limits)
    _ = plt.legend()

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            metric_range, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.1
        )

    plt.tight_layout()
    return fig


def make_strip_plot_multi(
    data: dict[str, Any],
    metric_label: str,
    metric_limits: list[float],
    metric_ranges: list[list[float]],
    y_label_strings: list[str] = Y_LABEL_STRINGS,
) -> Figure:
    """
    Create an extended strip plot for multiple tasks per configuration.

    Similar to make_strip_plot but with expanded y-axis to accommodate
    multiple tasks (4x spacing) for each configuration.

    Parameters
    ----------
    data
        Dictionary mapping submission names to (values, run_ids) tuples.
    metric_label
        Label for the x-axis describing the metric.
    metric_limits
        X-axis limits as [min, max].
    metric_ranges
        List of [min, max] ranges to highlight with gray shading.
    y_label_strings
        Labels for y-axis ticks corresponding to different configurations.
        Default is Y_LABEL_STRINGS.

    Returns
    -------
    fig
        Matplotlib Figure object containing the multi-task strip plot.

    Notes
    -----
    Y-axis spacing is 4x larger than make_strip_plot to separate multiple
    tasks per configuration.
    """
    fig = plt.figure()

    n_y_labels = len(y_label_strings)
    y_min = -0.5
    y_max = 4 * (n_y_labels) - 0.5

    for key, val in data.items():
        plt.scatter(val[0], val[1], label=key)

    _ = plt.yticks(np.linspace(1, 4 * (n_y_labels - 1), n_y_labels), y_label_strings)
    _ = plt.xlabel(metric_label)
    _ = plt.ylim(y_min, y_max)
    _ = plt.xlim(metric_limits)
    _ = plt.legend()

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            metric_range, [y_min, y_min], [y_max, y_max], color="gray", alpha=0.1
        )

    plt.tight_layout()
    return fig


def make_qq_pit_plot(
    data_dict: dict[str, Any],
    submissions: list[str],
) -> Figure:
    """
    Create a Q-Q plot using Probability Integral Transform (PIT) values.

    Generates quantile-quantile plots comparing estimated cumulative distributions
    to reference values for task 1, scenario 1, simulation 1.

    Parameters
    ----------
    data_dict
        Dictionary of submission data tables containing 'x_vals' and 'y_vals'
        columns with histogram data.
    submissions
        List of submission identifiers to plot.

    Returns
    -------
    fig
        Matplotlib Figure object (8x8 inches) containing the Q-Q plot.

    Notes
    -----
    - Only plots data for task=1, scenario=1, sim=1
    - Y-values are clipped to [0, 5] before computing CDF
    - Different line styles indicate different tasksets
    - Colors distinguish different submissions
    - Perfect calibration would follow the diagonal line
    """
    fig = plt.figure()

    lines = ["", "-", "dashed"]
    fig = plt.figure()
    for sub_ in submissions:
        if sub_ not in data_dict:
            continue
        for i_row, row_ in data_dict[sub_].iterrows():
            if row_["task"] != 1:
                continue
            if row_["scenario"] != 1:
                continue
            if row_["sim"] != 1:
                continue
            cdf = np.cumsum(row_["y_vals"].clip(0, 5))
            plt.plot(
                row_["x_vals"],
                cdf / cdf[-1],
                ls=lines[row_["taskset"]],
                label=f"{sub_} {row_['taskset']}",
            )
    _ = plt.plot([0, 1], [0, 1])
    _ = plt.xlim(0, 1)
    _ = plt.ylim(0, 1)
    _ = plt.xlabel("Q")
    _ = plt.ylabel(r"$p(z_{\rm ref} < z(Q))$")
    _ = plt.legend()

    plt.tight_layout()
    return fig


def make_point_v_redshift_plot(
    data_dict: dict[str, Any],
    submissions: list[str],
    label: str,
    metric: str,
    metric_limits: list[float],
    metric_ranges: list[list[float]],
) -> Figure:
    """
    Create a plot showing metric performance as a function of redshift.

    Generates line plots comparing algorithm performance across different
    redshift bins for task 1 (excluding scenario 1, simulation 1).

    Parameters
    ----------
    data_dict
        Dictionary of submission data tables containing redshift ('z_mean')
        and metric columns.
    submissions
        List of submission identifiers to plot.
    label
        Description label (currently unused in function).
    metric
        Name of the metric column to plot on y-axis.
    metric_limits
        Y-axis limits as [min, max].
    metric_ranges
        List of [min, max] ranges to highlight with horizontal gray shading
        across the full redshift range.

    Returns
    -------
    fig
        Matplotlib Figure object containing the redshift vs. metric plot.

    Notes
    -----
    - Only plots task=1 data, excluding scenario=1 and sim=1
    - Redshift range is fixed at [0, 2.5]
    - Different line styles indicate different tasksets
    - Colors distinguish different submissions
    - Y-axis label is hardcoded for photometric redshift bias
    """
    fig = plt.figure()

    lines = ["", "-", "dashed"]
    colors = ["blue", "orange", "green", "red"]
    for i_sub, sub_ in enumerate(submissions):
        task1_mask = data_dict[sub_]["task"] == 1
        for i_row, row_ in data_dict[sub_][task1_mask].iterrows():
            if row_["task"] != 1:
                continue
            if row_["scenario"] == 1:
                continue
            if row_["sim"] == 1:
                continue
            plt.plot(
                row_["z_mean"],
                row_[metric],
                ls=lines[row_["taskset"]],
                c=colors[i_sub],
                label=f"{sub_} {row_['taskset']}",
            )
    _ = plt.legend()
    _ = plt.xlim(0, 2.5)
    _ = plt.ylim(metric_limits)
    _ = plt.xlabel(r"$z_{\rm ref}$")
    _ = plt.ylabel(label)

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            [0.0, 2.5],
            [metric_range[0], metric_range[0]],
            [metric_range[1], metric_range[1]],
            color="gray",
            alpha=0.1,
        )

    plt.tight_layout()
    return fig


def make_point_v_mag_plot(
    data_dict: dict[str, Any],
    submissions: list[str],
    label: str,
    metric: str,
    metric_limits: list[float],
    metric_ranges: list[list[float]],
) -> Figure:
    """
    Create a plot showing metric performance as a function of magnitdue.

    Generates line plots comparing algorithm performance across different
    redshift bins for task 1 (excluding scenario 1, simulation 1).

    Parameters
    ----------
    data_dict
        Dictionary of submission data tables containing redshift ('z_mean')
        and metric columns.
    submissions
        List of submission identifiers to plot.
    label
        Description label (currently unused in function).
    metric
        Name of the metric column to plot on y-axis.
    metric_limits
        Y-axis limits as [min, max].
    metric_ranges
        List of [min, max] ranges to highlight with horizontal gray shading
        across the full redshift range.

    Returns
    -------
    fig
        Matplotlib Figure object containing the redshift vs. metric plot.

    Notes
    -----
    - Only plots task=1 data, excluding scenario=1 and sim=1
    - Mag range is fixed at [20, 24.5]
    - Different line styles indicate different tasksets
    - Colors distinguish different submissions
    - Y-axis label is hardcoded for photometric redshift bias
    """
    fig = plt.figure()

    lines = ["", "-", "dashed"]
    colors = ["blue", "orange", "green", "red"]
    for i_sub, sub_ in enumerate(submissions):
        task1_mask = data_dict[sub_]["task"] == 1
        for i_row, row_ in data_dict[sub_][task1_mask].iterrows():
            if row_["task"] != 1:
                continue
            if row_["scenario"] == 1:
                continue
            if row_["sim"] == 1:
                continue
            plt.plot(
                row_["i_mag"],
                row_[metric],
                ls=lines[row_["taskset"]],
                c=colors[i_sub],
                label=f"{sub_} {row_['taskset']}",
            )
    _ = plt.legend()
    _ = plt.xlim(20, 24.5)
    _ = plt.ylim(metric_limits)
    _ = plt.xlabel(r"$z_{\rm ref}$")
    _ = plt.ylabel(label)

    for metric_range in metric_ranges:
        _ = plt.fill_between(
            [20.0, 24.5],
            [metric_range[0], metric_range[0]],
            [metric_range[1], metric_range[1]],
            color="gray",
            alpha=0.1,
        )

    plt.tight_layout()
    return fig
