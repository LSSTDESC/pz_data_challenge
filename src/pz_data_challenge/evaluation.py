
import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import tables_io


def build_summary_data_dict(
    results_dir: str,
    submissions: list[str],
    summary_type: str,
) -> dict[str, Any]:    

    data_dict = {}
    for sub_ in submissions:
        submission_file = os.path.join(results_dir, f"summary_{sub_}_{summary_type}.parq")
        data_dict[sub_] = tables_io.read(submission_file)
    return data_dict


def make_algo_estimate_time_strip_plot(
    data: dict[str, Any],
    submissions: list[str],
    metric_label: str="Estimation time [ms/object]",
    metric_limits: list[float] = [1e-2, 1e3],
    metric_ranges: list[list[float]] = [[1e-2, 1], [1e-2, 5], [1e-2, 20]],
):

    n_sub = len(submissions)
    y_min = -0.5
    y_max = n_sub - 0.5
    for i_sub, sub_ in enumerate(submissions):
        task2_mask = data[sub_]['task'] == 2
        times = data[sub_]['time'][task2_mask]/20
        mean_task_2 = np.mean(times)
        std_task_2 = np.std(times)
    
        _ = plt.errorbar(mean_task_2, i_sub, xerr=std_task_2, label=sub_, ls="", marker=".")
    _ = plt.yticks(np.linspace(0, n_sub-1, n_sub), submissions)
    _ = plt.xlabel(metric_label)

    _ = plt.xlim(metric_limits)
    _ = plt.ylim(y_min, y_max)

    for metric_range in metric_ranges:
        _ = plt.fill_between(metric_range, [y_min, y_min], [y_max, y_max], color='gray', alpha=0.1)
    _ = plt.xscale('log')

    plt.tight_layout()
    return plt.gcf()
    

def make_algo_inform_time_strip_plot(
    data: dict[str, Any],
    submissions: list[str],
    metric_label: str="Inform time [s]",
    metric_limits: list[float] = [10, 1e4],
    metric_ranges: list[list[float]] = [[1, 60], [1, 300], [1, 1800]],
):

    n_sub = len(submissions)
    y_min = -0.5
    y_max = n_sub - 0.5
    
    estimate_only = {}
    inform_only = {}
    for i_sub, sub_ in enumerate(submissions):
        task2_mask = data[sub_]['task'] == 2 
        task3_mask = data[sub_]['task'] == 3

        mean_task_2 = np.mean(data[sub_]['time'][task2_mask])
        mean_task_3 = np.mean(data[sub_]['time'][task3_mask])

        std_task_3 = np.std(data[sub_]['time'][task3_mask])

        inform_time = max(mean_task_3-mean_task_2, 10)
    
        _ = plt.errorbar(inform_time, i_sub, xerr=max(std_task_3, 10), label=sub_, ls="", marker=".")

    _= plt.yticks(np.linspace(0, n_sub-1, n_sub), submissions)
    _ = plt.xlabel("Inform time [s]")
    _ = plt.ylim(y_min, y_max)

    _ = plt.xlim(metric_limits)
    for metric_range in metric_ranges:
        _ = plt.fill_between(metric_range, [y_min, y_min], [y_max, y_max], color='gray', alpha=0.1)
    _ = plt.xscale('log')

    plt.tight_layout()
    return plt.gcf()
    
    
def make_strip_plot(
    data: dict[str, Any],
    y_label_strings: list[str],
    metric_label: str,
    metric_limits: list[float],
    metric_ranges: list[list[float]],
):
    
    n_y_labels = len(y_label_strings)
    y_min = -0.5
    y_max = n_y_labels-0.5
    
    for key, val in data.items():
        plt.scatter(val[0], val[1], label=key)
        
    _ = plt.yticks(np.linspace(0, n_y_labels-1, n_y_labels), y_label_strings)
    _ = plt.xlabel(metric_label)
    _ = plt.ylim(y_min, y_max)
    _ = plt.xlim(metric_limits)
    
    for metric_range in metric_ranges:        
        _ = plt.fill_between(
            metric_range,
            [y_min, y_min],
            [y_max, y_max],
            color='gray',
            alpha=0.1
        )

    plt.tight_layout()
    return plt.gcf()

