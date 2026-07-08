"""
Scoring utilities for evaluating submission performance against metric thresholds.

This module provides functions for scoring algorithm submissions based on multiple
performance metrics across different tasksets. Scores are computed by checking if
metric values fall within predefined acceptable ranges.
"""

import numpy as np
import pandas as pd

metric_dict: dict[str, list[list[float]]] = dict(
    mean=[[-0.01, 0.01], [-0.025, 0.025], [-0.05, 0.05]],
    std=[[0.0, 0.025], [0.0, 0.050], [0.0, 0.10]],
    abs_outlier_rate=[[0.0, 0.025], [0.0, 0.1], [0.0, 0.2]],
    CvM=[[0, 250], [0, 500], [0, 1000]],
    outlier=[[0, 0.2], [0, 0.4], [0, 0.6]],
    ks=[[0, 0.2], [0, 0.4], [0, 0.6]],
    ksamp=[[0, 1e3], [0, 2e3], [0, 4e3]],
)


def score_metrics(
    data: pd.DataFrame,
    metric_dict: dict[str, list[list[float]]],
    n_tasksets: int = 2,
) -> dict[str, list]:
    """
    Score submission metrics against predefined acceptable ranges.

    Evaluates how many metric values fall within specified acceptable ranges
    for each taskset. Only processes task 1 data.

    Parameters
    ----------
    data
        DataFrame containing submission results with columns for 'task',
        'taskset', and metric names matching keys in metric_dict.
    metric_dict
        Dictionary mapping metric names to lists of acceptable [min, max] ranges.
        Each metric can have multiple ranges corresponding to different scoring tiers.
    n_tasksets
        Number of tasksets to evaluate. Default is 2.

    Returns
    -------
    score_dict
        Dictionary containing:
        - One entry per metric name with array of scores per taskset
        - 'norms': Total possible points per taskset
        - 'taskset_scores': Total achieved points per taskset
        - 'percentages': Percentage scores per taskset (taskset_scores/norms)

    Notes
    -----
    A point is awarded for each metric value that falls within any of its
    acceptable ranges. Multiple ranges per metric allow for tiered scoring.
    """
    score_dict = {k: np.zeros(n_tasksets, dtype=int) for k in metric_dict}
    norms = np.zeros(n_tasksets, dtype=int)
    taskset_scores = np.zeros(n_tasksets, dtype=int)
    for irow, row in data.iterrows():
        if row["task"] != 1:
            continue
        i_taskset = int(row["taskset"] - 1)

        for k, metric_ranges in metric_dict.items():
            norms[i_taskset] += len(metric_ranges)

            for metric_range in metric_ranges:
                if metric_range[0] <= row[k] and metric_range[1] >= row[k]:
                    score_dict[k][i_taskset] += 1
                    taskset_scores[i_taskset] += 1

    score_dict["norms"] = norms
    score_dict["taskset_scores"] = taskset_scores
    score_dict["percentages"] = taskset_scores / norms
    ret_dict = {k: v.tolist() for k, v in score_dict.items()}

    return ret_dict


def score_all_metrics(
    data_dict: dict[str, pd.DataFrame],
    metric_dict: dict[str, list[list[float]]],
) -> dict[str, dict[str, list]]:
    """
    Score metrics for all submissions in a data dictionary.

    Applies score_metrics to each submission's data and organizes results
    by submission name.

    Parameters
    ----------
    data_dict
        Dictionary mapping submission names to their result DataFrames.
    metric_dict
        Dictionary mapping metric names to lists of acceptable [min, max] ranges.

    Returns
    -------
    scores
        Dictionary mapping submission names to their score dictionaries
        (as returned by score_metrics).

    Examples
    --------
    >>> all_scores = score_all_metrics(data_dict, metric_dict)
    >>> print(all_scores['baseline']['percentages'])
    [0.85 0.92]
    """
    return {k: score_metrics(v, metric_dict) for k, v in data_dict.items()}


def extract_score(
    score_dict: dict[str, dict[str, list]],
    which_score: str,
    n_tasksets: int = 2,
) -> pd.DataFrame:
    """
    Extract specific score type into a formatted DataFrame.

    Creates a tabular view of a particular score component (e.g., 'percentages',
    'taskset_scores') across all submissions and tasksets.

    Parameters
    ----------
    score_dict
        Dictionary mapping submission names to their score dictionaries
        (as returned by score_all_metrics).
    which_score
        Name of the score component to extract (e.g., 'percentages',
        'taskset_scores', 'norms', or any metric name).
    n_tasksets
        Number of tasksets. Default is 2.

    Returns
    -------
    df
        DataFrame with columns:
        - 'Submission': Submission name
        - 'Taskset_1', 'Taskset_2', ...: Score values rounded to 2 decimals

    Examples
    --------
    >>> df = extract_score(all_scores, 'percentages')
    >>> print(df)
       Submission  Taskset_1  Taskset_2
    0    baseline       0.85       0.92
    1  improved_algo    0.91       0.95
    """
    _n_sub = len(score_dict)
    out_dict: dict[str, list] = dict(Submission=[])
    out_dict.update(**{f"Taskset_{i+1}": [] for i in range(n_tasksets)})
    for sub, scores in score_dict.items():
        out_dict["Submission"].append(sub)
        for i in range(n_tasksets):
            out_dict[f"Taskset_{i+1}"].append(np.round(scores[which_score][i], 2))
    return pd.DataFrame(out_dict)
