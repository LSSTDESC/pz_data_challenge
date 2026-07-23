"""Administrative utilities for submission processing and evaluation.

This module provides functions for processing, evaluating, and analyzing
submissions in the challenge framework.
"""

import os
from jinja2 import Template
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List
import yaml
import numpy as np

import glob
import pandas as pd
import tables_io

from pz_data_challenge import metrics, evaluation, scoring

SUBMISSION_TOP_DIR = "submissions"
RESULTS_TOP_DIR = "results"
SUBMISSION_TEMPLATE = "docs_results/_templates/entry_summary.rst.j2"

TASKSETS = ["taskset_1", "taskset_2"]
TASKS = ["", "outputs_2/", "outputs_3/"]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
TASKSETS_TIMING = ["taskset1", "taskset2"]
SIMS_DICT = dict(cardinal=1, flagship=2)
SCENARIOS_DICT = {"1yr": 1, "10yr": 2}


def copy_file(input_path: str, output_path: str) -> None:
    """
    Copy a text file to an output file

    Parameters:
    -----------
    input_path : str
        Path to the input_path file
    output_path : str
        Path where the modified file will be saved

    Returns:
    --------
    None
    """
    try:
        # Read the template file
        with open(input_path, "r", encoding="utf-8") as input_file:
            content = input_file.read()

        # Write to output file
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(content)

        print(f"Successfully created {output_path} from {input_path}\n")

    except FileNotFoundError:
        print(f"Warning: Input file '{input_path}' not found", file=sys.stderr)
    except PermissionError:
        print("Warning: Permission denied when accessing files", file=sys.stderr)
    except Exception as e:
        print(f"Warning: {str(e)}", file=sys.stderr)


def extract_dataframes(
    results_top_dir: str,
    submission_name: str,
) -> None:
    """Extract dataframes from results directory.

    Parameters
    ----------
    results_top_dir : str
        Top-level directory containing all results.
    submission_name : str
        Name identifier for the submission.

    """
    submission_file = os.path.join(results_top_dir, f"summary_{submission_name}.yaml")

    with open(submission_file) as fin:
        results_data = yaml.safe_load(fin)

    df_timing = get_timing_stats(results_data)
    df_point_stats = get_point_stats(results_data)
    df_point_v_redshift = get_point_v_redshift(results_data)
    df_point_v_mag = get_point_v_mag(results_data)
    df_pit_prob = get_pit_prob(results_data)

    out_dict = dict(
        timing=df_timing,
        point=df_point_stats,
        point_v_redshift=df_point_v_redshift,
        point_v_mag=df_point_v_mag,
        pit_prob=df_pit_prob,
    )

    outfile = os.path.join(results_top_dir, f"summary_{submission_name}_.parq")

    tables_io.write(out_dict, outfile)


def cleanup_submission_files(submission_name: str) -> None:
    """Clean up temporary submission files.

    Parameters
    ----------
    submission_name : str
        Name identifier for the submission.

    Notes
    -----
    Removes requirements file and test file associated with the submission.
    Silently ignores if files don't exist.
    """
    try:
        os.unlink(f"requirements_{submission_name}.txt")
    except Exception:
        pass

    try:
        os.unlink(f"tests/test_{submission_name}.py")
    except Exception:
        pass


def merge_results_summaries(
    results_dir: str,
    submission: str,
) -> None:
    """Merge all results files into a single summary YAML.

    Parameters
    ----------
    results_dir : str
        Directory containing individual result files.
    submission : str
        Name identifier for the submission.

    Notes
    -----
    Creates a summary_{submission}.yaml file in RESULTS_TOP_DIR containing
    all results merged into a single dictionary.
    """
    all_files = glob.glob(f"{results_dir}/*.yaml")
    all_files += glob.glob(f"{results_dir}/*/*.yaml")

    all_dict = {}
    for file_ in all_files:
        with open(file_) as fin:
            try:
                all_dict[file_.replace(f"{results_dir}/", "")] = yaml.safe_load(fin)
            except Exception:
                pass

    summary_file = os.path.join(RESULTS_TOP_DIR, f"summary_{submission}.yaml")
    with open(summary_file, "w", encoding="utf-8") as fout:
        yaml.dump(all_dict, fout)


def make_submission_eval_plots(
    reseved_data_dir: str,
    submission_data_dir: str,
    results_dir: str,
    *,
    force: bool = False,
) -> None:
    """Generate evaluation plot

    Parameters
    ----------
    reserved_data_dir : str
        Path to the reserved/validation data.
    submission_data_dir : str
        Path to the directory containing submission files.
    results_dir : str
        Path to the directory where results will be stored.

    Notes
    -----
    Processes three task outputs: default, outputs_2, and outputs_3.
    Generates plots for each task and merges all results into a summary.
    """

    data_dict = {}
    for taskset_ in TASKSETS:
        for sim_ in SIMS:
            for scenario_ in SCENARIOS:
                prefix = f"{taskset_}_{sim_}_{scenario_}"

                try:
                    sub_data_dict = metrics.get_truth_and_qp_ensemble(
                        reseved_data_dir,
                        submission_data_dir,
                        taskset_,
                        sim_,
                        scenario_,
                        test_label="test",
                        eval_label="pz_estimate",
                    )
                except Exception:
                    continue
                test_data = sub_data_dict[f"{prefix}_test"]
                submit_data = sub_data_dict[f"{prefix}_evaluate"]

                data_dict.update(
                    metrics.point_metrics_plot(
                        f"{prefix}_point", test_data, submit_data
                    )
                )
                data_dict.update(
                    metrics.point_v_redshfit_plot(
                        f"{prefix}_point_v_redshift", test_data, submit_data
                    )
                )
                data_dict.update(
                    metrics.point_v_mag_plot(
                        f"{prefix}_point_v_mag", test_data, submit_data
                    )
                )
                data_dict.update(
                    metrics.plot_pit_prob_plot(
                        f"{prefix}_pit_prob", test_data, submit_data
                    )
                )
                data_dict.update(
                    metrics.plot_pit_qq_plot(f"{prefix}_pit_qq", test_data, submit_data)
                )

    try:
        os.makedirs(results_dir)
    except Exception:
        pass

    for k, v in data_dict.items():
        v.savefig(k.replace("0.0", ""), results_dir)
        v.savedata(results_dir)

    print(f"\nSaved {len(data_dict)} plots to {results_dir}")


def make_eval_plots_and_summarize(
    submission_name: str,
    submission_dir: str,
    results_dir: str,
    reserved_data_path: str,
    *,
    force: bool = False,
) -> None:
    """Generate evaluation plots and create results summary.

    Parameters
    ----------
    submission_name : str
        Name identifier for the submission.
    submission_dir : str
        Path to the directory containing submission files.
    results_dir : str
        Path to the directory where results will be stored.
    reserved_data_path : str
        Path to the reserved/validation data.

    Notes
    -----
    Processes three task outputs: default, outputs_2, and outputs_3.
    Generates plots for each task and merges all results into a summary.
    """
    summary_file = os.path.join(RESULTS_TOP_DIR, f"summary_{submission_name}.yaml")
    if os.path.exists(summary_file) and not force:
        return

    for task in ["", "outputs_2", "outputs_3"]:
        try:
            make_submission_eval_plots(
                reserved_data_path,
                os.path.join(submission_dir, task),
                os.path.join(results_dir, task),
                force=force,
            )
        except Exception as exc:
            print(exc)
            raise

    merge_results_summaries(results_dir, submission_name)


def get_point_stats(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract point statistics from results data.

    Parameters
    ----------
    results_data : dict of str to Any
        Dictionary containing results data indexed by task keys.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing point statistics and PIT QQ data for all
        combinations of tasksets, tasks, simulations, and scenarios.

    Notes
    -----
    Combines point estimates and PIT (Probability Integral Transform) QQ
    statistics from all task combinations.
    """
    temp_list: List[Dict[str, Any]] = []

    for i_taskset, taskset_ in enumerate(TASKSETS):
        for i_task, task_ in enumerate(TASKS):
            for i_sim, sim_ in enumerate(SIMS):
                for i_scenario, scenario_ in enumerate(SCENARIOS):
                    key = f"{task_}{taskset_}_{sim_}_{scenario_}"

                    temp_dict: Dict[str, Any] = dict(
                        taskset=i_taskset + 1,
                        task=i_task + 1,
                        sim=i_sim + 1,
                        scenario=i_scenario + 1,
                    )
                    try:
                        point_data = results_data[f"{key}_point.yaml"]
                        temp_dict.update(**point_data)
                    except Exception:
                        temp_dict.update(
                            abs_outlier_rate=np.nan,
                            mean=np.nan,
                            mean_err=np.nan,
                            outlier_rate=np.nan,
                            std=np.nan,
                        )
                    try:
                        pit_data = results_data[f"{key}_pit_qq.yaml"]
                        temp_dict.update(**pit_data)
                    except KeyError:
                        temp_dict.update(
                            CvM=np.nan,
                            ks=np.nan,
                            ksamp=np.nan,
                            outlier=np.nan,
                        )
                    temp_list.append(temp_dict)

    out_dict: Dict[str, List[Any]] = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                out_dict[k] = [v]

    return pd.DataFrame(out_dict)


def get_point_v_redshift(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract point statistics versus redshift from results data.

    Parameters
    ----------
    results_data : dict of str to Any
        Dictionary containing results data indexed by task keys.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing point statistics as a function of redshift
        for all combinations of tasksets, tasks, simulations, and scenarios.
    """
    temp_list: List[Dict[str, Any]] = []

    for i_taskset, taskset_ in enumerate(TASKSETS):
        for i_task, task_ in enumerate(TASKS):
            for i_sim, sim_ in enumerate(SIMS):
                for i_scenario, scenario_ in enumerate(SCENARIOS):
                    key = f"{task_}{taskset_}_{sim_}_{scenario_}"

                    temp_dict: Dict[str, Any] = dict(
                        taskset=i_taskset + 1,
                        task=i_task + 1,
                        sim=i_sim + 1,
                        scenario=i_scenario + 1,
                    )
                    try:
                        point_data = results_data[f"{key}_point_v_redshift.yaml"]
                    except KeyError:
                        continue
                    temp_dict.update(**point_data)
                    temp_list.append(temp_dict)

    out_dict: Dict[str, List[Any]] = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                out_dict[k] = [v]

    return pd.DataFrame(out_dict)


def get_point_v_mag(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract point statistics versus magnitude from results data.

    Parameters
    ----------
    results_data : dict of str to Any
        Dictionary containing results data indexed by task keys.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing point statistics as a function of magnitude
        for all combinations of tasksets, tasks, simulations, and scenarios.
    """
    temp_list: List[Dict[str, Any]] = []

    for i_taskset, taskset_ in enumerate(TASKSETS):
        for i_task, task_ in enumerate(TASKS):
            for i_sim, sim_ in enumerate(SIMS):
                for i_scenario, scenario_ in enumerate(SCENARIOS):
                    key = f"{task_}{taskset_}_{sim_}_{scenario_}"

                    temp_dict: Dict[str, Any] = dict(
                        taskset=i_taskset + 1,
                        task=i_task + 1,
                        sim=i_sim + 1,
                        scenario=i_scenario + 1,
                    )
                    try:
                        point_data = results_data[f"{key}_point_v_mag.yaml"]
                    except KeyError:
                        continue
                    temp_dict.update(**point_data)
                    temp_list.append(temp_dict)

    out_dict: Dict[str, List[Any]] = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                out_dict[k] = [v]

    return pd.DataFrame(out_dict)


def get_timing_stats(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract timing statistics from results data.

    Parameters
    ----------
    results_data : dict of str to Any
        Dictionary containing results data indexed by task keys.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing timing statistics for all task combinations.

    """
    temp_list = []
    for i_taskset, taskset_ in enumerate(TASKSETS_TIMING):
        key = f"stats_{taskset_}.yaml"

        temp_dict = dict(
            taskset=i_taskset + 1,
        )
        try:
            stats = results_data[key]
        except KeyError:
            continue
        for k, the_time in stats.items():
            if k.find("time") < 0:
                continue
            tokens = k.split("_")
            dd = dict(
                sim=SIMS_DICT[tokens[0]],
                scenario=SCENARIOS_DICT[tokens[1]],
                task=int(tokens[3]),
                time=the_time,
            )
            temp_dict.update(**dd)
            temp_list.append(temp_dict.copy())

    out_dict: dict[str, Any] = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                out_dict[k] = [v]

    return pd.DataFrame(out_dict)


def get_pit_prob(results_data: Dict[str, Any]) -> pd.DataFrame:
    """Extract PIT probability from data.

    Parameters
    ----------
    results_data : dict of str to Any
        Dictionary containing results data indexed by task keys.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing timing statistics for all task combinations.
    """

    temp_list = []

    for i_taskset, taskset_ in enumerate(TASKSETS):
        for i_task, task_ in enumerate(TASKS):
            for i_sim, sim_ in enumerate(SIMS):
                for i_scenario, scenario_ in enumerate(SCENARIOS):
                    key = f"{task_}{taskset_}_{sim_}_{scenario_}"

                    temp_dict = dict(
                        taskset=i_taskset + 1,
                        task=i_task + 1,
                        sim=i_sim + 1,
                        scenario=i_scenario + 1,
                    )
                    try:
                        point_data = results_data[f"{key}_pit_prob.yaml"]
                    except KeyError:
                        continue
                    temp_dict.update(**point_data)
                    temp_list.append(temp_dict)

    out_dict: dict[str, Any] = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                out_dict[k] = [v]
    return pd.DataFrame(out_dict)


def run_submission(
    submission_name: str, submission_dir: str, results_dir: str, *, force: bool = False
) -> None:
    """Run the code for a submission.

    Parameters
    ----------
    submission_name : str
        Name identifier for the submission.
    submission_dir : str
        Path to the directory containing submission files.
    results_dir : str
        Path to the directory where results will be stored.
    """

    if os.environ.get("SKIP_RUN"):
        return

    if os.path.exists(os.path.join(results_dir, "pytest.log")) and not force:
        return

    if not os.environ.get("SKIP_INSTALL"):
        subprocess.run(
            ["pip", "install", "-r", f"requirements_{submission_name}.txt"], check=True
        )

    os.environ["NO_TEARDOWN"] = "1"

    output = subprocess.run(
        ["py.test", f"tests/test_{submission_name}.py"], check=True, capture_output=True
    )

    try:
        os.makedirs(results_dir)
    except Exception:
        pass

    with open(os.path.join(results_dir, "pytest.log"), "w", encoding="utf-8") as fout:
        fout.write(output.stdout.decode())

    try:
        copy_file(
            f"{submission_dir}/stats_taskset1.yaml",
            f"{results_dir}/stats_taskset1.txt",
        )
    except Exception:
        pass

    try:
        copy_file(
            f"{submission_dir}/stats_taskset2.yaml",
            f"{results_dir}/stats_taskset2.txt",
        )
    except Exception:
        pass


def evaluate_submission(
    accepted_dir: str,
    submission_name: str,
    submission_dir: str,
    results_dir: str,
    results_top_dir: str,
    reserved_data_path: str,
) -> None:
    """Evaluate submission results.

    Parameters
    ----------
    submission_name : str
        Name identifier for the submission.
    submission_dir : str
        Path to the directory containing submission files.
    results_dir : str
        Path to the directory where results will be stored.
    results_top_dir : str
        Path to the directory where results will be summarized.
    reserved_data_path : str
        Path to the reserved/validation data.
    """

    # copy files from accepted area back to where they were
    copy_file(
        f"{accepted_dir}/requirements_{submission_name}.txt",
        f"requirements_{submission_name}.txt",
    )

    copy_file(
        f"{accepted_dir}/test_{submission_name}.py",
        f"tests/test_{submission_name}.py",
    )

    failed = False
    # Run the submssion
    if not os.environ.get("SKIP_RUN"):
        try:
            run_submission(
                submission_name,
                submission_dir,
                results_dir,
            )
        except Exception:
            failed = True

    # Evaluate the results
    if not os.environ.get("SKIP_EVALUATE") and not failed:
        try:
            make_eval_plots_and_summarize(
                submission_name,
                submission_dir,
                results_dir,
                reserved_data_path,
            )
        except Exception:
            failed = True

    # Extract the results
    if not os.environ.get("SKIP_EXTRACT") and not failed:
        try:
            extract_dataframes(
                results_top_dir,
                submission_name,
            )
        except Exception:
            failed = True
            

    # clean up
    try:
        os.unlink(f"requirements_{submission_name}.txt")
    except Exception:
        pass

    try:
        os.unlink(f"tests/test_{submission_name}.py")
    except Exception:
        pass


def make_point_summaries(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Generate summary strip plots for point-estimate metrics.

    Parameters
    ----------
    results_dir : str
        Directory where summary plots will be saved.
    submissions : list[str]
        List of submission identifiers to include.
    """
    data_dict = evaluation.build_summary_data_dict(results_dir, submissions, "point")

    dd_outliers = evaluation.get_metric_summary_dict(
        data_dict, submissions, "abs_outlier_rate"
    )
    dd_mean = evaluation.get_metric_summary_dict(data_dict, submissions, "mean")
    dd_rms = evaluation.get_metric_summary_dict(data_dict, submissions, "std")

    fig_mean = evaluation.make_strip_plot(
        dd_mean,
        r"Mean $\frac{|z_{\rm est} - z_{\rm ref}|}{1 + z_{\rm ref}}$",
        [-0.05, 0.05],
        scoring.metric_dict["mean"],
    )
    fig_mean.savefig(f"{results_dir}/plot_summary_point_mean.png", dpi=200)

    fig_rms = evaluation.make_strip_plot(
        dd_rms,
        r"RMS $\frac{|z_{\rm est} - z_{\rm ref}|}{1 + z_{\rm ref}}$",
        [0, 0.3],
        scoring.metric_dict["std"],
    )
    fig_rms.savefig(f"{results_dir}/plot_summary_point_rms.png", dpi=200)

    fig_outliers = evaluation.make_strip_plot(
        dd_outliers,
        r"Outlier rate $\frac{|z_{\rm est} - z_{\rm ref}|}{1 + z_{\rm ref}} > 0.15$",
        [0, 0.5],
        scoring.metric_dict["abs_outlier_rate"],
    )
    fig_outliers.savefig(f"{results_dir}/plot_summary_point_outliers.png", dpi=200)


def make_PIT_summaries(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Generate summary strip plots for PIT Q-Q metrics.

    Parameters
    ----------
    results_dir : str
        Directory where summary plots will be saved.
    submissions : list[str]
        List of submission identifiers to include.
    """
    data_dict = evaluation.build_summary_data_dict(results_dir, submissions)

    dd_outlier = evaluation.get_metric_summary_dict(data_dict, submissions, "outlier")
    dd_CvM = evaluation.get_metric_summary_dict(data_dict, submissions, "CvM")
    dd_ks = evaluation.get_metric_summary_dict(data_dict, submissions, "ks")
    dd_ksamp = evaluation.get_metric_summary_dict(data_dict, submissions, "ksamp")

    fig_CvM = evaluation.make_strip_plot(
        dd_CvM,
        r"PIT Q-Q CvM Metric",
        [0, 2000],
        scoring.metric_dict["CvM"],
    )
    fig_CvM.savefig(f"{results_dir}/plot_summary_pit_CvV.png")

    fig_pit_outlier = evaluation.make_strip_plot(
        dd_outlier,
        r"PIT Q-Q Outliers",
        [0, 1],
        scoring.metric_dict["outlier"],
    )
    fig_pit_outlier.savefig(f"{results_dir}/plot_summary_pit_outlier.png")

    fig_ks = evaluation.make_strip_plot(
        dd_ks,
        r"PIT Q-Q KS",
        [0, 1],
        scoring.metric_dict["ks"],
    )
    fig_ks.savefig(f"{results_dir}/plot_summary_pit_ks.png", dpi=200)

    fig_ksamp = evaluation.make_strip_plot(
        dd_ksamp,
        r"PIT Q-Q ksamp",
        [0, 1e4],
        scoring.metric_dict["ksamp"],
    )
    fig_ksamp.savefig(f"{results_dir}/plot_summary_pit_ksamp.png")


def make_timing_summaries(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Generate summary strip plots for algorithm timing metrics.

    Parameters
    ----------
    results_dir : str
        Directory where summary plots will be saved.
    submissions : list[str]
        List of submission identifiers to include.
    """
    data_dict = evaluation.build_summary_data_dict(results_dir, submissions, "timing")

    fig_algo_estimate_time = evaluation.make_algo_estimate_time_strip_plot(
        data_dict,
        submissions,
    )
    fig_algo_estimate_time.savefig(f"{results_dir}/plot_summary_timing_estimate.png", dpi=200)

    fig_algo_inform_time = evaluation.make_algo_inform_time_strip_plot(
        data_dict,
        submissions,
    )
    fig_algo_inform_time.savefig(f"{results_dir}/plot_summary_timing_inform.png", dpi=200)


def make_PIT_plot(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Generate PIT Q-Q comparison plot across submissions.

    Parameters
    ----------
    results_dir : str
        Directory where the Q-Q plot will be saved.
    submissions : list[str]
        List of submission identifiers to include.
    """
    data_dict = evaluation.build_summary_data_dict(results_dir, submissions, "pit_prob")

    fig_qq_plot = evaluation.make_qq_pit_plot(data_dict, submissions)

    fig_qq_plot.savefig(f"{results_dir}/plot_summary_pit_qq.png")


def make_submission_summary_rst(
    results_dir: str,
    submissions: list[str],
    template_file: str,
) -> None:
    """Render RST summary pages for each submission from a Jinja2 template.

    Parameters
    ----------
    results_dir : str
        Base directory for output RST files.
    submissions : list[str]
        List of submission identifiers.
    template_file : str
        Path to the Jinja2 RST template file.
    """
    with open(template_file, "r") as f:
        template = Template(f.read())

    base_path = Path(results_dir)

    # Generate RST file for each directory
    for submission_name in submissions:
        # Render template
        content = template.render(dir_name=submission_name)

        # Write to file
        output_path = base_path / Path(submission_name) / "index.rst"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(content)

            print(f"Generated: {output_path}")


def make_scores(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Compute and save scoring summaries for all submissions.

    Parameters
    ----------
    results_dir : str
        Directory where score CSV and YAML files will be written.
    submissions : list[str]
        List of submission identifiers to score.
    """
    data_dict = evaluation.build_summary_data_dict(results_dir, submissions, "point")
    score_dict = scoring.score_all_metrics(data_dict, scoring.metric_dict)
    scores = scoring.extract_score(score_dict, "percentages")
    with open(f"{results_dir}/scores_full.csv", "w", encoding="utf-8") as fout:
        yaml.dump(score_dict, fout)
    scores.to_csv(f"{results_dir}/scores_summary.csv", index=False)


def make_all_summary_plots_and_files(
    results_dir: str,
    submissions: list[str],
) -> None:
    """Generate all summary plots, scores, and RST files.

    Parameters
    ----------
    results_dir : str
        Directory where all summary outputs will be written.
    submissions : list[str]
        List of submission identifiers to process.
    """
    if not os.environ.get("SKIP_SUMMARIZE"):
        make_point_summaries(results_dir, submissions)
        make_PIT_summaries(results_dir, submissions)
        make_timing_summaries(results_dir, submissions)
        make_PIT_plot(results_dir, submissions)
        make_scores(results_dir, submissions)
        make_submission_summary_rst(results_dir, submissions, SUBMISSION_TEMPLATE)
