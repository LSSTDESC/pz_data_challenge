"""Administrative utilities for submission processing and evaluation.

This module provides functions for processing, evaluating, and analyzing
submissions in the challenge framework.
"""

import os
import subprocess
import sys
from typing import Any, Dict, List
import yaml

import glob
import pandas as pd
import tables_io

from pz_data_challenge import metrics

SUBMISSION_TOP_DIR = 'submissions'
RESULTS_TOP_DIR = 'results'

TASKSETS = ["taskset_1", "taskset_2"]
TASKS = ["", "outputs_2/", "outputs_3/"]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]
TASKSETS_TIMING = ["taskset1", "taskset2"]
SIMS_DICT = dict(cardinal=1, flagship=2)
SCENARIOS_DICT = {"1yr":1, "10yr":2}


def copy_file(input_path, output_path): 
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
        with open(input_path, 'r', encoding='utf-8') as input_file:
            content = input_file.read()

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(content)

        print(f"Successfully created {output_path} from {input_path}\n")

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
    except PermissionError:
        print(f"Error: Permission denied when accessing files", file=sys.stderr)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)


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
def merge_results_summaries(results_dir, submission):

    all_files = glob.glob(f"{results_dir}/*.yaml")
    all_files += glob.glob(f"{results_dir}/*/*.yaml")

    all_dict = {}
    for file_ in all_files:
        with open(file_) as fin:
            all_dict[file_.replace(f"{results_dir}/", '')] = yaml.safe_load(fin)

    summary_file = os.path.join(RESULTS_TOP_DIR, f"summary_{submission}.yaml")
    with open(summary_file, 'w', encoding='utf-8') as fout:
        yaml.dump(all_dict, fout)


def make_submission_eval_plots(
    reseved_data_dir: str,
    submission_data_dir: str,
    results_dir: str,
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

                sub_data_dict = metrics.get_truth_and_qp_ensemble(
                    reseved_data_dir, submission_data_dir, taskset_, sim_, scenario_, test_label='test', eval_label='pz_estimate',
                )
                test_data = sub_data_dict[f"{prefix}_test"]
                submit_data = sub_data_dict[f"{prefix}_evaluate"]

                data_dict.update(
                    metrics.point_metrics_plot(f"{prefix}_point", test_data, submit_data)
                )
                data_dict.update(
                    metrics.point_v_redshfit_plot(f"{prefix}_point_v_redshift", test_data, submit_data)
                )
                data_dict.update(
                    metrics.point_v_mag_plot(f"{prefix}_point_v_mag", test_data, submit_data)
                )
                data_dict.update(
                    metrics.plot_pit_prob_plot(f"{prefix}_pit_prob", test_data, submit_data)
                )
                data_dict.update(
                    metrics.plot_pit_qq_plot(f"{prefix}_pit_qq", test_data, submit_data)
                )

    try:
        os.makedirs(results_dir)
    except Exception:
        pass

    for k, v in data_dict.items():
        v.savefig(k.replace("0.0",""), results_dir)
        v.savedata(results_dir)

    print(f"\nSaved {len(data_dict)} plots to {results_dir}")

        

def make_eval_plots_and_summarize(
    submission_name: str,
    submission_dir: str,
    results_dir: str,
    reserved_data_path: str,
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
    for task in ["", "outputs_2", "outputs_3"]:
        make_submission_eval_plots(
            reserved_data_path,
            os.path.join(submission_dir, task),
            os.path.join(results_dir, task),
        )

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
                    point_data = results_data[f"{key}_point.yaml"]
                    pit_data = results_data[f"{key}_pit_qq.yaml"]
                    temp_dict.update(**point_data)
                    temp_dict.update(**pit_data)
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
                    point_data = results_data[f"{key}_point_v_redshift.yaml"]
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
                    point_data = results_data[f"{key}_point_v_mag.yaml"]
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
            taskset=i_taskset+1,                        
        )
        stats = results_data[key]
        for k, the_time in stats.items():
            if k.find('time') < 0:
                continue
            tokens = k.split('_')
            dd = dict(
                sim=SIMS_DICT[tokens[0]],
                scenario=SCENARIOS_DICT[tokens[1]],
                task=int(tokens[3]),
                time=the_time,
            )
            temp_dict.update(**dd)
            temp_list.append(temp_dict.copy())

    out_dict = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                 out_dict[k] = [v]        

    return pd.DataFrame(out_dict)



def get_pit_prob(results_data):
    """Extract PIT probability fro data.
    
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
                        taskset=i_taskset+1,                        
                        task=i_task+1,
                        sim=i_sim+1,
                        scenario=i_scenario+1,
                    )
                    point_data = results_data[f"{key}_pit_prob.yaml"]
                    temp_dict.update(**point_data)
                    temp_list.append(temp_dict)    

    out_dict = {}
    for next_dict in temp_list:
        for k, v in next_dict.items():
            if k in out_dict:
                out_dict[k].append(v)
            else:
                 out_dict[k] = [v]        
    return pd.DataFrame(out_dict)


def run_submission(
    submission_name: str,
    submission_dir: str,
    results_dir: str,
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
    
    if os.environ.get('SKIP_RUN'):
        return

    subprocess.run(["pip", "install", "-r", f"requirements_{submission_name}.txt"], check=True)

    os.environ['NO_TEARDOWN'] = '1'

    output = subprocess.run(["py.test", f"tests/test_{submission_name}.py"], check=True, capture_output=True)

    try:
        os.makedirs(results_dir)
    except Exception:
        pass

    with open(os.path.join(results_dir, "pytest.log"), 'w', encoding='utf-8') as fout:
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
        f"accepted/requirements_{submission_name}.txt",
        f"requirements_{submission_name}.txt",
    )
    
    copy_file(
        f"accepted/test_{submission_name}.py",
        f"tests/test_{submission_name}.py",
    )

    # Run the submssion
    if not os.environ.get('SKIP_RUN'):
        run_submission(
            submission_name,
            submission_dir,
            results_dir,
        )

    # Evaluate the results
    if not os.environ.get('SKIP_EVALUATE'):
        make_eval_plots_and_summarize(
            submission_name,
            submission_dir,
            results_dir,
            reserved_data_path,
        )

    # Extract the results        
    if not os.environ.get('SKIP_EXTRACT'):
        extract_dataframes(
            results_top_dir,
            submission_name,
        )
        
        
    # clean up
    try:
        os.unlink(f"requirements_{submission_name}.txt")
    except Exception:
        pass

    try:
        os.unlink(f"tests/test_{submission_name}.py")
    except Exception:
        pass

    
