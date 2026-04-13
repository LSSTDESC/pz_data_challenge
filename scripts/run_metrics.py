import os

from pz_data_challenge import metrics

DATADIR = 'public'
SUBMISSION_DIR = 'submission'
PLOTS_DIR = 'metric_plots'

TASKSETS = ["taskset_1", "taskset_2"]
SIMS = ["cardinal", "flagship"]
SCENARIOS = ["1yr", "10yr"]


if __name__ == '__main__':
    
    data_dict = {}
    for taskset_ in TASKSETS:
        for sim_ in SIMS:
            for scenario_ in SCENARIOS:
                prefix = f"{taskset_}_{sim_}_{scenario_}"
                
                sub_data_dict = metrics.get_truth_and_qp_ensemble(
                    DATADIR, SUBMISSION_DIR, taskset_, sim_, scenario_
                )
                test_data = sub_data_dict[f"{prefix}_test"]
                submit_data = sub_data_dict[f"{prefix}_validate"]
                
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
        os.makedirs(PLOTS_DIR)
    except Exception:
        pass

    for k, v in data_dict.items():
        v.savefig(k.replace("0.0",""), PLOTS_DIR)

    print(f"\nSaved {len(data_dict)} plots to {PLOTS_DIR}")
            
