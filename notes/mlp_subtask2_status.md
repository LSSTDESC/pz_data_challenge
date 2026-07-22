# MLP Subtask 2 Status

## Current subtask 2 packaging

We use `tests/scripts/taskset1_mlp_submission.py` as the estimation-only export helper for the Simple MLP Flow line.

Implemented entry points:
- `run_taskset_1_estimation_only(model_file, test_file, output_file)`
- `run_taskset_2_estimation_only(model_file, test_file, output_file)`

Both entry points call the shared export function:
- `export_model_predictions_to_qp(...)`

## Current model line

- Model family: Simple MLP Flow
- Current best line: `pretrained_full`

## Current trained model files

### Taskset 1 / Cardinal / 1yr
- `/home/junofang/pz_dataChallenge/outputs/gpu_cardinal_1yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`

### Taskset 2 / Cardinal / 1yr
- `/home/junofang/pz_dataChallenge/outputs/gpu_ts2_cardinal_1yr_lsst6_pretrained_seed7_full_clean/simple_mlp_flow_model.pt`

## Current subtask 1 estimate files

### Taskset 1 / Cardinal / 1yr
- `/home/junofang/pzdc_shared/pz_data_challenge/outputs/subtask1_mlp_ts1_cardinal_1yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_1yr.hdf5`

### Taskset 2 / Cardinal / 1yr
- `/home/junofang/pzdc_shared/pz_data_challenge/outputs/subtask1_mlp_ts2_cardinal_1yr_pretrained/pz_challenge_taskset_2_cardinal_pz_estimate_1yr.hdf5`

## Purpose

This satisfies the core requirement for subtask 2:
- provide trained model files
- provide estimation-only callable functions
- allow challenge organizers to time inference and test generalization
