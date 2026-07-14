# MLP Artifact Delivery List

## Current MLP submission line

- Model family: Simple MLP Flow
- Current best line: `pretrained_full`

## Subtask 1 formal estimate files

### Taskset 1
- Cardinal / 1yr
  - `outputs/subtask1_mlp_ts1_cardinal_1yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_1yr.hdf5`
- Cardinal / 10yr
  - `outputs/subtask1_mlp_ts1_cardinal_10yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5`
- Flagship / 1yr
  - `outputs/subtask1_mlp_ts1_flagship_1yr_pretrained/pz_challenge_taskset_1_flagship_pz_estimate_1yr.hdf5`
- Flagship / 10yr
  - `outputs/subtask1_mlp_ts1_flagship_10yr_pretrained/pz_challenge_taskset_1_flagship_pz_estimate_10yr.hdf5`

### Taskset 2
- Cardinal / 1yr
  - `outputs/subtask1_mlp_ts2_cardinal_1yr_pretrained/pz_challenge_taskset_2_cardinal_pz_estimate_1yr.hdf5`
- Cardinal / 10yr
  - `outputs/subtask1_mlp_ts2_cardinal_10yr_pretrained/pz_challenge_taskset_2_cardinal_pz_estimate_10yr.hdf5`
- Flagship / 1yr
  - `outputs/subtask1_mlp_ts2_flagship_1yr_pretrained/pz_challenge_taskset_2_flagship_pz_estimate_1yr.hdf5`
- Flagship / 10yr
  - `outputs/subtask1_mlp_ts2_flagship_10yr_pretrained/pz_challenge_taskset_2_flagship_pz_estimate_10yr.hdf5`

## Subtask 2 estimation-only code

Primary script:
- `tests/scripts/taskset1_mlp_submission.py`

Implemented callable entry points:
- `run_taskset_1_estimation_only(model_file, test_file, output_file)`
- `run_taskset_2_estimation_only(model_file, test_file, output_file)`

## Current trained model files

### Taskset 1
- Cardinal / 1yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_cardinal_1yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`
- Cardinal / 10yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts1_cardinal_10yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`
- Flagship / 1yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts1_flagship_1yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`
- Flagship / 10yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts1_flagship_10yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`

### Taskset 2
- Cardinal / 1yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts2_cardinal_1yr_lsst6_pretrained_seed7_full_clean/simple_mlp_flow_model.pt`
- Cardinal / 10yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts2_cardinal_10yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`
- Flagship / 1yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts2_flagship_1yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`
- Flagship / 10yr
  - `/home/junofang/pz_dataChallenge/outputs/gpu_ts2_flagship_10yr_lsst6_pretrained_seed7_full/simple_mlp_flow_model.pt`

## Notes

- This is the current first-pass full-scenario MLP delivery list for subtask 1 and subtask 2 readiness.
- Large binary outputs are kept outside git tracking, while their paths and generation workflow are documented here.
