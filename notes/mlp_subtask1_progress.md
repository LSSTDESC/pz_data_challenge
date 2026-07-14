# MLP Subtask 1 Progress

## Completed formal estimate files

### Taskset 1 / Cardinal / 1yr
- Output file:
  - `outputs/subtask1_mlp_ts1_cardinal_1yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_1yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 2 / Cardinal / 1yr
- Output file:
  - `outputs/subtask1_mlp_ts2_cardinal_1yr_pretrained/pz_challenge_taskset_2_cardinal_pz_estimate_1yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

## Current MLP line
- Model family: Simple MLP Flow
- Current best line: pretrained_full
- Pretraining: simulated photometric catalog
- Fine-tuning: challenge labeled training data

## Next step
- Package subtask 2 deliverables:
  - trained model files
  - estimation-only function
  - minimal notes on training / pretraining workflow
