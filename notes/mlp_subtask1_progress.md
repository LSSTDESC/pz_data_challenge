# MLP Subtask 1 Progress

## Current MLP line

- Model family: Simple MLP Flow
- Current best line: `pretrained_full`
- Pretraining source: simulated photometric catalog
- Fine-tuning target: challenge labeled training data

## Completed formal estimate files

### Taskset 1 / Cardinal / 1yr
- Output file:
  - `outputs/subtask1_mlp_ts1_cardinal_1yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_1yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 1 / Cardinal / 10yr
- Output file:
  - `outputs/subtask1_mlp_ts1_cardinal_10yr_pretrained/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 1 / Flagship / 1yr
- Output file:
  - `outputs/subtask1_mlp_ts1_flagship_1yr_pretrained/pz_challenge_taskset_1_flagship_pz_estimate_1yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 1 / Flagship / 10yr
- Output file:
  - `outputs/subtask1_mlp_ts1_flagship_10yr_pretrained/pz_challenge_taskset_1_flagship_pz_estimate_10yr.hdf5`
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

### Taskset 2 / Cardinal / 10yr
- Output file:
  - `outputs/subtask1_mlp_ts2_cardinal_10yr_pretrained/pz_challenge_taskset_2_cardinal_pz_estimate_10yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 2 / Flagship / 1yr
- Output file:
  - `outputs/subtask1_mlp_ts2_flagship_1yr_pretrained/pz_challenge_taskset_2_flagship_pz_estimate_1yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

### Taskset 2 / Flagship / 10yr
- Output file:
  - `outputs/subtask1_mlp_ts2_flagship_10yr_pretrained/pz_challenge_taskset_2_flagship_pz_estimate_10yr.hdf5`
- Validation checks passed:
  - qp file loads successfully
  - ancillary keys include `object_id` and `zmode`
  - output object count = 20000
  - object_id values exactly match the test catalog

## Status

This first-pass `pretrained_full` Simple MLP Flow line now has full scenario coverage for subtask 1 across:

- taskset 1 / taskset 2
- cardinal / flagship
- 1yr / 10yr

## Next step

- finalize subtask 2 packaging
- organize final artifact delivery list
- decide whether to merge this branch as the current MLP submission line
