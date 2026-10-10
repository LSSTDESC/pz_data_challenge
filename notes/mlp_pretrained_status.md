# Simple MLP Flow Pretrained Status

## Current best MLP line

Current best Simple MLP Flow model: `pretrained_full`

Workflow:
1. Pretrain on simulated photometric data
2. Fine-tune on challenge labeled training data
3. Evaluate with overall metrics, RAIL-style metric plots, and PIT diagnostics

## Pretraining source

- Simulated photometric catalog
- Features used in pretraining:
  - `u_cModelMag`
  - `g_cModelMag`
  - `r_cModelMag`
  - `i_cModelMag`
  - `z_cModelMag`
  - `y_cModelMag`

## Fine-tuning setup

Challenge fine-tuning features:
- `mag_u_lsst`
- `mag_g_lsst`
- `mag_r_lsst`
- `mag_i_lsst`
- `mag_z_lsst`
- `mag_y_lsst`

Architecture / training setup:
- hidden size: 256
- depth: 3
- posterior samples: 128
- sample steps: 64
- seed: 7
- split seed: 7

## Taskset 1 result

Fine-tuning dataset:
- `pz_challenge_taskset_1_cardinal_training_1yr`

Overall metrics:
- bias: `0.00194`
- scatter: `0.02938`
- outlier rate: `0.01563`
- MSE: `0.00614`
- Energy Score: `0.03132`

Interpretation:
- This is the strongest current Simple MLP Flow result so far.
- It outperforms the baseline, longer-trained, and aligned-ensemble MLP variants in current overall metrics.

## Taskset 2 result

Fine-tuning dataset:
- `pz_challenge_taskset_2_cardinal_training_1yr`

Overall metrics:
- bias: `0.00256`
- scatter: `0.03577`
- outlier rate: `0.0455`
- MSE: `0.01751`
- Energy Score: `0.04711`

Interpretation:
- The pretrained MLP pipeline transfers successfully to taskset 2.
- Taskset 2 appears substantially harder than taskset 1.

## Current deliverable status

Completed:
- Pretraining pipeline for Simple MLP Flow
- Fine-tuned TS1 pretrained full result
- Fine-tuned TS2 cardinal 1yr pretrained full result
- RAIL-style plots and PIT diagnostics
- Subtask 1 / 2 helper scripts in progress

Still needed:
- Formal packaging of subtask 1 estimate files
- Formal packaging of subtask 2 model + estimation functions
- Further taskset/scenario coverage if needed
