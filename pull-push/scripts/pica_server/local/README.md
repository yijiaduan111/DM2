# Local PICA experiment scripts

These are the fixed scripts used on dyj server for the 45936 PICA experiments.

## Core scripts

- `env_cuda_kernel_eval.sh`
  - Activates the `cuda-kernel-eval` environment and sets project paths.
  - Source this before manual train/eval commands.

- `run_pica_drand12_large_wandb_earlystop.sh`
  - Main training launcher for the current best v2c drand12 setup.
  - Default config: `ppo/train_config_gla_pica_drand12_aux_v2c.yaml`, object `45936`, num_envs `64`, max_epochs `300`, GPU `5`.
  - Common override example:
    `RUN_NAME=my_run MAX_EPOCHS=500 GPU=5 nohup bash scripts/pica_server/local/run_pica_drand12_large_wandb_earlystop.sh > logs_output/my_run_nohup.log 2>&1 &`

- `monitor_train_wandb_earlystop.py`
  - Watches `runs/<run>/epoch_rewards.csv`, logs metrics to wandb, and optionally early-stops by `reward_mean` plateau.
  - Usually launched by `run_pica_drand12_large_wandb_earlystop.sh`.

- `eval_pica_v2c_damping_det_stoch_dyj.sh`
  - Runs deterministic and stochastic damping eval at x1/x2/x4 for a checkpoint directory.
  - Common example:
    `RUN_NAME=hand_drag_gla_45936_pica_drand12_aux_v2c_500ep_es OUT_ROOT=output/pica_v2c_500ep_eval bash scripts/pica_server/local/eval_pica_v2c_damping_det_stoch_dyj.sh`

- `summarize_damping_eval.py`
  - Builds `deterministic_damping_eval_table.md` and `stochastic_damping_eval_table.md` from eval JSON/CSV outputs.
  - Usually launched by the eval script.

- `summarize_v2c_150_500_table.py`
  - Rebuilds the 150ep-vs-500ep comparison table:
    `reports/pica_handoff/experimental_evidence_summary_45936_v2c_150_500.md`.

## Older calibration helpers

- `run_pica_gla_calibration_20ep_dyj.sh`
- `eval_pica_calibration_nominal_dyj.sh`

Kept for historical calibration runs. Not the main workflow anymore.

## Archive

One-off scripts used during exploration were moved to `archive/` for traceability:

- `archive/logs_output_scripts/`
- `archive/tmp_scripts/`
