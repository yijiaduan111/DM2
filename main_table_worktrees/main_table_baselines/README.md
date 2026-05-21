# Main Table Baselines

This directory is a lightweight baseline launcher area for the main-table experiments.
It does not duplicate the large code tree. `code` is a symlink to the current baseline code worktree:

- `code -> /data/dyj/zts/main_table_worktrees/main_table_v2c_clean`

The main method / ours should not be run here. Ours is maintained in:

- `/data/dyj/zts/pull-push`
- `/data/dyj/zts/pull-push_best_v2c150_v2d_both_ft50`

The old v2d main-table worktree has been archived to:

- `/data/dyj/zts/storage/experiment_artifacts/main_table_worktrees_legacy/main_table_v2d_clean`

## Layout

```text
main_table_baselines/
├── code -> ../main_table_v2c_clean
├── configs/
│   ├── trajectory_tracking_hand_config.yaml
│   ├── state_only_ppo.yaml
│   ├── flat_history_ppo.yaml
│   ├── gla_no_aux.yaml
│   └── pica_v2c150_reference_only.yaml
├── scripts/
│   ├── train_eval_trajectory_tracking.sh
│   ├── train_eval_state_only.sh
│   ├── train_eval_flat_history.sh
│   ├── train_eval_gla_no_aux.sh
│   └── eval_existing.sh
├── results/
└── logs/
```


## Canonical Data Contract

All runnable main-table entries must use the canonical manifest below, not the historical `output/hand_drag/<object_id>/trajectory.json` default path:

- `/data/dyj/zts/clean_data/v20260520/batch_manifest.csv`

Each row is keyed by `sample_id = object_id + handle`, and every training/eval call must pass the manifest trajectory explicitly via `--trajectory`. Current canonical samples are:

| sample_id | object_id | handle |
|---|---:|---|
| `45261_handle_7` | `45261` | `handle_7` |
| `45526_handle_1` | `45526` | `handle_1` |
| `45661_handle_3` | `45661` | `handle_3` |
| `45936_handle_1` | `45936` | `handle_1` |
| `46440_handle_5` | `46440` | `handle_5` |
| `7310_handle_1` | `7310` | `handle_1` |

The shared PPO baseline runner is `scripts/run_ppo_manifest_train_eval.sh`. It records the exact manifest used into each run directory as `canonical_manifest.tsv`.

## Standard Baseline Conditions

Current main-table baseline scripts use the following default setup from `main_table_v2c_clean`:

- Samples: loaded from `/data/dyj/zts/clean_data/v20260520/batch_manifest.csv`
- Conda env: `/data/dyj/miniconda3/envs/cuda-kernel-eval`
- IsaacGym: `/data/dyj/zts/isaacgym/python`
- PPO training envs: `--num_envs 64`
- PPO training epochs: `--max_epochs 150`
- PPO eval episodes: `20`
- PPO eval damping: `x1`, i.e. `object_damping_scale=1.0`
- PPO eval modes: deterministic and stochastic
- PPO eval checkpoint kind: `best`
- Trajectory tracking phase: `drag`

## How To Run

Run from anywhere:

```bash
cd /data/dyj/zts/main_table_worktrees/main_table_baselines
bash scripts/train_eval_trajectory_tracking.sh
bash scripts/train_eval_state_only.sh
bash scripts/train_eval_flat_history.sh
bash scripts/train_eval_gla_no_aux.sh
bash scripts/train_eval_pica_no_gla.sh
# Row06 with x1/x4 eval:
bash scripts/train_eval_pica_no_gla_x1_x4.sh
```

Evaluate existing runs:

```bash
bash scripts/eval_existing.sh flat_history [OUTDIR]
bash scripts/eval_existing.sh gla_no_aux [RUN_DIR]
```

## Experiment Mapping

| Main-table item | Status | Code root | Config snapshot | Wrapper |
|---|---|---|---|---|
| 01 Trajectory tracking | runnable | `code/` | `configs/trajectory_tracking_hand_config.yaml` | `scripts/train_eval_trajectory_tracking.sh` |
| 03 State-only PPO | runnable | `code/` | `configs/state_only_ppo.yaml` | `scripts/train_eval_state_only.sh` |
| 04 Flat-history PPO | runnable | `code/` | `configs/flat_history_ppo.yaml` | `scripts/train_eval_flat_history.sh` |
| 05 GLA no aux | runnable | `code/` | `configs/gla_no_aux.yaml` | `scripts/train_eval_gla_no_aux.sh` |
| 06 PICA no GLA ablation | runnable | `code/` | `configs/pica_no_gla_aux_v2c.yaml` | `scripts/train_eval_pica_no_gla.sh`, `scripts/train_eval_pica_no_gla_x1_x4.sh` |
| 07 PICA ours | run outside baselines | `/data/dyj/zts/pull-push` | main-method configs | main-method scripts |

Current missing main-table slots after this cleanup:

- `02`: not yet implemented/standardized here.

`PICA v2c150 reference` is kept only as a reference config snapshot, because the final ours pipeline should be run from `/data/dyj/zts/pull-push`.

## Notes

- This directory is an organizational layer. The actual executable code remains in `main_table_v2c_clean` for now.
- Do not add new ours experiments here; use `/data/dyj/zts/pull-push` for the main method.
- If a baseline requires code changes, create a named branch/worktree or record the patch before running.
- If a baseline only changes hyperparameters, add a new YAML under `configs/` and a small wrapper under `scripts/`.
- Historical trajectory tracking may return a non-zero IsaacGym cleanup code after writing valid summaries; the wrapper treats an existing `summary.csv` as success.
