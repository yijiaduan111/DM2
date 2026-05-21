# scripts/pica_server/

Server-side scripts for the PICA v1 calibration. **Do not run locally** —
the local machine has unstable power.

## What's here

| script | purpose |
|---|---|
| `run_pica_gla_calibration_20ep.sh` | 20-epoch PICA-GLA training × 3 `lambda_bound` settings ({1.0, 5.0, 20.0}); fixes contact_distance.weight=0.5; slip + smooth disabled |
| `summarize_pica_calibration.py` | reads each run's `epoch_rewards.csv` at epoch 20 and writes `reports/pica_handoff/pica_calibration_20ep_summary.md` |
| `eval_pica_calibration_nominal.sh` | nominal damping x1 deterministic eval of each calibration checkpoint (10 eps, seed 42, max_len 300) |

## How to run on the server

```bash
# 0. Verify environment
python -m py_compile ppo/hand_drag_task.py ppo/rlgames_wrapper.py ppo/train.py
python -c "import yaml; [yaml.safe_load(open(p)) for p in ['ppo/train_config_gla_pica.yaml','ppo/train_config_flat_pica.yaml']]"

# 1. Calibration (3 × 20-epoch runs, ~1-2 hours total on a single 24 GB GPU)
bash scripts/pica_server/run_pica_gla_calibration_20ep.sh

# 2. Summarize epoch-20 metrics
python scripts/pica_server/summarize_pica_calibration.py
# Result: reports/pica_handoff/pica_calibration_20ep_summary.md

# 3. (Optional) nominal eval of all three checkpoints
bash scripts/pica_server/eval_pica_calibration_nominal.sh
python scripts/eval_postprocess.py output/pica_calibration_45936/*_metrics.csv
```

## Decision protocol

Compare the three rows in `pica_calibration_20ep_summary.md` against the
disabled-PICA baseline at the same epoch (`runs/hand_drag_gla_45936_smoke20_bounds01/epoch_rewards.csv`,
or the local smoke runs in `reports/pica_handoff/smoke_csv/`).

Pick the largest `lambda_bound` such that:

- `r_phys_bound_mean` is visibly non-zero at epoch 20 (|value| > 1e-3 is a
  reasonable lower bar — at 20 epochs the policy hasn't fully saturated
  yet, but the term should already be active).
- Training `success_mean` and `normalized_progress_mean` are not collapsed
  versus disabled-PICA. A drop of ~5-10 % in progress is acceptable; a
  collapse to zero is not.

Tie-breakers (in order):
1. `r_phys_contact_mean` close to disabled baseline (contact stays well-behaved).
2. `r_phys_bound_mean` the most negative (strongest active penalty).
3. Highest `success_mean`.

## Results to send back to mentor / ChatGPT

After step 2 above:

- `reports/pica_handoff/pica_calibration_20ep_summary.md`
- The three `runs/<exp>/epoch_rewards.csv` files (zipped is fine)
- Decision: which `lambda_bound` was chosen, and why

After step 3 (optional but recommended):

- `output/pica_calibration_45936/*_summary.json`
- The `eval_postprocess.py` Markdown table

## Expected next step after a winner is chosen

1. Edit `ppo/train_config_gla_pica.yaml::physical_regularization.action_bound.weight`
   to the chosen value (or keep using `--lambda_bound`).
2. **PICA-GLA 100 epochs** with the same protocol as `gla_long_bounds01`.
3. **PICA-Flat 100 epochs** to confirm the framework is encoder-agnostic.
4. **Damping x1 / x2 / x4 deterministic eval** on both 100-epoch checkpoints,
   identical protocol to `output/gla_tuning_45936/long_damping/`.
5. Compare against `gla_long_bounds01` and `flat_history_long_bounds01` —
   the PICA win condition is **non-zero damping x2 success while keeping
   nominal x1 success ≥ 0.90**.

Steps 2-4 are not in this script set yet — write them after the
calibration winner is locked in.
