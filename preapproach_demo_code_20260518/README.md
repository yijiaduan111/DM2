# Pull-Push PICA Release Code

This is a cleaned release-style code package for the PICA hand-object pull/push experiments.

## What Is Included

- `ppo/`: PPO training code, task implementation, GLA/PICA network and training configs.
- `scripts/evaluate_ppo_baseline.py`: generic evaluation entry point.
- `flash-linear-attention/`: vendored GLA dependency used by the actor-critic network.
- `hand_object_gym.py`: hand/object environment and asset loading utilities.
- `hand_config.yaml`: default hand/object configuration using relative paths.
- `smplx_right_hand_floating.urdf`: hand URDF used by the simulator.
- `assets/`, `dataset/`, `output/`, `runs/`, `checkpoints/`: kept as repository layout placeholders; their contents are ignored by git.

## What Is Not Included

- Server-specific launch scripts, conda paths, GPU IDs, SSH details, and machine-local environment files.
- Trained checkpoints and run directories.
- GAPartNet assets and generated trajectory data.
- Historical experiment outputs, debugging logs, and cache files.

## Main Pipeline

The trusted experimental recipe is:

1. Train v2c base policy with `ppo/train_config_gla_pica_drand12_aux_v2c.yaml`.
2. Fine-tune with v2d Both using `ppo/train_config_gla_pica_v2d_both.yaml`.
3. Evaluate with `scripts/evaluate_ppo_baseline.py`.

The exact epoch counts used in our best internal pipeline were v2c 150 epochs followed by v2d Both fine-tuning for 50 epochs. This repository keeps the code and configs, but not private run outputs or checkpoints.

## Path Setup

`hand_config.yaml` uses relative paths by default:

- `hand.hand_asset_root: .`
- `asset.asset_root: assets`

Place object assets and trajectory files under `assets/` following the expected GAPartNet layout before training or evaluation.
