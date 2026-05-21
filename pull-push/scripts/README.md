# Scripts Layout

This directory now has a lightweight standard entrypoint layer for daily PICA experiments.
The cleanup does not change algorithm code. Existing historical scripts are kept in place for traceability.

## Standard Entrypoints

```text
scripts/
├── train/
│   ├── train_v2c150.sh
│   ├── finetune_v2d_both50.sh
│   └── train_v2c150_v2d_both50_pipeline.sh
├── eval/
│   └── eval_det_stoch_damping.sh
├── video/
│   └── README.md
├── utils/
│   └── check_checkpoint.sh
└── legacy/
```

## Main Method Training

Train the v2c 150ep stage:

```bash
cd /data/dyj/zts/pull-push
OBJECT_ID=45936 GPU=0 RUN_NAME=my_45936_v2c150 \
  bash scripts/train/train_v2c150.sh
```

Fine-tune v2d Both for 50 more epochs from a v2c checkpoint:

```bash
cd /data/dyj/zts/pull-push
OBJECT_ID=45936 GPU=0 RUN_NAME=my_45936_v2d_both50 \
CHECKPOINT=runs/my_45936_v2c150/nn/HandDrag.pth CHECKPOINT_KIND=best \
  bash scripts/train/finetune_v2d_both50.sh
```

Run the two-stage pipeline:

```bash
cd /data/dyj/zts/pull-push
OBJECT_ID=45936 GPU=0 TAG=test45936 \
  bash scripts/train/train_v2c150_v2d_both50_pipeline.sh
```

Default configs:

- v2c: `ppo/train_config_gla_pica_drand12_aux_v2c.yaml`
- v2d Both FT: `ppo/train_config_gla_pica_v2d_both.yaml`

Important resume note:

- `rl_games` resumes `epoch_num` from the checkpoint.
- For `v2c150 + 50ep FT`, the v2d `MAX_EPOCHS` default is `200`, not `50`.

## Main Method Eval

Evaluate deterministic and stochastic rollouts over damping x1/x2/x4:

```bash
cd /data/dyj/zts/pull-push
OBJECT_ID=45936 GPU=0 RUN_NAME=my_eval \
CHECKPOINT=runs/my_45936_v2d_both50/nn/HandDrag.pth CHECKPOINT_KIND=best \
OUT_ROOT=output/eval_standard/my_eval \
  bash scripts/eval/eval_det_stoch_damping.sh
```

Defaults:

- `EPISODES=20`
- `DAMPS="1.0 2.0 4.0"`
- `MAX_LEN=300`
- `GLA_POOL=last`

## Utilities

List checkpoints for a run or checkpoint directory:

```bash
bash scripts/utils/check_checkpoint.sh my_45936_v2d_both50
bash scripts/utils/check_checkpoint.sh runs/my_45936_v2d_both50/nn
```

## Historical Scripts

Older exploration scripts are still kept where they were, especially:

- `scripts/pica_server/local/`
- `scripts/pica_server/local/active/`
- top-level historical eval/screening scripts

Do not delete or move them until the corresponding experiment is confirmed obsolete.
If an old script is still useful, wrap it with a small standard entrypoint instead of editing it in place.

## Current Policy

- Use `scripts/train/` and `scripts/eval/` for new routine runs.
- Use `scripts/pica_server/local/active/` only for older batch jobs that have not yet been standardized.
- Do not modify algorithm code during script cleanup.
- If a new experiment requires changing `ppo/`, `hand_object_gym.py`, or `utils.py`, stop and record the intended code change first.
