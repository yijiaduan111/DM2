# PPO Demo With Pre-Approach

This worktree is copied from the trusted `pica_v2c150_v2d_both_ft50` code package and adds one visualization-only script:

- `scripts/record_preapproach_policy_rollout.py`

## What It Does

The script records a complete demo in two segments:

1. **Pre segment**: replay trajectory frames before the first `phase == "drag"` frame so the hand/palm approaches and grasps naturally.
2. **PPO segment**: reset to the original PPO drag-start state and run the unchanged PPO checkpoint.

It intentionally does **not** change:

- PPO reward
- PPO action space
- PPO network architecture
- checkpoint weights
- PPO start state

So this is for natural demo visualization, not for claiming the policy learned the approach motion.

## Example Command

Run from this project root on the server:

```bash
python scripts/record_preapproach_policy_rollout.py \
  --object_id 45936 \
  --trajectory output/hand_drag/45936/trajectory.json \
  --checkpoint /path/to/old_trusted_v2d_both_ft50/nn \
  --checkpoint-kind latest \
  --out_dir output/preapproach_demo/45936_det \
  --mode det \
  --max_steps 300 \
  --frame_stride 5 \
  --pre_frame_stride 3 \
  --pre_steps_per_frame 2 \
  --make_gif
```

Outputs:

- `frames/frame_*.png`: full pre-approach + PPO rollout frames
- `preapproach_policy_rollout.gif`: optional GIF if `imageio` works
- `summary.json`: PPO segment metrics and provenance
- `preapproach_trace.json`: trajectory frames used in the pre segment
- `ppo_metrics.csv`: step metrics after PPO takes over

## Interpretation

When presenting results, describe the video as:

> The approach/grasp segment is deterministic trajectory/algorithmic pre-roll. PPO starts from the original drag-start state and only controls the drag/attack segment.
