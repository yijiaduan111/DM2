# Parallel-Jaw Manipulation Evaluation

This evaluates a GAPartNet-style part-centric parallel-jaw baseline:

```text
GT Part Pose + Parallel-Jaw Pulling
```

The script reuses the original `run_arti_open` primitive:

1. load one GAPartNet articulated object,
2. read the annotated GAPart bounding box,
3. move the Franka parallel jaw to a pre-grasp pose,
4. close the gripper,
5. pull along the annotated handle outward direction,
6. report articulated object DOF progress.

Run from `GAPartNet/manipulation`:

```bash
PYTHONPATH=/home/plote/isaacgym/python \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121 \
LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH \
/home/plote/miniconda3/envs/isaacgym/bin/python evaluate_parallel_jaw.py \
  --object_id 45661 \
  --headless \
  --output output/parallel_jaw_eval/45661
```

Outputs:

- `summary.json`: paper-table level metrics.
- `metrics.csv`: per-stage DOF and gripper-target distance diagnostics.

Main fields:

- `success`: whether max normalized articulated DOF progress exceeds `--success_threshold`.
- `normalized_abs_progress`: largest normalized absolute DOF displacement.
- `target_dof`: DOF with largest normalized displacement.
- `min_gripper_target_dist`: closest observed distance from gripper hand frame to annotated part target point.
- `final_gripper_target_dist`: same proxy distance at the final stage.

The baseline should be reported as a part-centric manipulation primitive, not as a direct reimplementation of the full GAPartNet perception pipeline unless predicted part poses are used.
