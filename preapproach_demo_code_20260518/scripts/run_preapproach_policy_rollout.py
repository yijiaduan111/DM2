#!/usr/bin/env python3
"""Run the full pre-approach -> PPO-drag pipeline without recording video."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from hand_object_gym import HandObjectGym
from ppo.hand_drag_task import HandDragTask
from record_preapproach_policy_rollout import infer_include_handle_rot, resolve_checkpoint_path
from record_preapproach_policy_rollout import (
    apply_ood_dynamics_overrides,
    default_trajectory,
    eval_step,
    infer_phys_aux_cfg,
    load_cfg,
    make_actor,
    metric_row,
    resolve_ppo_trajectory,
    resolve_handle_link_name,
    summarize_episode,
)
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hand_config.yaml")
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--trajectory", default=None, help="Legacy alias for --pre_trajectory")
    parser.add_argument("--pre_trajectory", default=None)
    parser.add_argument("--ppo_trajectory", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"), default="latest")
    parser.add_argument("--target_joint_idx", type=int, default=None)
    parser.add_argument("--handle_link_name", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=("det", "stoch"), default="det")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--pre_steps_per_frame", type=int, default=1)
    parser.add_argument("--transition_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_alpha", type=float, default=1.0)
    parser.add_argument("--detach_arm_delay", type=int, default=0)
    parser.add_argument("--object_damping_scale", type=float, default=1.0)
    parser.add_argument("--object_friction_scale", type=float, default=1.0)
    parser.add_argument("--gla_pool", choices=("last", "mean"), default="last")
    parser.add_argument("--phys_aux", type=int, default=None)
    args, gym_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + gym_args
    return args


def resolve_pre_trajectory(args):
    path = args.pre_trajectory or args.trajectory
    return Path(path) if path else default_trajectory(args.object_id)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_kind)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pre_traj_path = resolve_pre_trajectory(args)
    ppo_traj_path = resolve_ppo_trajectory(args, pre_traj_path)

    with open(ppo_traj_path) as f:
        ppo_traj = json.load(f)
    include_handle_rot, include_prev_action = infer_include_handle_rot(checkpoint, ppo_traj)
    eval_phys_aux, eval_phys_aux_cfg = infer_phys_aux_cfg(args, checkpoint)

    cfg = load_cfg(args)
    cfg["cam"]["use_cam"] = False
    env = HandObjectGym(cfg)
    env.get_gapartnet_anno()
    env.run_steps(50, refresh_obs=True)
    handle_link_name = resolve_handle_link_name(env, args.handle_link_name)
    damp_after, fric_after = apply_ood_dynamics_overrides(
        env,
        damping_scale=float(args.object_damping_scale),
        friction_scale=float(args.object_friction_scale),
    )

    task = HandDragTask(
        env,
        trajectory_path=str(ppo_traj_path),
        target_joint_idx=args.target_joint_idx,
        handle_link_name=handle_link_name,
        include_handle_rot=include_handle_rot,
        is_eval_mode=True,
        epoch_log_path=None,
        include_prev_action_in_history=include_prev_action,
        physical_auxiliary=eval_phys_aux_cfg,
    )
    task.max_episode_length = int(args.max_steps)
    task.eval_action_ema_alpha = float(args.ema_alpha)

    pre_result = task.run_pre_policy_approach(
        trajectory_path=str(pre_traj_path),
        steps_per_frame=int(args.pre_steps_per_frame),
        transition_steps=int(args.transition_steps),
        frame_callback=None,
    )

    actor = make_actor(
        checkpoint,
        task,
        stochastic=(args.mode == "stoch"),
        gla_pool=args.gla_pool,
        phys_aux=eval_phys_aux,
    )

    rows = []
    obs = task.reset()
    done_step = None
    for step in range(args.max_steps):
        if step < args.detach_arm_delay:
            task.detach_armed_buf.zero_()
        with torch.no_grad():
            action = actor(obs).clamp(-1.0, 1.0)
        obs, reward, done = eval_step(task, action)
        rows.append(metric_row(task, 0, step, reward, done))
        if bool(done[0].item()):
            done_step = int(step)
            break

    summary = summarize_episode(rows) if rows else {}
    summary.update({
        "object_id": str(args.object_id),
        "pre_trajectory": str(pre_traj_path),
        "ppo_trajectory": str(ppo_traj_path),
        "checkpoint": str(ckpt_path),
        "mode": args.mode,
        "pre_drag_start_frame_idx": int(pre_result["drag_start_frame_idx"]),
        "ppo_drag_start_frame_idx": int(task.drag_start_frame_idx),
        "pre_frames_played": int(pre_result["pre_frames_played"]),
        "transition_steps": int(pre_result["transition_steps"]),
        "done_step": done_step,
        "handle_link_name": handle_link_name,
        "target_joint_idx": int(task.target_joint_idx),
        "object_damping_scale": float(args.object_damping_scale),
        "object_friction_scale": float(args.object_friction_scale),
        "object_damping_actual": damp_after,
        "object_friction_actual": fric_after,
        "note": "Full pipeline run: deterministic trajectory approach is executed before PPO drag-start; no video frames are recorded.",
    })

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "preapproach_trace.json", "w") as f:
        json.dump(pre_result["rows"], f, indent=2)
    if rows:
        with open(out_dir / "ppo_metrics.csv", "w") as f:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[key]) for key in keys) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()