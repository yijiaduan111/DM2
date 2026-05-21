#!/usr/bin/env python3
"""
Record the full pre-approach -> PPO-drag pipeline.

Runtime pipeline:
  1. HandDragTask.run_pre_policy_approach executes approach/grasp frames from
     --pre_trajectory inside Isaac Gym;
  2. optional transition moves to the PPO drag-start when pre/ppo trajectories
     differ;
  3. HandDragTask resets at --ppo_trajectory drag_start and runs the checkpoint.

The saved frames/GIF are observers of this runtime pipeline. PPO reward, action
space, network weights, and standard eval remain unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from isaacgym import gymapi, gymtorch  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_FLA_ROOT = str((PROJECT_ROOT / "flash-linear-attention").resolve())
if _FLA_ROOT not in sys.path:
    sys.path.insert(0, _FLA_ROOT)

from hand_object_gym import HandObjectGym  # noqa: E402
from ppo.hand_drag_task import HandDragTask  # noqa: E402
from ppo.rlgames_wrapper import resolve_handle_link_name  # noqa: E402
from scripts.evaluate_ppo_baseline import (  # noqa: E402
    GLAActor,
    RLGamesActor,
    _is_gla_checkpoint,
    apply_ood_dynamics_overrides,
    eval_step,
    infer_include_handle_rot,
    metric_row,
    resolve_checkpoint_path,
    summarize_episode,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record pre-approach trajectory frames + unchanged PPO rollout",
    )
    parser.add_argument("--config", default="hand_config.yaml")
    parser.add_argument("--object_id", required=True)
    parser.add_argument("--trajectory", default=None, help="Legacy alias for --pre_trajectory")
    parser.add_argument("--pre_trajectory", default=None, help="Trajectory used only for pre-roll approach/grasp")
    parser.add_argument("--ppo_trajectory", default=None, help="Trajectory defining the original PPO drag-start")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest"), default="latest")
    parser.add_argument("--target_joint_idx", type=int, default=None)
    parser.add_argument("--handle_link_name", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=("det", "stoch"), default="det")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--post_done_steps", type=int, default=0)
    parser.add_argument("--post_done_frame_stride", type=int, default=2)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--pre_frame_stride", type=int, default=3)
    parser.add_argument("--pre_steps_per_frame", type=int, default=2)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--transition_steps", type=int, default=30)
    parser.add_argument("--transition_frame_stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema_alpha", type=float, default=1.0)
    parser.add_argument("--detach_arm_delay", type=int, default=0)
    parser.add_argument("--object_damping_scale", type=float, default=1.0)
    parser.add_argument("--object_friction_scale", type=float, default=1.0)
    parser.add_argument("--gla_pool", choices=("last", "mean"), default="last")
    parser.add_argument("--phys_aux", type=int, default=None)
    parser.add_argument("--make_gif", action="store_true")
    args, gym_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + gym_args
    return args


def load_cfg(args):
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["HEADLESS"] = True
    cfg["num_envs"] = 1
    cfg.setdefault("cam", {})
    cfg["cam"]["use_cam"] = True
    cfg["cam"]["cam_w"] = 960
    cfg["cam"]["cam_h"] = 640
    cfg["asset"]["arti_gapartnet_ids"] = [int(args.object_id)]
    return cfg


def default_trajectory(object_id):
    return PROJECT_ROOT / "output" / "hand_drag" / str(object_id) / "trajectory.json"


def resolve_pre_trajectory(args):
    path = args.pre_trajectory or args.trajectory
    return Path(path) if path else default_trajectory(args.object_id)


def resolve_ppo_trajectory(args, pre_path):
    return Path(args.ppo_trajectory) if args.ppo_trajectory else pre_path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def first_drag_index(trajectory):
    for idx, frame in enumerate(trajectory):
        if frame.get("phase") == "drag":
            return idx
    return min(40, max(0, len(trajectory) - 1))


def hand_targets_from_frame(frame):
    return torch.tensor(
        frame["virtual_xyz"] + frame["wrist_rpy"] + frame["finger_dofs"],
        dtype=torch.float32,
    ).cpu().numpy()


def save_frame(env, frame_dir, frame_idx, tag, cam_idx=0):
    path = frame_dir / f"frame_{frame_idx:05d}_{tag}.png"
    env.save_camera_image(str(path), cam_idx=cam_idx)
    return frame_idx + 1


def play_preapproach(env, pre_traj, pre_drag_idx, ppo_start_hand, frame_dir, args):
    frame_idx = 0
    pre_end = max(0, pre_drag_idx)
    pre_rows = []
    print(
        f"  [pre] playing pre trajectory frames [0, {pre_end}); "
        f"pre drag idx={pre_drag_idx}"
    )

    current = None
    for traj_idx, frame in enumerate(pre_traj[:pre_end]):
        current = hand_targets_from_frame(frame)
        env.set_hand_dof_targets(current)
        env.run_steps(int(args.pre_steps_per_frame), refresh_obs=True)
        if traj_idx % args.pre_frame_stride == 0 or traj_idx == pre_end - 1:
            frame_idx = save_frame(env, frame_dir, frame_idx, "pre", args.cam_idx)
        pre_rows.append(
            {
                "segment": "pre",
                "trajectory_idx": int(traj_idx),
                "phase": frame.get("phase", ""),
                "step": int(frame.get("step", traj_idx)),
            }
        )

    if current is None:
        current = env.get_current_hand_targets()

    n_transition = max(0, int(args.transition_steps))
    if n_transition > 0:
        print(f"  [pre] interpolating to original PPO drag-start in {n_transition} steps")
    for step in range(n_transition):
        frac = float(step + 1) / float(n_transition)
        targets = (1.0 - frac) * current + frac * ppo_start_hand
        env.set_hand_dof_targets(targets)
        env.run_steps(int(args.pre_steps_per_frame), refresh_obs=True)
        if step % args.transition_frame_stride == 0 or step == n_transition - 1:
            frame_idx = save_frame(env, frame_dir, frame_idx, "to_ppo_start", args.cam_idx)
        pre_rows.append(
            {
                "segment": "transition_to_ppo_start",
                "trajectory_idx": None,
                "phase": "transition",
                "step": int(step),
            }
        )

    return frame_idx, pre_rows


def make_actor(checkpoint, task, stochastic, gla_pool, phys_aux):
    if _is_gla_checkpoint(checkpoint["model"]):
        return GLAActor(
            checkpoint,
            task.device,
            history_length=task.history_len,
            stochastic=stochastic,
            pool=gla_pool,
            phys_aux=phys_aux,
        )
    return RLGamesActor(checkpoint, task.device, stochastic=stochastic)


def infer_phys_aux_cfg(args, checkpoint):
    is_gla = _is_gla_checkpoint(checkpoint["model"])
    aux_keys = any(k.startswith("a2c_network.aux_head.") for k in checkpoint["model"])
    eval_phys_aux = bool(aux_keys) if args.phys_aux is None else bool(int(args.phys_aux))
    if eval_phys_aux and not is_gla:
        eval_phys_aux = False
    if not eval_phys_aux:
        return False, None

    pred_dim = 0
    if "a2c_network.aux_head.2.weight" in checkpoint["model"]:
        pred_dim = int(checkpoint["model"]["a2c_network.aux_head.2.weight"].shape[0])

    if pred_dim == 4:
        return True, {
            "enabled": True,
            "mode": "causal_horizon",
            "horizon": 5,
            "targets": {
                "q_response_K": {"enabled": True},
                "max_dist_K": {"enabled": True},
                "detach_proxy_K": {"enabled": True},
                "tracking_stress": {"enabled": True},
            },
            "gating": {"enabled": False, "d_valid": 0.10, "sharpness": 80.0},
            "warmup": {"enabled": False, "max_weight": 0.0},
        }

    return True, {
        "enabled": True,
        "mode": "current",
        "targets": {
            "dq_obj": {"enabled": True},
            "slip_proxy": {"enabled": True},
            "tracking_stress": {"enabled": True},
        },
        "gating": {"enabled": False, "d_valid": 0.10, "sharpness": 80.0},
        "warmup": {"enabled": False, "max_weight": 0.0},
    }


def maybe_write_gif(frame_dir, out_dir):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"  [gif][skip] imageio unavailable: {exc}")
        return None

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return None
    gif_path = out_dir / "preapproach_policy_rollout.gif"
    images = [imageio.imread(path) for path in frames]
    imageio.mimsave(gif_path, images, duration=0.08)
    return gif_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_kind)
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    pre_traj_path = resolve_pre_trajectory(args)
    ppo_traj_path = resolve_ppo_trajectory(args, pre_traj_path)
    pre_traj = load_json(pre_traj_path)
    ppo_traj = load_json(ppo_traj_path)
    pre_drag_idx = first_drag_index(pre_traj)
    ppo_drag_idx = first_drag_index(ppo_traj)
    ppo_start_hand = hand_targets_from_frame(ppo_traj[ppo_drag_idx])

    include_handle_rot, include_prev_action = infer_include_handle_rot(checkpoint, ppo_traj)
    eval_phys_aux, eval_phys_aux_cfg = infer_phys_aux_cfg(args, checkpoint)

    cfg = load_cfg(args)
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

    frame_idx = 0
    pre_rows = []

    def capture_pre_frame(row):
        nonlocal frame_idx
        pre_rows.append(row)
        segment = row["segment"]
        row_idx = len(pre_rows) - 1
        if segment == "pre":
            traj_idx = row["trajectory_idx"]
            if traj_idx % args.pre_frame_stride == 0 or traj_idx == pre_drag_idx - 1:
                frame_idx = save_frame(env, frame_dir, frame_idx, "pre", args.cam_idx)
        elif segment == "transition_to_ppo_start":
            if row_idx % args.transition_frame_stride == 0:
                frame_idx = save_frame(env, frame_dir, frame_idx, "to_ppo_start", args.cam_idx)

    pre_result = task.run_pre_policy_approach(
        trajectory_path=str(pre_traj_path),
        steps_per_frame=int(args.pre_steps_per_frame),
        transition_steps=int(args.transition_steps),
        frame_callback=capture_pre_frame,
    )
    pre_drag_idx = int(pre_result["drag_start_frame_idx"])
    pre_rows = pre_result["rows"]

    actor = make_actor(
        checkpoint,
        task,
        stochastic=(args.mode == "stoch"),
        gla_pool=args.gla_pool,
        phys_aux=eval_phys_aux,
    )

    rows = []
    obs = task.reset()
    frame_idx = save_frame(env, frame_dir, frame_idx, "ppo_start", args.cam_idx)
    done_step = None
    for step in range(args.max_steps):
        if step < args.detach_arm_delay:
            task.detach_armed_buf.zero_()
        with torch.no_grad():
            action = actor(obs).clamp(-1.0, 1.0)
        obs, reward, done = eval_step(task, action)
        row = metric_row(task, 0, step, reward, done)
        rows.append(row)
        if step % args.frame_stride == 0 or bool(done[0].item()):
            frame_idx = save_frame(env, frame_dir, frame_idx, "ppo", args.cam_idx)
        if bool(done[0].item()):
            done_step = int(step)
            break

    for hold_step in range(max(0, int(args.post_done_steps))):
        env.run_steps(1, refresh_obs=True)
        if hold_step % args.post_done_frame_stride == 0 or hold_step == args.post_done_steps - 1:
            frame_idx = save_frame(env, frame_dir, frame_idx, "post_done", args.cam_idx)

    summary = summarize_episode(rows) if rows else {}
    summary.update(
        {
            "object_id": str(args.object_id),
            "pre_trajectory": str(pre_traj_path),
            "ppo_trajectory": str(ppo_traj_path),
            "checkpoint": str(ckpt_path),
            "mode": args.mode,
            "pre_drag_start_frame_idx": int(pre_drag_idx),
            "ppo_drag_start_frame_idx": int(ppo_drag_idx),
            "pre_frames_played": int(pre_drag_idx),
            "transition_steps": int(args.transition_steps),
            "done_step": done_step,
            "post_done_steps": int(args.post_done_steps),
            "recorded_frames": int(frame_idx),
            "cam_idx": int(args.cam_idx),
            "handle_link_name": handle_link_name,
            "target_joint_idx": int(task.target_joint_idx),
            "object_damping_scale": float(args.object_damping_scale),
            "object_friction_scale": float(args.object_friction_scale),
            "object_damping_actual": damp_after,
            "object_friction_actual": fric_after,
            "note": (
                "pre segment is deterministic trajectory/algorithmic pre-roll; "
                "transition moves hand to original PPO drag-start; "
                "PPO policy then starts from unchanged ppo_trajectory drag_start"
            ),
        }
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "preapproach_trace.json", "w") as f:
        json.dump(pre_rows, f, indent=2)
    if rows:
        with open(out_dir / "ppo_metrics.csv", "w") as f:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[key]) for key in keys) + "\n")

    if args.make_gif:
        gif_path = maybe_write_gif(frame_dir, out_dir)
        if gif_path is not None:
            summary["gif"] = str(gif_path)
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  [gif] wrote {gif_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
