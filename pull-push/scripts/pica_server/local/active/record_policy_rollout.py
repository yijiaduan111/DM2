#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import numpy as np

# Isaac Gym before torch
from isaacgym import gymapi, gymtorch  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
    p = argparse.ArgumentParser()
    p.add_argument('--object_id', required=True)
    p.add_argument('--trajectory', default=None, help='Legacy alias for --ppo_trajectory and --pre_trajectory')
    p.add_argument('--use_preapproach', action='store_true', help='Replay dataset approach frames before the PPO drag rollout for demo videos only')
    p.add_argument('--pre_trajectory', default=None, help='Trajectory used for the pre-approach replay segment')
    p.add_argument('--ppo_trajectory', default=None, help='Trajectory used to define the unchanged PPO drag-start state')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-kind', default='latest', choices=['latest','best'])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mode', default='det', choices=['det','stoch'])
    p.add_argument('--damping', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=300)
    p.add_argument('--frame_stride', type=int, default=5)
    p.add_argument('--pre_frame_stride', type=int, default=3)
    p.add_argument('--pre_steps_per_frame', type=int, default=2)
    p.add_argument('--transition_steps', type=int, default=30)
    p.add_argument('--transition_frame_stride', type=int, default=3)
    p.add_argument('--cam_idx', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', default='hand_config.yaml')
    p.add_argument('--gla_pool', default='last')
    p.add_argument('--phys_aux', default=None)
    p.add_argument('--target_joint_idx', type=int, default=None)
    p.add_argument('--handle_link_name', default=None)
    p.add_argument('--ema_alpha', type=float, default=1.0)
    p.add_argument('--detach_arm_delay', type=int, default=0)
    args, gym_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + gym_args
    return args




def default_trajectory(object_id):
    return PROJECT_ROOT / 'output' / 'hand_drag' / str(object_id) / 'trajectory.json'


def resolve_pre_trajectory(args):
    path = args.pre_trajectory or args.trajectory
    return Path(path) if path else default_trajectory(args.object_id)


def resolve_ppo_trajectory(args):
    path = args.ppo_trajectory or args.trajectory
    return Path(path) if path else default_trajectory(args.object_id)


def first_drag_index(trajectory):
    for idx, frame in enumerate(trajectory):
        if frame.get('phase') == 'drag':
            return idx
    return min(40, max(0, len(trajectory) - 1))


def hand_targets_from_frame(frame):
    return np.asarray(
        frame['virtual_xyz'] + frame['wrist_rpy'] + frame['finger_dofs'],
        dtype=np.float32,
    )


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w') as f:
        keys = list(rows[0].keys())
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row[k]) for k in keys) + '\n')

def load_cfg(args):
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['HEADLESS'] = True
    cfg['num_envs'] = 1
    cfg['cam']['use_cam'] = True
    cfg['cam']['cam_w'] = 960
    cfg['cam']['cam_h'] = 640
    cfg['asset']['arti_gapartnet_ids'] = [int(args.object_id)]
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.checkpoint_kind)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    traj_path = resolve_ppo_trajectory(args)
    pre_traj_path = resolve_pre_trajectory(args)
    with open(traj_path) as f:
        trajectory = json.load(f)
    pre_trajectory = None
    if args.use_preapproach:
        with open(pre_traj_path) as f:
            pre_trajectory = json.load(f)

    inferred_obs_flags = infer_include_handle_rot(checkpoint, trajectory)
    if len(inferred_obs_flags) == 2:
        include_handle_rot, include_prev_action = inferred_obs_flags
        include_task_progress = False
    else:
        include_handle_rot, include_prev_action, include_task_progress = inferred_obs_flags
    is_gla = _is_gla_checkpoint(checkpoint['model'])
    aux_keys = any(k.startswith('a2c_network.aux_head.') for k in checkpoint['model'])
    if args.phys_aux is None:
        eval_phys_aux = bool(aux_keys)
    else:
        eval_phys_aux = bool(int(args.phys_aux))
    if eval_phys_aux and not is_gla:
        eval_phys_aux = False
    eval_phys_aux_cfg = None
    if eval_phys_aux:
        pred_dim = 0
        if 'a2c_network.aux_head.2.weight' in checkpoint['model']:
            pred_dim = int(checkpoint['model']['a2c_network.aux_head.2.weight'].shape[0])
        if pred_dim == 4:
            eval_phys_aux_cfg = {
                'enabled': True, 'mode': 'causal_horizon', 'horizon': 5,
                'targets': {
                    'q_response_K': {'enabled': True},
                    'max_dist_K': {'enabled': True},
                    'detach_proxy_K': {'enabled': True},
                    'tracking_stress': {'enabled': True},
                },
                'gating': {'enabled': False, 'd_valid': 0.10, 'sharpness': 80.0},
                'warmup': {'enabled': False, 'max_weight': 0.0},
            }
        else:
            eval_phys_aux_cfg = {
                'enabled': True, 'mode': 'current',
                'targets': {
                    'dq_obj': {'enabled': True},
                    'slip_proxy': {'enabled': True},
                    'tracking_stress': {'enabled': True},
                },
                'gating': {'enabled': False, 'd_valid': 0.10, 'sharpness': 80.0},
                'warmup': {'enabled': False, 'max_weight': 0.0},
            }

    cfg = load_cfg(args)
    env = HandObjectGym(cfg)
    env.get_gapartnet_anno()
    env.run_steps(50, refresh_obs=True)
    handle = resolve_handle_link_name(
        env,
        args.handle_link_name,
        target_joint_idx=args.target_joint_idx,
        trajectory_path=str(traj_path),
        env_config=cfg,
        object_id=args.object_id,
    )
    damp_after, fric_after = apply_ood_dynamics_overrides(env, damping_scale=args.damping, friction_scale=1.0)

    task = HandDragTask(
        env,
        trajectory_path=str(traj_path),
        target_joint_idx=args.target_joint_idx,
        handle_link_name=handle,
        include_handle_rot=include_handle_rot,
        is_eval_mode=True,
        epoch_log_path=None,
        include_prev_action_in_history=include_prev_action,
        physical_auxiliary=eval_phys_aux_cfg,
    )
    task.max_episode_length = int(args.max_steps)
    task.eval_action_ema_alpha = float(args.ema_alpha)

    if is_gla:
        actor = GLAActor(checkpoint, task.device, history_length=task.history_len, stochastic=(args.mode == 'stoch'), pool=args.gla_pool, phys_aux=eval_phys_aux)
    else:
        actor = RLGamesActor(checkpoint, task.device, stochastic=(args.mode == 'stoch'))
        # Some v2c no-GLA checkpoints kept aux-target normalization stats at the tail
        # even though the actor input itself is the shorter base observation.
        if hasattr(actor, 'running_mean') and actor.running_mean.shape[-1] > actor.obs_dim:
            actor.running_mean = actor.running_mean[:actor.obs_dim]
            actor.running_var = actor.running_var[:actor.obs_dim]

    rows = []
    obs = task.reset()
    ppo_start_hand = task.ready_grasp_hand.detach().cpu().numpy().astype(np.float32)
    frame_i = 0

    def save_frame(tag=None):
        nonlocal frame_i
        # IsaacGym headless cameras can otherwise keep returning the first image.
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        suffix = f'_{tag}' if tag else ''
        env.save_camera_image(str(frame_dir / f'frame_{frame_i:04d}{suffix}.png'), cam_idx=args.cam_idx)
        frame_i += 1

    def play_preapproach():
        nonlocal frame_i
        if not args.use_preapproach:
            return []
        pre_drag_idx = first_drag_index(pre_trajectory)
        pre_end = max(0, pre_drag_idx)
        pre_rows = []
        current = None
        print(
            f"  [pre] playing approach frames [0, {pre_end}); "
            f"pre_traj={pre_traj_path} ppo_traj={traj_path}"
        )
        for traj_idx, frame in enumerate(pre_trajectory[:pre_end]):
            current = hand_targets_from_frame(frame)
            env.set_hand_dof_targets(current)
            env.run_steps(int(args.pre_steps_per_frame), refresh_obs=True)
            if traj_idx % args.pre_frame_stride == 0 or traj_idx == pre_end - 1:
                save_frame('pre')
            pre_rows.append({
                'segment': 'pre',
                'trajectory_idx': int(traj_idx),
                'phase': frame.get('phase', ''),
                'step': int(frame.get('step', traj_idx)),
            })
        if current is None:
            current = env.get_current_hand_targets().astype(np.float32)
        n_transition = max(0, int(args.transition_steps))
        if n_transition > 0:
            print(f"  [pre] interpolating to PPO drag-start in {n_transition} steps")
        for step in range(n_transition):
            frac = float(step + 1) / float(n_transition)
            targets = ((1.0 - frac) * current + frac * ppo_start_hand).astype(np.float32)
            env.set_hand_dof_targets(targets)
            env.run_steps(int(args.pre_steps_per_frame), refresh_obs=True)
            if step % args.transition_frame_stride == 0 or step == n_transition - 1:
                save_frame('to_ppo_start')
            pre_rows.append({
                'segment': 'transition_to_ppo_start',
                'trajectory_idx': '',
                'phase': 'transition',
                'step': int(step),
            })
        return pre_rows

    try:
        pre_rows = play_preapproach()
        if args.use_preapproach:
            obs = task.reset()
        save_frame('ppo_start' if args.use_preapproach else None)
        for step in range(args.max_steps):
            if step < args.detach_arm_delay:
                task.detach_armed_buf.zero_()
            with torch.no_grad():
                action = actor(obs).clamp(-1.0, 1.0)
            obs, reward, done = eval_step(task, action)
            row = metric_row(task, 0, step, reward, done)
            rows.append(row)
            if step % args.frame_stride == 0 or bool(done[0].item()):
                save_frame('ppo' if args.use_preapproach else None)
            if bool(done[0].item()):
                break
        summary = summarize_episode(rows)
        summary.update({
            'object_id': str(args.object_id),
            'trajectory': str(traj_path),
            'preapproach_enabled': bool(args.use_preapproach),
            'pre_trajectory': str(pre_traj_path) if args.use_preapproach else None,
            'checkpoint': str(ckpt_path),
            'mode': args.mode,
            'damping': args.damping,
            'damping_actual': damp_after,
            'frames': frame_i,
            'cam_idx': args.cam_idx,
            'handle_link_name': handle,
            'success_any': int(any(r['success'] for r in rows)),
        })
        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        write_csv(out_dir / 'metrics.csv', rows)
        if args.use_preapproach:
            with open(out_dir / 'preapproach_trace.json', 'w') as f:
                json.dump(pre_rows, f, indent=2)
            write_csv(out_dir / 'preapproach_trace.csv', pre_rows)
        print(json.dumps(summary, indent=2))

    finally:
        pass

if __name__ == '__main__':
    main()
