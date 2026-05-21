#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

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
    p.add_argument('--trajectory', default=None)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--checkpoint-kind', default='latest', choices=['latest','best'])
    p.add_argument('--out_dir', required=True)
    p.add_argument('--mode', default='det', choices=['det','stoch'])
    p.add_argument('--damping', type=float, default=1.0)
    p.add_argument('--max_steps', type=int, default=300)
    p.add_argument('--frame_stride', type=int, default=5)
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
    traj_path = Path(args.trajectory) if args.trajectory else PROJECT_ROOT / 'output' / 'hand_drag' / args.object_id / 'trajectory.json'
    with open(traj_path) as f:
        trajectory = json.load(f)

    include_handle_rot, include_prev_action, include_task_progress = infer_include_handle_rot(checkpoint, trajectory)
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
    handle = resolve_handle_link_name(env, args.handle_link_name)
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
        include_task_progress_obs=include_task_progress,
        physical_auxiliary=eval_phys_aux_cfg,
    )
    task.max_episode_length = int(args.max_steps)
    task.eval_action_ema_alpha = float(args.ema_alpha)

    if is_gla:
        actor = GLAActor(checkpoint, task.device, history_length=task.history_len, stochastic=(args.mode == 'stoch'), pool=args.gla_pool, phys_aux=eval_phys_aux)
    else:
        actor = RLGamesActor(checkpoint, task.device, stochastic=(args.mode == 'stoch'))

    rows = []
    obs = task.reset()
    frame_i = 0
    try:
        env.save_camera_image(str(frame_dir / f'frame_{frame_i:04d}.png'), cam_idx=args.cam_idx)
        frame_i += 1
        for step in range(args.max_steps):
            if step < args.detach_arm_delay:
                task.detach_armed_buf.zero_()
            with torch.no_grad():
                action = actor(obs).clamp(-1.0, 1.0)
            obs, reward, done = eval_step(task, action)
            row = metric_row(task, 0, step, reward, done)
            rows.append(row)
            if step % args.frame_stride == 0 or bool(done[0].item()):
                env.save_camera_image(str(frame_dir / f'frame_{frame_i:04d}.png'), cam_idx=args.cam_idx)
                frame_i += 1
            if bool(done[0].item()):
                break
        summary = summarize_episode(rows)
        summary.update({
            'object_id': str(args.object_id),
            'trajectory': str(traj_path),
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
        with open(out_dir / 'metrics.csv', 'w') as f:
            keys=list(rows[0].keys())
            f.write(','.join(keys)+'\n')
            for r in rows:
                f.write(','.join(str(r[k]) for k in keys)+'\n')
        print(json.dumps(summary, indent=2))

    finally:
        pass

if __name__ == '__main__':
    main()
