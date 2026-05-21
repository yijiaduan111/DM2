#!/usr/bin/env python3
"""Monitor PICA training CSVs, log to wandb, and optionally early-stop."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import time
from typing import Dict, Iterable, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--project', default='pica-pull-push')
    p.add_argument('--entity', default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--mode', default=os.environ.get('WANDB_MODE', 'online'), choices=['online', 'offline', 'disabled'])
    p.add_argument('--poll-seconds', type=float, default=30.0)
    p.add_argument('--metric', default='reward_mean')
    p.add_argument('--min-epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--min-delta', type=float, default=1.0)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--no-early-stop', action='store_true')
    p.add_argument('--step-offset', type=int, default=0)
    p.add_argument('--grace-seconds', type=float, default=180.0)
    return p.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    for _ in range(3):
        try:
            with path.open(newline='') as f:
                return list(csv.DictReader(f))
        except Exception:
            time.sleep(1)
    return []


def to_number(value: str):
    if value is None or value == '':
        return value
    try:
        f = float(value)
    except Exception:
        return value
    if f.is_integer():
        return int(f)
    return f


def numeric_row(row: Dict[str, str], prefix: str = '') -> Dict[str, object]:
    out = {}
    for k, v in row.items():
        if not k:
            continue
        out[prefix + k] = to_number(v)
    return out


def latest_checkpoint(run_dir: Path) -> Optional[str]:
    nn = run_dir / 'nn'
    if not nn.exists():
        return None
    ckpts = sorted(nn.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None


def terminate_training(pid: int, grace_seconds: float):
    if not pid_alive(pid):
        return 'already_exit'
    os.kill(pid, signal.SIGINT)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not pid_alive(pid):
            return 'sigint_exit'
        time.sleep(2)
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline:
        if not pid_alive(pid):
            return 'sigterm_exit'
        time.sleep(2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return 'sigkill_exit'


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_name = args.name or run_dir.name
    state_path = run_dir / 'early_stop_state.json'
    monitor_log = run_dir / 'monitor_earlystop.log'
    logged_epochs = set()
    logged_aux_epochs = set()
    rewards = []
    best_score = None
    best_epoch = None
    bad_epochs = 0
    stopped = False

    wandb = None
    if args.mode != 'disabled':
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            wandb.init(
                project=args.project,
                entity=args.entity,
                name=run_name,
                group=args.group,
                dir=str(run_dir),
                mode=args.mode,
                config={
                    'run_dir': str(run_dir),
                    'train_pid': args.pid,
                    'early_stop_metric': args.metric,
                    'early_stop_min_epochs': args.min_epochs,
                    'early_stop_patience': args.patience,
                    'early_stop_min_delta': args.min_delta,
                    'early_stop_window': args.window,
                    'step_offset': args.step_offset,
                },
                resume='allow',
            )
        except Exception as exc:
            print(f'[monitor][warn] wandb disabled: {exc}', flush=True)
            wandb = None

    def log_line(msg: str):
        text = '[{}] {}'.format(time.strftime('%F %T'), msg)
        print(text, flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with monitor_log.open('a') as f:
            f.write(text + '\n')

    log_line(f'start monitor run={run_name} pid={args.pid} mode={args.mode}')

    while True:
        rows = read_csv_rows(run_dir / 'epoch_rewards.csv')
        for row in rows:
            try:
                epoch = int(float(row.get('epoch', 'nan')))
            except Exception:
                continue
            if epoch in logged_epochs:
                continue
            metrics = numeric_row(row)
            logged_epochs.add(epoch)
            metric_value = metrics.get(args.metric)
            if isinstance(metric_value, (int, float)):
                rewards.append((epoch, float(metric_value)))
            ckpt = latest_checkpoint(run_dir)
            payload = dict(metrics)
            if ckpt:
                payload['latest_checkpoint'] = ckpt
            if wandb:
                wandb.log(payload, step=epoch + args.step_offset)
            log_line('epoch={} {}={} success={} progress={}'.format(
                epoch, args.metric, metric_value,
                metrics.get('success_mean'), metrics.get('normalized_progress_mean')))

            if (not args.no_early_stop) and epoch >= args.min_epochs and len(rewards) >= args.window:
                window_vals = [v for _, v in rewards[-args.window:]]
                score = sum(window_vals) / len(window_vals)
                if best_score is None or score > best_score + args.min_delta:
                    best_score = score
                    best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= args.patience:
                    reason = {
                        'reason': 'plateau',
                        'metric': args.metric,
                        'epoch': epoch,
                        'best_epoch': best_epoch,
                        'best_window_score': best_score,
                        'current_window_score': score,
                        'patience': args.patience,
                        'min_delta': args.min_delta,
                        'window': args.window,
                        'latest_checkpoint': ckpt,
                    }
                    state_path.write_text(json.dumps(reason, indent=2, sort_keys=True))
                    log_line('early stop triggered: ' + json.dumps(reason, sort_keys=True))
                    if wandb:
                        wandb.summary.update(reason)
                        wandb.log({'early_stop/triggered': 1, 'early_stop/best_epoch': best_epoch or -1}, step=epoch + args.step_offset)
                    terminate_result = terminate_training(args.pid, args.grace_seconds)
                    log_line(f'terminate result: {terminate_result}')
                    stopped = True
                    break

        aux_rows = read_csv_rows(run_dir / 'aux_log.csv')
        for row in aux_rows:
            try:
                epoch = int(float(row.get('epoch', 'nan')))
            except Exception:
                continue
            if epoch in logged_aux_epochs:
                continue
            logged_aux_epochs.add(epoch)
            if wandb:
                wandb.log(numeric_row(row, prefix='aux/'), step=epoch)

        if stopped:
            break
        if not pid_alive(args.pid):
            log_line('training process exited; monitor stopping')
            break
        time.sleep(args.poll_seconds)

    if wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
