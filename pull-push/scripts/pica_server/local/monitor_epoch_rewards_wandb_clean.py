#!/usr/bin/env python3
import argparse
import csv
import os
import signal
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--project', default='pica-pull-push-clean')
    p.add_argument('--name', default=None)
    p.add_argument('--mode', default=os.environ.get('WANDB_MODE', 'online'))
    p.add_argument('--poll-seconds', type=float, default=30.0)
    p.add_argument('--step-offset', type=int, default=0)
    p.add_argument('--metric', default='reward_mean')
    p.add_argument('--min-epochs', type=int, default=72)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--min-delta', type=float, default=1.0)
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--no-early-stop', action='store_true')
    p.add_argument('--grace-seconds', type=float, default=180.0)
    return p.parse_args()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def to_number(value):
    if value is None or value == '':
        return value
    try:
        f = float(value)
    except Exception:
        return value
    return int(f) if f.is_integer() else f


def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    for _ in range(3):
        try:
            with path.open(newline='') as f:
                return list(csv.DictReader(f))
        except Exception:
            time.sleep(1)
    return []


def terminate_training(pid, grace_seconds):
    if not pid_alive(pid):
        return 'already_exit'
    os.kill(pid, signal.SIGINT)
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not pid_alive(pid):
            return 'sigint_exit'
        time.sleep(2)
    os.kill(pid, signal.SIGTERM)
    return 'sigterm_sent'


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_name = args.name or run_dir.name
    log_path = run_dir / 'monitor_wandb_clean.log'
    state_path = run_dir / 'early_stop_clean_state.json'
    logged = set()
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
                name=run_name,
                dir=str(run_dir),
                mode=args.mode,
                config={
                    'run_dir': str(run_dir),
                    'train_pid': args.pid,
                    'step_offset': args.step_offset,
                    'early_stop_metric': args.metric,
                    'early_stop_min_local_epoch': args.min_epochs,
                    'early_stop_total_epoch': args.min_epochs + args.step_offset,
                    'early_stop_patience': args.patience,
                    'early_stop_min_delta': args.min_delta,
                    'early_stop_window': args.window,
                    'logs_aux': False,
                },
                resume='never',
            )
            wandb.define_metric('total_epoch')
            wandb.define_metric('*', step_metric='total_epoch')
        except Exception as exc:
            print(f'[clean-monitor][warn] wandb disabled: {exc}', flush=True)
            wandb = None

    def log(msg):
        text = '[{}] {}'.format(time.strftime('%F %T'), msg)
        print(text, flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as f:
            f.write(text + '\n')

    log(f'start clean monitor run={run_name} pid={args.pid} step_offset={args.step_offset}')
    while True:
        rows = read_rows(run_dir / 'epoch_rewards.csv')
        for row in rows:
            try:
                epoch = int(float(row.get('epoch', 'nan')))
            except Exception:
                continue
            if epoch in logged:
                continue
            logged.add(epoch)
            total_epoch = epoch + args.step_offset
            payload = {k: to_number(v) for k, v in row.items() if k}
            payload['local_epoch'] = epoch
            payload['total_epoch'] = total_epoch
            metric_value = payload.get(args.metric)
            if isinstance(metric_value, (int, float)):
                rewards.append((epoch, float(metric_value)))
            if wandb:
                wandb.log(payload)
            log('local_epoch={} total_epoch={} {}={} success={} progress={} clip099={}'.format(
                epoch, total_epoch, args.metric, metric_value,
                payload.get('success_mean'), payload.get('normalized_progress_mean'), payload.get('clip099_mean')))

            if (not args.no_early_stop) and epoch >= args.min_epochs and len(rewards) >= args.window:
                score = sum(v for _, v in rewards[-args.window:]) / args.window
                if best_score is None or score > best_score + args.min_delta:
                    best_score = score
                    best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                with state_path.open('w') as f:
                    f.write(str({
                        'local_epoch': epoch,
                        'total_epoch': total_epoch,
                        'score': score,
                        'best_score': best_score,
                        'best_local_epoch': best_epoch,
                        'best_total_epoch': best_epoch + args.step_offset if best_epoch is not None else None,
                        'bad_epochs': bad_epochs,
                    }))
                if bad_epochs >= args.patience and not stopped:
                    log(f'early stop trigger local_epoch={epoch} total_epoch={total_epoch} bad_epochs={bad_epochs}')
                    if wandb:
                        wandb.log({'early_stop_triggered': 1, 'total_epoch': total_epoch})
                    terminate_training(args.pid, args.grace_seconds)
                    stopped = True

        if not pid_alive(args.pid):
            log('training process exited; monitor stopping')
            break
        time.sleep(args.poll_seconds)

    if wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
