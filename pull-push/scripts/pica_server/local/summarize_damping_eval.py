#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def clip_detach(path):
    rows = list(csv.DictReader(open(path, newline='')))
    by_episode = {}
    for row in rows:
        by_episode.setdefault(int(row['episode']), []).append(row)
    clip099 = sum(1 for row in rows if float(row['action_abs_max']) >= 0.99) / len(rows)
    detach = 0
    for episode_rows in by_episode.values():
        if any(int(row['success']) for row in episode_rows):
            continue
        if min(float(row['reward']) for row in episode_rows) <= -40.0:
            detach += 1
    detach /= max(1, len(by_episode))
    return clip099, detach


def build_table(run_name, directory, suffix, title):
    directory = Path(directory)
    lines = [
        f'# {title} -- {run_name}',
        '',
        '| method | damping | success | progress | return | steps | action_l2 | clip099 | detach |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for damping in ['1.0', '2.0', '4.0']:
        summary = directory / f'{run_name}_damp{damping}_{suffix}_summary.json'
        metrics = directory / f'{run_name}_damp{damping}_{suffix}_metrics.csv'
        if not summary.exists():
            lines.append(f'| {run_name} | {damping} | MISSING | | | | | | |')
            continue
        data = json.load(open(summary))
        clip099, detach = clip_detach(metrics)
        lines.append(
            f"| {run_name} | {damping} | {data.get('success_rate', 0):.2f} | "
            f"{data.get('normalized_progress_mean', 0):.3f} | {data.get('return_mean', 0):.2f} | "
            f"{data.get('steps_mean', 0):.1f} | {data.get('mean_action_l2', 0):.3f} | "
            f"{clip099:.3f} | {detach:.2f} |"
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--det-dir', required=True)
    parser.add_argument('--stoch-dir', required=True)
    args = parser.parse_args()
    det_dir = Path(args.det_dir)
    stoch_dir = Path(args.stoch_dir)
    det_text = build_table(args.run_name, det_dir, 'det', 'deterministic damping eval')
    stoch_text = build_table(args.run_name, stoch_dir, 'stoch', 'stochastic damping eval')
    (det_dir / 'deterministic_damping_eval_table.md').write_text(det_text)
    (stoch_dir / 'stochastic_damping_eval_table.md').write_text(stoch_text)
    print(det_text)
    print(stoch_text)


if __name__ == '__main__':
    main()
