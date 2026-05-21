#!/usr/bin/env python3
import csv
import json
from pathlib import Path

runs = [
    {
        'label': 'hand_drag_gla_45936_pica_drand12_aux_v2c_150ep',
        'det_dir': Path('output/pica_v2b_eval/hand_drag_gla_45936_pica_drand12_aux_v2c_150ep'),
        'stoch_dir': Path('output/pica_v2c_stochastic_eval/hand_drag_gla_45936_pica_drand12_aux_v2c_150ep'),
    },
    {
        'label': 'hand_drag_gla_45936_pica_drand12_aux_v2c_500ep_es',
        'det_dir': Path('output/pica_v2c_500ep_eval/hand_drag_gla_45936_pica_drand12_aux_v2c_500ep_es'),
        'stoch_dir': Path('output/pica_v2c_500ep_eval/hand_drag_gla_45936_pica_drand12_aux_v2c_500ep_es_stochastic'),
    },
]


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
        if min(float(row['reward']) for row in episode_rows) <= -40:
            detach += 1
    return clip099, detach / max(1, len(by_episode))


def append_section(lines, title, suffix, dir_key):
    lines += [
        '',
        f'## {title}',
        '',
        '| method | damping | success | progress | return | steps | action_l2 | clip099 | detach |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for run in runs:
        for damping in ['1.0', '2.0', '4.0']:
            summary = run[dir_key] / f"{run['label']}_damp{damping}_{suffix}_summary.json"
            metrics = run[dir_key] / f"{run['label']}_damp{damping}_{suffix}_metrics.csv"
            if not summary.exists():
                continue
            data = json.load(open(summary))
            clip099, detach = clip_detach(metrics)
            lines.append(
                f"| {run['label']} | {damping} | {data.get('success_rate', 0):.2f} | "
                f"{data.get('normalized_progress_mean', 0):.3f} | {data.get('return_mean', 0):.2f} | "
                f"{data.get('steps_mean', 0):.1f} | {data.get('mean_action_l2', 0):.3f} | "
                f"{clip099:.3f} | {detach:.2f} |"
            )


def main():
    lines = ['# 45936 PICA v2c results']
    append_section(lines, 'Deterministic', 'det', 'det_dir')
    append_section(lines, 'Stochastic', 'stoch', 'stoch_dir')
    out = Path('reports/pica_handoff/experimental_evidence_summary_45936_v2c_150_500.md')
    out.write_text('\n'.join(lines) + '\n')
    print(out)


if __name__ == '__main__':
    main()
