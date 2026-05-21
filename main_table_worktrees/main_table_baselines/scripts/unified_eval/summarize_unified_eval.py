#!/usr/bin/env python3
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [
    ('success', 'success_rate'),
    ('progress', 'normalized_progress_mean'),
    ('return', 'return_mean'),
    ('steps', 'steps_mean'),
    ('action_l2', 'mean_action_l2'),
    ('clip099', 'clip099'),
    ('detach', 'detach_rate'),
]


def fnum(x):
    if x == '' or x is None:
        return ''
    try:
        return f'{float(x):.3f}'
    except Exception:
        return str(x)


def mean(vals):
    vals = [float(v) for v in vals if v != '' and v is not None]
    return sum(vals) / len(vals) if vals else ''


def main():
    run_dir = Path(sys.argv[1])
    eval_manifest = run_dir / 'eval_manifest.tsv'
    rows = []
    with eval_manifest.open(newline='') as f:
        for rec in csv.DictReader(f, delimiter='\t'):
            data = {}
            js = Path(rec['summary_json'])
            if js.exists():
                data = json.loads(js.read_text())
            row = dict(rec)
            for out_key, json_key in METRICS:
                row[out_key] = data.get(json_key, '')
            row['episodes'] = data.get('episodes', '')
            row['checkpoint'] = data.get('checkpoint', '')
            row['trajectory_used'] = data.get('trajectory', '')
            rows.append(row)

    flat_fields = [
        'method_id', 'method_label', 'sample_id', 'object_id', 'handle', 'mode',
        'damping_scale', 'status', 'episodes', 'success', 'progress', 'return',
        'steps', 'action_l2', 'clip099', 'detach', 'checkpoint', 'trajectory_used',
        'summary_json', 'log_csv', 'log'
    ]
    with (run_dir / 'unified_eval_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in flat_fields})

    groups = defaultdict(list)
    for row in rows:
        groups[(row['method_id'], row['method_label'], row['mode'], row['damping_scale'])].append(row)

    compact = ['# Unified Eval Compact Summary', '', f'Run dir: `{run_dir}`', '', '| method | mode | x | n | success | progress | return | steps | action_l2 | clip099 | detach |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key in sorted(groups):
        method_id, label, mode, damp = key
        items = groups[key]
        compact.append('| ' + ' | '.join([
            label, mode, str(damp), str(len(items)),
            fnum(mean([r.get('success', '') for r in items])),
            fnum(mean([r.get('progress', '') for r in items])),
            fnum(mean([r.get('return', '') for r in items])),
            fnum(mean([r.get('steps', '') for r in items])),
            fnum(mean([r.get('action_l2', '') for r in items])),
            fnum(mean([r.get('clip099', '') for r in items])),
            fnum(mean([r.get('detach', '') for r in items])),
        ]) + ' |')
    (run_dir / 'compact_summary.md').write_text('\n'.join(compact) + '\n')

    per_obj = ['# Unified Eval Per-object Results', '', f'Run dir: `{run_dir}`', '', '| method | sample | mode | x | success | progress | return | steps | action_l2 | clip099 | detach | status |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in sorted(rows, key=lambda r: (r['method_id'], r['sample_id'], r['mode'], float(r['damping_scale']))):
        per_obj.append('| ' + ' | '.join([
            row.get('method_label', ''), row.get('sample_id', ''), row.get('mode', ''), row.get('damping_scale', ''),
            fnum(row.get('success', '')), fnum(row.get('progress', '')), fnum(row.get('return', '')),
            fnum(row.get('steps', '')), fnum(row.get('action_l2', '')), fnum(row.get('clip099', '')),
            fnum(row.get('detach', '')), row.get('status', ''),
        ]) + ' |')
    (run_dir / 'per_object_results.md').write_text('\n'.join(per_obj) + '\n')
    print(run_dir / 'compact_summary.md')
    print(run_dir / 'per_object_results.md')
    print(run_dir / 'unified_eval_results.csv')


if __name__ == '__main__':
    main()
