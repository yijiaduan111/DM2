#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

BASELINE_ROOT = Path('/data/dyj/zts/main_table_worktrees/main_table_baselines')
BASELINE_CODE = BASELINE_ROOT / 'code'
PULL_PUSH_ROOT = Path('/data/dyj/zts/pull-push')
DEFAULT_DATA_MANIFEST = Path('/data/dyj/zts/clean_data/v20260520/batch_manifest.csv')

ROW_ALIASES = {
    '03': '03_state_only_ppo',
    '03_state_only': '03_state_only_ppo',
    '03_state_only_ppo': '03_state_only_ppo',
    '04': '04_flat_history_ppo',
    '04_flat': '04_flat_history_ppo',
    '04_flat_history_ppo': '04_flat_history_ppo',
    '05': '05_gla_no_aux',
    '05_gla_no_aux': '05_gla_no_aux',
    '06': '06_pica_no_gla_aux_v2c',
    '06_pica_no_gla': '06_pica_no_gla_aux_v2c',
    '06_pica_no_gla_aux_v2c': '06_pica_no_gla_aux_v2c',
    '07': '07_ours_v2c150_v2d50',
    '07_ours': '07_ours_v2c150_v2d50',
    '07_ours_v2c150_v2d50': '07_ours_v2c150_v2d50',
}

ROW_LABELS = {
    '03_state_only_ppo': '03 State-only PPO',
    '04_flat_history_ppo': '04 Flat-history PPO',
    '05_gla_no_aux': '05 GLA no aux',
    '06_pica_no_gla_aux_v2c': '06 PICA no GLA',
    '07_ours_v2c150_v2d50': '07 Ours v2c150+v2d50FT',
}

ROW_RESULT_DIRS = {
    '03_state_only_ppo': BASELINE_ROOT / 'results/03_state_only_ppo',
    '04_flat_history_ppo': BASELINE_ROOT / 'results/04_flat_history_ppo',
    '05_gla_no_aux': BASELINE_ROOT / 'results/05_gla_no_aux',
    '06_pica_no_gla_aux_v2c': BASELINE_ROOT / 'results/06_pica_no_gla_aux_v2c',
}


def parse_rows(raw: str):
    rows = []
    for item in raw.replace(',', ' ').split():
        key = item.strip()
        if not key:
            continue
        if key not in ROW_ALIASES:
            raise SystemExit(f'Unknown row alias: {key}')
        row = ROW_ALIASES[key]
        if row not in rows:
            rows.append(row)
    if not rows:
        raise SystemExit('No rows selected')
    return rows


def load_data_manifest(path: Path, only_sample: str = ''):
    samples = []
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if row.get('enabled', '1').strip() != '1':
                continue
            if only_sample and row['sample_id'] != only_sample:
                continue
            samples.append(row)
    if not samples:
        raise SystemExit(f'No enabled samples from {path}; only_sample={only_sample!r}')
    return samples


def latest_parallel_dir(row_id: str) -> Path:
    parent = ROW_RESULT_DIRS[row_id]
    dirs = sorted(parent.glob('parallel_canonical_*'))
    if not dirs:
        raise SystemExit(f'No parallel_canonical_* run under {parent}')
    return dirs[-1]


def read_train_manifest(path: Path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows:
        raise SystemExit(f'Empty train manifest: {path}')
    return rows[-1]


def baseline_ckpt(row_id: str, sample):
    run_dir = latest_parallel_dir(row_id)
    manifest = run_dir / sample['sample_id'] / 'train_manifest.tsv'
    if not manifest.exists():
        raise SystemExit(f'Missing train manifest: {manifest}')
    train = read_train_manifest(manifest)
    if train.get('status') != 'ok':
        raise SystemExit(f'Train status is not ok in {manifest}: {train.get("status")}')
    ckpt_dir = Path(train['checkpoint_dir'])
    best = ckpt_dir / 'HandDrag.pth'
    if not best.exists():
        raise SystemExit(f'Missing best checkpoint: {best}')
    return ckpt_dir, run_dir


def ours_ckpt(sample):
    run_name = f"batch_{sample['sample_id']}_v2d_both_ft50_200total"
    ckpt_dir = PULL_PUSH_ROOT / 'runs' / run_name / 'nn'
    best = ckpt_dir / 'HandDrag.pth'
    if not best.exists():
        raise SystemExit(f'Missing ours best checkpoint: {best}')
    train_run = PULL_PUSH_ROOT / 'output/canonical_ours_v2c150_v2d50_parallel_20260520_115709'
    return ckpt_dir, train_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', default='03,04,05,06,07')
    ap.add_argument('--data-manifest', default=str(DEFAULT_DATA_MANIFEST))
    ap.add_argument('--only-sample', default='')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rows = parse_rows(args.rows)
    samples = load_data_manifest(Path(args.data_manifest), args.only_sample)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        'method_id', 'method_label', 'sample_id', 'object_id', 'handle',
        'trajectory', 'checkpoint_dir', 'checkpoint_kind', 'train_run_dir', 'eval_code_root'
    ]
    records = []
    for row_id in rows:
        for sample in samples:
            if row_id == '07_ours_v2c150_v2d50':
                ckpt_dir, train_run = ours_ckpt(sample)
            else:
                ckpt_dir, train_run = baseline_ckpt(row_id, sample)
            records.append({
                'method_id': row_id,
                'method_label': ROW_LABELS[row_id],
                'sample_id': sample['sample_id'],
                'object_id': sample['object_id'],
                'handle': sample['handle'],
                'trajectory': sample['trajectory'],
                'checkpoint_dir': str(ckpt_dir),
                'checkpoint_kind': 'best',
                'train_run_dir': str(train_run),
                'eval_code_root': str(BASELINE_CODE),
            })

    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(records)
    print(out)
    print(f'rows={len(records)}')


if __name__ == '__main__':
    main()
