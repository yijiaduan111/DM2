#!/usr/bin/env python3
import csv
from pathlib import Path
BASELINE_ROOT = Path('/data/dyj/zts/main_table_worktrees/main_table_baselines')
CODE_ROOT = BASELINE_ROOT / 'code'
RUNS = Path('/data/dyj/zts/main_table_worktrees/main_table_v2c_clean/runs')
DATA_MANIFEST = BASELINE_ROOT / 'manifests/main_table_7obj_manifest.csv'
OUT = BASELINE_ROOT / 'manifests/checkpoint_manifest_03_06_7obj.tsv'
rows = {
    '03_state_only_ppo': ('03 State-only PPO', {
        '45936_handle_1':'main_table_03_state_only_ppo_45936_handle_1_150ep',
        '7310_handle_1':'main_table_03_state_only_ppo_7310_handle_1_150ep',
        '45661_handle_3':'main_table_03_state_only_ppo_45661_handle_3_150ep',
        '45261_handle_7':'main_table_03_state_only_ppo_45261_handle_7_150ep',
        '46440_handle_5':'main_table_03_state_only_ppo_46440_handle_5_150ep',
        '12583_handle_1':'explore_datanew_03_state_only_ppo_12583_handle_1_150ep',
        '48513_handle_2':'explore_datanew_03_state_only_ppo_48513_handle_2_150ep',
    }),
    '04_flat_history_ppo': ('04 Flat-history PPO', {
        '45936_handle_1':'main_table_04_flat_history_ppo_45936_handle_1_150ep',
        '7310_handle_1':'main_table_04_flat_history_ppo_7310_handle_1_150ep',
        '45661_handle_3':'main_table_04_flat_history_ppo_45661_handle_3_150ep',
        '45261_handle_7':'main_table_04_flat_history_ppo_45261_handle_7_150ep',
        '46440_handle_5':'main_table_04_flat_history_ppo_46440_handle_5_150ep',
        '12583_handle_1':'explore_datanew_04_flat_history_ppo_12583_handle_1_150ep',
        '48513_handle_2':'explore_datanew_04_flat_history_ppo_48513_handle_2_150ep',
    }),
    '05_gla_no_aux': ('05 GLA no aux', {
        '45936_handle_1':'main_table_05_gla_no_aux_45936_handle_1_150ep',
        '7310_handle_1':'main_table_05_gla_no_aux_7310_handle_1_150ep',
        '45661_handle_3':'main_table_05_gla_no_aux_45661_handle_3_150ep',
        '45261_handle_7':'main_table_05_gla_no_aux_45261_handle_7_150ep',
        '46440_handle_5':'main_table_05_gla_no_aux_46440_handle_5_150ep',
        '12583_handle_1':'explore_datanew_05_gla_no_aux_12583_handle_1_150ep',
        '48513_handle_2':'explore_datanew_05_gla_no_aux_48513_handle_2_150ep',
    }),
    '06_pica_no_gla_aux_v2c': ('06 PICA no GLA', {
        '45936_handle_1':'main_table_06_pica_no_gla_aux_v2c_45936_handle_1_150ep',
        '7310_handle_1':'main_table_06_pica_no_gla_aux_v2c_7310_handle_1_150ep',
        '45661_handle_3':'main_table_06_pica_no_gla_aux_v2c_45661_handle_3_150ep',
        '45261_handle_7':'main_table_06_pica_no_gla_aux_v2c_45261_handle_7_150ep',
        '46440_handle_5':'main_table_06_pica_no_gla_aux_v2c_46440_handle_5_150ep',
        '12583_handle_1':'explore_datanew_06_pica_no_gla_aux_v2c_12583_handle_1_150ep',
        '48513_handle_2':'explore_datanew_06_pica_no_gla_aux_v2c_48513_handle_2_150ep',
    }),
}
samples = list(csv.DictReader(DATA_MANIFEST.open()))
fields = ['method_id','method_label','sample_id','object_id','handle','trajectory','checkpoint_dir','checkpoint_kind','train_run_dir','eval_code_root']
records=[]
missing=[]
for method_id,(label,mapping) in rows.items():
    for sample in samples:
        sid=sample['sample_id']
        run_dir=RUNS / mapping[sid]
        ckpt_dir=run_dir / 'nn'
        best=ckpt_dir / 'HandDrag.pth'
        if not best.exists():
            missing.append(str(best))
        records.append({
            'method_id': method_id,
            'method_label': label,
            'sample_id': sid,
            'object_id': sample['object_id'],
            'handle': sample['handle'],
            'trajectory': sample['trajectory'],
            'checkpoint_dir': str(ckpt_dir),
            'checkpoint_kind': 'best',
            'train_run_dir': str(run_dir),
            'eval_code_root': str(CODE_ROOT),
        })
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    w.writeheader(); w.writerows(records)
print(OUT)
print(f'records={len(records)}')
if missing:
    print('MISSING:')
    print('\n'.join(missing))
    raise SystemExit(1)
for r in records:
    print(f"{r['method_id']}\t{r['sample_id']}\t{r['checkpoint_dir']}")
