import csv, glob, json, os, re
from pathlib import Path
sources=[
 ('45936_default','output/pica_manifest_singletraj_v2d_both_ft50_eval/45936_default/**/*.json'),
 ('45661_default','output/pica_manifest_singletraj_v2d_both_ft50_eval/45661_default/**/*.json'),
 ('7310_handle_1_rerun','output/pica_single_7310_rerun_eval/7310_handle_1/**/*.json'),
 ('45261_handle_7_clean','output/pica_single_45261_handle7_clean_eval/45261_handle_7/**/*.json'),
 ('27044_handle_1_rerun2','output/pica_rerun_27044_41529_eval/27044_handle_1/**/*.json'),
 ('41529_handle_1_rerun2','output/pica_rerun_27044_41529_eval/41529_handle_1/**/*.json'),
]

def pick(d,*keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return ''

def fmt(v):
    if v is None or v=='': return ''
    if isinstance(v,str):
        try: f=float(v)
        except Exception: return v
    else:
        f=float(v)
    return f'{f:.4f}'.rstrip('0').rstrip('.') if abs(f)<1000 else f'{f:.2f}'

def eval_clip_detach(metrics_path):
    if not os.path.exists(metrics_path):
        return '', ''
    rows=list(csv.DictReader(open(metrics_path, newline='')))
    if not rows:
        return '', ''
    clip=sum(1 for r in rows if float(r.get('action_abs_max',0) or 0) >= 0.99)/len(rows)
    by_ep={}
    for r in rows:
        by_ep.setdefault(int(float(r.get('episode',0) or 0)), []).append(r)
    detach=0
    for ep_rows in by_ep.values():
        if any(int(float(r.get('success',0) or 0)) for r in ep_rows):
            continue
        rewards=[float(r.get('reward',0) or 0) for r in ep_rows]
        if rewards and min(rewards) <= -40:
            detach += 1
    return clip, detach/max(1,len(by_ep))

rows=[]
order={s:i for i,(s,_) in enumerate(sources)}
for sample, pat in sources:
    for p in sorted(glob.glob(pat, recursive=True)):
        if not p.endswith('_summary.json'):
            continue
        data=json.load(open(p))
        base=os.path.basename(p)
        m=re.search(r'damp([0-9.]+)_(det|stoch)_summary', base)
        if not m:
            continue
        damp, mode=m.group(1), m.group(2)
        metrics=p.replace('_summary.json','_metrics.csv')
        clip, detach=eval_clip_detach(metrics)
        rows.append({
            'sample': sample,
            'mode': mode,
            'damp': damp,
            'success': pick(data,'success_rate','success'),
            'progress': pick(data,'normalized_progress_mean','progress'),
            'return': pick(data,'return_mean','return'),
            'steps': pick(data,'steps_mean','steps'),
            'action_l2': pick(data,'mean_action_l2','action_l2'),
            'clip099': clip,
            'detach': detach,
        })
rows.sort(key=lambda r:(order.get(r['sample'],99), 0 if r['mode']=='det' else 1, float(r['damp'])))
out=Path('/data/dyj/zts/pull-push/reports/pica_handoff/current_multiobject_eval_table.md')
out.parent.mkdir(parents=True, exist_ok=True)
lines=['# Current Multi-object Eval Core Table','', '| method | damping | success | progress | return | steps | action_l2 | clip099 | detach |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    method=f"{r['sample']}_{r['mode']}"
    vals=[method, r['damp'], r['success'], r['progress'], r['return'], r['steps'], r['action_l2'], r['clip099'], r['detach']]
    lines.append('| ' + ' | '.join(fmt(v) for v in vals) + ' |')
out.write_text('\n'.join(lines)+'\n')
Path('/data/dyj/zts/pull-push/reports/pica_handoff/current_multiobject_eval_core_table.md').write_text(out.read_text())
print(out)
print('\n'.join(lines[:14]))