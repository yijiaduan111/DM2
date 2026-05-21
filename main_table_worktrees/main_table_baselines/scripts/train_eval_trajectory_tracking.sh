#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
CODE_ROOT=${CODE_ROOT:-$BASELINE_ROOT/code}
ENVROOT=${ENVROOT:-/data/dyj/miniconda3/envs/cuda-kernel-eval}
CANONICAL_MANIFEST=${CANONICAL_MANIFEST:-/data/dyj/zts/clean_data/v20260520/batch_manifest.csv}
GPU_ID=${GPU_ID:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$BASELINE_ROOT/results/01_trajectory_tracking/run_$RUN_TAG}
PHASE=${PHASE:-drag}
ONLY_SAMPLE=${ONLY_SAMPLE:-}

mkdir -p "$RUN_DIR/logs"
cd "$CODE_ROOT"

export PATH=$ENVROOT/bin:$PATH
export PYTHONPATH=/data/dyj/zts/isaacgym/python:$CODE_ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$ENVROOT/lib:${LD_LIBRARY_PATH:-}
export TORCH_EXTENSIONS_DIR=/data/dyj/.cache/torch_extensions/py38_cu121
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/run_status.txt"; }

log "RUN_DIR=$RUN_DIR"
log "CODE_ROOT=$CODE_ROOT"
log "CANONICAL_MANIFEST=$CANONICAL_MANIFEST"
log "GPU_ID=$GPU_ID PHASE=$PHASE"

git rev-parse HEAD > "$RUN_DIR/git_head.txt" 2>/dev/null || true
git status --short > "$RUN_DIR/git_status_short.txt" 2>/dev/null || true
git diff -- scripts/track_trajectory_baseline.py hand_config.yaml > "$RUN_DIR/trajectory_tracking_patch.diff" 2>/dev/null || true
cp hand_config.yaml "$RUN_DIR/hand_config.yaml"
cp scripts/track_trajectory_baseline.py "$RUN_DIR/track_trajectory_baseline.py"

printf 'sample_id\tobject_id\thandle\ttrajectory\tasset\tstatus\tsummary_json\tlog\n' > "$RUN_DIR/run_manifest.tsv"
failures=0
while IFS=, read -r sample_id object_id handle trajectory asset enabled; do
  sample_id=${sample_id%$'\r'}; object_id=${object_id%$'\r'}; handle=${handle%$'\r'}
  trajectory=${trajectory%$'\r'}; asset=${asset%$'\r'}; enabled=${enabled%$'\r'}
  [[ "$sample_id" == "sample_id" ]] && continue
  [[ "${enabled:-1}" == "1" ]] || continue
  [[ -n "$ONLY_SAMPLE" && "$sample_id" != "$ONLY_SAMPLE" ]] && continue
  [[ -f "$trajectory" ]] || { log "MISSING trajectory sample=$sample_id path=$trajectory"; failures=$((failures+1)); continue; }
  [[ -d "$asset" ]] || { log "MISSING asset sample=$sample_id path=$asset"; failures=$((failures+1)); continue; }
  log "TRACK sample=$sample_id object=$object_id handle=$handle trajectory=$trajectory"
  set +e
  CUDA_VISIBLE_DEVICES=$GPU_ID "$ENVROOT/bin/python" scripts/track_trajectory_baseline.py \
    --object_id "$object_id" \
    --trajectory "$trajectory" \
    --phase "$PHASE" \
    --log_csv "$RUN_DIR/${sample_id}.csv" \
    --summary_json "$RUN_DIR/${sample_id}.json" \
    > "$RUN_DIR/logs/${sample_id}.log" 2>&1
  rc=$?
  set -e
  if [[ -s "$RUN_DIR/${sample_id}.json" ]]; then
    status=ok
    log "TRACK_OK sample=$sample_id rc=$rc"
  else
    status=failed
    failures=$((failures+1))
    log "TRACK_FAILED sample=$sample_id rc=$rc log=$RUN_DIR/logs/${sample_id}.log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$sample_id" "$object_id" "$handle" "$trajectory" "$asset" "$status" "$RUN_DIR/${sample_id}.json" "$RUN_DIR/logs/${sample_id}.log" >> "$RUN_DIR/run_manifest.tsv"
done < "$CANONICAL_MANIFEST"

"$ENVROOT/bin/python" - <<'PY' "$RUN_DIR"
from pathlib import Path
import csv, json, sys
run_dir=Path(sys.argv[1])
rows=[]
with (run_dir/'run_manifest.tsv').open() as f:
    for rec in csv.DictReader(f, delimiter='\t'):
        data={}
        p=Path(rec['summary_json'])
        if p.exists():
            try: data=json.load(open(p))
            except Exception: data={}
        def pick(*keys):
            for k in keys:
                if k in data: return data[k]
            return ''
        rows.append({
            'sample_id': rec['sample_id'], 'object_id': rec['object_id'], 'handle': rec['handle'], 'status': rec['status'],
            'success': pick('success_rate','success'),
            'progress': pick('progress_mean','normalized_progress_mean','progress'),
            'return': pick('return_mean','mean_return','return'),
            'steps': pick('steps_mean','mean_steps','steps'),
            'action_l2': pick('action_l2_mean','mean_action_l2','action_l2'),
            'trajectory': rec['trajectory'],
        })
with (run_dir/'summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['sample_id']); w.writeheader(); w.writerows(rows)
md=['# Trajectory Tracking Canonical Summary','',f'Run dir: `{run_dir}`','', '| sample | object | handle | success | progress | return | steps |', '|---|---:|---|---:|---:|---:|---:|']
def fmt(v):
    if v == '' or v is None: return ''
    try: return f'{float(v):.3g}'
    except Exception: return str(v)
for r in rows:
    md.append('| ' + ' | '.join([r['sample_id'], r['object_id'], r['handle'], fmt(r['success']), fmt(r['progress']), fmt(r['return']), fmt(r['steps'])]) + ' |')
(run_dir/'summary.md').write_text('\n'.join(md)+'\n')
print(run_dir/'summary.md')
PY
log "FAILURES=$failures"
log "SUMMARY=$RUN_DIR/summary.md"
[[ $failures -eq 0 ]] || exit 1
