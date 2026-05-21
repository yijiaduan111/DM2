#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
CODE_ROOT=${CODE_ROOT:-$BASELINE_ROOT/code}
ENVROOT=${ENVROOT:-/data/dyj/miniconda3/envs/cuda-kernel-eval}
CANONICAL_MANIFEST=${CANONICAL_MANIFEST:-/data/dyj/zts/clean_data/v20260520/batch_manifest.csv}
ROW_NAME=${ROW_NAME:?set ROW_NAME, e.g. 03_state_only_ppo}
CONFIG=${CONFIG:?set CONFIG, e.g. ppo/train_config_main_state_only_ppo.yaml}
EXP_PREFIX=${EXP_PREFIX:?set EXP_PREFIX}
MAX_EPOCHS=${MAX_EPOCHS:-150}
EPISODES=${EPISODES:-20}
DAMPS=${DAMPS:-1}
GPU_ID=${GPU_ID:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$BASELINE_ROOT/results/$ROW_NAME/run_$RUN_TAG}
TRAIN_LOG_DIR=$RUN_DIR/train_logs
EVAL_DIR=$RUN_DIR/eval
ONLY_SAMPLE=${ONLY_SAMPLE:-}

mkdir -p "$TRAIN_LOG_DIR" "$EVAL_DIR/logs" "$RUN_DIR/metadata"
cd "$CODE_ROOT"

export PATH=$ENVROOT/bin:$PATH
export PYTHONPATH=/data/dyj/zts/isaacgym/python:$CODE_ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$ENVROOT/lib:${LD_LIBRARY_PATH:-}
export TORCH_EXTENSIONS_DIR=/data/dyj/.cache/torch_extensions/py38_cu121
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 PYTORCH_JIT=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/run_status.txt"; }

load_manifest() {
  sample_ids=(); object_ids=(); handles=(); trajectories=(); assets=()
  while IFS=, read -r sample_id object_id handle trajectory asset enabled; do
    sample_id=${sample_id%$'\r'}; object_id=${object_id%$'\r'}; handle=${handle%$'\r'}
    trajectory=${trajectory%$'\r'}; asset=${asset%$'\r'}; enabled=${enabled%$'\r'}
    [[ "$sample_id" == "sample_id" ]] && continue
    [[ "${enabled:-1}" == "1" ]] || continue
    [[ -n "$ONLY_SAMPLE" && "$sample_id" != "$ONLY_SAMPLE" ]] && continue
    [[ -f "$trajectory" ]] || { echo "missing trajectory for $sample_id: $trajectory" >&2; exit 2; }
    [[ -d "$asset" ]] || { echo "missing asset for $sample_id: $asset" >&2; exit 2; }
    sample_ids+=("$sample_id"); object_ids+=("$object_id"); handles+=("$handle"); trajectories+=("$trajectory"); assets+=("$asset")
  done < "$CANONICAL_MANIFEST"
  [[ ${#sample_ids[@]} -gt 0 ]] || { echo "no enabled samples in $CANONICAL_MANIFEST" >&2; exit 2; }
}

load_manifest
printf 'sample_id\tobject_id\thandle\ttrajectory\tasset\n' > "$RUN_DIR/canonical_manifest.tsv"
for i in "${!sample_ids[@]}"; do
  printf '%s\t%s\t%s\t%s\t%s\n' "${sample_ids[$i]}" "${object_ids[$i]}" "${handles[$i]}" "${trajectories[$i]}" "${assets[$i]}" >> "$RUN_DIR/canonical_manifest.tsv"
done

log "RUN_DIR=$RUN_DIR"
log "CODE_ROOT=$CODE_ROOT"
log "CONFIG=$CONFIG"
log "CANONICAL_MANIFEST=$CANONICAL_MANIFEST"
log "samples=${sample_ids[*]}"
log "MAX_EPOCHS=$MAX_EPOCHS EPISODES=$EPISODES DAMPS=$DAMPS GPU_ID=$GPU_ID"
cp "$CONFIG" "$RUN_DIR/$(basename "$CONFIG")"
git rev-parse HEAD > "$RUN_DIR/metadata/git_head.txt" 2>/dev/null || true
git status --short > "$RUN_DIR/metadata/git_status_short.txt" 2>/dev/null || true
git diff -- ppo/train.py ppo/hand_drag_task.py ppo/rlgames_wrapper.py scripts/evaluate_ppo_baseline.py "$CONFIG" > "$RUN_DIR/metadata/patch.diff" 2>/dev/null || true

printf 'sample_id\tobject_id\thandle\ttrajectory\trun_name\tcheckpoint_dir\tbest_checkpoint\ttrain_log\tstatus\n' > "$RUN_DIR/train_manifest.tsv"
printf 'sample_id\tobject_id\thandle\tmode\tdamping_scale\tcheckpoint_dir\tsummary_json\tlog_csv\tlog\tstatus\n' > "$RUN_DIR/eval_manifest.tsv"

train_failures=0
eval_failures=0
for i in "${!sample_ids[@]}"; do
  sample_id=${sample_ids[$i]}; obj=${object_ids[$i]}; handle=${handles[$i]}; traj=${trajectories[$i]}
  exp="${EXP_PREFIX}_${sample_id}_${MAX_EPOCHS}ep"
  train_log="$TRAIN_LOG_DIR/${sample_id}.train.log"
  log "TRAIN sample=$sample_id object=$obj handle=$handle trajectory=$traj exp=$exp"
  rm -rf "runs/$exp"
  set +e
  CUDA_VISIBLE_DEVICES=$GPU_ID "$ENVROOT/bin/python" ppo/train.py \
    --train_config "$CONFIG" \
    --object_id "$obj" \
    --trajectory "$traj" \
    --num_envs 64 \
    --max_epochs "$MAX_EPOCHS" \
    --experiment_name "$exp" \
    > "$train_log" 2>&1
  train_rc=$?
  set -e
  ckpt_dir="$CODE_ROOT/runs/$exp/nn"
  best_ckpt="$ckpt_dir/$(basename "${CONFIG%.yaml}").pth"
  if [[ -d "$ckpt_dir" ]] && compgen -G "$ckpt_dir/*.pth" >/dev/null; then
    status=ok
    log "TRAIN_OK sample=$sample_id rc=$train_rc ckpt_dir=$ckpt_dir"
  else
    status=failed
    train_failures=$((train_failures + 1))
    log "TRAIN_FAILED sample=$sample_id rc=$train_rc log=$train_log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$sample_id" "$obj" "$handle" "$traj" "$exp" "$ckpt_dir" "$best_ckpt" "$train_log" "$status" >> "$RUN_DIR/train_manifest.tsv"
  [[ "$status" == ok ]] || continue

  for damp in $DAMPS; do
    for mode in det stoch; do
      objdir="$EVAL_DIR/$sample_id"
      mkdir -p "$objdir"
      prefix="$objdir/${mode}_x${damp}"
      extra=()
      [[ "$mode" == stoch ]] && extra+=(--stochastic)
      log "EVAL sample=$sample_id mode=$mode damping=$damp checkpoint=$ckpt_dir"
      set +e
      CUDA_VISIBLE_DEVICES=$GPU_ID "$ENVROOT/bin/python" scripts/evaluate_ppo_baseline.py \
        --object_id "$obj" \
        --trajectory "$traj" \
        --checkpoint "$ckpt_dir" \
        --checkpoint-kind best \
        --episodes "$EPISODES" \
        --object_damping_scale "$damp" \
        --log_csv "${prefix}.csv" \
        --summary_json "${prefix}.json" \
        "${extra[@]}" \
        > "$EVAL_DIR/logs/${sample_id}_${mode}_x${damp}.log" 2>&1
      eval_rc=$?
      set -e
      if [[ -s "${prefix}.json" ]] && grep -q 'wrote summary:' "$EVAL_DIR/logs/${sample_id}_${mode}_x${damp}.log"; then
        estatus=ok
        log "EVAL_OK sample=$sample_id mode=$mode damping=$damp rc=$eval_rc"
      else
        estatus=failed
        eval_failures=$((eval_failures + 1))
        log "EVAL_FAILED sample=$sample_id mode=$mode damping=$damp rc=$eval_rc"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$sample_id" "$obj" "$handle" "$mode" "$damp" "$ckpt_dir" "${prefix}.json" "${prefix}.csv" "$EVAL_DIR/logs/${sample_id}_${mode}_x${damp}.log" "$estatus" >> "$RUN_DIR/eval_manifest.tsv"
      sleep 2
    done
  done
done

"$ENVROOT/bin/python" - <<'PY' "$RUN_DIR" "$EVAL_DIR"
from pathlib import Path
import csv, json, sys
run_dir=Path(sys.argv[1]); eval_dir=Path(sys.argv[2])
rows=[]
with (run_dir/'canonical_manifest.tsv').open() as f:
    for rec in csv.DictReader(f, delimiter='\t'):
        row={'sample_id':rec['sample_id'],'object_id':rec['object_id'],'handle':rec['handle']}
        for mode in ('det','stoch'):
            for js in sorted((eval_dir/rec['sample_id']).glob(f'{mode}_x*.json')):
                damp=js.stem.split('_x')[-1]
                try: data=json.load(open(js))
                except Exception: data={}
                def pick(*keys):
                    for k in keys:
                        if k in data: return data[k]
                    return ''
                prefix=f'{mode}_x{damp}'
                row[f'{prefix}_success']=pick('success_rate','success')
                row[f'{prefix}_progress']=pick('progress_mean','normalized_progress_mean','progress')
                row[f'{prefix}_return']=pick('return_mean','mean_return','return')
                row[f'{prefix}_steps']=pick('steps_mean','mean_steps','steps')
                row[f'{prefix}_action_l2']=pick('action_l2_mean','mean_action_l2','action_l2')
                row[f'{prefix}_clip099']=pick('clip099','clip_099','action_clip099','clip_fraction_099')
                row[f'{prefix}_detach']=pick('detach_rate','detach','detach_ratio')
        rows.append(row)
keys=['sample_id','object_id','handle']
for row in rows:
    for k in row:
        if k not in keys: keys.append(k)
with (run_dir/'summary.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
md=['# Canonical Manifest Eval Summary','',f'Run dir: `{run_dir}`','', '| sample | object | handle | det x1 success | stoch x1 success | det x4 success | stoch x4 success |', '|---|---:|---|---:|---:|---:|---:|']
def fmt(v):
    if v == '' or v is None: return ''
    try: return f'{float(v):.2f}'
    except Exception: return str(v)
for r in rows:
    md.append('| ' + ' | '.join([r.get('sample_id',''), r.get('object_id',''), r.get('handle',''), fmt(r.get('det_x1_success','')), fmt(r.get('stoch_x1_success','')), fmt(r.get('det_x4_success','')), fmt(r.get('stoch_x4_success',''))]) + ' |')
(run_dir/'summary.md').write_text('\n'.join(md)+'\n')
print(run_dir/'summary.md')
PY
log "TRAIN_FAILURES=$train_failures EVAL_FAILURES=$eval_failures"
log "SUMMARY=$RUN_DIR/summary.md"
exit 0
