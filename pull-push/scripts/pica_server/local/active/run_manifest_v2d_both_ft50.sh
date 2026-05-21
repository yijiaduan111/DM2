#!/usr/bin/env bash
set -u
cd /data/dyj/zts/pull-push || exit 1
source scripts/pica_server/local/env_cuda_kernel_eval.sh
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 CUDA_MODULE_LOADING=LAZY
export PYTORCH_JIT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1

MANIFEST=${MANIFEST:-/data/dyj/zts/clean_data/v20260520/batch_manifest.csv}
OUT_ROOT=${OUT_ROOT:-output/pica_manifest_singletraj_v2d_both_ft50_eval}
MASTER=${MASTER:-logs_output/manifest_v2d_both_ft50.log}
V2C_CFG=${V2C_CFG:-ppo/train_config_gla_pica_drand12_aux_v2c.yaml}
V2D_CFG=${V2D_CFG:-ppo/train_config_gla_pica_v2d_both.yaml}
EPISODES=${EPISODES:-20}
SEED=${SEED:-42}
DAMPS=${DAMPS:-"1.0 2.0 4.0"}
GPU=${GPU:-4}
V2C_EPOCHS=${V2C_EPOCHS:-150}
V2D_TOTAL_EPOCHS=${V2D_TOTAL_EPOCHS:-200}
DRY_RUN=${DRY_RUN:-0}
EVAL_CHECKPOINT_KIND=${EVAL_CHECKPOINT_KIND:-best}
ONLY_SAMPLE=${ONLY_SAMPLE:-}

mkdir -p logs_output "$OUT_ROOT"
LOCK_DIR=${LOCK_DIR:-logs_output/manifest_v2d_both_ft50.lock}
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another manifest batch appears to be running: $LOCK_DIR" | tee -a "$MASTER"
  exit 9
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
: > "$MASTER"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }
latest_ckpt() {
  local run="$1"
  find "runs/${run}/nn" -maxdepth 1 -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}


run_train() {
  local oid=$1 traj=$2 run=$3 cfg=$4 max_epochs=$5 checkpoint=${6:-}
  local ckpt_arg="" ec=0 last_epoch=0
  [ -n "$checkpoint" ] && ckpt_arg="--checkpoint ${checkpoint} --checkpoint-kind latest"
  log "train start oid=${oid} run=${run} epochs=${max_epochs} gpu=${GPU}"
  rm -rf "runs/${run}"
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run train python ppo/train.py --train_config ${cfg} --object_id ${oid} --trajectory ${traj} --max_epochs ${max_epochs} --experiment_name ${run} ${ckpt_arg}"
    return 0
  fi
  for attempt in 1 2 3; do
    log "train attempt=${attempt} run=${run}"
    CUDA_VISIBLE_DEVICES=${GPU} python -X faulthandler ppo/train.py \
      --train_config "${cfg}" --object_id "${oid}" --trajectory "${traj}" --num_envs 64 --max_epochs "${max_epochs}" \
      --bounds_loss_coef 0.01 --experiment_name "${run}" ${ckpt_arg} \
      > "logs_output/${run}_train.log" 2>&1
    ec=$?
    log "train exit run=${run} attempt=${attempt} exit=${ec}"
    if [ -f "runs/${run}/epoch_rewards.csv" ]; then
      last_epoch=$(tail -n +2 "runs/${run}/epoch_rewards.csv" | awk -F, 'NF>21 {e=$22} END{print e+0}')
      log "train check run=${run} last_epoch=${last_epoch} ckpt=$(latest_ckpt "$run")"
      if [ "$last_epoch" -ge "$max_epochs" ] || [ -n "$(latest_ckpt "$run")" ]; then
        break
      fi
    fi
    sleep 15
  done
  tail -3 "runs/${run}/epoch_rewards.csv" 2>/dev/null | tee -a "$MASTER" || true
}

run_eval_one() {
  local sample_id="$1" oid="$2" traj="$3" run="$4" mode="$5" damp="$6"
  local suffix="det" extra=""
  if [ "$mode" = "stoch" ]; then suffix="stoch"; extra="--stochastic"; fi
  local outdir="${OUT_ROOT}/${sample_id}/${run}_${suffix}"
  local prefix="${run}_damp${damp}_${suffix}"
  mkdir -p "$outdir"
  log "eval start sample=${sample_id} oid=${oid} run=${run} mode=${mode} damp=${damp} ckpt_kind=${EVAL_CHECKPOINT_KIND}"
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run eval ${prefix}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=${GPU} python -X faulthandler scripts/evaluate_ppo_baseline.py \
    --checkpoint "runs/${run}/nn" --checkpoint-kind ${EVAL_CHECKPOINT_KIND} \
    --object_id "${oid}" --trajectory "${traj}" --episodes "${EPISODES}" --seed "${SEED}" \
    --max_episode_length 300 --object_damping_scale "${damp}" --gla_pool last ${extra} \
    --log_csv "${outdir}/${prefix}_metrics.csv" \
    --summary_json "${outdir}/${prefix}_summary.json" \
    > "${outdir}/${prefix}.log" 2>&1
  log "eval exit prefix=${prefix} exit=$?"
  grep -E 'success rate|normalized progress mean|return mean|steps mean|mean action l2|clip099|detach|checkpoint:' "${outdir}/${prefix}.log" | tail -30 | tee -a "$MASTER" || true
}

run_eval_all() {
  local sample_id="$1" oid="$2" traj="$3" run="$4"
  local damp
  for damp in ${DAMPS}; do run_eval_one "$sample_id" "$oid" "$traj" "$run" det "$damp"; done
  for damp in ${DAMPS}; do run_eval_one "$sample_id" "$oid" "$traj" "$run" stoch "$damp"; done
}

summarize_results() {
  local md="${OUT_ROOT}/batch_eval_summary.md"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$PYBIN" - <<'PY' "$OUT_ROOT" > "$md"
import csv, json, os, sys, glob
root=sys.argv[1]
print('| sample | run | mode | damp | success | progress | return | steps | action_l2 | clip099 | detach |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for path in sorted(glob.glob(os.path.join(root, '**', '*_summary.json'), recursive=True)):
    try:
        data=json.load(open(path))
    except Exception:
        continue
    rel=os.path.relpath(path, root).split(os.sep)
    sample=rel[0] if rel else ''
    name=os.path.basename(path).replace('_summary.json','')
    mode='stoch' if '_stoch' in name else 'det'
    damp=''
    for part in name.split('_'):
        if part.startswith('damp'):
            damp=part[4:]
    def pick(*keys):
        for key in keys:
            if key in data: return data[key]
        return ''
    vals=[
        sample,
        rel[1] if len(rel)>1 else '',
        mode,
        damp,
        pick('success_rate','success rate','success'),
        pick('normalized_progress_mean','progress_mean','normalized progress mean','progress'),
        pick('return_mean','return mean','return'),
        pick('steps_mean','steps mean','steps'),
        pick('mean_action_l2','action_l2','mean action l2'),
        pick('clip099','clip099_mean','obs_clip099'),
        pick('detach','detach_mean'),
    ]
    def fmt(v):
        if isinstance(v, float): return f'{v:.4g}'
        return str(v)
    print('| ' + ' | '.join(fmt(v) for v in vals) + ' |')
PY
  log "summary written: ${md}"
}

run_sample() {
  local sample_id="$1" oid="$2" handle="$3" traj="$4" asset="$5"
  local v2c_run="batch_${sample_id}_v2c_${V2C_EPOCHS}ep"
  local v2d_run="batch_${sample_id}_v2d_both_ft50_${V2D_TOTAL_EPOCHS}total"
  log "sample start sample=${sample_id} oid=${oid} handle=${handle} traj=${traj}"
  if [ ! -f "$traj" ]; then log "fatal missing trajectory: $traj"; return 2; fi
  if [ ! -d "$asset" ]; then log "fatal missing asset: $asset"; return 2; fi
  run_train "$oid" "$traj" "$v2c_run" "$V2C_CFG" "$V2C_EPOCHS"
  local ckpt
  ckpt=$(latest_ckpt "$v2c_run")
  if [ "$DRY_RUN" != "1" ] && [ -z "$ckpt" ]; then
    log "fatal no v2c checkpoint sample=${sample_id}"
    return 3
  fi
  run_train "$oid" "$traj" "$v2d_run" "$V2D_CFG" "$V2D_TOTAL_EPOCHS" "$ckpt"
  run_eval_all "$sample_id" "$oid" "$traj" "$v2d_run"
  log "sample end sample=${sample_id}"
}

log "batch start canonical_manifest=${MANIFEST} out=${OUT_ROOT} gpu=${GPU} episodes=${EPISODES} dry_run=${DRY_RUN}"
if [ ! -f "$MANIFEST" ]; then log "fatal missing manifest: $MANIFEST"; exit 2; fi
while IFS=, read -r sample_id oid handle traj asset enabled; do
  [ "$sample_id" = "sample_id" ] && continue
  [ "${enabled:-1}" = "1" ] || continue
  [ -n "$ONLY_SAMPLE" ] && [ "$sample_id" != "$ONLY_SAMPLE" ] && continue
  run_sample "$sample_id" "$oid" "$handle" "$traj" "$asset"
done < "$MANIFEST"
summarize_results
log "batch end"
