#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/data/dyj/zts/pull-push}
BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
BASELINE_CODE=${BASELINE_CODE:-$BASELINE_ROOT/code}
ENVROOT=${ENVROOT:-/data/dyj/miniconda3/envs/cuda-kernel-eval}
ISAACGYM_ROOT=${ISAACGYM_ROOT:-/data/dyj/zts/isaacgym/python}
MANIFEST=${MANIFEST:-/data/dyj/zts/clean_data/v20260520/batch_manifest_plus_datanew.csv}
OUT_ROOT=${OUT_ROOT:-$ROOT/output/ours7_true_jointaware}
MODE=${MODE:-all}              # preflight | dryrun | train | eval | all | summarize
RUN_TAG=${RUN_TAG:-ours7_true_jointaware}
SAMPLES=${SAMPLES:-45936_handle_1 7310_handle_1 45661_handle_3 45261_handle_7 46440_handle_5 12583_handle_1 48513_handle_2}
GPUS=${GPUS:-0 1 2 3 4 5 6}
BATCH_SIZE=${BATCH_SIZE:-7}
V2C_EPOCHS=${V2C_EPOCHS:-150}
V2D_TOTAL_EPOCHS=${V2D_TOTAL_EPOCHS:-200}
EPISODES=${EPISODES:-20}
DAMPS=${DAMPS:-"1 2 4"}
EVAL_MODES=${EVAL_MODES:-"det stoch"}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTH:-300}
SEED=${SEED:-42}
NUM_ENVS=${NUM_ENVS:-64}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status" "$OUT_ROOT/eval/$RUN_TAG" "$OUT_ROOT/meta" "$OUT_ROOT/summary"
STATUS="$OUT_ROOT/status/status_${RUN_TAG}.tsv"
if [ ! -f "$STATUS" ]; then
  printf 'time\tevent\tstage\tsample\tgpu\trc\tdetail\n' > "$STATUS"
fi

log_status() {
  local event="$1" stage="$2" sample="$3" gpu="$4" rc="$5" detail="${6:-}"
  (
    flock -x 201
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$event" "$stage" "$sample" "$gpu" "$rc" "$detail" >> "$STATUS"
  ) 201>"$OUT_ROOT/status/status.lock"
}

setup_common_env() {
  export PATH=$ENVROOT/bin:$PATH
  export LD_LIBRARY_PATH=$ENVROOT/lib:${LD_LIBRARY_PATH:-}
  export TORCH_EXTENSIONS_DIR=/data/dyj/.cache/torch_extensions/py38_cu121
  export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
  export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 PYTORCH_JIT=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
}

setup_train_env() {
  setup_common_env
  export PYTHONPATH=$ISAACGYM_ROOT:$ROOT:$ROOT/flash-linear-attention:${PYTHONPATH:-}
}

setup_eval_env() {
  setup_common_env
  export PYTHONPATH=$ISAACGYM_ROOT:$BASELINE_CODE:${PYTHONPATH:-}
}

manifest_value() {
  local sample="$1" field="$2"
  "$ENVROOT/bin/python" - <<'PY' "$MANIFEST" "$sample" "$field"
import csv, sys
manifest, sample, field = sys.argv[1:4]
with open(manifest, newline='') as f:
    for row in csv.DictReader(f):
        if row['sample_id'] == sample:
            print(row[field])
            break
    else:
        raise SystemExit(f'missing sample {sample} in {manifest}')
PY
}

run_name_v2c() { echo "ours7_true_jointaware_${1}_v2c${V2C_EPOCHS}_${RUN_TAG}"; }
run_name_v2d() { echo "ours7_true_jointaware_${1}_v2d50_${V2D_TOTAL_EPOCHS}total_${RUN_TAG}"; }

latest_v2c_ckpt() {
  local run="$1" dir
  dir="$ROOT/runs/$run/nn"
  ls -1t "$dir"/last_HandDragep*rew*.pth "$dir"/last_HandDrag_ep_${V2C_EPOCHS}*.pth "$dir"/HandDrag.pth 2>/dev/null | head -1
}

reward_ckpt() {
  local sample="$1" run dir ckpt
  run=$(run_name_v2d "$sample")
  dir="$ROOT/runs/$run/nn"
  ckpt="$dir/HandDrag.pth"
  [ -f "$ckpt" ] && echo "$ckpt"
}

preflight() {
  setup_common_env
  echo "[preflight] root=$ROOT"
  echo "[preflight] manifest=$MANIFEST"
  echo "[preflight] out=$OUT_ROOT run_tag=$RUN_TAG"
  "$ENVROOT/bin/python" - <<'PY' "$MANIFEST" "$SAMPLES" "$ROOT"
import csv, pathlib, sys
manifest=pathlib.Path(sys.argv[1]); samples=sys.argv[2].split(); root=pathlib.Path(sys.argv[3])
rows={r['sample_id']: r for r in csv.DictReader(open(manifest, newline=''))}
missing=[]
for sample in samples:
    if sample not in rows:
        missing.append(f'{sample}: not in manifest'); continue
    r=rows[sample]
    traj=pathlib.Path(r['trajectory']); asset=pathlib.Path(r['asset']); urdf=asset/'mobility.urdf'
    for label,path,kind in [('trajectory',traj,'file'),('asset',asset,'dir'),('urdf',urdf,'file')]:
        ok=path.is_dir() if kind=='dir' else path.exists()
        if not ok: missing.append(f'{sample}: missing {label}: {path}')
    print(f"[sample] {sample} object={r['object_id']} handle={r['handle']} traj={traj}")
if missing:
    print('\n'.join(missing)); raise SystemExit(2)
print(f'[preflight] ok samples={len(samples)}')
PY
  "$ENVROOT/bin/python" -m py_compile "$ROOT/ppo/hand_drag_task.py"
  cp "$MANIFEST" "$OUT_ROOT/meta/manifest_${RUN_TAG}.csv" 2>/dev/null || true
}

train_one() {
  local sample="$1" gpu="$2" oid traj v2c v2d log rc ckpt
  oid=$(manifest_value "$sample" object_id)
  traj=$(manifest_value "$sample" trajectory)
  v2c=$(run_name_v2c "$sample")
  v2d=$(run_name_v2d "$sample")
  log_status start train "$sample" "$gpu" 0 "$v2c -> $v2d"
  echo "[$(date '+%F %T')] TRAIN sample=$sample gpu=$gpu oid=$oid traj=$traj"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[dryrun] v2c=$v2c v2d=$v2d gpu=$gpu"
    log_status end train "$sample" "$gpu" 0 dryrun
    return 0
  fi

  setup_train_env
  cd "$ROOT" || return 1

  if [ -d "$ROOT/runs/$v2c" ] && [ "$FORCE" != "1" ]; then
    echo "[skip] existing v2c run $v2c; set FORCE=1 to rerun"
  else
    log="$OUT_ROOT/logs/train_${sample}_v2c.log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$ENVROOT/bin/python" ppo/train.py \
      --train_config ppo/train_config_gla_pica_drand12_aux_v2c.yaml \
      --object_id "$oid" \
      --trajectory "$traj" \
      --num_envs "$NUM_ENVS" \
      --max_epochs "$V2C_EPOCHS" \
      --bounds_loss_coef 0.01 \
      --experiment_name "$v2c" > "$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 139 ]; then
      log_status end train "$sample" "$gpu" "$rc" "$log"
      return "$rc"
    fi
  fi

  ckpt=$(latest_v2c_ckpt "$v2c")
  if [ -z "${ckpt:-}" ] || [ ! -f "$ckpt" ]; then
    log_status end train "$sample" "$gpu" 18 "missing_v2c_ckpt:$v2c"
    return 18
  fi
  echo "[$(date '+%F %T')] v2c_ckpt=$ckpt"

  if [ -d "$ROOT/runs/$v2d" ] && [ "$FORCE" != "1" ]; then
    echo "[skip] existing v2d run $v2d; set FORCE=1 to rerun"
  else
    log="$OUT_ROOT/logs/train_${sample}_v2d.log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$ENVROOT/bin/python" ppo/train.py \
      --train_config ppo/train_config_gla_pica_v2d_both_jointaware_ablate.yaml \
      --object_id "$oid" \
      --trajectory "$traj" \
      --checkpoint "$ckpt" \
      --num_envs "$NUM_ENVS" \
      --max_epochs "$V2D_TOTAL_EPOCHS" \
      --bounds_loss_coef 0.01 \
      --experiment_name "$v2d" > "$log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 139 ]; then
      log_status end train "$sample" "$gpu" "$rc" "$log"
      return "$rc"
    fi
  fi

  if [ ! -f "$ROOT/runs/$v2d/nn/HandDrag.pth" ]; then
    log_status end train "$sample" "$gpu" 19 "missing_reward_ckpt:$v2d/nn/HandDrag.pth"
    return 19
  fi
  log_status end train "$sample" "$gpu" 0 "$ROOT/runs/$v2d/nn/HandDrag.pth"
  return 0
}

run_sample_batch() {
  local stage=$1 batch_samples=$2 gpus=($GPUS) i=0 sample gpu pid rc=0 pids=()
  echo [$(date '+%F %T')] batch $stage samples=[$batch_samples]
  for sample in $batch_samples; do
    gpu=${gpus[$i]}
    if [ $stage = train ]; then
      train_one $sample $gpu > $OUT_ROOT/logs/worker_train_${sample}.log 2>&1 &
      pid=$!
      echo $pid > $OUT_ROOT/status/train_${sample}.pid
    else
      eval_one_sample $sample $gpu > $OUT_ROOT/logs/worker_eval_${sample}.log 2>&1 &
      pid=$!
      echo $pid > $OUT_ROOT/status/eval_${sample}.pid
    fi
    pids+=($pid:$sample)
    i=$((i+1))
  done
  for item in ${pids[@]}; do
    pid=${item%%:*}
    sample=${item#*:}
    if ! wait $pid; then
      echo [error] $stage failed sample=$sample pid=$pid >&2
      rc=1
    fi
  done
  return $rc
}

run_train_all() {
  preflight || return $?
  log_status pipeline_start train all - 0 "$OUT_ROOT"
  local arr=($SAMPLES) start=0 total=${#arr[@]} batch samples i
  while [ "$start" -lt "$total" ]; do
    samples=""
    for ((i=start; i<start+BATCH_SIZE && i<total; i++)); do samples+="${arr[$i]} "; done
    if ! run_sample_batch train "$samples"; then
      log_status pipeline_end train all - 1 "failed_batch:$samples"
      return 1
    fi
    start=$((start+BATCH_SIZE))
  done
  log_status pipeline_end train all - 0 "$OUT_ROOT"
}

eval_one_sample() {
  local sample="$1" gpu="$2" oid traj ckpt out_dir mode damp extra prefix rc log_file
  oid=$(manifest_value "$sample" object_id)
  traj=$(manifest_value "$sample" trajectory)
  ckpt=$(reward_ckpt "$sample")
  if [ -z "${ckpt:-}" ] || [ ! -f "$ckpt" ]; then
    log_status end eval "$sample" "$gpu" 20 "missing_reward_ckpt:$sample"
    return 20
  fi
  out_dir="$OUT_ROOT/eval/$RUN_TAG/07_ours_v2c150_v2d50/$sample"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] EVAL sample=$sample gpu=$gpu ckpt=$ckpt"
  log_status start eval "$sample" "$gpu" 0 "$out_dir"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[dryrun] eval sample=$sample gpu=$gpu ckpt=$ckpt" > "$out_dir/dryrun.log"
    log_status end eval "$sample" "$gpu" 0 dryrun
    return 0
  fi

  setup_eval_env
  cd "$BASELINE_CODE" || return 1
  for damp in $DAMPS; do
    for mode in $EVAL_MODES; do
      extra=()
      [ "$mode" = "stoch" ] && extra=(--stochastic)
      prefix="$out_dir/${mode}_x${damp}"
      log_file="$prefix.log"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" "$ENVROOT/bin/python" "$BASELINE_CODE/scripts/evaluate_ppo_unified.py" \
        --object_id "$oid" \
        --trajectory "$traj" \
        --checkpoint "$ckpt" \
        --checkpoint-kind latest \
        --episodes "$EPISODES" \
        --max_episode_length "$MAX_EPISODE_LENGTH" \
        --seed "$SEED" \
        --object_damping_scale "$damp" \
        --log_csv "$prefix.csv" \
        --summary_json "$prefix.json" \
        "${extra[@]}" > "$log_file" 2>&1
      rc=$?
      set -e
      if [ ! -s "$prefix.json" ]; then
        log_status end eval "$sample" "$gpu" "$rc" "$log_file"
        return "$rc"
      fi
      sleep 2
    done
  done
  log_status end eval "$sample" "$gpu" 0 "$out_dir"
}

run_eval_all() {
  preflight || return $?
  log_status pipeline_start eval all - 0 "$OUT_ROOT/eval/$RUN_TAG"
  local arr=($SAMPLES) start=0 total=${#arr[@]} samples i
  while [ "$start" -lt "$total" ]; do
    samples=""
    for ((i=start; i<start+BATCH_SIZE && i<total; i++)); do samples+="${arr[$i]} "; done
    if ! run_sample_batch eval "$samples"; then
      log_status pipeline_end eval all - 1 "failed_batch:$samples"
      summarize || true
      return 1
    fi
    start=$((start+BATCH_SIZE))
  done
  summarize
  log_status pipeline_end eval all - 0 "$OUT_ROOT/eval/$RUN_TAG"
}

summarize() {
  setup_common_env
  "$ENVROOT/bin/python" - <<'PY' "$OUT_ROOT" "$RUN_TAG" "$SAMPLES"
import json, pathlib, sys, csv
out=pathlib.Path(sys.argv[1]); tag=sys.argv[2]; samples=sys.argv[3].split()
root=out/'eval'/tag/'07_ours_v2c150_v2d50'
rows=[]; missing=[]
for sample in samples:
    for mode in ['det','stoch']:
        for damp in ['1','2','4']:
            p=root/sample/f'{mode}_x{damp}.json'
            if not p.exists():
                missing.append(f'{sample}/{mode}_x{damp}.json')
                continue
            d=json.load(open(p))
            rows.append({
                'sample_id': sample, 'mode': mode, 'damping': damp,
                'success': d.get('success_rate'),
                'progress': d.get('normalized_progress_mean'),
                'return': d.get('return_mean'),
                'steps': d.get('steps_mean'),
                'action_l2': d.get('mean_action_l2'),
                'clip099': d.get('clip099'),
                'detach': d.get('detach_rate'),
                'checkpoint': d.get('checkpoint'),
                'summary_json': str(p),
            })
summary_dir=out/'summary'; summary_dir.mkdir(parents=True, exist_ok=True)
csv_path=summary_dir/f'ours12_eval_summary_{tag}.csv'
with open(csv_path,'w',newline='') as f:
    fieldnames=['sample_id','mode','damping','success','progress','return','steps','action_l2','clip099','detach','checkpoint','summary_json']
    w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
md_path=summary_dir/f'ours12_eval_summary_{tag}.md'
with open(md_path,'w') as f:
    f.write(f'# Ours12 eval summary {tag}\n\n')
    f.write(f'- JSON found: {len(rows)}/72\n')
    f.write(f'- Missing: {len(missing)}\n\n')
    f.write('| sample | det x1 | det x2 | det x4 | stoch x1 | stoch x2 | stoch x4 |\n')
    f.write('|---|---:|---:|---:|---:|---:|---:|\n')
    by={(r['sample_id'],r['mode'],r['damping']):r for r in rows}
    for sample in samples:
        vals=[]
        for mode,damp in [('det','1'),('det','2'),('det','4'),('stoch','1'),('stoch','2'),('stoch','4')]:
            r=by.get((sample,mode,damp))
            vals.append('NA' if not r else f"{float(r['success']):.2f}/{float(r['progress']):.3f}")
        f.write('| '+sample+' | '+' | '.join(vals)+' |\n')
missing_path=summary_dir/f'missing_{tag}.txt'
missing_path.write_text('\n'.join(missing)+'\n' if missing else '')
print(f'[summary] {csv_path}')
print(f'[summary] {md_path}')
print(f'[summary] json_found={len(rows)}/72 missing={len(missing)}')
if missing:
    print('[missing]'); print('\n'.join(missing))
PY
}

case "$MODE" in
  preflight) preflight ;;
  dryrun) DRY_RUN=1; preflight && run_train_all && run_eval_all ;;
  train) run_train_all ;;
  eval) run_eval_all ;;
  all) run_train_all && run_eval_all ;;
  summarize) summarize ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac
