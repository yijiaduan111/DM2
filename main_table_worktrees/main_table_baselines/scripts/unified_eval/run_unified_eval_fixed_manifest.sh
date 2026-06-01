#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
CODE_ROOT=${CODE_ROOT:-$BASELINE_ROOT/code}
ENVROOT=${ENVROOT:-/data/dyj/miniconda3/envs/cuda-kernel-eval}
CHECKPOINT_MANIFEST=${CHECKPOINT_MANIFEST:-$BASELINE_ROOT/manifests/checkpoint_manifest_03_06_7obj.tsv}
CANONICAL_MANIFEST=${CANONICAL_MANIFEST:-$BASELINE_ROOT/manifests/main_table_7obj_manifest.csv}
EPISODES=${EPISODES:-20}
MAX_EPISODE_LENGTH=${MAX_EPISODE_LENGTH:-300}
SEED=${SEED:-42}
DAMPS=${DAMPS:-"1 2 4"}
MODES=${MODES:-"det stoch"}
GPU_ID=${GPU_ID:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$BASELINE_ROOT/results/unified_eval_03_06_7obj/run_$RUN_TAG}
EVALUATOR=${EVALUATOR:-$CODE_ROOT/scripts/evaluate_ppo_unified.py}
mkdir -p "$RUN_DIR/results" "$RUN_DIR/logs" "$RUN_DIR/meta"
cd "$CODE_ROOT"
export PATH=$ENVROOT/bin:$PATH
export PYTHONPATH=/data/dyj/zts/isaacgym/python:$CODE_ROOT:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$ENVROOT/lib:${LD_LIBRARY_PATH:-}
export TORCH_EXTENSIONS_DIR=/data/dyj/.cache/torch_extensions/py38_cu121
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 PYTORCH_JIT=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HAND_DRAG_MANIFEST=$CANONICAL_MANIFEST
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/run_status.txt"; }
log "RUN_DIR=$RUN_DIR"
log "CHECKPOINT_MANIFEST=$CHECKPOINT_MANIFEST"
log "EPISODES=$EPISODES DAMPS=$DAMPS MODES=$MODES GPU_ID=$GPU_ID"
cp "$CHECKPOINT_MANIFEST" "$RUN_DIR/checkpoint_manifest.tsv"
cp "$CANONICAL_MANIFEST" "$RUN_DIR/meta/main_table_7obj_manifest.csv"
cp "$EVALUATOR" "$RUN_DIR/meta/evaluate_ppo_unified.py"
git rev-parse HEAD > "$RUN_DIR/meta/eval_code_git_head.txt" 2>/dev/null || true
git status --short > "$RUN_DIR/meta/eval_code_git_status_short.txt" 2>/dev/null || true
printf 'method_id\tmethod_label\tsample_id\tobject_id\thandle\tmode\tdamping_scale\tcheckpoint_dir\tcheckpoint_kind\tsummary_json\tlog_csv\tlog\tstatus\n' > "$RUN_DIR/eval_manifest.tsv"
eval_failures=0
while IFS=$'\t' read -r method_id method_label sample_id object_id handle trajectory checkpoint_dir checkpoint_kind train_run_dir eval_code_root; do
  [[ "$method_id" == "method_id" ]] && continue
  for damp in $DAMPS; do
    for mode in $MODES; do
      out_dir="$RUN_DIR/results/$method_id/$sample_id"
      mkdir -p "$out_dir"
      prefix="$out_dir/${mode}_x${damp}"
      log_file="$prefix.log"
      extra=()
      [[ "$mode" == "stoch" ]] && extra+=(--stochastic)
      log "EVAL method=$method_id sample=$sample_id mode=$mode x=$damp ckpt=$checkpoint_dir"
      set +e
      CUDA_VISIBLE_DEVICES=$GPU_ID "$ENVROOT/bin/python" "$EVALUATOR" \
        --object_id "$object_id" \
        --trajectory "$trajectory" \
        --checkpoint "$checkpoint_dir" \
        --checkpoint-kind "$checkpoint_kind" \
        --episodes "$EPISODES" \
        --max_episode_length "$MAX_EPISODE_LENGTH" \
        --seed "$SEED" \
        --object_damping_scale "$damp" \
        --log_csv "$prefix.csv" \
        --summary_json "$prefix.json" \
        "${extra[@]}" \
        > "$log_file" 2>&1
      rc=$?
      set -e
      if [[ -s "$prefix.json" ]] && grep -q 'wrote summary:' "$log_file"; then
        status=ok
        log "EVAL_OK method=$method_id sample=$sample_id mode=$mode x=$damp rc=$rc"
      else
        status=failed
        eval_failures=$((eval_failures + 1))
        log "EVAL_FAILED method=$method_id sample=$sample_id mode=$mode x=$damp rc=$rc log=$log_file"
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$method_id" "$method_label" "$sample_id" "$object_id" "$handle" "$mode" "$damp" \
        "$checkpoint_dir" "$checkpoint_kind" "$prefix.json" "$prefix.csv" "$log_file" "$status" \
        >> "$RUN_DIR/eval_manifest.tsv"
      sleep 1
    done
  done
done < "$RUN_DIR/checkpoint_manifest.tsv"
python "$BASELINE_ROOT/scripts/unified_eval/summarize_unified_eval.py" "$RUN_DIR" | tee "$RUN_DIR/meta/summarize.log" || true
log "EVAL_FAILURES=$eval_failures"
log "SUMMARY=$RUN_DIR/compact_summary.md"
exit 0
