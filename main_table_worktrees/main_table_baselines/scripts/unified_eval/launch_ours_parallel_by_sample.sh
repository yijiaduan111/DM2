#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
RUNNER=${RUNNER:-$BASELINE_ROOT/scripts/unified_eval/run_unified_eval.sh}
SUMMARIZER=${SUMMARIZER:-$BASELINE_ROOT/scripts/unified_eval/summarize_unified_eval.py}
CANONICAL_MANIFEST=${CANONICAL_MANIFEST:-/data/dyj/zts/clean_data/v20260520/batch_manifest.csv}
ENVROOT=${ENVROOT:-/data/dyj/miniconda3/envs/cuda-kernel-eval}
GPU_LIST_STR=${GPU_LIST:-"0 1 2 3 4 5"}
EPISODES=${EPISODES:-20}
DAMPS=${DAMPS:-"1 2 4"}
MODES=${MODES:-"det stoch"}
ROWS=${ROWS:-07}
RUN_TAG=${RUN_TAG:-ours_unified_parallel_$(date +%Y%m%d_%H%M%S)}
PARENT_RUN_DIR=${PARENT_RUN_DIR:-$BASELINE_ROOT/results/unified_eval/run_$RUN_TAG}

mkdir -p "$PARENT_RUN_DIR/logs" "$PARENT_RUN_DIR/by_sample" "$PARENT_RUN_DIR/meta"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PARENT_RUN_DIR/run_status.txt"; }

read -r -a GPUS <<< "$GPU_LIST_STR"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

mapfile -t SAMPLES < <(python3 - <<'PY' "$CANONICAL_MANIFEST"
import csv, sys
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        if row.get('enabled', '1').strip() == '1':
            print(row['sample_id'])
PY
)
if [[ ${#SAMPLES[@]} -eq 0 ]]; then
  echo "No enabled samples in $CANONICAL_MANIFEST" >&2
  exit 2
fi

cp "$CANONICAL_MANIFEST" "$PARENT_RUN_DIR/meta/batch_manifest.csv"
cat > "$PARENT_RUN_DIR/launch_config.txt" <<EOF
ROWS=$ROWS
EPISODES=$EPISODES
DAMPS=$DAMPS
MODES=$MODES
GPU_LIST=$GPU_LIST_STR
RUN_TAG=$RUN_TAG
PARENT_RUN_DIR=$PARENT_RUN_DIR
CANONICAL_MANIFEST=$CANONICAL_MANIFEST
EOF

log "PARENT_RUN_DIR=$PARENT_RUN_DIR"
log "ROWS=$ROWS EPISODES=$EPISODES DAMPS=$DAMPS MODES=$MODES"
log "SAMPLES=${SAMPLES[*]}"
log "GPUS=${GPUS[*]}"

printf 'sample_id\tgpu\tpid\trun_dir\tstdout_log\n' > "$PARENT_RUN_DIR/pids.tsv"
pids=()
idx=0
for sample in "${SAMPLES[@]}"; do
  gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
  child="$PARENT_RUN_DIR/by_sample/$sample"
  stdout="$PARENT_RUN_DIR/logs/$sample.nohup.out"
  mkdir -p "$child"
  log "LAUNCH sample=$sample gpu=$gpu child=$child"
  nohup env \
    ROWS="$ROWS" \
    ONLY_SAMPLE="$sample" \
    EPISODES="$EPISODES" \
    DAMPS="$DAMPS" \
    MODES="$MODES" \
    GPU_ID="$gpu" \
    RUN_TAG="${RUN_TAG}_${sample}" \
    RUN_DIR="$child" \
    CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
    "$RUNNER" > "$stdout" 2>&1 &
  pid=$!
  pids+=("$pid")
  printf '%s\t%s\t%s\t%s\t%s\n' "$sample" "$gpu" "$pid" "$child" "$stdout" >> "$PARENT_RUN_DIR/pids.tsv"
  idx=$((idx + 1))
  sleep 3
done

failures=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    log "CHILD_OK pid=$pid"
  else
    rc=$?
    failures=$((failures + 1))
    log "CHILD_FAILED pid=$pid rc=$rc"
  fi
done

# Merge child eval manifests into parent manifest. Summary paths are absolute.
first=1
for sample in "${SAMPLES[@]}"; do
  child="$PARENT_RUN_DIR/by_sample/$sample"
  if [[ -f "$child/eval_manifest.tsv" ]]; then
    if [[ $first -eq 1 ]]; then
      cat "$child/eval_manifest.tsv" > "$PARENT_RUN_DIR/eval_manifest.tsv"
      first=0
    else
      tail -n +2 "$child/eval_manifest.tsv" >> "$PARENT_RUN_DIR/eval_manifest.tsv"
    fi
  else
    log "MISSING eval_manifest sample=$sample child=$child"
    failures=$((failures + 1))
  fi
  if [[ -f "$child/checkpoint_manifest.tsv" ]]; then
    if [[ ! -f "$PARENT_RUN_DIR/checkpoint_manifest.tsv" ]]; then
      cat "$child/checkpoint_manifest.tsv" > "$PARENT_RUN_DIR/checkpoint_manifest.tsv"
    else
      tail -n +2 "$child/checkpoint_manifest.tsv" >> "$PARENT_RUN_DIR/checkpoint_manifest.tsv"
    fi
  fi
done

if [[ -f "$PARENT_RUN_DIR/eval_manifest.tsv" ]]; then
  "$ENVROOT/bin/python" "$SUMMARIZER" "$PARENT_RUN_DIR" | tee "$PARENT_RUN_DIR/meta/summarize.log"
fi

ok=0; failed=0; total=0
if [[ -f "$PARENT_RUN_DIR/eval_manifest.tsv" ]]; then
  total=$(( $(wc -l < "$PARENT_RUN_DIR/eval_manifest.tsv") - 1 ))
  ok=$(awk -F'\t' 'NR>1 && $NF=="ok"{c++} END{print c+0}' "$PARENT_RUN_DIR/eval_manifest.tsv")
  failed=$(awk -F'\t' 'NR>1 && $NF=="failed"{c++} END{print c+0}' "$PARENT_RUN_DIR/eval_manifest.tsv")
fi
log "DONE total_eval_rows=$total ok=$ok failed=$failed child_failures=$failures"
log "SUMMARY=$PARENT_RUN_DIR/compact_summary.md"
exit 0
