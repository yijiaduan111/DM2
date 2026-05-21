#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=/data/dyj/zts/main_table_worktrees/main_table_baselines
CODE_ROOT=${CODE_ROOT:-$BASELINE_ROOT/code}
cd "$CODE_ROOT"

usage() {
  cat <<USAGE
Usage:
  $0 flat_history [OUTDIR]
  $0 gla_no_aux [RUN_DIR]

Examples:
  $0 flat_history /data/dyj/zts/main_table_worktrees/main_table_v2c_clean/output/main_table_runs/01_flat_history_ppo/eval_best_6obj_20260514_225149
  $0 gla_no_aux /data/dyj/zts/main_table_worktrees/main_table_v2c_clean/output/main_table_runs/05_gla_no_aux/run_20260515_004843
USAGE
}

kind=${1:-}
shift || true
case "$kind" in
  flat_history)
    exec bash scripts/main_table/run_01_flat_history_missing_eval_single_gpu.sh "$@"
    ;;
  gla_no_aux)
    exec bash scripts/main_table/run_05_gla_no_aux_eval_x1_only.sh "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
