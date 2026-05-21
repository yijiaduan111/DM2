#!/usr/bin/env bash
set -euo pipefail
echo "[deprecated] This legacy entry used object_id-only/default trajectory paths."
echo "[canonical] Redirecting to /data/dyj/zts/main_table_worktrees/main_table_baselines/scripts/train_eval_state_only.sh"
exec /data/dyj/zts/main_table_worktrees/main_table_baselines/scripts/train_eval_state_only.sh "$@"
