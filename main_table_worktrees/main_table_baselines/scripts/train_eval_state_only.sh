#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT=${BASELINE_ROOT:-/data/dyj/zts/main_table_worktrees/main_table_baselines}
export ROW_NAME=${ROW_NAME:-03_state_only_ppo}
export CONFIG=${CONFIG:-ppo/train_config_main_state_only_ppo.yaml}
export EXP_PREFIX=${EXP_PREFIX:-main_table_03_state_only_ppo}
export DAMPS=${DAMPS:-"1"}
exec "$BASELINE_ROOT/scripts/run_ppo_manifest_train_eval.sh" "$@"
