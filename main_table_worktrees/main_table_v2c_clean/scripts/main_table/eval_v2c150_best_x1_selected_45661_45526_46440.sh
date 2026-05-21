#!/usr/bin/env bash
set -euo pipefail
echo "[blocked] This selected-object eval script used old object_id-only/default trajectory paths."
echo "Use canonical manifest entrypoints instead:"
echo "  /data/dyj/zts/main_table_worktrees/main_table_baselines/scripts/train_eval_pica_no_gla_x1_x4.sh"
echo "  /data/dyj/zts/pull-push/scripts/pica_server/local/active/run_manifest_v2d_both_ft50.sh"
exit 2
