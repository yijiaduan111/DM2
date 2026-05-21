#!/usr/bin/env bash
set -euo pipefail
echo "[deprecated] This legacy entry used object_id-only/default trajectory paths."
echo "[canonical] Redirecting to /data/dyj/zts/pull-push/scripts/pica_server/local/active/run_manifest_v2d_both_ft50.sh"
exec /data/dyj/zts/pull-push/scripts/pica_server/local/active/run_manifest_v2d_both_ft50.sh "$@"
