#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_cuda_kernel_eval.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="${ISAACGYM_ROOT}/python:${PROJECT_ROOT}/flash-linear-attention:${PYTHONPATH:-}"
bash "${PROJECT_ROOT}/scripts/pica_server/run_pica_gla_calibration_20ep.sh"
