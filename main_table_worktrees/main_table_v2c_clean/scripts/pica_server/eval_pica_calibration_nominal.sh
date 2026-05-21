#!/bin/bash
# scripts/pica_server/eval_pica_calibration_nominal.sh
#
# Nominal deterministic eval of the latest checkpoint of each PICA-GLA
# calibration run. To be run AFTER the 20-epoch calibration has finished.
#
# Eval protocol matches output/gla_tuning_45936/long_damping/:
#   object 45936, 10 episodes, seed 42, max_episode_length 300,
#   eval_mode True, deterministic mu, NOMINAL damping (x1) only.
#
# DO NOT run on a power-unstable local machine.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/plote/new/2}"
PYBIN="${PYBIN:-/home/plote/miniconda3/envs/isaacgym/bin/python}"

export PYTHONPATH="/home/plote/isaacgym/python:${PROJECT_ROOT}/flash-linear-attention"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions/py38_cu121}"
export LD_LIBRARY_PATH="/home/plote/miniconda3/envs/isaacgym/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"

OBJ=45936
EPISODES=10
SEED=42
MAX_LEN=300

OUTDIR="output/pica_calibration_${OBJ}"
mkdir -p "${OUTDIR}"

declare -a EXP_NAMES=(
    "hand_drag_gla_45936_pica_ab1_contact05_smoke20"
    "hand_drag_gla_45936_pica_ab5_contact05_smoke20"
    "hand_drag_gla_45936_pica_ab20_contact05_smoke20"
)

eval_one() {
    local exp_name="$1"
    local ckpt_dir="runs/${exp_name}/nn"
    local prefix="${exp_name}_damp1_det"
    local log_csv="${OUTDIR}/${prefix}_metrics.csv"
    local sum_json="${OUTDIR}/${prefix}_summary.json"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "[skip] no checkpoint dir: ${ckpt_dir}"
        return
    fi
    if [ -z "$(ls -A "${ckpt_dir}"/*.pth 2>/dev/null || true)" ]; then
        echo "[skip] no .pth files in: ${ckpt_dir}"
        return
    fi
    if [ -f "${sum_json}" ]; then
        echo "[skip] already have summary: ${sum_json}"
        return
    fi

    echo "[eval] ${prefix}"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${ckpt_dir}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EPISODES}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --gla_pool last \
        --log_csv "${log_csv}" --summary_json "${sum_json}" \
        2>&1 | tail -8
}

for exp in "${EXP_NAMES[@]}"; do
    eval_one "${exp}"
done

echo
echo "Eval done. Compute saturation / detach metrics with:"
echo "    ${PYBIN} scripts/eval_postprocess.py ${OUTDIR}/*_metrics.csv"
