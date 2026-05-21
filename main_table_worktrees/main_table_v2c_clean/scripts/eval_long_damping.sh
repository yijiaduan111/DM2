#!/bin/bash
# Stage C: 20-episode deterministic damping sweep for the two tuned 100-epoch
# GLA candidates at damping_scale ∈ {1, 2, 4}. Extends to {8, 16} only when
# called with EXTEND=1.
set -e

OBJ=45936
EP=20
SEED=42
MAX_LEN=300

PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

cd /home/plote/new/2
OUTDIR="output/gla_tuning_45936/long_damping"
mkdir -p "${OUTDIR}"

eval_one() {
    local TAG="$1"        # config tag e.g. b0.015_poolmean
    local CKPT_DIR="$2"
    local D="$3"
    local PREFIX="${TAG}_damp${D}_det"
    local LOG_CSV="${OUTDIR}/${PREFIX}_metrics.csv"
    local SUM_JSON="${OUTDIR}/${PREFIX}_summary.json"
    if [ -f "${SUM_JSON}" ]; then
        echo "[skip] ${PREFIX}"
        return
    fi
    echo "[eval] ${PREFIX}"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${CKPT_DIR}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EP}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --object_damping_scale "${D}" \
        --gla_pool mean \
        --log_csv "${LOG_CSV}" --summary_json "${SUM_JSON}" \
        2>&1 | tail -8
}

CONFIGS=(
    "b0.015_poolmean runs/hand_drag_gla_45936_long_b0.015_poolmean/nn"
    "b0.03_poolmean  runs/hand_drag_gla_45936_long_b0.03_poolmean/nn"
)

DAMPS_BASE=(1 2 4)
if [ "${EXTEND:-0}" = "1" ]; then
    DAMPS_BASE=(8 16)
fi

for CFG in "${CONFIGS[@]}"; do
    set -- ${CFG}
    TAG="$1"; CKPT="$2"
    for D in "${DAMPS_BASE[@]}"; do
        eval_one "${TAG}" "${CKPT}" "${D}"
    done
done
echo "DONE damping eval"
