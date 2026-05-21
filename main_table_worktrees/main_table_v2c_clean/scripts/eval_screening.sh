#!/bin/bash
# Stage A: lightweight deterministic eval (10 episodes) for each screening
# checkpoint at nominal damping/friction.
set -e

OBJ=45936
EP=10
SEED=42
MAX_LEN=300

PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

cd /home/plote/new/2
OUTDIR="output/gla_tuning_45936/screen"
mkdir -p "${OUTDIR}"

eval_one() {
    local TAG="$1"
    local CKPT_DIR="runs/${TAG}/nn"
    local LOG_CSV="${OUTDIR}/${TAG}_metrics.csv"
    local SUM_JSON="${OUTDIR}/${TAG}_summary.json"
    if [ -f "${SUM_JSON}" ]; then
        echo "[skip] ${TAG} already evaluated"
        return
    fi
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "[miss] ${CKPT_DIR}"
        return
    fi
    echo "[eval] ${TAG}"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${CKPT_DIR}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EP}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --log_csv "${LOG_CSV}" --summary_json "${SUM_JSON}" \
        2>&1 | tail -8
}

for B in 0.015 0.02 0.03; do
    for P in last mean; do
        eval_one "hand_drag_gla_${OBJ}_screen_b${B}_pool${P}"
    done
done
echo "DONE eval_screening"
