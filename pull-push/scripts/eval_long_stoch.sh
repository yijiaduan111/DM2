#!/bin/bash
# Stage D: stochastic deterministic eval for the best candidate at damping x1, x2.
# Best per nominal det eval = b=0.015 + pool=mean.
set -e

OBJ=45936
EP=20
SEED=42
MAX_LEN=300
TAG="b0.015_poolmean"
CKPT="runs/hand_drag_gla_45936_long_b0.015_poolmean/nn"

PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

cd /home/plote/new/2
OUTDIR="output/gla_tuning_45936/long_damping"
mkdir -p "${OUTDIR}"

eval_one() {
    local D="$1"
    local PREFIX="${TAG}_damp${D}_stoch"
    local LOG_CSV="${OUTDIR}/${PREFIX}_metrics.csv"
    local SUM_JSON="${OUTDIR}/${PREFIX}_summary.json"
    if [ -f "${SUM_JSON}" ]; then
        echo "[skip] ${PREFIX}"
        return
    fi
    echo "[eval] ${PREFIX}"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${CKPT}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EP}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --object_damping_scale "${D}" \
        --gla_pool mean --stochastic \
        --log_csv "${LOG_CSV}" --summary_json "${SUM_JSON}" \
        2>&1 | tail -8
}

eval_one 1
eval_one 2
echo "DONE stochastic eval"
