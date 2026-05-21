#!/bin/bash
# Stage A: 20-epoch screening sweep over (bounds_loss_coef, gla.pool).
# Object: 45936. Trains 6 configs sequentially.
set -e

OBJ=45936
EPOCHS=20
NUM_ENVS=64
PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

cd /home/plote/new/2

run_one() {
    local BOUNDS="$1"
    local POOL="$2"
    local TAG="hand_drag_gla_${OBJ}_screen_b${BOUNDS}_pool${POOL}"
    if [ -d "runs/${TAG}/nn" ] && [ -n "$(ls runs/${TAG}/nn/*.pth 2>/dev/null)" ]; then
        echo "[skip] ${TAG} already trained"
        return
    fi
    if [ "${POOL}" = "last" ]; then
        TRAIN_CFG="ppo/train_config_gla.yaml"
    else
        TRAIN_CFG="ppo/train_config_gla_pool_mean.yaml"
    fi
    echo "[train] ${TAG} (cfg=${TRAIN_CFG} bounds=${BOUNDS})"
    mkdir -p "logs/${TAG}"
    "${PYBIN}" ppo/train.py \
        --train_config "${TRAIN_CFG}" \
        --object_id "${OBJ}" \
        --num_envs "${NUM_ENVS}" \
        --max_epochs "${EPOCHS}" \
        --bounds_loss_coef "${BOUNDS}" \
        --experiment_name "${TAG}" \
        2>&1 | tail -200 > "logs/${TAG}/train.log"
    echo "  -> $(tail -3 logs/${TAG}/train.log | tr '\n' ' ')"
}

for B in 0.015 0.02 0.03; do
    for P in last mean; do
        run_one "${B}" "${P}"
    done
done
echo "DONE screening"
