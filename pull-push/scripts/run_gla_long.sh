#!/bin/bash
# Stage B: 100-epoch training for the top 2 screening configs.
# Object: 45936. Both runs use pool=mean.
set -e

OBJ=45936
EPOCHS=100
NUM_ENVS=64
PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

cd /home/plote/new/2

run_one() {
    local BOUNDS="$1"
    local POOL="$2"
    local TAG="hand_drag_gla_${OBJ}_long_b${BOUNDS}_pool${POOL}"
    if [ -d "runs/${TAG}/nn" ] && [ -n "$(ls runs/${TAG}/nn/*.pth 2>/dev/null)" ]; then
        local epochs_done=$(($(wc -l < runs/${TAG}/epoch_rewards.csv 2>/dev/null) - 2))
        if [ "${epochs_done}" -ge "${EPOCHS}" ]; then
            echo "[skip] ${TAG} already trained for ${epochs_done} epochs"
            return
        fi
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
}

run_one 0.015 mean
run_one 0.03  mean
echo "DONE long_runs"
