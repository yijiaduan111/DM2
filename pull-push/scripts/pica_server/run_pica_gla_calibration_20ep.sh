#!/bin/bash
# scripts/pica_server/run_pica_gla_calibration_20ep.sh
#
# Reward-scale calibration for PICA-GLA on a remote GPU server.
#
# Bracket: action_bound.weight in {1.0, 5.0, 20.0}; everything else fixed
# at the train_config_gla_pica.yaml defaults. The aim is to find the
# weight at which r_phys_bound_mean is visibly non-zero at epoch 20
# without crushing nominal training progress.
#
# Each run produces:
#   runs/<exp_name>/                             checkpoint dir
#   runs/<exp_name>/epoch_rewards.csv            per-epoch component log
#   logs_output/pica_calibration/<exp_name>.log  full stdout/stderr
#   logs_output/pica_calibration/<exp_name>.cmd  exact command used
#
# DO NOT run locally; the local machine has power instability.

set -euo pipefail

# ---- Activate / paths ----------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/home/plote/new/2}"
PYBIN="${PYBIN:-/home/plote/miniconda3/envs/isaacgym/bin/python}"

export PYTHONPATH="/home/plote/isaacgym/python:${PROJECT_ROOT}/flash-linear-attention"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions/py38_cu121}"
export LD_LIBRARY_PATH="/home/plote/miniconda3/envs/isaacgym/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs_output/pica_calibration

# ---- Fixed protocol ------------------------------------------------------
OBJ=45936
NUM_ENVS=64
MAX_EPOCHS=20
BOUNDS=0.01
TRAIN_CFG="ppo/train_config_gla_pica.yaml"

# ---- Calibration bracket -------------------------------------------------
declare -a LAMBDA_BOUNDS=(1.0 5.0 20.0)
declare -a EXP_TAGS=(ab1 ab5 ab20)

run_one() {
    local lb="$1"
    local tag="$2"
    local exp_name="hand_drag_gla_${OBJ}_pica_${tag}_contact05_smoke20"
    local log_file="logs_output/pica_calibration/${exp_name}.log"
    local cmd_file="logs_output/pica_calibration/${exp_name}.cmd"

    if [ -f "runs/${exp_name}/epoch_rewards.csv" ] && [ "$(wc -l <"runs/${exp_name}/epoch_rewards.csv")" -ge $((MAX_EPOCHS+1)) ]; then
        echo "[skip] ${exp_name} already has >= ${MAX_EPOCHS} epochs"
        return
    fi

    local cmd=(
        "${PYBIN}" ppo/train.py
        --train_config "${TRAIN_CFG}"
        --object_id "${OBJ}"
        --num_envs "${NUM_ENVS}"
        --max_epochs "${MAX_EPOCHS}"
        --bounds_loss_coef "${BOUNDS}"
        --phys_reg 1
        --lambda_bound "${lb}"
        --lambda_contact 0.5
        --lambda_slip 0.0
        --lambda_smooth 0.0
        --experiment_name "${exp_name}"
    )

    # Persist the exact command for reproducibility.
    printf '%q ' "${cmd[@]}" > "${cmd_file}"
    echo >> "${cmd_file}"

    echo "[run] ${exp_name}  lambda_bound=${lb}"
    printf '       %q ' "${cmd[@]}"; echo
    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

for i in "${!LAMBDA_BOUNDS[@]}"; do
    run_one "${LAMBDA_BOUNDS[$i]}" "${EXP_TAGS[$i]}"
done

echo
echo "DONE. After all three runs finish, summarize with:"
echo "    ${PYBIN} scripts/pica_server/summarize_pica_calibration.py"
