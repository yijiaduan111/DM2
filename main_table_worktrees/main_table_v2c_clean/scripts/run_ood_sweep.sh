#!/bin/bash
# OOD damping/friction sweep driver. Runs deterministic eval at each scale
# for both flat-history and GLA checkpoints with identical protocol.
#
# Usage:
#   bash scripts/run_ood_sweep.sh damping
#   bash scripts/run_ood_sweep.sh friction
#   bash scripts/run_ood_sweep.sh combined
set -e

MODE="${1:-damping}"
EP=20
SEED=42
MAX_LEN=300
OBJ=45936

FLAT_CKPT="runs/hand_drag_history_45936_long_bounds01/nn"
GLA_CKPT="runs/hand_drag_gla_45936_long_bounds01/nn"

PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

run_one() {
    local TAG="$1"; local CKPT="$2"; local D="$3"; local F="$4"
    local OUTDIR="output/ood_dynamics_${OBJ}/${MODE}_sweep"
    mkdir -p "${OUTDIR}"
    local LOG_CSV="${OUTDIR}/${TAG}_metrics.csv"
    local SUM_JSON="${OUTDIR}/${TAG}_summary.json"
    if [ -f "${SUM_JSON}" ]; then
        echo "  [skip] ${TAG} already done"
        return
    fi
    echo "  [run]  ${TAG}  damp=${D}  fric=${F}"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "${CKPT}" --checkpoint-kind latest \
        --object_id "${OBJ}" --episodes "${EP}" --seed "${SEED}" \
        --max_episode_length "${MAX_LEN}" \
        --object_damping_scale "${D}" \
        --object_friction_scale "${F}" \
        --log_csv "${LOG_CSV}" --summary_json "${SUM_JSON}" \
        2>&1 | tail -3
}

if [ "${MODE}" = "damping" ]; then
    for D in 1 2 4 8 16; do
        run_one "flat_damp${D}_det" "${FLAT_CKPT}" "${D}" 1.0
        run_one "gla_damp${D}_det"  "${GLA_CKPT}"  "${D}" 1.0
    done
elif [ "${MODE}" = "friction" ]; then
    for F in 0.5 1 2 4; do
        run_one "flat_fric${F}_det" "${FLAT_CKPT}" 1.0 "${F}"
        run_one "gla_fric${F}_det"  "${GLA_CKPT}"  1.0 "${F}"
    done
elif [ "${MODE}" = "combined" ]; then
    for COMBO in "2 2" "4 2" "4 4" "8 2"; do
        D=$(echo $COMBO | awk '{print $1}')
        F=$(echo $COMBO | awk '{print $2}')
        run_one "flat_damp${D}_fric${F}_det" "${FLAT_CKPT}" "${D}" "${F}"
        run_one "gla_damp${D}_fric${F}_det"  "${GLA_CKPT}"  "${D}" "${F}"
    done
else
    echo "Unknown mode: ${MODE}"
    exit 1
fi
echo "DONE ${MODE}"
