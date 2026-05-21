#!/usr/bin/env bash
set -u
source /data/dyj/zts/pull-push/scripts/pica_server/local/env_cuda_kernel_eval.sh
cd /data/dyj/zts/pull-push

RUN_NAME="${RUN_NAME:-hand_drag_gla_45936_pica_drand12_aux_v2c_500ep_es}"
CKPT_DIR="${CKPT_DIR:-runs/${RUN_NAME}/nn}"
OUT_ROOT="${OUT_ROOT:-output/pica_v2c_eval}"
EPISODES="${EPISODES:-20}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-300}"
GLA_POOL="${GLA_POOL:-last}"
DAMPS="${DAMPS:-1.0 2.0 4.0}"

DET_OUT="${OUT_ROOT}/${RUN_NAME}"
STO_OUT="${OUT_ROOT}/${RUN_NAME}_stochastic"
mkdir -p "${DET_OUT}" "${STO_OUT}"

run_eval() {
  local mode="$1"
  local outdir="$2"
  local extra="$3"
  local suffix="det"
  [ "$mode" = "stoch" ] && suffix="stoch"

  for D in ${DAMPS}; do
    local prefix="${RUN_NAME}_damp${D}_${suffix}"
    local log_csv="${outdir}/${prefix}_metrics.csv"
    local sum_json="${outdir}/${prefix}_summary.json"
    local stdout_log="${outdir}/${prefix}.log"
    local cmd_log="${outdir}/${prefix}.cmd"
    if [ -f "${sum_json}" ]; then
      echo "[skip] ${prefix}"
      continue
    fi
    echo "[eval] ${prefix}"
    cat > "${cmd_log}" <<EOF
python scripts/evaluate_ppo_baseline.py --checkpoint "${CKPT_DIR}" --checkpoint-kind latest --object_id 45936 --episodes ${EPISODES} --seed ${SEED} --max_episode_length ${MAX_LEN} --object_damping_scale ${D} --gla_pool ${GLA_POOL} ${extra} --log_csv "${log_csv}" --summary_json "${sum_json}"
EOF
    python scripts/evaluate_ppo_baseline.py \
      --checkpoint "${CKPT_DIR}" --checkpoint-kind latest \
      --object_id 45936 --episodes "${EPISODES}" --seed "${SEED}" \
      --max_episode_length "${MAX_LEN}" \
      --object_damping_scale "${D}" \
      --gla_pool "${GLA_POOL}" ${extra} \
      --log_csv "${log_csv}" --summary_json "${sum_json}" \
      2>&1 | tee "${stdout_log}" | tail -12 || true
  done
}

run_eval det "${DET_OUT}" ""
run_eval stoch "${STO_OUT}" "--stochastic"
python scripts/pica_server/local/summarize_damping_eval.py --run-name "${RUN_NAME}" --det-dir "${DET_OUT}" --stoch-dir "${STO_OUT}"
