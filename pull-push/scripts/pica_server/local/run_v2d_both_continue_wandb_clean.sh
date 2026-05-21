#!/usr/bin/env bash
set -u
cd /data/dyj/zts/pull-push
source scripts/pica_server/local/env_cuda_kernel_eval.sh

RUN_NAME=${RUN_NAME:-v2c150_v2d_both_continue400_clean_45936}
GPU=${GPU:-0}
MAX_EPOCHS=${MAX_EPOCHS:-400}
NUM_ENVS=${NUM_ENVS:-64}
TRAIN_CFG=${TRAIN_CFG:-ppo/train_config_gla_pica_v2d_both_ft_continue_save25.yaml}
OBJECT_ID=${OBJECT_ID:-45936}
BOUNDS_LOSS_COEF=${BOUNDS_LOSS_COEF:-0.01}
CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-best}
WANDB_PROJECT=${WANDB_PROJECT:-pica-pull-push-clean}
WANDB_MODE=${WANDB_MODE:-online}
STEP_OFFSET=${STEP_OFFSET:-188}
EARLY_MIN_EPOCHS=${EARLY_MIN_EPOCHS:-72}
EARLY_PATIENCE=${EARLY_PATIENCE:-50}
EARLY_MIN_DELTA=${EARLY_MIN_DELTA:-1.0}
EARLY_WINDOW=${EARLY_WINDOW:-5}

mkdir -p logs_output runs/${RUN_NAME}
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 CUDA_MODULE_LOADING=LAZY
export PYTORCH_JIT=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export WANDB_MODE

cat > logs_output/${RUN_NAME}_command.txt <<EOF
RUN_NAME=${RUN_NAME}
GPU=${GPU}
MAX_EPOCHS=${MAX_EPOCHS}
NUM_ENVS=${NUM_ENVS}
TRAIN_CFG=${TRAIN_CFG}
OBJECT_ID=${OBJECT_ID}
BOUNDS_LOSS_COEF=${BOUNDS_LOSS_COEF}
CHECKPOINT=${CHECKPOINT}
CHECKPOINT_KIND=${CHECKPOINT_KIND}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_MODE=${WANDB_MODE}
STEP_OFFSET=${STEP_OFFSET}
EARLY_MIN_EPOCHS=${EARLY_MIN_EPOCHS}
EARLY_PATIENCE=${EARLY_PATIENCE}
EARLY_MIN_DELTA=${EARLY_MIN_DELTA}
EARLY_WINDOW=${EARLY_WINDOW}
EOF

echo "[launcher] starting ${RUN_NAME} on GPU ${GPU} max_epochs=${MAX_EPOCHS}"
CUDA_VISIBLE_DEVICES=${GPU} python -X faulthandler ppo/train.py \
  --train_config "${TRAIN_CFG}" \
  --object_id "${OBJECT_ID}" --num_envs "${NUM_ENVS}" --max_epochs "${MAX_EPOCHS}" \
  --bounds_loss_coef "${BOUNDS_LOSS_COEF}" \
  --experiment_name "${RUN_NAME}" \
  --checkpoint "${CHECKPOINT}" --checkpoint-kind "${CHECKPOINT_KIND}" \
  > logs_output/${RUN_NAME}_train.log 2>&1 &
TRAIN_PID=$!
echo ${TRAIN_PID} > logs_output/${RUN_NAME}_train.pid

echo "[launcher] train pid=${TRAIN_PID}; starting clean wandb monitor"
WANDB_MODE=${WANDB_MODE} python scripts/pica_server/local/monitor_epoch_rewards_wandb_clean.py \
  --run-dir runs/${RUN_NAME} \
  --pid ${TRAIN_PID} \
  --project ${WANDB_PROJECT} \
  --name ${RUN_NAME} \
  --mode ${WANDB_MODE} \
  --step-offset ${STEP_OFFSET} \
  --min-epochs ${EARLY_MIN_EPOCHS} \
  --patience ${EARLY_PATIENCE} \
  --min-delta ${EARLY_MIN_DELTA} \
  --window ${EARLY_WINDOW} \
  > logs_output/${RUN_NAME}_monitor_clean.log 2>&1 &
MONITOR_PID=$!
echo ${MONITOR_PID} > logs_output/${RUN_NAME}_monitor_clean.pid

wait ${TRAIN_PID}
TRAIN_EXIT=$?
echo "[launcher] train exit=${TRAIN_EXIT}" | tee -a logs_output/${RUN_NAME}_launcher.log
wait ${MONITOR_PID} 2>/dev/null || true
echo "[launcher] done $(date)" | tee -a logs_output/${RUN_NAME}_launcher.log
exit ${TRAIN_EXIT}
