#!/bin/bash
set -e
PYBIN=/home/plote/miniconda3/envs/isaacgym/bin/python
export PYTHONPATH=/home/plote/isaacgym/python:/home/plote/new/2/flash-linear-attention
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions/py38_cu121
export LD_LIBRARY_PATH=/home/plote/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH
cd /home/plote/new/2
mkdir -p output/gla_tuning_45936/screen
for B in 0.015 0.02 0.03; do
    TAG="hand_drag_gla_45936_screen_b${B}_poolmean"
    LOG="output/gla_tuning_45936/screen/${TAG}_metrics.csv"
    SUM="output/gla_tuning_45936/screen/${TAG}_summary.json"
    if [ -f "$SUM" ]; then echo "[skip] $TAG"; continue; fi
    echo "[eval] $TAG"
    "${PYBIN}" scripts/evaluate_ppo_baseline.py \
        --checkpoint "runs/${TAG}/nn" --checkpoint-kind latest \
        --object_id 45936 --episodes 10 --seed 42 --max_episode_length 300 \
        --gla_pool mean \
        --log_csv "$LOG" --summary_json "$SUM" 2>&1 | tail -3
done
