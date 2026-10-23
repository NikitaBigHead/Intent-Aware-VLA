#!/usr/bin/env bash
# Finetune GR00T-N1.6 on the Cosmos3 + RoboHaMeR right-arm dataset.
set -euo pipefail

source /opt/Isaac-GR00T/.venv/bin/activate
cd /opt/Isaac-GR00T
mkdir -p /workspace/logs /workspace/checkpoints

export VLM_REJECTED_JSONL="${VLM_REJECTED_JSONL:-}"

MAX_STEPS=${MAX_STEPS:-10000} \
SAVE_STEPS=${SAVE_STEPS:-2500} \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16} \
USE_WANDB=0 \
DATALOADER_NUM_WORKERS=4 \
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/*/ \
  --dataset-path /data/project/cosmos3_g1right_lerobot_rawvideo \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/gr00t_cosmos_right13_config.py \
  --output-dir /workspace/checkpoints/cosmos_right13_rawvideo \
  2>&1 | tee /workspace/logs/train_cosmos_$(date +%Y%m%d_%H%M%S).log
