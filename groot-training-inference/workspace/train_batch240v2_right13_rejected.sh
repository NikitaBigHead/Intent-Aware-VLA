#!/usr/bin/env bash
set -euo pipefail
cd /opt/Isaac-GR00T
source /opt/Isaac-GR00T/.venv/bin/activate
export VLM_REJECTED_JSONL=/data/project/batch240v2_8objects_rejected_video_lerobot_mapped_20261002.jsonl
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/data/project/hf_cache
export HF_XET_CACHE=/data/project/hf_cache/xet
export TMPDIR=/data/project/tmp
mkdir -p /data/project/hf_cache/xet /data/project/tmp /data/project/logs
export MAX_STEPS=20000
export SAVE_STEPS=5000
export SAVE_TOTAL_LIMIT=2
export GLOBAL_BATCH_SIZE=16
export USE_WANDB=0
export DATALOADER_NUM_WORKERS=4
LOG=/data/project/logs/train_batch240v2_right13_rejected106_20261002.log
bash examples/finetune.sh \
  --base-model-path /root/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/d0814e7ecb19202e7c8468b46098b0b7ef3a6d61/ \
  --dataset-path /data/project/batch240v2_8objects_lerobot_rawvideo_groot_right13 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /workspace/gr00t_cosmos_right13_config.py \
  --output-dir /workspace/checkpoints/batch240v2_8objects_right13_rejected106_20261002 \
  -- \
     --upload_checkpoints \
     --upload_every 5000 \
     --hf_repo_id Arcwy/gr00t-cosmos-right13-batch240v2-8objects \
     --hf_token_file /data/project/.hf_token_batch240v2 \
  2>&1 | tee "$LOG"
