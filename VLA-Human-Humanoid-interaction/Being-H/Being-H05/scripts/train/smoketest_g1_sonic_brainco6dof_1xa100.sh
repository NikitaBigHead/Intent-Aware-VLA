#!/bin/bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Short smoke test for train_g1_sonic_brainco6dof_1xa100.sh: same
# single-GPU (1x A100 80GB) config, but a handful of steps writing to a
# throwaway output dir, so you can confirm the whole pipeline (checkpoint
# loading, dataset build against the real "sep-10-2026-cleaned-merged"
# dataset, a forward/backward step, a checkpoint save) actually comes up --
# and watch `nvidia-smi` for peak VRAM -- before committing to the full
# 40k-step run. Every setting that affects per-step memory (MAX_NUM_TOKENS,
# EXPECTED_NUM_TOKENS, ACTION_CHUNK_LENGTH, FORCE_IMAGE_SIZE,
# GRAD_CHECKPOINT, USE_GRADIENT_CHECKPOINT, FREEZE_MLLM, FREEZE_VIT_MLP,
# FREEZE_VIT, VIT_TRAINABLE_WITHIN_MLLM -- the last of these unfreezes the
# vision tower, see the full run's own "Freezing Configuration" / "Batch
# size / VRAM reasoning" comments for the why and the VRAM math)
# intentionally matches the full run exactly (see
# train_g1_sonic_brainco6dof_1xa100.sh's "Batch size / VRAM reasoning"
# comment for why those are the real memory-controlling knobs here, not a
# --global-batch-size flag), so the peak memory you observe is the real
# number for the full run, not an approximation. Only MAX_STEPS,
# SAVE_STEPS/SAVE_STEPS_START, and OUTPUT_DIR differ.
#
# This is also the closest thing to a config sanity check available without
# a GPU/the real dataset: docs/data_configuration.md's own "Step 5: Verify
# Setup" recommends exactly this pattern (torchrun ... --max_steps 10
# --output_dir /tmp/...) for validating a new DataConfig/YAML/dataset
# registration before trusting a long run.
#
# Usage:
#   DATASET_PATH=/path/to/sep-10-2026-cleaned-merged \
#     bash scripts/train/smoketest_g1_sonic_brainco6dof_1xa100.sh
#
# While it runs, in another terminal on the same machine:
#   watch -n1 nvidia-smi
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING -- run it yourself.
# It was not run while writing it either: no GPU and no copy of the
# dataset were available in this environment.

set -euo pipefail

export PYTHONPATH=.
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=1

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

PRETRAIN_MODEL="${PRETRAIN_MODEL:-/path/to/InternVL3_5-2B}"
EXPERT_MODEL="${EXPERT_MODEL:-/path/to/Qwen3-0.6B}"
RESUME_PATH="${RESUME_PATH:-/path/to/model_zoo/Being-H05-2B}"

: "${DATASET_PATH:?Set DATASET_PATH to your LeRobot v2.1 dataset root, e.g. DATASET_PATH=/path/to/sep-10-2026-cleaned-merged bash $0}"
export DATASET_PATH

DATASET_CONFIG_FILE="configs/posttrain/unitree_g1_sonic_brainco6dof/unitree_g1_sonic_brainco6dof.yaml"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/outputs/g1_sonic_brainco6dof_smoketest_1xa100}"
LOG_DIR="results/tensorboard/smoketest"
LOG_FILE="${OUTPUT_DIR}/training.log"
mkdir -p "$OUTPUT_DIR"

# --- Everything below this line matches train_g1_sonic_brainco6dof_1xa100.sh ---
FORCE_IMAGE_SIZE=224
MAX_VIEW_NUM=-1
USE_FIXED_VIEW=False
DOWN_SAMPLE_RATIO=0.5
ACTION_CHUNK_LENGTH=40
FREEZE_MLLM=True
FREEZE_VIT_MLP=False
FREEZE_VIT=False
VIT_TRAINABLE_WITHIN_MLLM=True
GRAD_CHECKPOINT=True
USE_GRADIENT_CHECKPOINT=True
MAX_NUM_TOKENS=7680
EXPECTED_NUM_TOKENS=7168
PREFER_BUFFER_BEFORE=3584
MAX_BUFFER_SIZE=4
ATTN_MODE="causal"
GRADIENT_ACCUMULATION_STEPS=1
NUM_WORKERS=12
PREFETCH_FACTOR=8
USE_MPG=True
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=1.0
MPG_USE_STOP_GRADIENT=True
USE_TRAINING_TIME_RTC=False
SIMULATED_DELAY=0
RTC_DELAY_EXP_WEIGHT=True
USE_INFERENCE_PREFIX_OVERWRITE=True

# --- Smoke-test-only overrides ---
MAX_STEPS=2
SAVE_STEPS=2
SAVE_STEPS_START=0
SAVE_LAST=False
SAVE_MODEL_ONLY=True
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05
SAVE_MERGED_META=False

MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

echo "SMOKE TEST (2 steps, 1x A100 80GB, throwaway output dir):"
echo "torchrun --nnodes=1 --node_rank=0 --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} \\"
echo "  BeingH/train/train.py \\"
echo "  --mllm_path ${PRETRAIN_MODEL} \\"
echo "  --expert_path ${EXPERT_MODEL} \\"
echo "  --resume_from ${RESUME_PATH} \\"
echo "  --resume_model_only True \\"
echo "  --layer_module Qwen3MoTDecoderLayer \\"
echo "  --use_expert True \\"
echo "  --use_flow_matching True \\"
echo "  --llm_qk_norm True \\"
echo "  --freeze_mllm ${FREEZE_MLLM} \\"
echo "  --freeze_vit_mlp ${FREEZE_VIT_MLP} \\"
echo "  --freeze_vit ${FREEZE_VIT} \\"
echo "  --vit_trainable_within_mllm ${VIT_TRAINABLE_WITHIN_MLLM} \\"
echo "  --grad_checkpoint ${GRAD_CHECKPOINT} \\"
echo "  --action_chunk_length ${ACTION_CHUNK_LENGTH} \\"
echo "  --max_num_tokens ${MAX_NUM_TOKENS} \\"
echo "  --max_num_tokens_per_sample ${MAX_NUM_TOKENS} \\"
echo "  --expected_num_tokens ${EXPECTED_NUM_TOKENS} \\"
echo "  --prefer_buffer_before ${PREFER_BUFFER_BEFORE} \\"
echo "  --max_buffer_size ${MAX_BUFFER_SIZE} \\"
echo "  --attn_mode ${ATTN_MODE} \\"
echo "  --max_view_num ${MAX_VIEW_NUM} \\"
echo "  --use_fixed_view ${USE_FIXED_VIEW} \\"
echo "  --force_image_size ${FORCE_IMAGE_SIZE} \\"
echo "  --down_sample_ratio ${DOWN_SAMPLE_RATIO} \\"
echo "  --dataset_config_file ${DATASET_CONFIG_FILE} \\"
echo "  --save_merged_metadata ${SAVE_MERGED_META} \\"
echo "  --conv_style being_h0 \\"
echo "  --vision_select_layer -1 \\"
echo "  --prompt_template long \\"
echo "  --output_dir ${OUTPUT_DIR} \\"
echo "  --logging_dir ${LOG_DIR} \\"
echo "  --num_workers ${NUM_WORKERS} \\"
echo "  --prefetch_factor ${PREFETCH_FACTOR} \\"
echo "  --max_steps ${MAX_STEPS} \\"
echo "  --save_model_only ${SAVE_MODEL_ONLY} \\"
echo "  --save_steps ${SAVE_STEPS} \\"
echo "  --save_steps_start ${SAVE_STEPS_START} \\"
echo "  --save_last ${SAVE_LAST} \\"
echo "  --logging_steps 1 \\"
echo "  --learning_rate ${LEARNING_RATE} \\"
echo "  --weight_decay ${WEIGHT_DECAY} \\"
echo "  --warmup_ratio ${WARMUP_RATIO} \\"
echo "  --lr_scheduler cosine \\"
echo "  --use_gradient_checkpoint ${USE_GRADIENT_CHECKPOINT} \\"
echo "  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \\"
echo "  --use_mpg ${USE_MPG} \\"
echo "  --mpg_lambda ${MPG_LAMBDA} \\"
echo "  --mpg_num_projections ${MPG_NUM_PROJECTIONS} \\"
echo "  --mpg_refinement_iters ${MPG_REFINEMENT_ITERS} \\"
echo "  --mpg_gate_temperature ${MPG_GATE_TEMPERATURE} \\"
echo "  --mpg_use_stop_gradient ${MPG_USE_STOP_GRADIENT} \\"
echo "  --use_training_time_rtc ${USE_TRAINING_TIME_RTC} \\"
echo "  --simulated_delay ${SIMULATED_DELAY} \\"
echo "  --rtc_delay_exp_weight ${RTC_DELAY_EXP_WEIGHT} \\"
echo "  --use_inference_prefix_overwrite ${USE_INFERENCE_PREFIX_OVERWRITE} \\"
echo "  2>&1 | tee ${LOG_FILE}"
echo ""
echo "DATASET_PATH=${DATASET_PATH}"
echo ""

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=${MASTER_PORT} \
  BeingH/train/train.py \
  --mllm_path ${PRETRAIN_MODEL} \
  --expert_path ${EXPERT_MODEL} \
  --resume_from ${RESUME_PATH} \
  --resume_model_only True \
  --layer_module Qwen3MoTDecoderLayer \
  --use_expert True \
  --use_flow_matching True \
  --llm_qk_norm True \
  --freeze_mllm ${FREEZE_MLLM} \
  --freeze_vit_mlp ${FREEZE_VIT_MLP} \
  --freeze_vit ${FREEZE_VIT} \
  --vit_trainable_within_mllm ${VIT_TRAINABLE_WITHIN_MLLM} \
  --grad_checkpoint ${GRAD_CHECKPOINT} \
  --action_chunk_length ${ACTION_CHUNK_LENGTH} \
  --max_num_tokens ${MAX_NUM_TOKENS} \
  --max_num_tokens_per_sample ${MAX_NUM_TOKENS} \
  --expected_num_tokens ${EXPECTED_NUM_TOKENS} \
  --prefer_buffer_before ${PREFER_BUFFER_BEFORE} \
  --max_buffer_size ${MAX_BUFFER_SIZE} \
  --attn_mode ${ATTN_MODE} \
  --max_view_num ${MAX_VIEW_NUM} \
  --use_fixed_view ${USE_FIXED_VIEW} \
  --force_image_size ${FORCE_IMAGE_SIZE} \
  --down_sample_ratio ${DOWN_SAMPLE_RATIO} \
  --dataset_config_file ${DATASET_CONFIG_FILE} \
  --save_merged_metadata ${SAVE_MERGED_META} \
  --conv_style being_h0 \
  --vision_select_layer -1 \
  --prompt_template long \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${LOG_DIR} \
  --num_workers ${NUM_WORKERS} \
  --prefetch_factor ${PREFETCH_FACTOR} \
  --max_steps ${MAX_STEPS} \
  --save_model_only ${SAVE_MODEL_ONLY} \
  --save_steps ${SAVE_STEPS} \
  --save_steps_start ${SAVE_STEPS_START} \
  --save_last ${SAVE_LAST} \
  --logging_steps 1 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --lr_scheduler cosine \
  --use_gradient_checkpoint ${USE_GRADIENT_CHECKPOINT} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --use_mpg ${USE_MPG} \
  --mpg_lambda ${MPG_LAMBDA} \
  --mpg_num_projections ${MPG_NUM_PROJECTIONS} \
  --mpg_refinement_iters ${MPG_REFINEMENT_ITERS} \
  --mpg_gate_temperature ${MPG_GATE_TEMPERATURE} \
  --mpg_use_stop_gradient ${MPG_USE_STOP_GRADIENT} \
  --use_training_time_rtc ${USE_TRAINING_TIME_RTC} \
  --simulated_delay ${SIMULATED_DELAY} \
  --rtc_delay_exp_weight ${RTC_DELAY_EXP_WEIGHT} \
  --use_inference_prefix_overwrite ${USE_INFERENCE_PREFIX_OVERWRITE} \
  2>&1 | tee "${LOG_FILE}"

echo "=========================================="
echo "Smoke test complete!"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "If this succeeded (2 steps ran, a checkpoint landed in"
echo "${OUTPUT_DIR}/0000002, and nvidia-smi showed a peak"
echo "well under 80GB), the full run in train_g1_sonic_brainco6dof_1xa100.sh"
echo "is ready to launch."
echo "=========================================="
