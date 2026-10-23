#!/bin/bash
# Short smoke test for finetune_g1_sonic_brainco6dof.sh: same config, but
# a handful of steps writing to a throwaway output dir, so you can confirm
# the whole pipeline (model download, dataset build, DeepSpeed init,
# optimizer step, checkpoint save) actually comes up before committing to
# a full run. Also exercises the exact-steps checkpoint callback: with
# --max-steps 10 and RLDX_EXACT_SAVE_STEPS="5,10" you should see exactly
# two checkpoints (5 and 10), not three (i.e. no extra one at "end of
# training" beyond the last requested step).
#
# Usage:
#   DATASET_PATH=/path/to/your/lerobot_dataset \
#     bash run_scripts/train/custom/smoketest_g1_sonic_brainco6dof.sh

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1
# See finetune_g1_sonic_brainco6dof.sh for why this is needed.
export HF_HUB_DISABLE_XET=1
export RLDX_EXACT_SAVE_STEPS="5,10"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2

BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$BASE_DIR"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
: "${DATASET_PATH:?Set DATASET_PATH to your LeRobot v2.1 dataset root, e.g. DATASET_PATH=/path/to/dataset bash $0}"
EMBODIMENT_TAG="UNITREE_G1_SONIC_BRAINCO6DOF"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py"

CKPT_DIR="${CKPT_DIR:-$BASE_DIR/outputs/g1_sonic_brainco6dof_smoketest}"
export HF_HOME="${HF_HOME:-$BASE_DIR/.hf_cache}"
mkdir -p "$CKPT_DIR"

ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --video-length 4
    --n-cog-tokens 64
    --action-horizon 40
    --global-batch-size $((6 * NUM_GPUS))
    --gradient-accumulation-steps 1
    --num-gpus $NUM_GPUS
    --dataloader-num-workers 16
    --dataset-mode standard
    --learning-rate 1e-4
    --lr-scheduler-type cosine
    --weight-decay 0.01
    --warmup-ratio 0.0125
    --max-grad-norm 1.0
    --max-steps 10
    --save-steps 1000000
    --save-total-limit 5
    --output-dir "$CKPT_DIR"
    --experiment-name g1_sonic_brainco6dof_smoketest
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --random-rotation-angle 8
    --random-crop-fraction 0.9
)

export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"

echo "SMOKE TEST (10 steps): pixi run --environment rldx torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT rldx/experiment/launch_train.py ${ARGS[*]}"

pixi run --environment rldx torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py "${ARGS[@]}"
