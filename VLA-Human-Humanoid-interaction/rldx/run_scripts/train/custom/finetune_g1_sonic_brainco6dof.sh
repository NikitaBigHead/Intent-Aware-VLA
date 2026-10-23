#!/bin/bash
# Fine-tune RLDX-1 on a Unitree G1 + BrainCo Revo2 (6-DOF hand) SONIC
# motion-token dataset (LeRobot v2.1 format).
#
# Action target = 76-dim continuous vector:
#   64  action.motion_token   (SONIC latent, live-encoded during teleop)
#  + 6  teleop.left_hand_joints
#  + 6  teleop.right_hand_joints
#  (action.wbc is intentionally NOT used — it is a debug/reference field)
#
# Requires the one-time source patches + new files already applied under
# this checkout (see README.md "Unitree G1 + SONIC fine-tune" section for
# the full list and reasoning):
#   - rldx/data/embodiment_tags.py            (+ UNITREE_G1_SONIC_BRAINCO6DOF)
#   - rldx/model/core/processing_rldx.py       (+ projector-index mapping,
#                                                + max_action_dim/max_state_dim
#                                                override whitelist fix)
#   - rldx/model/core/setup.py                 (max_action_dim/max_state_dim
#                                                override fix for the
#                                                from-pretrained-checkpoint path)
#   - rldx/configs/model/rldx.py               (max_action_dim 64 -> 76)
#   - rldx/configs/training/training_config.py (gradient_checkpointing -> True)
#   - rldx/experiment/utils.py                 (+ ExactStepsCheckpointCallback)
#   - rldx/experiment/experiment.py            (wires in the callback above)
#   - rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py (new)
#
# Usage:
#   DATASET_PATH=/path/to/your/lerobot_dataset \
#     bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof.sh
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING — run it yourself
# when ready, after checking GPU availability and the env vars below.

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1

# The hf_xet accelerated downloader (huggingface_hub's default) writes its
# own state/logs to ~/.cache/huggingface/xet/ regardless of HF_HOME. If that
# directory isn't writable by your user (e.g. pre-existing, owned by a
# different user on a shared machine), the download can silently hang
# instead of erroring. Force the plain HTTP downloader instead, which stays
# entirely inside HF_HOME.
export HF_HUB_DISABLE_XET=1

# Checkpoints are large (full model weights + DeepSpeed optimizer state per
# save) — save at these exact, irregularly-spaced steps instead of a
# uniform interval, so disk usage is bounded to exactly
# len(RLDX_EXACT_SAVE_STEPS) checkpoints rather than churning through more
# intermediate ones. See ExactStepsCheckpointCallback in
# rldx/experiment/utils.py. --save-steps below is set far past --max-steps
# so the normal uniform-interval path never fires on its own. Adjust the
# step list (and --max-steps / --save-total-limit below) to taste — the
# last entry should normally equal --max-steps.
export RLDX_EXACT_SAVE_STEPS="10000,20000,30000,35000,40000"

# --- GPU pinning -------------------------------------------------------
# Adjust to whatever GPUs are actually free on your machine (check
# `nvidia-smi` first) and keep NUM_GPUS in sync.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2

BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$BASE_DIR"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"

# Required: path to your LeRobot v2.1 dataset (see README.md "Dataset
# layout" for the expected meta/modality.json schema this embodiment
# config assumes — 41-dim state, 76-dim action, 2 camera keys).
: "${DATASET_PATH:?Set DATASET_PATH to your LeRobot v2.1 dataset root, e.g. DATASET_PATH=/path/to/dataset bash $0}"

EMBODIMENT_TAG="UNITREE_G1_SONIC_BRAINCO6DOF"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py"

CKPT_NAME="g1_sonic_brainco6dof_ft"
RUN_NAME="$CKPT_NAME"
CKPT_DIR="${CKPT_DIR:-$BASE_DIR/outputs/$CKPT_NAME}"

# Keep HF/model downloads self-contained under the repo checkout by default.
export HF_HOME="${HF_HOME:-$BASE_DIR/.hf_cache}"

mkdir -p "$CKPT_DIR"

# Micro-batch 6/GPU x NUM_GPUS GPUs, no grad-accum. Tuned for GPUs with
# ~96GB VRAM and this recipe's backbone-freeze (only top-4 LLM layers +
# projector + full action head trainable, gradient checkpointing on) —
# comfortable headroom there, not a tight fit. Lower --global-batch-size
# (and/or NUM_GPUS above) if your GPUs have less VRAM.
GLOBAL_BATCH_SIZE=$((6 * NUM_GPUS))

ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"

    # RLDX-1-PT is the video-input checkpoint -> video-length 4 (matches
    # its pretraining; see README.md "Choosing the base checkpoint").
    --video-length 4
    --n-cog-tokens 64

    # Action chunk: 40 steps @ 50 Hz = 0.8s lookahead. Must match
    # ACTION_HORIZON in unitree_g1_sonic_brainco6dof_config.py.
    --action-horizon 40

    --global-batch-size $GLOBAL_BATCH_SIZE
    --gradient-accumulation-steps 1
    --num-gpus $NUM_GPUS
    --dataloader-num-workers 16

    # "sharded" (the repo default) is built for web-scale pretraining
    # corpora: it pre-shards episodes and only samples a fraction of them
    # per shard (episode_sampling_rate=0.1 default). For a small
    # (hundreds-of-episodes) dataset, "standard" is the correct choice
    # instead — a plain map-style dataset that includes every valid
    # (episode, step) pair, so every frame is guaranteed to actually be
    # seen, with clean/deterministic epoch semantics. See
    # rldx/data/dataset/factory.py's DatasetFactory docstring for the mode
    # comparison. Switch back to "sharded" for a large/pretraining-scale
    # dataset.
    --dataset-mode standard

    # Optimizer / schedule. RLDX-1's Trainer uses a single global LR (no
    # per-parameter-group / differential-LR mechanism exists in this
    # codebase). The backbone-freeze defaults below
    # (tune-top-llm-layers=4, tune-llm=False, tune-visual=False,
    # tune-projector=True, tune-diffusion-model=True) are RLDX-1's
    # built-in substitute for a low backbone LR: they simply never expose
    # most of the backbone to gradients at all.
    --learning-rate 1e-4
    --lr-scheduler-type cosine
    --weight-decay 0.01
    --warmup-ratio 0.0125   # e.g. ~500 steps at max-steps=40000 — adjust to taste
    --max-grad-norm 1.0

    --max-steps 40000
    --save-steps 1000000   # sentinel, past max-steps: real save timing comes from RLDX_EXACT_SAVE_STEPS above
    --save-total-limit 5   # should match the number of entries in RLDX_EXACT_SAVE_STEPS, so none get rotated out

    --output-dir "$CKPT_DIR"
    --experiment-name "$RUN_NAME"

    # No wandb by default. No validation split either: eval-strategy
    # defaults to "no" in rldx/configs/training/training_config.py, so
    # nothing to pass here. Add --use-wandb --wandb-project <name> if you
    # want logging.

    # Image augmentation (applied to all configured camera keys; there is
    # no horizontal-flip stage in rldx/data/augmentations.py at all, so
    # left/right hand semantics are preserved by construction, not by a
    # flag — relevant for any bimanual embodiment).
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --random-rotation-angle 8
    --random-crop-fraction 0.9
)

export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}"

echo "About to launch (NOT auto-run): "
echo "pixi run --environment rldx torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT rldx/experiment/launch_train.py ${ARGS[*]}"

pixi run --environment rldx torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py "${ARGS[@]}"
