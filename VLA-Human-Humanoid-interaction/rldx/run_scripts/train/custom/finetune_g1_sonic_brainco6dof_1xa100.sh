#!/bin/bash
# Fine-tune RLDX-1 on a Unitree G1 + BrainCo Revo2 (6-DOF hand) SONIC
# motion-token dataset (LeRobot v2.1 format) — single-GPU variant for
# 1x NVIDIA A100 80GB.
#
# Same recipe as finetune_g1_sonic_brainco6dof.sh (that script's header
# comment lists every source patch this depends on — still required here,
# nothing GPU-count-specific was added). The only differences from that
# script are GPU pinning and batch/accumulation, explained below.
#
# Action target = 76-dim continuous vector:
#   64  action.motion_token   (SONIC latent, live-encoded during teleop)
#  + 6  teleop.left_hand_joints
#  + 6  teleop.right_hand_joints
#  (action.wbc is intentionally NOT used — it is a debug/reference field)
#
# Usage:
#   DATASET_PATH=/path/to/your/lerobot_dataset \
#     bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING — run it yourself
# when ready, after checking GPU availability and the env vars below.

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1

# See finetune_g1_sonic_brainco6dof.sh for why this is forced (hf_xet can
# hang silently writing to an unwritable ~/.cache/huggingface/xet/).
export HF_HUB_DISABLE_XET=1

export RLDX_EXACT_SAVE_STEPS="10000,20000,30000,35000,40000"

# --- GPU pinning ---------------------------------------------------------
# Single GPU by design (this script targets 1x A100 80GB). Override
# CUDA_VISIBLE_DEVICES if that GPU isn't index 0 on your box (check
# `nvidia-smi` first) — but keep exactly one index here; NUM_GPUS below
# must stay 1 to match.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=1

BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$BASE_DIR"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"

# Required: path to your LeRobot v2.1 dataset (see README.md "Dataset
# layout" for the expected meta/modality.json schema this embodiment
# config assumes — 41-dim state, 76-dim action, 2 camera keys).
: "${DATASET_PATH:?Set DATASET_PATH to your LeRobot v2.1 dataset root, e.g. DATASET_PATH=/path/to/dataset bash $0}"

EMBODIMENT_TAG="UNITREE_G1_SONIC_BRAINCO6DOF"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py"

CKPT_NAME="g1_sonic_brainco6dof_ft_1xa100"
RUN_NAME="$CKPT_NAME"
CKPT_DIR="${CKPT_DIR:-$BASE_DIR/outputs/$CKPT_NAME}"

# Keep HF/model downloads self-contained under the repo checkout by default.
export HF_HOME="${HF_HOME:-$BASE_DIR/.hf_cache}"

mkdir -p "$CKPT_DIR"

# --- Batch size / VRAM reasoning -----------------------------------------
# UPDATED to also unfreeze the vision tower (--tune-visual below), per the
# task owner's call: this dataset now includes tasks (hug/handshake/
# fist-bump) whose close-proximity, human-in-frame visual domain is well
# outside what any of this backbone's pretraining (general VLM data, plus
# whatever SONIC/G1 teleop data RLDX-1-PT itself saw) is likely to cover
# well from a frozen vision tower alone.
#
# `tune_visual` is all-or-nothing in this codebase (see
# rldx/model/modules/backbone/adapter.py::set_trainable_parameters — the
# whole `qwen_model.model.visual` submodule is unfrozen together, there is
# no top-N-layers partial option the way `tune_top_llm_layers` gives the
# LLM). Qwen3-VL-8B's vision tower size could not be measured directly (no
# GPU/model access while writing this), but Qwen-VL-family ViTs in this
# parameter range are typically ~0.4-0.7B params — budgeting for the
# heavier end (~0.7B) below to stay conservative:
#   +~2.8GB  fp32 master copy of the vision tower (0.7B x 4 bytes)
#   +~5.6GB  AdamW (m, v) for it (0.7B x 8 bytes)
#   +~2.8GB  gradients for it (0.7B x 4 bytes, fp32 to match
#            backbone_trainable_params_fp32=True)
#   +~3-6GB  extra activations (vision-token count stays small even
#            trainable — image_max_area=65536 means each frame is only
#            ~80 tokens, x2 cameras x video_length=4 = ~640 tokens through
#            the tower — assuming gradient checkpointing extends to the
#            visual submodule, which was not independently verified here)
#   -----------------------------------------------------------------
#   +~14-17GB on top of the vision-frozen estimate below.
#
# To keep real headroom under that addition, this script drops the
# per-device micro-batch from the vision-frozen recipe's 6 to 4, and
# raises --gradient-accumulation-steps from 2 to 3 so the effective batch
# stays 12 (same as before — keeps the LR/warmup schedule meaningful
# without re-deriving it):
#
#   global-batch-size 4 (= per_device_train_batch_size, since num-gpus=1)
#   x gradient-accumulation-steps 3 = effective batch 12
#
# Rough VRAM budget with the vision tower now trainable too (top-4 LLM
# layers + projector + vlln + diffusion action head + full vision tower;
# rest of the LLM frozen; bf16 weights, fp32 trainable master + AdamW
# state, gradient_checkpointing=True):
#   ~18GB    frozen + trainable backbone/action-model weights (bf16)
#   ~9GB     fp32 master copy of the ~2.0-2.2B trainable params (was
#            ~1.3-1.5B before adding the vision tower)
#   ~18GB    AdamW (m, v) for those trainable params
#   ~9GB     gradients
#   ~7-14GB  activations at micro-batch 4 (scaled down from the
#            vision-frozen estimate's 10-20GB at batch 6, roughly by the
#            batch ratio, plus the vision-tower activation addition above)
#   -----------------------------------------------------------------
#   ~61-68GB estimated, leaving ~12-19GB headroom under 80GB — tighter
#   than the vision-frozen recipe's ~55-65GB, by design (this is a
#   reasoned estimate, not measured; there was no GPU access when this
#   was written).
#
# Run the matching smoketest script first and watch `nvidia-smi` during it
# before trusting this for a multi-hour run. If it OOMs anyway, drop
# --global-batch-size to 2 and raise --gradient-accumulation-steps to 6
# (keeps effective batch 12); if that's still not enough, the next lever
# is reverting --tune-visual to false (back to the original vision-frozen
# recipe) rather than fighting for headroom on this backbone size.
#
# NOTE: with num-gpus=1, rldx bypasses DeepSpeed entirely (see
# `if config.training.num_gpus > 1` in rldx/experiment/experiment.py) and
# runs as a plain HF Trainer with bf16 + gradient checkpointing. This is
# expected and not a problem: ZeRO's stage-2 partitioning has nothing to
# shard across on a single GPU anyway, so there is no capability lost.
GLOBAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=3

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
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS
    --num-gpus $NUM_GPUS
    --dataloader-num-workers 8

    # "standard" (not the repo-default "sharded") — see
    # finetune_g1_sonic_brainco6dof.sh for why: this is a
    # hundreds-of-episodes dataset, not a web-scale pretraining corpus.
    --dataset-mode standard

    # Vision tower unfrozen (overrides the tune-visual=False default) —
    # see "Batch size / VRAM reasoning" above for why and the batch-size
    # tradeoff this required.
    --tune-visual

    # Optimizer / schedule. RLDX-1's Trainer uses a single global LR (no
    # per-parameter-group / differential-LR mechanism exists in this
    # codebase). The backbone-freeze defaults (tune-top-llm-layers=4,
    # tune-llm=False, tune-projector=True, tune-diffusion-model=True),
    # plus --tune-visual above, are RLDX-1's built-in substitute for a low
    # backbone LR: everything not explicitly unfrozen never sees gradients
    # at all.
    --learning-rate 1e-4
    --lr-scheduler-type cosine
    --weight-decay 0.01
    --warmup-ratio 0.0125   # ~500 steps at max-steps=40000
    --max-grad-norm 1.0

    --max-steps 40000
    --save-steps 1000000   # sentinel, past max-steps: real save timing comes from RLDX_EXACT_SAVE_STEPS above
    --save-total-limit 5   # matches the number of entries in RLDX_EXACT_SAVE_STEPS, so none get rotated out

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
