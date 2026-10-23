#!/bin/bash
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# Post-train Being-H05-2B on a Unitree G1 + BrainCo Revo2 (6-DOF hand)
# SONIC motion-token dataset (LeRobot v2.1, "sep-10-2026-cleaned-merged",
# 535 episodes / ~182k frames @ 50Hz, teleop'd through NVlabs
# GR00T-WholeBodyControl's GEAR-SONIC whole-body controller) -- single-GPU
# variant for 1x NVIDIA A100 80GB.
#
# This is a NEW script, following the shape of scripts/train/
# train_cross_emb_example.sh and scripts/train/train_libero_example.sh, but
# adapted to a single dataset/embodiment and a single GPU.
#
# Action target = 76-dim continuous vector:
#   64  action.motion_token   (SONIC latent, live-encoded during teleop)
#  + 6  teleop.left_hand_joints
#  + 6  teleop.right_hand_joints
#  (action.wbc, the 41-dim raw joint-target field, is intentionally NOT
#   used -- it's a debug/reference field only)
# Predicted as a 40-step chunk (0.8s lookahead @ 50Hz), not single-step.
#
# ==========================================================================
# What this depends on (source patches + new config -- see each file's own
# comments for the full "why"):
#   - BeingH.utils.constants.EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF
#     (BeingH/utils/constants.py) -- also flags a pre-existing embodiment_id
#     resolution quirk found during this integration (not fixed here, out
#     of scope for a minimal patch, affects every embodiment equally).
#   - configs.data_config.UnitreeG1SonicBrainco6DofDataConfig
#     (configs/data_config.py) -- the 76-dim action / 41-dim state / 2-camera
#     modality mapping. Read its docstring: several UNIFIED_MAPPING slot
#     choices and the observation.state column ordering are DOCUMENTED
#     GUESSES (meta/info.json for the real dataset was not available to
#     this integration) -- verify against the real dataset before trusting
#     a long run.
#   - configs/dataset_info.py: DATASET_REGISTRY + DATASET_INFO entries for
#     'g1_sonic_brainco6dof_posttrain'. The dataset path is read from the
#     DATASET_PATH env var at import time (not hardcoded, unlike every
#     other entry in that file).
#   - configs/posttrain/unitree_g1_sonic_brainco6dof/unitree_g1_sonic_brainco6dof.yaml
#     -- the single-embodiment dataset_config_file this script points at.
#   - BeingH/dataset/preprocess.py::build_vit_transform_base -- two new,
#     OPT-IN kwargs (crop_scale_min, random_rotation_degrees) so this
#     recipe can get RandomResizedCrop scale (0.85, 1.0) and +/-8deg
#     rotation without changing augmentation for any existing recipe
#     (LIBERO/RoboCasa YAMLs don't set these keys, so their behavior is
#     byte-for-byte unchanged). ColorJitter(0.3, 0.4, 0.5, 0.08) was
#     already hardcoded in that function and already matches this recipe's
#     required values, so it was left alone.
#   - BeingH/train/train.py::apply_model_freezing() -- a new, OPT-IN
#     --vit_trainable_within_mllm flag (default False, byte-for-byte
#     unchanged for every other recipe) that excludes vit_model.*
#     parameter names from freeze_mllm's bundle freeze, so the ViT trains
#     while the base LLM (and, via freeze_vit_mlp, the connector) stay
#     frozen. Added after tracing the freeze path all the way through
#     (train.py -> train_utils.py::loading_ckpt_from_pretrained ->
#     model/llm/qwen3_navit.py) -- see the "Freezing Configuration"
#     section below for the full writeup of why no existing flag
#     combination could do this and why freeze_llm is NOT a safe
#     substitute (it would also freeze the action expert).
#
# Usage:
#   DATASET_PATH=/path/to/sep-10-2026-cleaned-merged \
#     bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING -- run it yourself
# when ready, after checking GPU availability and the env vars below.
# NO training was run to produce this script: no GPU was available while
# writing it, and even a smoke test would try to download an 8B+... err,
# ~2.7B-parameter checkpoint and touch a dataset that no longer exists on
# this machine. See smoketest_g1_sonic_brainco6dof_1xa100.sh for how to do
# a real (cheap) sanity check once you have both.

set -euo pipefail

export PYTHONPATH=.
export NCCL_IB_DISABLE=0
export NO_ALBUMENTATIONS_UPDATE=1

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

# --- GPU pinning ----------------------------------------------------------
# Single GPU by design (this script targets 1x A100 80GB). Override
# CUDA_VISIBLE_DEVICES if that GPU isn't index 0 on your box (check
# `nvidia-smi` first) -- but keep exactly one index here; NUM_GPUS below
# must stay 1 to match. FSDP itself has no problem running with
# world_size=1 (BeingH/train/fsdp_utils.py's SHARD_GRAD_OP/FULL_SHARD paths
# both handle a single rank correctly -- shard_index=0, total_shards=1;
# only HYBRID_SHARD assumes >1 rank, and this script doesn't use it).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS=1

# ============ Model and Data Paths ============
# These three mirror scripts/train/train_cross_emb_example.sh's own
# placeholders -- Being-H05 always needs the base InternVL MLLM checkpoint,
# the base Qwen3 expert checkpoint, AND a --resume_from checkpoint (the
# released Being-H05-2B weights) together: mllm_path/expert_path establish
# the model architecture/config, resume_from (loaded via
# FSDPCheckpoint.try_load_ckpt) supplies the actual post-trained weights.
# None of these three live in this repo or on this machine -- set them to
# wherever you downloaded https://huggingface.co/collections/BeingBeyond/being-h05
# and its InternVL3_5-2B / Qwen3-0.6B dependencies.
PRETRAIN_MODEL="${PRETRAIN_MODEL:-/path/to/InternVL3_5-2B}"
EXPERT_MODEL="${EXPERT_MODEL:-/path/to/Qwen3-0.6B}"
RESUME_PATH="${RESUME_PATH:-/path/to/model_zoo/Being-H05-2B}"

# Required: path to the LeRobot v2.1 "sep-10-2026-cleaned-merged" dataset
# root (containing meta/, data/, videos/). Not hardcoded -- read by
# configs/dataset_info.py at import time via this same env var.
: "${DATASET_PATH:?Set DATASET_PATH to your LeRobot v2.1 dataset root, e.g. DATASET_PATH=/path/to/sep-10-2026-cleaned-merged bash $0}"
export DATASET_PATH

EMBODIMENT="unitree_g1_sonic_brainco6dof"
EMBODIMENT_DATASET="unitree_g1_sonic_brainco6dof"
DATASET_CONFIG_FILE="configs/posttrain/${EMBODIMENT}/${EMBODIMENT_DATASET}.yaml"

# Single embodiment/dataset -- the merged cross-embodiment metadata format
# is documented as needed only for cross-embodiment inference (multiple
# embodiments sharing one checkpoint); this run has exactly one, so the
# standard per-sub-dataset metadata format saved by default is sufficient.
SAVE_MERGED_META=False

# ============ Training Configuration ============
MAX_STEPS=40000
# Checkpoint at 30k/40k, using model-only saves to fit the server disk,
# Being-H05's own native uniform-interval mechanism
# (curr_step > save_steps_start and curr_step % save_steps == 0 --
# BeingH/train/train.py) rather than a custom callback: unlike the sibling
# RLDX-1 adaptation of this same dataset (which needed a bespoke
# ExactStepsCheckpointCallback because RLDX-1's own save mechanism can't
# express an irregular step list), Being-H05's save_steps/save_steps_start
# pair already expresses exactly the "checkpoint every N steps starting
# from step S" shape this recipe needs, so no source patch was required
# here.
SAVE_STEPS="${SAVE_STEPS:-10000}"
SAVE_STEPS_START=20000
SAVE_LAST="${SAVE_LAST:-False}"
SAVE_MODEL_ONLY="${SAVE_MODEL_ONLY:-True}"
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05

# ============ Data Loading ============
NUM_WORKERS=12
PREFETCH_FACTOR=8

# ============ Sequence Configuration (this is what actually controls
# per-step VRAM here, not a --global-batch-size flag -- see "Batch size /
# VRAM reasoning" below) ============
# TRIMMED ~12.5% from the vendor's reference 8704/8192 defaults (used
# unchanged by the vision-frozen version of this recipe) now that the ViT
# is trainable too -- see "Batch size / VRAM reasoning" below for why this
# margin isn't strictly required by the reasoned estimate but is kept
# anyway, erring conservative given nothing here has been measured on a
# real GPU. PREFER_BUFFER_BEFORE kept at exactly half of
# EXPECTED_NUM_TOKENS, same ratio as the original values.
MAX_NUM_TOKENS=7680
EXPECTED_NUM_TOKENS=7168
PREFER_BUFFER_BEFORE=3584
MAX_BUFFER_SIZE=4
ATTN_MODE="causal"
GRADIENT_ACCUMULATION_STEPS=1

# ============ Image Configuration ============
# 224px matches the Being-H05-2B release ("the vision part is 224px by
# default" -- README.md). max_view_num=-1 means "use every camera key
# UnitreeG1SonicBrainco6DofDataConfig.VIDEO_SOURCE_COLUMNS declares" -- i.e.
# exactly depth_camera_rgb + webcam, since external_view_camera was never
# declared there in the first place (see that class's docstring).
FORCE_IMAGE_SIZE=224
MAX_VIEW_NUM=-1
USE_FIXED_VIEW=False
DOWN_SAMPLE_RATIO=0.5

# ============ Action Configuration ============
# 40 steps @ 50Hz = 0.8s lookahead. train.py auto-syncs action_token_num to
# this value for flow matching (see the "CRITICAL FIX" block in
# BeingH/train/train.py's main()), so it doesn't need to be set separately.
ACTION_CHUNK_LENGTH=40

# ============ Freezing Configuration ============
# UPDATED to unfreeze the vision tower (--vit_trainable_within_mllm below),
# per the task owner's call: the newer dataset adds tasks (hug/handshake/
# fist-bump) with close-proximity, human-in-frame visual content that a
# frozen ViT -- pretrained on general VLM data, never adapted to this
# dataset's visual domain -- may not represent well.
#
# This time the freeze path was traced ALL THE WAY THROUGH (not just
# flagged as uncertain like the prior integration's freeze_vit_mlp caveat
# below) -- BeingH/train/train.py::apply_model_freezing(),
# BeingH/train/train_utils.py::loading_ckpt_from_pretrained(), and
# BeingH/model/llm/qwen3_navit.py's Qwen3MoTDecoderLayer. Findings:
#   freeze_mllm=True   -> freezes every param whose name is in
#                         mllm_layer_names, a list built from EVERY key in
#                         the pretrained InternVL3_5-2B checkpoint's
#                         state_dict (ViT + base LLM + connector, via
#                         `k.replace("vision_model", "vit_model")` --
#                         train_utils.py). There is NO per-submodule
#                         opt-out for the ViT in the stock freeze loop (only
#                         "connector" gets one, via freeze_vit_mlp below) --
#                         confirmed by reading the loop itself:
#                         `if name in mllm_layer_names: requires_grad =
#                         False`, unconditional.
#   freeze_llm=True    -> NOT a substitute for "freeze the base LLM but
#                         keep the ViT and the expert trainable": this flag
#                         operates on `model.language_model` directly
#                         (`for param in model.language_model.parameters():
#                         requires_grad = False`), and
#                         model.language_model's decoder layers ARE
#                         Qwen3MoTDecoderLayer instances (--layer_module
#                         below) that hold BOTH the base LLM's weights AND
#                         the action-expert's "_mot_gen"-suffixed weights
#                         (mlp_mot_gen, input_layernorm_mot_gen, etc. --
#                         BeingH/model/llm/qwen3_navit.py) in the SAME
#                         nn.Module, at every layer. freeze_llm=True would
#                         freeze the expert too -- unacceptable, the expert
#                         is this recipe's whole trainable action head.
#   (nothing needed to keep the action expert trainable) -> the Qwen3-0.6B
#                         expert weights are added via init_mot()/
#                         init_expert() AFTER mllm_layer_names is captured
#                         from the base checkpoint, so they are NEVER in
#                         mllm_layer_names and stay trainable automatically
#                         regardless of freeze_mllm -- unaffected by any of
#                         the changes below.
#   freeze_vit_mlp=False -> keeps the connector (vision->LLM projector)
#                         trainable. Same caveat as the prior integration
#                         flagged (still not independently re-verified this
#                         round, out of scope for this pass): whether the
#                         connector's own param names actually appear in
#                         mllm_layer_names in the first place (BeingH's
#                         `connector` submodule vs. the original InternVL
#                         checkpoint's own key name, commonly `mlp1`) was
#                         not traced to a definitive answer. Left at False
#                         since it can only help, never hurt -- CHECK the
#                         "Trainable parameters: X / Total: Y (Z%)" line
#                         BeingH/train/train.py prints at startup before
#                         trusting a long run, same as before.
#   vit_trainable_within_mllm=True (NEW FLAG, added by this integration to
#                         BeingH/train/train.py -- see that file's own
#                         comment on the field and inside
#                         apply_model_freezing() for the source patch) ->
#                         the ONE thing the stock freeze loop couldn't do:
#                         excludes vit_model.*-named params from
#                         freeze_mllm's bundle freeze while leaving every
#                         other mllm_layer_names entry (base LLM, and
#                         connector when freeze_vit_mlp=True) frozen exactly
#                         as before. This is opt-in and additive -- default
#                         False, so every other recipe using freeze_mllm
#                         (LIBERO/RoboCasa/etc.) is byte-for-byte unaffected
#                         by this patch unless it explicitly sets the flag.
#   freeze_vit left at its default (False) -- freeze_vit=True would always
#                         win over vit_trainable_within_mllm (an explicit
#                         request to keep the ViT frozen overrides the
#                         opt-out; see the source patch), so it must stay
#                         False here for the ViT to actually train.
FREEZE_MLLM=True
FREEZE_VIT_MLP=False
FREEZE_VIT=False
VIT_TRAINABLE_WITHIN_MLLM=True

# Gradient checkpointing: OFF in both reference example scripts (they
# target more/bigger GPUs), ON here for safety margin on a single 80GB
# card, per the recipe's explicit ask. Being-H05 actually has TWO
# gradient-checkpointing switches that both matter:
#   grad_checkpoint (ModelArguments)      -> calls
#       model.language_model._set_gradient_checkpointing() (checkpoints
#       inside the LLM/expert decoder stack itself)
#   use_gradient_checkpoint (TrainingArguments) -> wraps FSDP submodules
#       matching grad_checkpoint_check_fn() (InternVisionEncoderLayer,
#       InternVLConnector, ProprioProjector, CategorySpecificLinear/MLP,
#       MultiEmbodimentActionEncoder -- BeingH/train/fsdp_utils.py) with
#       torch.distributed's activation-checkpoint wrapper.
# Both are enabled here; leaving either off would only checkpoint half the
# model.
GRAD_CHECKPOINT=True
USE_GRADIENT_CHECKPOINT=True

# ============ MPG Configuration ============
# Kept identical to both reference example scripts (train_cross_emb_example.sh,
# train_libero_example.sh) -- this is Being-H05's own tuned recipe for
# post-training a NEW embodiment, the closest existing analog to what this
# script does, and nothing about our 76-dim/2-camera/chunk-40 setup gives a
# reason to deviate from it.
USE_MPG=True
MPG_LAMBDA=0.1
MPG_NUM_PROJECTIONS=32
MPG_REFINEMENT_ITERS=1
MPG_GATE_TEMPERATURE=1.0
MPG_USE_STOP_GRADIENT=True

# ============ Training-Time RTC Configuration ============
# Off in both reference scripts; left off here too -- not part of this
# recipe's requirements.
USE_TRAINING_TIME_RTC=False
SIMULATED_DELAY=0
RTC_DELAY_EXP_WEIGHT=True
USE_INFERENCE_PREFIX_OVERWRITE=True

# ============ Output Configuration ============
MODEL_NAME="post-${EMBODIMENT_DATASET}_1xa100_chunk-${ACTION_CHUNK_LENGTH}_mpg-${USE_MPG}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/outputs/${MODEL_NAME}}"
LOG_DIR="results/tensorboard/post"
LOG_FILE="${OUTPUT_DIR}/training.log"

# No wandb: BeingH/train/train.py has no wandb/report_to integration at all
# (grepped the whole repo -- the only wandb references are in the LIBERO
# eval scripts, unrelated to training), and it isn't in requirements.txt
# either. TensorBoard (--logging_dir above) is the only training-time
# monitoring this codebase natively supports, so that's what this recipe
# uses -- `tail -f "$LOG_FILE"` or `tensorboard --logdir "$LOG_DIR"`.
# No validation split either: BeingH/train/train.py has no
# --eval_strategy/held-out-split mechanism to disable, so "train-loss only"
# falls out automatically, nothing to turn off.

mkdir -p "$OUTPUT_DIR"
cp "$0" "${OUTPUT_DIR}/"
mkdir -p "${OUTPUT_DIR}/code"
cp -r BeingH "${OUTPUT_DIR}/code/"

# --- Batch size / VRAM reasoning ------------------------------------------
# Unlike the sibling RLDX-1 adaptation (and unlike most VLA training
# scripts), Being-H05's DataLoader batch_size is HARDCODED to 1
# (BeingH/train/train.py) -- BeingH/dataset/base_dataset.py's PackedDataset
# is a TOKEN-BUDGET SEQUENCE PACKER (the same idea as LLM pretraining
# packing), not a fixed-sample-count batcher: each training step packs as
# many (vision + state + action + text) samples as fit under
# EXPECTED_NUM_TOKENS/MAX_NUM_TOKENS tokens into ONE forward/backward pass.
# There is no --global-batch-size flag to reason about; the token budget
# above IS the memory-controlling knob, and it is a PER-RANK setting, not a
# global/aggregate one.
#
# UPDATED for the vision tower now being trainable (vit_trainable_within_
# mllm=True above) -- per the task owner's call, motivated by the same
# hug/handshake/fist-bump visual-domain-shift concern noted in "Freezing
# Configuration" above. MAX_NUM_TOKENS/EXPECTED_NUM_TOKENS were trimmed
# ~12.5% from the vendor's reference 8704/8192 (used unchanged by the
# vision-frozen version of this recipe, which independently reproduced the
# vendor's own validated per-GPU footprint on this hardware class) as an
# extra safety margin on top of the reasoning below -- see that section's
# own note on why this isn't strictly required by the math but is kept
# anyway.
#
# Per-sample token cost estimate (for context, not a config knob): with
# force_image_size=224, down_sample_ratio=0.5, img_patch_size=14 (from the
# base InternVL ViT config): num_image_tokens = (224/14 * 0.5)^2 = 64
# tokens/view. 2 declared camera views -> ~128 vision tokens. Plus ~1-3
# state tokens (history_num=1), ACTION_CHUNK_LENGTH=40 action tokens,
# ~20-40 instruction/system/boilerplate text tokens -> roughly ~210-230
# tokens/sample, so EXPECTED_NUM_TOKENS=7168 packs on the order of
# ~30-33 samples into each optimizer step -- still a reasonably large
# effective batch (down from ~35-40 at the vendor default, a minor,
# deliberate trade-off for VRAM margin, not expected to meaningfully affect
# convergence for a hundreds-of-episodes dataset).
#
# ViT parameter count: NOT a guess this round -- fetched InternVL3_5-2B's
# actual vision_config (hidden_size=1024, num_hidden_layers=24,
# num_attention_heads=16, intermediate_size=4096, patch_size=14), which
# matches the well-known "InternViT-300M-448px" vision encoder used across
# the InternVL 2B tier. Per-layer dense-weight count (ignoring biases/norms,
# which are negligible here): attention (q/k/v/out, each hidden*hidden)
# = 4 * 1024^2 ~= 4.19M, MLP (fc1+fc2, each hidden*intermediate)
# = 2 * 1024*4096 ~= 8.39M -> ~12.58M/layer * 24 layers ~= 302M, plus patch
# embed (1024*3*14*14 ~= 0.6M) and position embeddings (~1025*1024 ~= 1.05M)
# -> **~304M total**, consistent with "InternViT-300M" naming. This
# replaces the prior integration's "~300-400M ViT" placeholder range with a
# real, computed number.
#
# Rough VRAM budget at these settings (Being-H05-2B = InternVL3_5-2B
# backbone [~304M ViT (see above) + ~1.7-1.8B Qwen3-1.7B-class LLM + a
# small connector] + Qwen3-0.6B action-expert weights threaded through
# every MoT decoder layer + small robot-specific heads [proprio encoder,
# action encoder/decoder, flow-matching head] -- roughly 2.7-2.8B
# parameters total; with the ViT now trainable, FREEZE_MLLM=True still
# freezes the ~1.7-1.8B base LLM, leaving ~0.95-1.0B trainable [expert
# (~0.6B) + ViT (~0.304B) + robot heads + possibly the connector, see the
# freeze_vit_mlp caveat above], per BeingH/train/fsdp_utils.py's
# MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)):
#   ~5.5GB   all params (frozen + trainable), bf16 -- unchanged by the
#            freeze/unfreeze split, this is the TOTAL backbone regardless
#   ~4.0GB   gradients for the ~1.0B trainable params (up from ~2.8GB for
#            ~0.65-0.7B before adding the ViT), budgeted fp32 (conservative
#            -- no explicit bf16-grad-only path was confirmed in this
#            codebase, so budgeting the worse case, same as before)
#   ~8.0GB   AdamW (m, v) for those trainable params, fp32 (up from ~5.6GB)
#   ~4.0GB   extra conservative buffer, in case an fp32 "master weight"
#            copy of the trainable params also exists somewhere in this
#            training loop that wasn't spotted while reading it (up from
#            ~2.8GB, same logic scaled to the new ~1.0B trainable count)
#   -----------------------------------------------------------------
#   ~21.5GB  weights + optimizer + gradients (conservative sum, up from
#            ~16.7GB pre-unfreeze -- a ~4.8GB addition from the ViT's own
#            optimizer state, not from its weights, which were already
#            counted in the ~5.5GB total-params line above whether frozen
#            or not)
#   + activations for an up-to-7680-token packed sequence through the
#     ~2.7B-parameter model, WITH gradient checkpointing on both the LLM/
#     expert stack and the FSDP-wrapped submodules (see GRAD_CHECKPOINT/
#     USE_GRADIENT_CHECKPOINT above -- grad_checkpoint_check_fn in
#     BeingH/train/fsdp_utils.py already wraps InternVisionEncoderLayer
#     UNCONDITIONALLY, i.e. structurally, regardless of its requires_grad
#     state, so the ViT's forward/backward was ALREADY being
#     activation-checkpointed even before this change -- the only NEW
#     activation cost from unfreezing it is that autograd now has to
#     actually retain/recompute those checkpointed boundaries for backward
#     at all, since a fully-frozen leaf subgraph needs no backward pass
#     through it. Ballparked at +~2-4GB for this (per-layer checkpoint
#     boundary activations at ~128 vision tokens/sample * ~30-33
#     samples/step * hidden_size=1024 * bf16, plus recompute transients,
#     freed layer-by-layer) -- not independently measured, deliberately on
#     the generous side. This is a materially smaller model than the
#     sibling RLDX-1 recipe's Qwen3-VL-8B backbone (which itself fit an
#     equivalent token-budget-class workload in ~61-68GB out of 80GB WITH
#     its own, much larger, vision tower unfrozen), so ballparking
#     ~22-39GB for this ~2.7B model's activations (the prior ~20-35GB
#     estimate, +~2-4GB for the ViT addition above) at the trimmed token
#     budget is a plausible, deliberately generous estimate, not a tight
#     one.
#   -----------------------------------------------------------------
#   ~44-61GB estimated total, leaving ~19-36GB headroom under 80GB -- LESS
#   tight than the sibling RLDX-1 recipe's unfreeze (~12-19GB headroom
#   there), because Being-H05's whole backbone (2.7B) and ViT (~0.3B) are
#   both much smaller than RLDX-1's (8B+ backbone, ~0.4-0.7B ViT estimate).
#   Given this real headroom, the MAX_NUM_TOKENS/EXPECTED_NUM_TOKENS trim
#   above (~12.5%) is a precautionary margin on top of an already-comfortable
#   estimate, not a change forced by the arithmetic -- kept anyway per
#   "erring conservative" and because nothing here has been measured on a
#   real GPU.
#
# This is a reasoned estimate, not a measured one -- no GPU was available
# while writing this script, and the real dataset isn't on this machine
# either. Run smoketest_g1_sonic_brainco6dof_1xa100.sh first and watch
# `nvidia-smi` during it before trusting this for a multi-hour/multi-day
# run. If it OOMs anyway, the first things to try, in order: (1) lower
# MAX_NUM_TOKENS/EXPECTED_NUM_TOKENS further (this is the actual per-step
# memory knob in this framework, unlike a batch-size flag), (2) confirm
# GRAD_CHECKPOINT and USE_GRADIENT_CHECKPOINT are both still True, (3)
# consider CPU_OFFLOAD=True (training_args.cpu_offload) as a next-to-last
# resort (slows training down noticeably), (4) only as a final fallback,
# revert VIT_TRAINABLE_WITHIN_MLLM to False (back to the original
# vision-frozen recipe) rather than fighting for headroom on this backbone
# size.
#
# FSDP-specific interaction check (task explicitly asked this be checked,
# not silently skipped): NO SCRIPT CHANGE NEEDED. Read
# BeingH/train/fsdp_utils.py's fsdp_wrapper() -- (1) FSDP is constructed
# with use_orig_params=True, which is exactly the setting that lets a
# single FSDP-wrapped unit hold a MIX of trainable and frozen parameters
# correctly (this codebase already relies on this for the base-LLM-frozen/
# expert-trainable split inside each Qwen3MoTDecoderLayer, so mixing
# frozen/trainable params within one wrapped unit is already an exercised
# code path, not a new risk). (2) FSDP_WRAPPER_CLASSES already includes
# InternVisionEncoderLayer and InternVisionModel as their OWN auto-wrap
# units, separate from the LLM's decoder layers -- so the ViT was already
# wrapped as its own (relatively small) FSDP unit(s) regardless of freeze
# status; unfreezing it doesn't merge it into a bigger indivisible unit or
# change the wrapping policy at all. (3) MixedPrecision(param_dtype=bf16,
# reduce_dtype=bf16, buffer_dtype=bf16) applies uniformly regardless of
# requires_grad, so no bucket/precision change either. Nothing here
# required a source or script change beyond the freeze-flag patch itself.

MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

echo "About to launch (NOT auto-run):"
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
echo "  --logging_steps 10 \\"
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
  --logging_steps 10 \
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
echo "Training Complete!"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "=========================================="
