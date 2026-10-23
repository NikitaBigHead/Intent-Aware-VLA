#!/bin/bash
# Fine-tune UnifoLM-VLA-0 on the Unitree G1 + BrainCo Revo2 (6-DOF hand)
# SONIC motion-token dataset "sep-10-2026-cleaned-merged" -- single-GPU
# variant for 1x NVIDIA A100 80GB, bf16.
#
# Action target = 76-dim continuous vector, chunked over 40 steps (0.8s @ 50Hz):
#   64  action.motion_token       (SONIC latent, live-encoded during teleop)
#  + 6  teleop.left_hand_joints
#  + 6  teleop.right_hand_joints
#  (action.wbc is intentionally NOT used -- debug/reference field only)
# See src/unifolm_vla/rlds_dataloader/constants.py's G1_SONIC_BRAINCO6DOF_CONSTANTS
# for the full reasoning (including the BOUNDS_Q99-vs-BOUNDS normalization-type
# risk flagged there) and oxe/configs.py's g1_sonic_brainco6dof entry for the
# camera-key naming gotcha (`depth_camera_rgb` is RGB, not a depth map).
#
# BEFORE running this: the 3-stage data conversion (LeRobot -> HDF5 -> RLDS/TFDS)
# must already be done. See convert_g1_sonic_brainco6dof_dataset.sh (companion
# script, also not auto-executed) for those exact commands.
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING -- run it yourself when
# ready, after checking GPU availability and the required env vars below.
#
# None of this has been run -- there is no GPU and no copy of this dataset on
# the machine this was written on. Every number below is a reasoned estimate
# (see "VRAM budget reasoning" further down), not a measured one. Run the
# smoke-test invocation printed at the bottom of this file's companion notes
# first, with `nvidia-smi -l 1` open in another terminal, before trusting this
# for a multi-hour unattended run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Required inputs (no hardcoded paths -- must be set by the caller)
# ---------------------------------------------------------------------------
# RLDS dataset root: the SAME --data_dir you passed to `tfds build` in stage 3
# of the conversion (NOT the dataset-specific subfolder). After a successful
# build you should be able to `ls $OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0/`
# and see .tfrecord shard files there -- verify that by hand before launching;
# this script does a best-effort check below but cannot fully validate a TFDS
# dataset without importing tensorflow_datasets.
: "${OXE_DATA_ROOT:?Set OXE_DATA_ROOT to the RLDS data_dir root, e.g. OXE_DATA_ROOT=/data/rlds bash $0}"

# UnifoLM-VLM-Base: local path (recommended, avoids a cold HF download before
# every run) or the HF hub id. Confirmed via the model's hosted config.json
# (hidden_size=3584, 28 layers -> Qwen2.5-VL-7B-Instruct architecture, ~8.3B
# params total incl. vision encoder, tagged "8B params" on its HF card) --
# NOT the 3B variant the docstring example in
# src/unifolm_vla/model/modules/vlm/QWen2_5.py might suggest. This size drives
# the VRAM reasoning below.
BASE_VLM="${BASE_VLM:-unitreerobotics/Unifolm-VLM-Base}"

# ---------------------------------------------------------------------------
# GPU pinning -- single GPU by design (this script targets 1x A100 80GB)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES=1

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$BASE_DIR"

DATA_MIX="g1_sonic_brainco6dof"

RUN_ID="${RUN_ID:-g1_sonic_brainco6dof_ft_1xa100}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$BASE_DIR/results/Checkpoints}"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/"

if [ ! -d "$OXE_DATA_ROOT/$DATA_MIX/1.0.0" ]; then
    echo "Missing TFDS dataset: $OXE_DATA_ROOT/$DATA_MIX/1.0.0" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Single-GPU bf16 Accelerate config. The server has no nvcc/CUDA toolkit,
# which DeepSpeed requires even to import; plain Accelerate uses torch AdamW.
ACCEL_CONFIG_1GPU="$BASE_DIR/src/unifolm_vla/config/single_gpu_bf16.yaml"

# ---------------------------------------------------------------------------
# VRAM budget reasoning (A100 80GB, bf16) -- read before changing batch size
# ---------------------------------------------------------------------------
# The single biggest lever here is FREEZE_MODULES below, which is NOT how the
# checked-in multi-GPU example script (run_unifolm_vla_train.sh) runs -- that
# script leaves freeze_module_list empty, i.e. full-parameter fine-tuning of
# the ~8.3B-param backbone, spread with DeepSpeed ZeRO2 across 8 GPUs. On a
# SINGLE 80GB GPU there is no rank to shard optimizer state across, so full
# fine-tuning of the whole backbone is not an option:
#   bf16 weights                    8.3e9 * 2B                     ~16.6 GB
#   fp32 AdamW state (m + v)        8.3e9 * 4B * 2                 ~66.4 GB
#   fp32 gradients                  8.3e9 * 4B                     ~33.2 GB
#   ------------------------------------------------------------------------
#   >116 GB before a single activation -- does not fit in 80GB.
#
# UPDATED (task owner request: unfreeze the vision encoder "everywhere
# feasible" to address a distribution-shift concern -- the newer ~1000-episode
# dataset adds hug/handshake/fist-bump tasks with close-proximity,
# human-in-frame visual content that a vision tower never trained on this data
# may not represent well). This required actually tracing freeze granularity
# in this codebase rather than assuming the prior integration's coarse
# qwen_vl_interface=all-or-nothing cut was the only option:
#
# freeze_backbones (trainer_utils/trainer_tools.py) freezes whatever
# comma-separated DOTTED SUBMODULE PATHS are passed in `freeze_modules`
# (getattr-chained off the top-level model, e.g. "qwen_vl_interface" or
# "action_model.net") -- it has no "top K layers" granularity, but nothing
# stops passing a MORE SPECIFIC path than "qwen_vl_interface" as a whole. The
# open question was purely whether Qwen2.5-VL's vision tower is a separately
# addressable submodule under that path or fused into one opaque block. It is
# separable: verified against the actual transformers==4.52.3 source pinned
# in pyproject.toml (github.com/huggingface/transformers, tag v4.52.3,
# modeling_qwen2_5_vl.py and configuration_qwen2_5_vl.py -- not assumed from
# memory or from an unrelated version), the module tree under this repo's
# `qwen_vl_interface` (_QWen_VL_Interface in model/modules/vlm/QWen2_5.py) is:
#   qwen_vl_interface                        (_QWen_VL_Interface wrapper)
#     .model                                 (Qwen2_5_VLForConditionalGeneration)
#       .model                               (Qwen2_5_VLModel)
#         .visual                            (Qwen2_5_VisionTransformerPretrainedModel -- the ViT)
#         .language_model                    (Qwen2_5_VLTextModel -- the 28-layer text decoder)
#       .lm_head                             (nn.Linear(3584, 152064), untied -- ~545M params on its own)
# So `--trainer.freeze_modules qwen_vl_interface` (the prior integration's
# choice) freezes all three as one blob. Passing
# "qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head"
# instead freezes exactly the text decoder + output head and leaves
# `qwen_vl_interface.model.model.visual` untouched -- untouched means
# requires_grad stays at its post-`from_pretrained` default (True), so it
# trains. This is real vision-only unfreezing, not a coarser stand-in for it.
#
# Qwen2.5-VL-7B's ViT parameter count: computed (not guessed) from the same
# verified transformers==4.52.3 source plus the real
# Qwen/Qwen2.5-VL-7B-Instruct config.json (huggingface.co, vision_config:
# depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
# out_hidden_size=3584, patch_size=14) -- summing PatchEmbed (Conv3d, no
# bias) + 32 Qwen2_5_VLVisionBlocks (qkv+proj attention at dim 1280 + a
# SwiGLU-style gate/up/down MLP at intermediate 3420) + the PatchMerger
# (Linear(5120,5120)+Linear(5120,3584), context_dim*spatial_merge_size^2 =
# 1280*4 = 5120) gives ~676.3M params (~0.68B) -- NOT a range guess, an actual
# parameter sum from the real config, cross-checked against the ~675M figure
# commonly cited for this ViT elsewhere. This leaves ~7.62B for the frozen
# text decoder + lm_head (8.3B total - 0.68B ViT), consistent with the
# ~7.6B commonly cited for Qwen2.5-7B's text-only parameters.
#
# So this script now freezes only the text decoder + lm_head
# (`qwen_vl_interface.model.model.language_model` +
# `qwen_vl_interface.model.lm_head`, ~7.62B params) and trains both the ViT
# (~0.68B) and the flow-matching DiT action head (`action_model`). Freezing
# the WHOLE backbone (the prior integration's choice) is no longer the
# constraint here -- it was a "coarser-but-verifiable cut chosen over a
# finer-but-unverified one" per that integration's own comment; the
# finer cut is now verified, so there is no remaining reason to leave the ViT
# frozen. The text decoder + lm_head stay frozen because, per the math above,
# full-backbone fine-tuning (all 8.3B) still does not fit in 80GB -- unfreezing
# just the ~0.68B ViT is the largest slice that plausibly does (worked out
# below), and this codebase's Trainer has no 8-bit/paged-AdamW option to
# revisit that tradeoff with (checked cfg.trainer.optimizer in
# unifolm_vla_train.yaml and setup_optimizer_and_scheduler in
# train_unifolm_vla.py -- it hardcodes `torch.optim.AdamW`, no bitsandbytes
# or paged-optimizer path exists to make full-backbone fine-tuning cheaper).
#
# With the text decoder + lm_head frozen (requires_grad=False on every one of
# their params, and none of their inputs require grad either), PyTorch's
# autograd builds no backward graph for them at all -- their forward
# activations are freed as soon as each layer finishes, so their memory
# footprint is just their bf16 weights (this part of the reasoning is
# unchanged from the prior full-backbone-frozen version):
#   backbone weights (bf16, ALL 8.3B incl. the now-trainable ViT -- the ViT's
#     own weight tensor is still bf16 dtype regardless of trainability; only
#     its optimizer/gradient buffers below are new)                ~16.6 GB
#   DiT action head (`action_model`) -- see DiT_ActionHeader.py /
#     cross_attention_dit.py: inner_dim = num_attention_heads(32) *
#     attention_head_dim(48) = 1536, 16 layers, alternating self-attn-only
#     and cross-attn-to-backbone(3584-dim) blocks. This module's __init__
#     PRINTS its own exact trainable parameter count at startup ("Total
#     number of DiT parameters: ...") -- watch the log for that line and
#     compare against the estimate below rather than trusting the estimate
#     alone. Rough order-of-magnitude: several hundred million params. The
#     action branch of Unifolm_VLA.forward() runs under
#     `torch.autocast("cuda", dtype=torch.float32)`, i.e. in fp32, not bf16 --
#     conservatively budgeting for a ~1B-param upper bound in fp32:
#       weights                     1.0e9 * 4B                      ~4.0 GB
#       fp32 AdamW state (m + v)    1.0e9 * 4B * 2                  ~8.0 GB
#       fp32 gradients              1.0e9 * 4B                      ~4.0 GB
#   ------------------------------------------------------------------------
#   ~32.6 GB fixed so far (backbone bf16 weights + DiT head's own fixed cost)
#   -- this is the same ~32.6 GB the prior, fully-frozen-backbone version of
#   this script computed, unaffected by which part of the backbone trains.
#
# NEW fixed cost from unfreezing the ~676M-param ViT (`qwen_vl_interface.
# model.model.visual`, see above): its forward runs under this codebase's
# bf16 autocast (unifolm_vla.py wraps the whole `qwen_vl_interface(...)` call
# in `torch.autocast("cuda", dtype=torch.bfloat16)`), and its weight tensors
# are already bf16-dtype from `from_pretrained(torch_dtype=torch.bfloat16)`
# -- so its bf16 weights are already inside the ~16.6 GB backbone line above,
# nothing new there. What's new is its optimizer/gradient footprint, since it
# now has requires_grad=True:
#   fp32 master weight copy      676e6 * 4B                       ~2.7 GB
#   fp32 AdamW state (m + v)     676e6 * 4B * 2                    ~5.4 GB
#   fp32 gradients                676e6 * 4B                       ~2.7 GB
#   ------------------------------------------------------------------------
#   ~10.8 GB additional fixed cost, ~43.4 GB fixed total (batch-independent).
# CAVEAT on the fp32-master-copy line: this assumes DeepSpeed's ZeRO 1/2 bf16
# path (deepspeed.runtime.bf16_optimizer.BF16_Optimizer) is actually what
# engages here, which is documented DeepSpeed behavior for
# `ds_config.yaml`'s `"bf16": {"enabled": "auto"}` -- but "auto" resolves off
# of accelerate's own mixed_precision setting, and NEITHER
# deepspeed_zero2.yaml nor this script's generated 1-GPU copy of it sets
# `mixed_precision: bf16` explicitly anywhere. This was NOT independently
# confirmed end-to-end (no GPU available to actually launch DeepSpeed and
# inspect the resulting optimizer's state dtypes). If bf16 mode does NOT
# engage and DeepSpeed instead runs the ViT's already-bf16 weights straight
# through plain AdamW with no fp32 master copy, this ~10.8 GB shrinks to
# ~8.1 GB (drop the master-copy line, keep AdamW state + gradients, which
# still need to be fp32-precision buffers separate from the bf16 params
# either way for numerically sane accumulation) -- budgeting for the larger
# (~10.8 GB) figure here to stay conservative, flagging the assumption rather
# than silently picking whichever number is smaller.
#
# Activations scale with PER_DEVICE_BATCH_SIZE below, and now have TWO
# sources instead of one:
# (1) DiT head activations -- same as the fully-frozen-backbone version: the
#     action chunk is only 40 steps (NUM_ACTIONS_CHUNK) at inner_dim 1536,
#     16 layers, and `Unifolm_VLA.forward()` repeats each sample
#     `repeated_diffusion_steps` times (config: trainer.repeated_diffusion_steps,
#     default 4 -- note this is a SEPARATE field from
#     framework.action_model.repeated_diffusion_steps=8 in the yaml, which
#     forward() does not read; left both as their yaml defaults here,
#     flagging the duplicate/dead field rather than silently picking one) --
#     on the order of a few GB even at batch 8, per the prior integration's
#     estimate.
# (2) ViT activations -- NEW, was exactly zero when the whole backbone was
#     frozen (no backward graph at all through it). Unlike the DiT head, the
#     ViT's sequence length is driven by image resolution, not action-chunk
#     length: unifolm_vla_train.yaml sets `image_size: [224,224]`, and at
#     patch_size=14 that's 16x16=256 patches/image *before* the 2x2
#     PatchMerger reduction (the ViT's own 32 transformer blocks operate on
#     the pre-merge 256-patch sequence). This dataset uses 2 camera keys
#     (depth_camera_rgb, webcam) per sample, window_size=1 (single timestep,
#     not video) -- so 2 images x 256 patches = 512 ViT tokens/sample.
#     Using the standard dense-transformer activation-memory approximation
#     (Korthikanti et al., "Reducing Activation Recomputation in Large
#     Transformer Models": ~34 x hidden x tokens bytes/layer in bf16, plus a
#     smaller attention-matrix term) as an ORDER-OF-MAGNITUDE estimate -- NOT
#     a precise measurement, and flagged as such: it doesn't specifically
#     model this ViT's windowed local attention (only 4 of 32 layers,
#     `fullatt_block_indexes=[7,15,23,31]` per the real vision config, attend
#     across a full 256-patch image; the other 28 use 112px-window-restricted
#     local attention over much shorter sequences) or FlashAttention-2's
#     reduced memory footprint (this codebase does request
#     `attn_implementation="flash_attention_2"` for the whole backbone
#     including the ViT, per QWen2_5.py) -- both effects mean the real number
#     is likely lower than this estimate, so treat it as a conservative-ish
#     ceiling:
#       ~34 * 1280(hidden) * (tokens/layer) * 2B(bf16), summed over 32 layers
#     At PER_DEVICE_BATCH_SIZE=6 (chosen below): 6 samples * 512 tokens/sample
#     = 3072 tokens/step -> roughly ~9 GB of ViT activations. (At batch 8 this
#     scales to roughly ~12 GB; scaling is ~linear in batch since the
#     dominant term is per-token, not per-attention-pair.)
#
# IMPORTANT CAVEAT: there is no gradient-accumulation lever available to trade
# against this. `Accelerator()` is constructed at module-import time in
# train_unifolm_vla.py, BEFORE cfg is loaded, so cfg.trainer.gradient_accumulation_steps
# is dead -- accelerator.gradient_accumulation_steps is always 1 (its default) and
# `self.accelerator.accumulate(self.model)` in _train_step never actually
# accumulates. PER_DEVICE_BATCH_SIZE below IS the real global batch size (with
# NUM_PROCESSES=1), not a micro-batch you can multiply up via accumulation
# steps -- unlike RLDX's 1xA100 script, which used accumulation to reach a
# larger effective batch on top of a small, safe micro-batch.
#
# Also dead/inert, so not relied upon: `trainer.enable_gradient_checkpointing:
# true` in unifolm_vla_train.yaml is never wired to a `.gradient_checkpointing_enable()`
# call anywhere in this codebase (grep confirms only `_supports_gradient_checkpointing
# = True` class attrs + an unused `self.gradient_checkpointing = False` default
# inside cross_attention_dit.py -- nothing ever flips it). This now matters
# MORE than it did in the fully-frozen-backbone version, since the ViT's
# activations (see above) are the first real per-layer backward-activation
# cost in this script that gradient checkpointing could have traded away --
# but wiring it in remains a deeper, unverifiable change to shared training
# code (Qwen2_5_VLModel's own `.gradient_checkpointing_enable()` would need
# to be called and confirmed to actually reduce peak memory, with no GPU
# available here to check), so the batch size below is chosen conservatively
# instead, and this is flagged rather than silently left as a TODO.
#
# ------------------------------------------------------------------------
# Total estimated budget @ PER_DEVICE_BATCH_SIZE=6:
#   16.6 GB  backbone bf16 weights (whole 8.3B, frozen + trainable alike)
#   16.0 GB  DiT head fixed (fp32 weights + AdamW state + gradients)
#   10.8 GB  ViT fixed (fp32 master + AdamW state + gradients, ~676M params)
#  ~3.8 GB   DiT activations (scaled down from the ~5GB/batch-8 estimate)
#  ~9.0 GB   ViT activations (see above)
#   ------------------------------------------------------------------------
#  ~56.2 GB estimated, leaving ~23.8 GB headroom under 80GB -- tighter than
#  the fully-frozen-backbone version's ~45GB+ headroom, by design (unfreezing
#  ~676M more params was never going to be free), but with real margin left
#  over, none of it measured. PER_DEVICE_BATCH_SIZE was dropped from the
#  fully-frozen-backbone version's 8 to 6 specifically to keep that margin
#  comfortable given how approximate the activation math above is (windowed
#  attention + FlashAttention-2 not precisely modeled, DeepSpeed bf16-master-
#  copy behavior not confirmed) -- run the smoke-test command first, watch
#  `nvidia-smi -l 1` for the first ~50 steps, and drop to 4 immediately if
#  usage is climbing toward 80GB rather than plateauing. If it comfortably
#  plateaus well under 80GB, batch could likely go back up toward 8, but that
#  was not the conservative default chosen here.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"

# ---------------------------------------------------------------------------
# Checkpointing / schedule
# ---------------------------------------------------------------------------
# save_steps is handled directly by VLATrainer and takes precedence over
# save_interval. This run saves exactly at steps 30k and 40k; each is a full
# state_dict of the WHOLE model as currently written
# (`self.accelerator.get_state_dict(self.model)` in _save_checkpoint saves
# everything, including the ~7.62B still-frozen text decoder + lm_head, not
# just the trainable parts (ViT + DiT action head) -- there is no
# partial/trainable-only checkpoint save path in this codebase). Budget disk
# space accordingly (tens of GB per checkpoint).
SAVE_STEPS="${SAVE_STEPS:-[30000,40000]}"

# max_train_steps=40000 as specified. num_warmup_steps is set directly
# (~1.25% of max_train_steps, matching the RLDX-1 adaptation's warmup ratio)
# because trainer.warmup_ratio in unifolm_vla_train.yaml is a DEAD field --
# setup_optimizer_and_scheduler() only ever reads cfg.trainer.num_warmup_steps,
# never warmup_ratio.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-40000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"

# eval_interval default in the yaml is 100 steps, but VLATrainer.eval_action_model
# is NOT a held-out validation metric -- it just calls predict_action() on
# whatever batch _get_next_batch() returns next from the SAME training
# dataloader iterator (no separate eval dataset is ever constructed in
# prepare_data()). Raised to reduce the extra inference-mode forward pass's
# overhead without removing the (weak) training-fit sanity signal entirely.
EVAL_INTERVAL=1000
LOGGING_FREQUENCY=50

ARGS=(
    --config_yaml "$BASE_DIR/src/unifolm_vla/config/training/unifolm_vla_train.yaml"

    --framework.framework_py unifolm_vla
    --framework.qwenvl.base_vlm "$BASE_VLM"
    --framework.qwenvl.model_type qwen2_5_vl
    --framework.qwenvl.attn_implementation sdpa

    --datasets.vla_data.data_root_dir "$OXE_DATA_ROOT"
    --datasets.vla_data.data_mix "$DATA_MIX"
    --datasets.vla_data.window_size 1
    --datasets.vla_data.per_device_batch_size $PER_DEVICE_BATCH_SIZE

    # Freeze the text decoder + lm_head only -- see "VRAM budget reasoning"
    # above for the full module-tree trace and why this is real vision-only
    # unfreezing, not a coarser stand-in for it. The ViT
    # (qwen_vl_interface.model.model.visual) is left untouched here, so it
    # keeps requires_grad=True from `from_pretrained` and trains, alongside
    # the flow-matching DiT action head (`action_model`).
    --trainer.freeze_modules "qwen_vl_interface.model.model.language_model,qwen_vl_interface.model.lm_head"

    --trainer.max_train_steps $MAX_TRAIN_STEPS
    --trainer.skip_final_save "${SKIP_FINAL_SAVE:-False}"
    --trainer.num_warmup_steps $NUM_WARMUP_STEPS
    --trainer.save_steps "$SAVE_STEPS"
    --trainer.eval_interval $EVAL_INTERVAL
    --trainer.logging_frequency $LOGGING_FREQUENCY
    --trainer.use_wrist_image True
    --trainer.use_proprio True
    --trainer.shuffle_buffer_size 1000

    # Turns on the image-augmentation hook patched into RLDSDataset.__init__
    # (datasets.py) and train_unifolm_vla.py's prepare_data() -- see those
    # files' comments for exactly which of the requested augmentation params
    # (ColorJitter brightness/contrast/saturation/hue, RandomResizedCrop) made
    # it into dlimp's hook as-is vs. approximated, and why random rotation
    # could NOT be included (no rotation op exists in this pipeline's only
    # augmentation hook, and this adaptation does not hand-roll a new one).
    --trainer.image_aug True

    --run_root_dir "$RUN_ROOT_DIR"
    --run_id "$RUN_ID"

    # No wandb by default (avoids requiring an account/API key just to smoke
    # test). Add --wandb_project <name> --wandb_entity <name> if you want it --
    # _init_wandb() in train_unifolm_vla.py runs unconditionally on the main
    # process, so leaving these as the yaml's placeholder values will still
    # attempt an (offline-mode) wandb.init() rather than skipping it outright.
)

echo "About to launch (NOT auto-run):"
echo "accelerate launch --config_file $ACCEL_CONFIG_1GPU --num_processes $NUM_PROCESSES \\"
echo "  src/unifolm_vla/training/train_unifolm_vla.py ${ARGS[*]}"

accelerate launch --config_file "$ACCEL_CONFIG_1GPU" --num_processes $NUM_PROCESSES \
    src/unifolm_vla/training/train_unifolm_vla.py "${ARGS[@]}"
