# RLDX-1 — Humanoid (Unitree G1 + SONIC) Fine-tune

This is a fork of [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1) with a
ready-made recipe for fine-tuning on a Unitree G1 + BrainCo Revo2 (6-DOF
hand) dataset, where the action target is a **SONIC motion-token latent**
(from NVlabs/GR00T-WholeBodyControl's GEAR-SONIC whole-body controller)
instead of raw joint commands.

**For everything else about RLDX-1** (installation, architecture,
inference server, benchmarks, general fine-tuning flags, citation, license)
**see [README_backup.md](README_backup.md)** — the original upstream
README, unmodified.

## Action target

76-dim continuous vector:
- 64 dims: `action.motion_token` (SONIC latent, encoded live during teleop)
- 6 dims: `teleop.left_hand_joints`
- 6 dims: `teleop.right_hand_joints`
- (`action.wbc`, the raw 41-dim joint-target field, is intentionally **not**
  used — it's a debug/reference field only)

## Dataset

Any [LeRobot](https://github.com/huggingface/lerobot) v2.1 dataset with
this schema — 41-dim `observation.state` (legs/waist/arms/hands), the
76-dim action above, and at least the two camera keys referenced in the
embodiment's modality config (adjust
`rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py` if your camera
keys or joint-group boundaries differ — it reads them from your dataset's
own `meta/modality.json`, not a hardcoded template).

## What's wired up in this fork

See the files themselves for the reasoning in each:
- `EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF` (`rldx/data/embodiment_tags.py`)
- Its modality config (`rldx/configs/data/unitree_g1_sonic_brainco6dof_config.py`)
- `max_action_dim` bumped to 76 (`rldx/configs/model/rldx.py`) — plus the
  corresponding processor-side override fix so this actually takes effect
  when fine-tuning from a pretrained checkpoint
  (`rldx/model/core/setup.py`, `rldx/model/core/processing_rldx.py`)
- `ExactStepsCheckpointCallback` (`rldx/experiment/utils.py` +
  `rldx/experiment/experiment.py`) — save checkpoints at an explicit,
  irregularly-spaced set of steps (`RLDX_EXACT_SAVE_STEPS` env var) instead
  of only a uniform interval, useful when each checkpoint is tens of GB


## How to run

```bash
# Smoke test first (a handful of steps, throwaway output dir):
DATASET_PATH=/path/to/your/lerobot_dataset \
  bash run_scripts/train/custom/smoketest_g1_sonic_brainco6dof.sh

# Full fine-tune:
DATASET_PATH=/path/to/your/lerobot_dataset \
CUDA_VISIBLE_DEVICES=0,1 \
  bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof.sh
```

Both scripts are plain `torchrun` + DeepSpeed under the hood (same as any
other fine-tune in this repo — see
[README_backup.md → Fine-tuning](README_backup.md#fine-tuning)) and print
the resolved command before running. Open either script for the full flag
list and inline comments — GPU count, batch size, LR schedule, checkpoint
steps, and augmentation are all adjustable in-place.

For a long run, launch inside `tmux`/`screen` so it survives a dropped
connection:

```bash
tmux new-session -d -s rldx-g1-ft -c "$(pwd)" \
  "DATASET_PATH=/path/to/your/lerobot_dataset bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof.sh; exec bash"
```

### Single-GPU variant (1x A100 80GB)

`finetune_g1_sonic_brainco6dof_1xa100.sh` / `smoketest_g1_sonic_brainco6dof_1xa100.sh`
are the same recipe pinned to exactly one GPU (`--num-gpus 1`), for
deployments where the checkpoint is trained and/or served on a single
A100 80GB rather than the multi-GPU boxes the default scripts assume.

With `num_gpus == 1`, `rldx/experiment/experiment.py` skips DeepSpeed
entirely (`if config.training.num_gpus > 1 and not config.training.use_ddp:`)
and trains as a plain HF `Trainer` with bf16 + gradient checkpointing — so
there's no ZeRO stage to pick for a single GPU, and none is lost either
(stage 2 partitions across ranks, and there's only one rank here).

Batch size: per-device micro-batch 4 with `--gradient-accumulation-steps 3`
(effective batch 12, matching this fork's 2-GPU default recipe's effective
batch, so the LR/warmup schedule stays meaningful without re-deriving it).
This was lowered from an earlier micro-batch-6 version of this script after
the vision tower was unfrozen (`--tune-visual`, see below) to keep VRAM
headroom comfortable. The script's own comments walk through the VRAM
budget (~61-68GB estimated on the Qwen3-VL-8B backbone with the top-4 LLM
layers + projector + vlln + diffusion action head + the full vision tower
trainable) — this is a reasoned estimate, not a measured one (no GPU was
available while writing it), so **run the smoke-test variant first and
watch `nvidia-smi`** before trusting it for a multi-hour run. `--max-steps`
is already 40000, matching the default script.

**Vision tower unfrozen (`--tune-visual`):** added after the task owner
flagged that the fine-tune now includes tasks (hug/handshake/fist-bump)
whose close-proximity, human-in-frame visual domain a frozen, purely
manipulation-pretrained vision tower may not represent well.
`tune_visual` is all-or-nothing in this codebase (no top-N-layers partial
option the way `tune_top_llm_layers` gives the LLM) — see
`rldx/model/modules/backbone/adapter.py::set_trainable_parameters`.

```bash
DATASET_PATH=/path/to/your/lerobot_dataset \
  bash run_scripts/train/custom/smoketest_g1_sonic_brainco6dof_1xa100.sh

DATASET_PATH=/path/to/your/lerobot_dataset \
CUDA_VISIBLE_DEVICES=0 \
  bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh
```

For a long run, launch inside `tmux`/`screen` so it survives a dropped
connection:

```bash
tmux new-session -d -s rldx-g1-ft-1xa100 -c "$(pwd)" \
  "DATASET_PATH=/path/to/your/lerobot_dataset CUDA_VISIBLE_DEVICES=0 bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```
