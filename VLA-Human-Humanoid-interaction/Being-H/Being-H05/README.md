# Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization

<p align="center">
    <img src="assets/being-h05.png" width="300"/>
<p>

<div align="center">

[![Blog](https://img.shields.io/badge/Blog-Being--H05-green)](https://research.beingbeyond.com/being-h05)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2601.12993)
[![Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/BeingBeyond/being-h05)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

</div>

Being-H0.5 is BeingBeyond's flagship **VLA** model, scaling human-centric learning with a unified action space to enable robust cross-embodiment robot control.
This directory contains the released H0.5 codebase inside the unified `Being-H` repository layout.

<div align="center">
<video src="https://github.com/user-attachments/assets/36714389-e737-4b11-8dcf-9076cc9f1d69" controls>
</video>
</div>

## Model Checkpoints

Download models from Hugging Face [Model Collections](https://huggingface.co/collections/BeingBeyond/being-h05):

| Model Type | Model Name | Parameters | Description |
|------------|------------|------------|-------------|
| **VLA Pretrained** | [Being-H05-2B](https://huggingface.co/BeingBeyond/Being-H05-2B) | 2B | Base vision-language-action model (preview) |
| **VLA Specialist** | [Being-H05-2B_libero](https://huggingface.co/BeingBeyond/Being-H05-2B_libero) | 2B | Post-trained on LIBERO benchmark |
| **VLA Specialist** | [Being-H05-2B_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_robocasa) | 2B | Post-trained on RoboCasa kitchen tasks |
| **VLA Generalist** | [Being-H05-2B_libero_robocasa](https://huggingface.co/BeingBeyond/Being-H05-2B_libero_robocasa) | 2B | Post-trained on both LIBERO and RoboCasa |

Note: the vision part is 224px by default.

## Quick Start

### Installation

```bash
git clone https://github.com/BeingBeyond/Being-H.git
cd Being-H/Being-H05
conda create -n beingh python=3.10
conda activate beingh
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Training

```bash
# Single-embodiment training (e.g., LIBERO)
bash scripts/train_libero_example.sh

# Cross-embodiment training (multiple robots)
bash scripts/train_cross_emb_example.sh
```

**Important for cross-embodiment training:** Enable `--save_merged_metadata True` to save hierarchical metadata for inference. See [docs/training.md](docs/training.md) for details.

### Inference

```python
from BeingH.inference.beingh_policy import BeingHPolicy

# Load a pre-trained policy
policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",      # Path to Being-H checkpoint
    data_config_name="<config-name>",       # e.g., "libero_nonorm", "robocasa_human"
    dataset_name="<dataset-name>",          # For loading normalization stats
    embodiment_tag="<robot-tag>",           # Robot identifier
    instruction_template="<prompt>",        # Task instruction template
)

# Run inference
actions = policy.get_action(observations)
```

See [docs/inference.md](docs/inference.md) for the complete API reference.

## Supported Robots

Being-H currently provides example configurations for **LIBERO** and **RoboCasa** benchmarks. We will gradually release more pre-built configurations for additional robot platforms.

To add your own robot, refer to our example configurations and the [Unified Action Space](docs/unified_action_space.md) slot layout, then follow the guide in [Data Configuration](docs/data_configuration.md).

Don't see your robot? [Open an issue](https://github.com/BeingBeyond/Being-H/issues) with your robot specs and a data sample - we're happy to help add support.

## How It Works: Unified Action Space

Being-H uses a **200-dimensional unified action space** that maps different robots to a shared semantic representation. This is what enables cross-embodiment transfer.

**The key insight**: Similar robot components (e.g., end-effector position) always map to the same dimensions, regardless of the robot type. This allows knowledge to transfer between robots.

For most users, you don't need to understand the details - just use one of the pre-built configurations. For advanced users who want to add custom robots, see the complete documentation:

**[Unified Action Space Guide](docs/unified_action_space.md)** - Complete slot layout and configuration examples

## Cross-Embodiment Metadata

For cross-embodiment models, Being-H saves **metadata** during training that is essential for inference. This metadata contains normalization statistics for each task/embodiment.

When running inference on a cross-embodiment model, specify which metadata variant to use:

```python
policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",
    dataset_name="uni_posttrain",              # Cross-embodiment dataset
    metadata_variant="<task-or-embodiment>",   # Select normalization stats
    stats_selection_mode="task",               # "task", "embodiment", or "auto"
    # ... other parameters
)
```

See [docs/inference.md](docs/inference.md#cross-embodiment-metadata) for details.

## Documentation

| Document | Description |
|----------|-------------|
| [Unified Action Space](docs/unified_action_space.md) | How cross-embodiment transfer works |
| [Data Configuration](docs/data_configuration.md) | Adding custom robots and datasets |
| [Training](docs/training.md) | Training parameters and scripts |
| [Inference](docs/inference.md) | BeingHPolicy API reference |
| [Evaluation](docs/evaluation.md) | LIBERO and RoboCasa benchmarks |

## Unitree G1 + BrainCo Revo2 (SONIC motion-token) fine-tune

This checkout also has a ready-made recipe for post-training Being-H05-2B
on a Unitree G1 + BrainCo Revo2 (6-DOF hand) dataset
("sep-10-2026-cleaned-merged", LeRobot v2.1, 535 episodes / ~182k frames
@ 50Hz), where the action target is a **SONIC motion-token latent** (from
NVlabs/GR00T-WholeBodyControl's GEAR-SONIC whole-body controller) instead
of raw joint commands, teleop'd through the same controller. This mirrors
an equivalent adaptation already done for a different VLA codebase
(RLDX-1); the mechanism here is fully native to Being-H05 (the two
codebases don't share an API), but the underlying recipe -- 76-dim
SONIC-latent action target, 40-step chunking, 2 cameras only, no
validation split, a specific augmentation recipe, mostly-frozen-backbone
fine-tuning -- is the same.

**The real dataset was not available while this was written** (it had
already been deleted locally) -- everything below was built from a written
schema description, not by loading real Parquet/video files. Several
pieces are therefore documented, flagged guesses rather than verified
facts -- see `configs/data_config.py`'s
`UnitreeG1SonicBrainco6DofDataConfig` docstring for the full list (mainly:
the exact column order inside the 41-dim `observation.state` vector, and
which "unified action space" slots to reuse for a bipedal humanoid + a
non-physical latent, since [docs/unified_action_space.md](docs/unified_action_space.md)'s
documented 200-dim layout has no official slot for either). **Verify
against the real dataset's `meta/info.json` before trusting a long run.**

### Action target

76-dim continuous vector, predicted as a 40-step chunk (0.8s lookahead @
50Hz), not single-step:
- 64 dims: `action.motion_token` (SONIC latent, encoded live during teleop)
- 6 dims: `teleop.left_hand_joints`
- 6 dims: `teleop.right_hand_joints` (exactly 0.0 across all 535 episodes
  in the *original* dataset; per the task owner this isn't purely an old-
  task-set artifact -- the right hand's own DOF range is hardware-limited
  and can't close into a full fist or a functional handshake grip, even in
  newer hug/handshake/fist-bump episodes where the right ARM does move via
  the motion-token/`observation.state.right_arm` channel, not this field.
  May still be a narrow/near-degenerate signal in the expanded dataset --
  re-profile before trusting either "all-zero" or "now varies" as fact.
  Still included per this recipe, not silently dropped)
- (`action.wbc`, the raw 41-dim joint-target field, is intentionally **not**
  used -- it's a debug/reference field only)

### What's wired up in this checkout

See each file's own comments for the full reasoning:
- `EmbodimentTag.UNITREE_G1_SONIC_BRAINCO6DOF` (`BeingH/utils/constants.py`)
  -- also documents a pre-existing `embodiment_id` resolution quirk found
  during this integration (an `Enum`-vs-`.value` key mismatch in
  `BeingH/dataset/datasets/vla_dataset.py`'s `__iter__`), which affects
  every embodiment equally and was flagged rather than fixed, being out of
  scope for a minimal patch.
- `UnitreeG1SonicBrainco6DofDataConfig`
  (`configs/data_config.py`) -- the 76-dim action / 41-dim state / 2-camera
  modality mapping into Being-H05's 200-dim unified action/state space.
- `configs/dataset_info.py` -- `DATASET_REGISTRY`/`DATASET_INFO` entries
  for `g1_sonic_brainco6dof_posttrain`. Unlike every other entry in that
  file, the dataset path is **not** hardcoded -- it's read from the
  `DATASET_PATH` env var at import time, since the real dataset lives
  outside this repo and isn't at a fixed path across machines.
- `configs/posttrain/unitree_g1_sonic_brainco6dof/unitree_g1_sonic_brainco6dof.yaml`
  -- single-embodiment dataset config (shaped like `configs/posttrain/libero/libero_all.yaml`,
  not the multi-embodiment `cross-embodiment/` example, since this is one
  dataset/one embodiment).
- `BeingH/dataset/preprocess.py`'s `build_vit_transform_base` -- two new,
  **opt-in** kwargs (`crop_scale_min`, `random_rotation_degrees`) so this
  recipe can get `RandomResizedCrop` scale `(0.85, 1.0)` and a ±8° random
  rotation stage, neither of which existed/was reachable before. Every
  existing recipe (LIBERO/RoboCasa) is byte-for-byte unaffected, since
  their YAMLs don't set these new keys and the defaults reproduce the
  exact previous hardcoded behavior. `ColorJitter(brightness=0.3,
  contrast=0.4, saturation=0.5, hue=0.08)` was already hardcoded in that
  same function and already matches this recipe's required values, so
  nothing needed to change there. There is no horizontal-flip stage
  anywhere in this pipeline, so left/right hand semantics (separate
  `left_hand_joints`/`right_hand_joints` action dims) are preserved by
  construction, not by a flag. Optional gaussian blur/noise was
  deliberately **not** added -- no ready-made transform exists in this
  codebase to opt into, and the task asked not to build one from scratch
  just for this.
- `BeingH/train/train.py`'s `apply_model_freezing()` -- a new, **opt-in**
  `--vit_trainable_within_mllm` flag (default `False`, byte-for-byte
  unchanged behavior for every other recipe) that lets `--freeze_mllm True`
  freeze the base LLM (and, via `--freeze_vit_mlp`, the connector) while
  excluding `vit_model.*` parameter names from that same bundle freeze, so
  the vision tower trains too. Added because `mllm_layer_names` (built from
  every key in the pretrained InternVL checkpoint) previously had no
  per-submodule opt-out for the ViT the way it already had one for the
  connector -- see this recipe's launch scripts for the full trace (why
  `--freeze_llm` alone can't substitute: it would also freeze the action
  expert, whose weights live in the same `Qwen3MoTDecoderLayer` modules as
  the base LLM's).
- Camera exclusion (`external_view_camera` must never be decoded): this
  needed **no source patch at all** -- `BaseDataConfig.add_video_modality()`
  only ever adds video keys that are explicitly declared in a
  `DataConfig`'s `VIDEO_SOURCE_COLUMNS`, and
  `LeRobotIterableDataset._extract_and_sample_step()` only opens/decodes an
  episode's mp4 for keys present in `modality_defs` (i.e. returned by
  `define_modalities()`). `UnitreeG1SonicBrainco6DofDataConfig` simply never
  declares `external_view_camera`, so its videos are never opened, decoded,
  or otherwise touched.
- No custom checkpoint-cadence callback was needed (unlike the sibling
  RLDX-1 adaptation, which needed a bespoke `ExactStepsCheckpointCallback`):
  Being-H05's native `--save_steps`/`--save_steps_start` pair
  (`BeingH/train/train.py`) already expresses "checkpoint every N steps
  starting from step S", which is all this recipe needs for "a handful of
  checkpoints across the run".

### How to run

```bash
# Smoke test first (10 steps, throwaway output dir):
DATASET_PATH=/path/to/sep-10-2026-cleaned-merged \
  bash scripts/train/smoketest_g1_sonic_brainco6dof_1xa100.sh

# Full post-train (40k steps, 1x A100 80GB):
DATASET_PATH=/path/to/sep-10-2026-cleaned-merged \
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh
```

Both scripts are plain `torchrun` calls into `BeingH/train/train.py` (same
entrypoint as every other recipe in this repo -- see [Training](docs/training.md)),
and both **print the fully-resolved command before running it** so you can
copy-paste it elsewhere (e.g. into `tmux`) or inspect it before committing
to a multi-hour run. Neither script auto-executes from anywhere else in
this repo; you run them yourself. Open either script for the full flag
list, the freeze/LR/MPG configuration and why each value was picked, and
the VRAM budget reasoning (batch size here is governed by a **token-budget
sequence packer**, not a `--global-batch-size` flag -- see the "Batch size
/ VRAM reasoning" comment in `train_g1_sonic_brainco6dof_1xa100.sh` for
why that's the right mental model in this codebase specifically).

You'll also need local copies of the base checkpoints referenced by both
scripts (`PRETRAIN_MODEL`=InternVL3_5-2B, `EXPERT_MODEL`=Qwen3-0.6B,
`RESUME_PATH`=the released Being-H05-2B checkpoint from
[Hugging Face](https://huggingface.co/collections/BeingBeyond/being-h05)) --
override those three env vars to wherever you've downloaded them.

Training-time monitoring is TensorBoard only (`--logging_dir`) -- this
codebase has no wandb/`report_to` integration for training at all (only
its LIBERO eval scripts reference wandb), and there's no validation split
either (`BeingH/train/train.py` has no `--eval_strategy`/held-out-split
mechanism to disable, so "train-loss only" falls out automatically). For a
long run, launch inside `tmux`/`screen` so it survives a dropped
connection:

```bash
tmux new-session -d -s beingh-g1-ft -c "$(pwd)" \
  "DATASET_PATH=/path/to/sep-10-2026-cleaned-merged bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```

## TODO

The following features are planned for future implementation:

- [ ] Out-of-the-box real robot pretrained checkpoints
- [ ] Complete pretraining scripts and documentation
- [x] Complete post-training scripts for all benchmarks
- [x] Detailed training and data documentation
- [x] Benchmark evaluation scripts for all supported tasks

## Contributing and Building on Being-H

We encourage researchers and practitioners to leverage Being-H as a foundation for their own experiments and applications. Whether you're adapting Being-H to new robotic platforms, exploring novel manipulation tasks, or extending the model to new domains, our modular codebase is designed to support your innovations. We welcome contributions of all kinds - from bug fixes and documentation improvements to new features and model architectures. By building on Being-H together, we can advance the field of vision-language-action modeling and enable robots to perform more complex and diverse manipulation tasks. Join us in making robotic manipulation more capable, robust, and accessible to all.

## Acknowledgments

Being-H builds on the following excellent open-source projects:

- [InternVL](https://github.com/OpenGVLab/InternVL): Vision-Language model backbone
- [Bagel](https://github.com/ByteDance-Seed/Bagel): Training framework
- [Qwen](https://github.com/QwenLM/Qwen): Language model and MoE expert
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): Benchmark for lifelong robot learning
- [RoboCasa](https://github.com/robocasa/robocasa): Large-scale simulation benchmark for everyday tasks

We thank the authors for their contributions to the robotics and machine learning communities.

## License

Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.

SPDX-License-Identifier: Apache-2.0

## Citation

If you find our work useful, please consider citing us and give a star to our repository! 🌟🌟🌟

**Being-H0.5**

```bibtex
@article{beingbeyond2026beingh05,
  title={Being-H0. 5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization},
  author={Luo, Hao and Wang, Ye and Zhang, Wanpeng and Zheng, Sipeng and Xi, Ziheng and Xu, Chaoyi and Xu, Haiweng and Yuan, Haoqi and Zhang, Chi and Wang, Yiqing and others},
  journal={arXiv preprint arXiv:2601.12993},
  year={2026}
}
```
