# VLA training adaptations

Project directory:
[`../VLA-Human-Humanoid-interaction/`](../VLA-Human-Humanoid-interaction/)

The repository contains three model adaptations trained on the shared Unitree
G1 + BrainCo Revo2 dataset. Each predicts a 76-dimensional action target:
64 SONIC motion-token dimensions plus six joint values for each hand, with a
40-step action chunk.

## Provenance

- The umbrella repository, Being-H05, and RLDX were copied from server 2,
  `/workspace/project/VLA-Human-Humanoid-interaction`, at base commit `760fb85`.
- `unifolm-vla/` was copied from the server 3 version at
  `/workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla`, also based on
  `760fb85` and including its local conversion, training, model, and dataset
  changes.
- Local uncommitted changes are intentionally present in the snapshot.
- Git history, checkpoints, datasets, logs, caches, generated results, the
  UnifoLM demo GIF, and RLDX external simulator dependencies are omitted.

The shared overview is in the
[project README](../VLA-Human-Humanoid-interaction/README.md).

## UnifoLM-VLA

Directory:
[`unifolm-vla/`](../VLA-Human-Humanoid-interaction/unifolm-vla/)

Convert the LeRobot dataset and train on one A100:

```bash
cd VLA-Human-Humanoid-interaction/unifolm-vla

LEROBOT_DATASET_ROOT=/path/to/lerobot_dataset \
HDF5_OUTPUT_DIR=/path/to/hdf5 \
OXE_DATA_ROOT=/path/to/rlds \
  bash scripts/run_scripts/convert_g1_sonic_brainco6dof_dataset.sh

OXE_DATA_ROOT=/path/to/rlds \
BASE_VLM=/path/to/Unifolm-VLM-Base \
  bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh
```

The server-3-specific setup and 40k-step launchers are
[`scripts/run_scripts/SERVER_SETUP.md`](../VLA-Human-Humanoid-interaction/unifolm-vla/scripts/run_scripts/SERVER_SETUP.md),
[`run_40k.sh`](../VLA-Human-Humanoid-interaction/unifolm-vla/scripts/run_scripts/run_40k.sh),
and [`start_40k_tmux.sh`](../VLA-Human-Humanoid-interaction/unifolm-vla/scripts/run_scripts/start_40k_tmux.sh).
For inference, use
[`deployment/model_server/run_real_eval_server.py`](../VLA-Human-Humanoid-interaction/unifolm-vla/deployment/model_server/run_real_eval_server.py).
See the [adaptation README](../VLA-Human-Humanoid-interaction/unifolm-vla/README.md)
for the full recipe.

## Being-H05

Directory:
[`Being-H/`](../VLA-Human-Humanoid-interaction/Being-H/)

```bash
cd VLA-Human-Humanoid-interaction/Being-H/Being-H05

DATASET_PATH=/path/to/lerobot_dataset \
  bash scripts/train/smoketest_g1_sonic_brainco6dof_1xa100.sh

DATASET_PATH=/path/to/lerobot_dataset \
  bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh
```

The server-2 launcher is
[`run_single_gpu.sh`](../VLA-Human-Humanoid-interaction/Being-H/Being-H05/scripts/train/run_single_gpu.sh).
Read the [short adaptation guide](../VLA-Human-Humanoid-interaction/Being-H/README.md)
and the [full Being-H05 README](../VLA-Human-Humanoid-interaction/Being-H/Being-H05/README.md).

## RLDX

Directory: [`rldx/`](../VLA-Human-Humanoid-interaction/rldx/)

```bash
cd VLA-Human-Humanoid-interaction/rldx

DATASET_PATH=/path/to/lerobot_dataset \
  bash run_scripts/train/custom/smoketest_g1_sonic_brainco6dof_1xa100.sh

DATASET_PATH=/path/to/lerobot_dataset CUDA_VISIBLE_DEVICES=0 \
  bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh
```

Detailed documentation:

- [adaptation README](../VLA-Human-Humanoid-interaction/rldx/README.md)
- [installation](../VLA-Human-Humanoid-interaction/rldx/docs/installation.md)
- [training](../VLA-Human-Humanoid-interaction/rldx/docs/training.md)
- [inference server](../VLA-Human-Humanoid-interaction/rldx/docs/inference_server.md)
- [evaluation](../VLA-Human-Humanoid-interaction/rldx/docs/evaluation.md)

Restore each model's base checkpoint and the shared LeRobot dataset before
running these commands. Those artifacts are intentionally not part of the
compact source submission.
