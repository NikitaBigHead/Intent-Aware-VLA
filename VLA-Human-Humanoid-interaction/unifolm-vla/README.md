# UnifoLM-VLA-0 — Unitree G1 + SONIC fine-tune

Adapted from [unitreerobotics/unifolm-vla](https://github.com/unitreerobotics/unifolm-vla).
Full background: [README_legacy.md](README_legacy.md).

Action target: 76-dim (64 `action.motion_token` + 6 `left_hand_joints` + 6
`right_hand_joints`), chunk=40, 1x A100 80GB, 40k steps.

## Run

```bash
# 1. Convert LeRobot -> HDF5 -> RLDS
LEROBOT_DATASET_ROOT=/path/to/lerobot_dataset \
HDF5_OUTPUT_DIR=/path/to/hdf5 \
OXE_DATA_ROOT=/path/to/rlds \
  bash scripts/run_scripts/convert_g1_sonic_brainco6dof_dataset.sh

# 2. Fine-tune
OXE_DATA_ROOT=/path/to/rlds \
BASE_VLM=/path/to/Unifolm-VLM-Base \
  bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh
```

Both steps can take hours — run each inside `tmux` so it survives a
dropped connection:

```bash
tmux new-session -d -s unifolm-g1-convert -c "$(pwd)" \
  "LEROBOT_DATASET_ROOT=/path/to/lerobot_dataset HDF5_OUTPUT_DIR=/path/to/hdf5 OXE_DATA_ROOT=/path/to/rlds bash scripts/run_scripts/convert_g1_sonic_brainco6dof_dataset.sh; exec bash"

tmux new-session -d -s unifolm-g1-ft -c "$(pwd)" \
  "OXE_DATA_ROOT=/path/to/rlds BASE_VLM=/path/to/Unifolm-VLM-Base bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```
