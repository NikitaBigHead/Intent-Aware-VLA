# Being-H05 — Unitree G1 + SONIC fine-tune

Adapted from [BeingBeyond/Being-H](https://github.com/BeingBeyond/Being-H)
(Being-H05). Full background: [README_legacy.md](README_legacy.md) and
[Being-H05/README.md](Being-H05/README.md).

Action target: 76-dim (64 `action.motion_token` + 6 `left_hand_joints` + 6
`right_hand_joints`), chunk=40, 1x A100 80GB, 40k steps.

## Run

```bash
cd Being-H05

# Smoke test
DATASET_PATH=/path/to/lerobot_dataset \
  bash scripts/train/smoketest_g1_sonic_brainco6dof_1xa100.sh

# Full fine-tune
DATASET_PATH=/path/to/lerobot_dataset \
  bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh
```

The full fine-tune runs for hours — launch it inside `tmux` so it survives
a dropped connection:

```bash
tmux new-session -d -s beingh-g1-ft -c "$(pwd)/Being-H05" \
  "DATASET_PATH=/path/to/lerobot_dataset bash scripts/train/train_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```
