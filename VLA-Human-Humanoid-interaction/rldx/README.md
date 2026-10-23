# RLDX-1 — Unitree G1 + SONIC fine-tune

Fork of [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1). Full background and
reasoning: [README_legacy.md](README_legacy.md).

Action target: 76-dim (64 `action.motion_token` + 6 `left_hand_joints` + 6
`right_hand_joints`), chunk=40, 1x A100 80GB, 40k steps.

## Run

```bash
# Smoke test
DATASET_PATH=/path/to/lerobot_dataset \
  bash run_scripts/train/custom/smoketest_g1_sonic_brainco6dof_1xa100.sh

# Full fine-tune
DATASET_PATH=/path/to/lerobot_dataset \
CUDA_VISIBLE_DEVICES=0 \
  bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh
```

The full fine-tune runs for hours — launch it inside `tmux` so it survives
a dropped connection:

```bash
tmux new-session -d -s rldx-g1-ft -c "$(pwd)" \
  "DATASET_PATH=/path/to/lerobot_dataset CUDA_VISIBLE_DEVICES=0 bash run_scripts/train/custom/finetune_g1_sonic_brainco6dof_1xa100.sh; exec bash"
```
