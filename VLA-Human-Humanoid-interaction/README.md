# VLA Human-Humanoid Interaction

Fine-tuning setups for three VLA models on a shared Unitree G1 + BrainCo
Revo2 dataset (LeRobot v2.1, teleop'd through NVlabs GR00T-WholeBodyControl's
GEAR-SONIC whole-body controller). All three predict the same 76-dim action
target — 64-dim `action.motion_token` (SONIC continuous latent) + 6-dim
`left_hand_joints` + 6-dim `right_hand_joints`, chunked over 40 steps.

## Models

- [rldx/](rldx/) — adapted from [RLWRLD/RLDX-1](https://github.com/RLWRLD/RLDX-1)
- [Being-H/](Being-H/) — adapted from [BeingBeyond/Being-H](https://github.com/BeingBeyond/Being-H) (Being-H05)
- [unifolm-vla/](unifolm-vla/) — adapted from [unitreerobotics/unifolm-vla](https://github.com/unitreerobotics/unifolm-vla)

Each folder has its own `README.md` with ready-to-run commands, and a
`README_legacy.md` with the original project documentation.
