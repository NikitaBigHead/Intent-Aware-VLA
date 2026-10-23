# Short-paper code package

This directory is the compact code package for the short-paper submission. Start
with the documentation index in [`docs/README.md`](docs/README.md).

## Components

| Directory | Purpose | Instructions |
|---|---|---|
| `hri_video_robot/` | Default two-camera RGB intent model | [`docs/hri_video_robot.md`](docs/hri_video_robot.md) |
| `hri_video_robot_all/` | Code and configurations for all six RGB variants | [`docs/hri_video_robot_all.md`](docs/hri_video_robot_all.md) |
| `dual_camera_robot/` | Original pose-based two-camera baseline | [`docs/dual_camera_robot.md`](docs/dual_camera_robot.md) |
| `dual_camera_robot_v2/` | Updated pose-based baseline | [`docs/dual_camera_robot_v2.md`](docs/dual_camera_robot_v2.md) |
| `Teleop-Inference-WBC/` | Robot control, teleoperation, and VLA integration | [`docs/teleop_inference_wbc.md`](docs/teleop_inference_wbc.md) |
| `VLA-Human-Humanoid-interaction/` | UnifoLM-VLA, Being-H05, and RLDX training adaptations | [`docs/vla_models.md`](docs/vla_models.md) |
| `groot-training-inference/` | Compact Isaac-GR00T training and inference snapshot from `tactile-train` | [`docs/groot_training_inference.md`](docs/groot_training_inference.md) |
| `tactile_test_rmse/` | Cross-model offline trajectory RMSE evaluation | [`docs/tactile_rmse.md`](docs/tactile_rmse.md) |
| `supplementary_video.mp4` | Anonymized 84-second qualitative demonstration | Faces are blurred in all human-facing views |

The package is intentionally source-oriented. Generated logs, caches, virtual
environments, Git history, training runs, datasets, checkpoints, reference
episodes, large simulator/robot assets, and non-default model weights are not
duplicated here. The included and omitted artifacts are documented in
[`docs/external_assets.md`](docs/external_assets.md).

Submission-specific names, participant identifiers, hostnames, network
addresses, and local filesystem paths have been anonymized. Participant IDs use
the `subject_XX` form, direct demographic fields were removed from the sample
metadata, and faces in the included sample videos are blurred or excluded by
the camera crop. Required third-party copyright, license, and citation
attributions are retained. See [`docs/anonymization.md`](docs/anonymization.md).

For archiving and submission, follow
[`docs/packaging.md`](docs/packaging.md). In particular, use `tar` rather than
ZIP so the shared RGB checkpoint remains stored only once.

> **Safety:** robot-control software can cause sudden motion. Test perception
> components offline first and use a clear safety zone and a ready emergency
> stop when running on hardware.
