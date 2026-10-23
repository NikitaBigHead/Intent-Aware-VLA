# RGB intent-model training

This directory contains the training implementation, dataset extraction and
audit utilities, experiment configurations, and tests used for the RGB model
comparison.

## Environment

Create an isolated Python environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data preparation

The training code expects an episode index with synchronized `head` and
`ego_view` streams, interaction labels, split assignments, and optional onset
and contact times. Dataset locations are environment-specific and are not part
of the source submission.

Use the scripts under `scripts/` to audit the source dataset, extract frames,
prepare a local experiment, and validate the resulting index. Replace all
example paths with the local dataset root.

## Training

Configuration files are stored in `configs/`. The main training entry point is
`scripts/train.py`; `scripts/launch_training_job.py` provides the corresponding
job launcher. Checkpoints and generated run directories are intentionally
excluded from the compact package.

The default deployable checkpoint and the six-model comparison package are
documented in [`../../docs/hri_video_robot.md`](../../docs/hri_video_robot.md)
and [`../../docs/hri_video_robot_all.md`](../../docs/hri_video_robot_all.md).
