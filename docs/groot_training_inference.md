# Isaac-GR00T training and inference

Project directory:
[`../groot-training-inference/`](../groot-training-inference/)

This directory is a compact snapshot of the project training container on the
primary GPU server. The original image is represented as
`<GR00T_FINETUNE_IMAGE>` in this anonymized package.

## Layout and provenance

- [`Isaac-GR00T/`](../groot-training-inference/Isaac-GR00T/) is the modified
  `/opt/Isaac-GR00T` source tree, based on commit `ead5283` of
  <https://github.com/NVIDIA/Isaac-GR00T.git>.
- [`workspace/`](../groot-training-inference/workspace/) contains the custom
  training configurations, launchers, data-processing utilities, inference
  wrappers, and offline validation code from the host bind mount
  `/root/tactile_ws` (container path `/workspace`).
- Local changes to GR00T configuration, dataset loading, fine-tuning,
  checkpoint upload, and server code are included.

The 14-GB virtual environment, 223 GB of checkpoints, 33 GB of datasets,
Hugging Face cache, logs, generated validation media, demo data, Git history,
and external simulator repositories are not included.

## Environment

Use the original container image or follow the upstream
[Isaac-GR00T README](../groot-training-inference/Isaac-GR00T/README.md).
Inside a compatible container, place or mount these directories at the paths
expected by the launchers:

```text
/opt/Isaac-GR00T   <- groot-training-inference/Isaac-GR00T
/workspace         <- groot-training-inference/workspace
```

Dataset and checkpoint paths in the launchers are machine-specific examples.
Replace `/workspace/datasets`, `/workspace/checkpoints`, and `/data/project`
before running on another system.

## Training

The primary upstream entry point is
[`examples/finetune.sh`](../groot-training-inference/Isaac-GR00T/examples/finetune.sh),
which calls the modified training stack under `gr00t/experiment/`.

Submission-specific configurations:

- [`affective_g1_config_two_cam.py`](../groot-training-inference/workspace/affective_g1_config_two_cam.py)
  — two-camera G1 affective-interaction configuration;
- [`gr00t_cosmos_right13_config.py`](../groot-training-inference/workspace/gr00t_cosmos_right13_config.py)
  — Cosmos/RoboHaMeR right-arm configuration;
- [`tactile_g1_config.py`](../groot-training-inference/workspace/tactile_g1_config.py)
  — tactile G1 configuration.

Ready-made launchers include:

- [`train_cosmos.sh`](../groot-training-inference/workspace/train_cosmos.sh)
- [`train_cosmos_batch400.sh`](../groot-training-inference/workspace/train_cosmos_batch400.sh)
- [`train_batch240v2_right13_rejected.sh`](../groot-training-inference/workspace/train_batch240v2_right13_rejected.sh)
- [`train_batch240v2_right13_resume_no_upload.sh`](../groot-training-inference/workspace/train_batch240v2_right13_resume_no_upload.sh)

`workspace/train.sh` is a record of several experiment commands rather than a
single portable launcher; use one of the focused scripts above for a new run.

## Inference server

The canonical server entry point is
[`gr00t/eval/run_gr00t_server.py`](../groot-training-inference/Isaac-GR00T/gr00t/eval/run_gr00t_server.py).
The workspace wrappers register the custom embodiment configuration and supply
the checkpoint and port:

- [`run_server_cosmos.py`](../groot-training-inference/workspace/run_server_cosmos.py)
- [`run_server_cosmos_gearsonic.py`](../groot-training-inference/workspace/run_server_cosmos_gearsonic.py)
- [`run_server_gearsonic_local20000.py`](../groot-training-inference/workspace/run_server_gearsonic_local20000.py)
- [`run_server_rawvideo20000.py`](../groot-training-inference/workspace/run_server_rawvideo20000.py)

With the original path layout restored:

```bash
source /opt/Isaac-GR00T/.venv/bin/activate
cd /opt/Isaac-GR00T
python /workspace/run_server_cosmos.py
```

Change the hard-coded checkpoint and port in the selected wrapper before use.
For deployment/export utilities, see
[`scripts/deployment/`](../groot-training-inference/Isaac-GR00T/scripts/deployment/).

## Validation and data preparation

- [`workspace/validation_VLA/README.md`](../groot-training-inference/workspace/validation_VLA/README.md)
  documents open-loop validation.
- [`workspace/dataset_process/`](../groot-training-inference/workspace/dataset_process/)
  contains interaction-class dataset transformations.
- `convert_robohamer_340.py`, `rebuild_lerobot_true_raw.py`, and
  `fix_lerobot_video_banner.py` preserve the dataset preparation utilities
  used by the experiments.
