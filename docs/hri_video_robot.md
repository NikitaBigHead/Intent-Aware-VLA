# Default RGB intent model

Project directory: [`../hri_video_robot/`](../hri_video_robot/)

This is the self-contained default RGB model. Its trained checkpoint is
included at `hri_video_robot/checkpoint/model.pt`.

## Setup and verification

Use Python 3.12. Install a platform-compatible PyTorch/torchvision pair first,
then install the remaining dependencies:

```bash
cd hri_video_robot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_video.py verify --device auto
```

For a CPU-only check, pass `--device cpu`. On Jetson, use the PyTorch build
matching the installed JetPack release.

## Run against the robot camera server

```bash
./run_on_robot.sh <CAMERA_SERVER_IP> --show
```

The model consumes synchronized `head` and `ego_view` RGB streams over ZMQ and
predicts `no_interaction`, `hug`, `handshake`, or `fist_bump`. It does not send
motion commands.

The full operating guide, ZMQ schema, thresholds, replay mode, VLA bridge, and
validation notes are in the [project README](../hri_video_robot/README.md).
