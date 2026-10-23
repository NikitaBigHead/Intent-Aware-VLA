# Original pose-based baseline

Project directory: [`../dual_camera_robot/`](../dual_camera_robot/)

This snapshot includes the source code, trained intent-classifier checkpoint,
tests, and dependency metadata. Large RTMW3D/YOLOX ONNX pose models and
reference episodes are omitted from the compact copy.

## Restore models and run

Restore the two files listed under **Pose-estimation weights** in
[`external_assets.md`](external_assets.md) into `dual_camera_robot/models/`.
Then:

```bash
cd dual_camera_robot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_dualcam.py verify --device auto
./run_on_robot.sh <CAMERA_SERVER_IP> --show
```

See the [project README](../dual_camera_robot/README.md) for
the camera protocol, replay/verification commands, thresholds, and known model
limitations. The compact copy includes local working-tree changes relative to
base commit `d8595f6`.
