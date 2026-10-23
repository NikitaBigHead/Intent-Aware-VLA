# Updated pose-based baseline

Project directory: [`../dual_camera_robot_v2/`](../dual_camera_robot_v2/)

This is the updated pose-based classifier. Its classifier checkpoint is
included. The two large pose-estimation ONNX files and reference episodes are
omitted because the ONNX files are identical to those used by the original
baseline.

Restore the two files listed under **Pose-estimation weights** in
[`external_assets.md`](external_assets.md) into `dual_camera_robot_v2/models/`,
then run:

```bash
cd dual_camera_robot_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_dualcam.py verify --device auto
./run_on_robot.sh <CAMERA_SERVER_IP> --show
```

See the [project README](../dual_camera_robot_v2/README.md)
for the full protocol, model behavior, and validation results.
