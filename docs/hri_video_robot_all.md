# Six RGB model variants

Project directory: [`../hri_video_robot_all/`](../hri_video_robot_all/)

The source code, model configurations, reports, and training source are
included. To keep the submission compact, only the default
`tinyvit_videolightformer` checkpoint is included. It is the same file as the
checkpoint in `hri_video_robot/` and is stored as a hardlink.

## Run the included default model

```bash
cd hri_video_robot_all
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_video.py verify --device auto
./run_on_robot.sh <CAMERA_SERVER_IP> --show
```

## Run another variant

Restore its `model.pt` under `models/<variant>/` first, then select it with the
`--model` option described in the project README. The expected checksums are in
[`external_assets.md`](external_assets.md).

The detailed guide and model comparison are in the
[project README](../hri_video_robot_all/README.md).
Training instructions are in
[`hri_video_robot_all/training_source/README.md`](../hri_video_robot_all/training_source/README.md).
