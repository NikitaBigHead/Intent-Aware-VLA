# Included and external assets

This compact copy keeps source code, configurations, tests, reports, small
classifier checkpoints, and the default RGB checkpoint. The following large
or generated content is intentionally omitted:

- `.git/`, virtual environments, Python/test caches, logs, and temporary files;
- RGB training runs and packaged archive duplicates;
- pose reference episodes;
- duplicate or non-default model weights;
- Teleop documentation videos, generated output, and large training/simulation
  robot assets.
- VLA/GR00T base models, fine-tuned checkpoints, training and test datasets,
  Hugging Face caches, generated evaluation results, and external simulator
  repositories.

## Included default RGB checkpoint

`hri_video_robot/checkpoint/model.pt` and
`hri_video_robot_all/models/tinyvit_videolightformer/model.pt` are two hardlink
names for one 25,235,459-byte file.

SHA-256:

```text
61dd39a8c94c2c2dd70dab9bd7562bebeb7d28d1b8f919fe78dc5caa1e38e8af
```

## Omitted RGB variant checkpoints

| Relative path under `hri_video_robot_all/` | SHA-256 |
|---|---|
| `models/efficientformer_videolightformer/model.pt` | `985ef144c11459ac90746e8cc609b6012ebd995dbd5ac85f3b87c697b99a24c8` |
| `models/efficientformer_vtn/model.pt` | `1caba1add567d50fdee49d26dfa93f107be5a8106340022cd19ee193aee39892` |
| `models/mobilevit_videolightformer/model.pt` | `ddb1e19a53cac2cba6057afdb863d37c84f4f37849660966f1bbcbce604abfe1` |
| `models/mobilevit_vtn/model.pt` | `4ec77bb8d933449e60abe87dd1a9991b0028d295898741b5ecf1820fabe7319e` |
| `models/tinyvit_vtn/model.pt` | `ba6d1e14564c991cfd441bc6d7e32b3858bba3a60b17a2f0a7a178c68db97836` |

From this workspace, restore them with:

```bash
rsync -a ../hri_video_robot_all/models/ hri_video_robot_all/models/
```

Run that command from the `submit/` directory. It is a local maintainer command,
not a reviewer download path.

## Pose-estimation weights

Both pose projects and `hri_video_robot_all/baseline_pose/` use the same two
files:

| Filename | Size | SHA-256 |
|---|---:|---|
| `yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` | 101,409,718 bytes | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` |
| `rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx` | 369,307,483 bytes | `4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd` |

Public sources:

- YOLOX-m archive: <https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip>
- RTMW3D-x ONNX: <https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx>

Place the extracted/downloaded files in the `models/` directory of the pose
project being run. From the original local workspace, all three copies can be
restored without downloading:

```bash
cp -a ../dual_camera_robot/models/. dual_camera_robot/models/
cp -al dual_camera_robot/models/. dual_camera_robot_v2/models/
cp -al dual_camera_robot/models/. hri_video_robot_all/baseline_pose/models/
```

The `-l` copies are hardlinks and therefore do not consume another 449 MiB per
project on the same filesystem.

## Teleop assets

The compact Teleop tree omits these large directories:

- `gear_sonic/data/robots/`
- `gear_sonic/data/assets/`
- `decoupled_wbc/control/robot_model/`
- `docs/source/_static/`
- `media/`, local virtual environments, temporary output, and archives

Restore them from the repository/commit described in
[`teleop_inference_wbc.md`](teleop_inference_wbc.md), then use the project's
`download_from_hf.py` for model checkpoints as documented upstream.

## VLA model artifacts

The directory `VLA-Human-Humanoid-interaction/` contains source code and
configuration only. Restore these separately for the relevant experiment:

- the shared Unitree G1 + BrainCo LeRobot dataset;
- the UnifoLM-VLM base model and converted HDF5/RLDS dataset;
- the Being-H05 base checkpoint;
- the RLDX base checkpoint;
- any fine-tuned checkpoints used for evaluation;
- the external LIBERO/RoboCasa repositories when simulator evaluation is
  required.

See [`vla_models.md`](vla_models.md) for the expected paths and model-specific
entry points.

## Isaac-GR00T artifacts

`groot-training-inference/` excludes the original container environment,
datasets, checkpoints, caches, and external simulator repositories. Recreate
the environment from the original `<GR00T_FINETUNE_IMAGE>` image or from
the included Isaac-GR00T packaging files, download the NVIDIA base model, and
mount the desired LeRobot dataset and fine-tuned checkpoint. See
[`groot_training_inference.md`](groot_training_inference.md).

## RMSE evaluation artifacts

`tactile_test_rmse/` requires the external test dataset and the selected
backend's checkpoint/environment. Generated JSON reports and logs were not
copied. See [`tactile_rmse.md`](tactile_rmse.md).
