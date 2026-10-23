import sys
sys.path.insert(0, "/workspace/validation_VLA")
from overlay_openloop_rollout import build_tuned_camera, build_qpos_index, action13_to_qpos_row
import cv2
import mujoco
import numpy as np
import pandas as pd

df = pd.read_parquet("/data/project/cosmos3_g1right_lerobot/data/chunk-000/episode_000000.parquet")
gt = np.stack(df["observation.state"].to_numpy())

model, render_width, K, D, option = build_tuned_camera("/workspace/validation_VLA/assets/g1_brainco.xml", 640, 392)
data = mujoco.MjData(model)
qidx = build_qpos_index(model)
data.qpos[:] = 0.0
data.qpos[qidx] = action13_to_qpos_row(gt[0])
mujoco.mj_forward(model, data)

print("nbody:", model.nbody, "render_width:", render_width)
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tuned_cam")
print("torso world pos:", data.xpos[torso_id])
print("tuned_cam world pos:", data.cam_xpos[cam_id])
print("tuned_cam world mat:\n", data.cam_xmat[cam_id].reshape(3,3))
rsh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")
print("right_shoulder_pitch_link world pos:", data.xpos[rsh])

# full-body, no geom-group filter, no undistort/shift -- raw sanity check
with mujoco.Renderer(model, height=392, width=render_width) as r:
    full_option = mujoco.MjvOption()  # default: all groups visible
    r.update_scene(data, camera="tuned_cam", scene_option=full_option)
    img = r.render()
    cv2.imwrite("/workspace/validation_VLA/out/diag_raw_full.png", cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (640, 392)))

    r.update_scene(data, camera="tuned_cam", scene_option=option)
    img2 = r.render()
    cv2.imwrite("/workspace/validation_VLA/out/diag_raw_armonly.png", cv2.resize(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), (640, 392)))
print("done")
