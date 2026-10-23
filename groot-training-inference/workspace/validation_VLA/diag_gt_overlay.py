import sys
sys.path.insert(0, "/workspace/validation_VLA")
from overlay_openloop_rollout import MujocoOverlayRenderer
import cv2
import numpy as np
import pandas as pd

df = pd.read_parquet("/data/project/cosmos3_g1right_lerobot/data/chunk-000/episode_000000.parquet")
gt = np.stack(df["observation.state"].to_numpy())

cap = cv2.VideoCapture("/data/project/cosmos3_g1right_lerobot/videos/chunk-000/observation.images.ego_view/episode_000000.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
h, w = frames[0].shape[:2]

r = MujocoOverlayRenderer("/workspace/validation_VLA/assets/g1_brainco.xml", w, h, alpha=0.55)
for idx in [0, 20, 40, 60, 80]:
    bg = r.undistort(frames[idx])
    out = r.render_overlay(bg, gt[idx], label=f"GT pose (TUNED cam), frame {idx}")
    cv2.imwrite(f"/workspace/validation_VLA/out/diag_tuned_f{idx}.png", out)
r.close()
print("done")
