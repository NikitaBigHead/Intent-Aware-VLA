#!/usr/bin/env python
"""Open-loop RGB+MuJoCo overlay rollout for cosmos_right13 (GR00T policy).

Runs the trained GR00T policy chunk-by-chunk against REAL ground-truth video
+ state from Lisa's LeRobot dataset (open-loop: each new chunk re-conditions
on the REAL next GT frame/state, not on a generated one -- there is no video
generator in this loop, so re-feeding real dataset video each chunk is the
only way to keep rolling out past the first chunk; this is the pre-hardware
sanity check requested in chat on 2026-09-29). For every predicted frame,
renders the 13 right-arm+hand joints in MuJoCo (g1_brainco.xml,
cam_right_high -- the single ego camera for this rig, see
reference_g1_head_camera memory) and alpha-blends the robot silhouette onto
the REAL ego_view video frame at that index -- same segmentation-mask +
alpha-blend technique as run_ui_app/live_overlay.py's _overlay_render_once()
and run_ui_app/video/mesh_overlay.py (the "SixGrid"-adjacent overlay code in
this repo; those two files are the actual precedent, ported here standalone
since this runs on GPU-02 without the run_ui_app package installed).

Data loading (LeRobotEpisodeLoader, extract_step_data, Gr00tPolicy/PolicyClient,
parse_observation_gr00t/parse_action_gr00t) is reused AS-IS from NVIDIA's own
gr00t.eval.open_loop_eval -- already installed in this container, and it
already computes MSE/MAE the "official" way (see run_mse_mae_eval.sh, a thin
wrapper around that exact script). This file only adds the piece
open_loop_eval.py does not have: a visual RGB overlay (it only plots
per-dimension joint-value curves).

Right-arm+hand joint order / qpos-index technique matches
scripts/sim_replay/sim_replay_unified.py's ACTION_JOINT_NAMES / build_qpos_index
(direct26d mode: action IS the absolute qpos, no IK) -- ACTION_JOINT_NAMES_13D
below is exactly that file's ACTION_JOINT_NAMES[7:14] + [20:26], confirmed to
match Lisa's meta/info.json feature order 1:1 (right_shoulder_pitch_joint...
right_wrist_yaw_joint, then thumb_aux/thumb/index/middle/ring/pinky).

Hand [0,1] -> radians conversion matches run_ui_app/inference/mirror.py's
_hand_fraction_to_radians exactly (same _HAND_RAD_MIN/_HAND_RAD_MAX table,
confirmed there against g1_brainco.xml's own joint <range>s). Arm channels
are used as-is (already radians per gr00t_cosmos_right13_config.py's
docstring) -- unlike live_overlay.py this does NOT apply RIGHT_ARM_OFFSETS,
since those correct a REAL robot's hardware zero-point against the URDF, and
RoboHaMeR's action space is simulation-native.

CAMERA: uses the TUNED chessboard calibration from robot_video2pose's
render_tuned_overlay.py (external/scripts_human2robot/pipeline/), the SAME
constant that file's build_model()/resolve_episode_params() use as the
shared base for every episode profile there. Two earlier guesses were tried
and ruled out first (both empirically, both wrong -- kept here since the
next person touching this will be tempted to retry them):
  1. g1_brainco.xml's own named cameras (cam_right_high, cam_right_wrist,
     cam_left_wrist): rendering the dataset's own GROUND-TRUTH pose through
     all three did not line up with the real ego_view frame in any of them
     (diag_gt_overlay.py, not deployed, run ad hoc 2026-09-29).
  2. Assuming TUNED was real-hardware-only and therefore inapplicable to
     RoboHaMeR's synthetic render -- wrong per user correction (2026-09-29):
     RoboHaMeR's generation is itself built FROM real calibrated-camera
     footage, so its ego_view render uses that same calibrated camera, and
     TUNED is exactly the parameters to use.
TUNED / pose_to_R / the tuned_cam construction (torso_link-relative pos +
quat, fovy from fy, render_width from fx/fy aspect) are ported verbatim from
render_tuned_overlay.py's build_model() -- see build_tuned_camera() below.
Lens undistortion (cv2.undistort with TUNED's K/dist_coeffs) is applied to
every real background frame, and the render is warpAffine-shifted by
(cx-w/2, cy-h/2) before compositing, exactly like run_ui_app/live_overlay.py
's _overlay_render_once() does for the real robot.

Usage (inside the tactile-train container):
  /opt/Isaac-GR00T/.venv/bin/python overlay_openloop_rollout.py \
    --dataset-path /data/project/cosmos3_g1right_lerobot \
    --episode-id 0 \
    --action-horizon 16 --steps 96 \
    --out-dir /workspace/validation_VLA/out
  # (--model-path defaults to the newest complete checkpoint-N/ under
  #  --ckpt-root /workspace/checkpoints/cosmos_right13; pass --use-server
  #  to query a running PolicyClient on --host/--port instead)
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient


# 13D action -> joint name, matches Lisa's info.json feature order exactly.
ACTION_JOINT_NAMES_13D = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_thumb_metacarpal_joint", "right_thumb_proximal_joint",
    "right_index_proximal_joint", "right_middle_proximal_joint",
    "right_ring_proximal_joint", "right_pinky_proximal_joint",
]
# thumb_aux, thumb, index, middle, ring, pinky -- [0,1] fraction -> radians.
_HAND_RAD_MIN = np.zeros(6, dtype=np.float64)
_HAND_RAD_MAX = np.array([1.52, 1.05, 1.47, 1.47, 1.47, 1.47], dtype=np.float64)


def hand_fraction_to_radians(frac: np.ndarray) -> np.ndarray:
    frac = np.clip(np.asarray(frac, dtype=np.float64), 0.0, 1.0)
    return _HAND_RAD_MIN + frac * (_HAND_RAD_MAX - _HAND_RAD_MIN)


def hand_radians_to_fraction(rad: np.ndarray) -> np.ndarray:
    """Inverse of hand_fraction_to_radians -- for sources (e.g. dataset_a_v2's raw
    qpos capture) that store the hand in direct26d joint radians instead of the
    [0,1] fraction cosmos_right13's own training dataset uses."""
    rad = np.asarray(rad, dtype=np.float64)
    return np.clip((rad - _HAND_RAD_MIN) / (_HAND_RAD_MAX - _HAND_RAD_MIN), 0.0, 1.0)


def action13_to_qpos_row(action13: np.ndarray) -> np.ndarray:
    """[13] raw dataset action (7 arm rad + 6 hand [0,1]) -> [13] MuJoCo qpos values."""
    out = np.asarray(action13, dtype=np.float64).copy()
    out[7:13] = hand_fraction_to_radians(out[7:13])
    return out


def build_qpos_index(model: mujoco.MjModel) -> np.ndarray:
    idx = np.zeros(len(ACTION_JOINT_NAMES_13D), dtype=int)
    for i, name in enumerate(ACTION_JOINT_NAMES_13D):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint not found in model: {name}")
        idx[i] = model.jnt_qposadr[jid]
    return idx


# Chessboard calibration, ported verbatim from robot_video2pose's
# external/scripts_human2robot/pipeline/render_tuned_overlay.py TUNED dict.
# offset_x/y/z + pitch/yaw/roll are torso_link-relative extrinsics; fx/fy/cx/cy
# + k1/k2/p1/p2/k3 are the fixed physical lens intrinsics/distortion.
TUNED = {
    "offset_x": 0.05089367658181505,
    "offset_y": 0.01979194460368823,
    "offset_z": 0.4068842946812315,
    "pitch": 62.73136993633539,
    "yaw": 2.807081014807754,
    "roll": 2.733422248132668,
    "fx": 907.8439649,
    "fy": 970.4315490,
    "cx": 656.2385080,
    "cy": 372.8413207,
    "k1": 0.1366030432,
    "k2": -0.4573869777,
    "p1": -0.0034289401,
    "p2": 0.0026317655,
    "k3": 0.4560740813,
}
# OpenCV optical axes expressed in torso_link at zero Euler correction.
_R_REF = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _pose_to_R(p: dict) -> np.ndarray:
    return _Rz(np.radians(p["yaw"])) @ _Ry(np.radians(p["pitch"])) @ _Rx(np.radians(p["roll"])) @ _R_REF


def _is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
    while body_id > 0:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


TUNED_CALIB_WIDTH = 1280  # the sensor width TUNED's fx/fy/cx/cy were fit at
                           # (user-confirmed 2026-09-29). Lisa's ego_view is
                           # 640x392 -- a uniform resize of the whole frame
                           # (also user-confirmed), so fx/fy/cx/cy scale by
                           # width/TUNED_CALIB_WIDTH; extrinsics
                           # (offset_x/y/z, pitch/yaw/roll) are physical
                           # mount pose and don't change with resolution;
                           # k1/k2/p1/p2/k3 are dimensionless in OpenCV's
                           # model and stay valid under uniform image scaling.


def build_tuned_camera(xml_path: Path, width: int, height: int, params: dict = TUNED):
    """Adds a "tuned_cam" to torso_link from the TUNED calibration, exactly
    as render_tuned_overlay.py's build_model() does -- INCLUDING its
    arm-only geom_group restriction (torso/head/legs hidden, group 3;
    arm geoms shown, group 0), which is load-bearing: without it the
    segmentation mask/render includes the whole body and the arm silhouette
    ends up in the wrong place relative to the real frame (found the hard
    way, 2026-09-29 -- the first TUNED attempt skipped this and still didn't
    line up) -- AND rescaling fx/fy/cx/cy for width != TUNED_CALIB_WIDTH
    (second bug found the same day: applying TUNED's intrinsics unscaled to
    a 640-wide render produced a wildly over-zoomed camera, hands filling
    the whole frame -- see README.md's camera section for the empirical
    trail). Returns (model, render_width, K, dist_coeffs, scene_option).
    Render at render_width x height (NOT width x height: fx != fy needs a
    different internal aspect, corrected back to width x height by resize
    afterward, matching the original)."""
    scale = width / TUNED_CALIB_WIDTH
    p = dict(params)
    p["fx"] *= scale
    p["fy"] *= scale
    p["cx"] *= scale
    p["cy"] *= scale
    render_width = max(2, round(width * p["fy"] / p["fx"]))
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.visual.global_.offwidth = render_width
    spec.visual.global_.offheight = height
    torso = next(b for b in spec.bodies if b.name == "torso_link")
    R_cv = _pose_to_R(p)
    R_mj = R_cv @ np.diag([1.0, -1.0, -1.0])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R_mj.flatten())
    fovy = 2 * np.degrees(np.arctan(height / (2 * p["fy"])))
    torso.add_camera(name="tuned_cam", pos=[p["offset_x"], p["offset_y"], p["offset_z"]],
                      quat=quat.tolist(), fovy=fovy)
    model = spec.compile()

    roots = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
             for name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link")]
    arm_bodies = {b for b in range(model.nbody) if any(_is_descendant(model, b, r) for r in roots)}
    arm_geoms = np.array([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies],
                          dtype=np.int32)
    model.geom_group[:] = 3
    model.geom_group[arm_geoms] = 0
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1

    K = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]], dtype=np.float32)
    D = np.array([p["k1"], p["k2"], p["p1"], p["p2"], p["k3"]], dtype=np.float32)
    return model, render_width, K, D, option


class MujocoOverlayRenderer:
    """Segmentation-mask + alpha-blend overlay using the TUNED calibrated
    camera -- same technique (undistort bg + cx/cy warpAffine-shift the
    render) as run_ui_app/live_overlay.py's _overlay_render_once()."""

    def __init__(self, xml_path: Path, width: int, height: int, alpha: float = 0.55):
        self.model, render_width, self.K, self.D, self.option = build_tuned_camera(xml_path, width, height)
        self.data = mujoco.MjData(self.model)
        self.qidx = build_qpos_index(self.model)
        self.alpha = alpha
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=render_width)
        cx, cy = self.K[0, 2], self.K[1, 2]  # already scaled for this width/height, see build_tuned_camera
        self._shift = np.float32([[1, 0, cx - width / 2], [0, 1, cy - height / 2]])

    def undistort(self, bg_bgr: np.ndarray) -> np.ndarray:
        return cv2.undistort(bg_bgr, self.K, self.D)

    def _set_pose(self, action13: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[self.qidx] = action13_to_qpos_row(action13)
        mujoco.mj_forward(self.model, self.data)

    def render_overlay(self, bg_bgr_undistorted: np.ndarray, action13: np.ndarray, *, label: str = "") -> np.ndarray:
        self._set_pose(action13)
        self.renderer.update_scene(self.data, camera="tuned_cam", scene_option=self.option)
        robot_rgb = self.renderer.render()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="tuned_cam", scene_option=self.option)
        seg = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        w, h = self.width, self.height
        robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
        robot_bgr = cv2.resize(robot_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize((seg[:, :, 0] >= 0).astype(np.uint8) * 255, (w, h),
                           interpolation=cv2.INTER_NEAREST)
        robot_bgr = cv2.warpAffine(robot_bgr, self._shift, (w, h))
        mask = cv2.warpAffine(mask, self._shift, (w, h)) > 127

        out = bg_bgr_undistorted.copy()
        blend = (self.alpha * robot_bgr.astype(np.float32)
                 + (1 - self.alpha) * bg_bgr_undistorted.astype(np.float32)).astype(np.uint8)
        out[mask] = blend[mask]
        if label:
            cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    def render_mesh(self, action13: np.ndarray) -> np.ndarray:
        """Robot render only (BGR, tuned_cam, shift-corrected) -- no
        background blend. Used by --mode panel."""
        self._set_pose(action13)
        self.renderer.update_scene(self.data, camera="tuned_cam", scene_option=self.option)
        rgb = self.renderer.render()
        w, h = self.width, self.height
        bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.warpAffine(bgr, self._shift, (w, h))

    def close(self):
        self.renderer.close()


def make_panel(bg_bgr: np.ndarray, gt_mesh_bgr: np.ndarray, pred_mesh_bgr: np.ndarray,
                *, label: str = "") -> np.ndarray:
    """3-wide strip: real GT video | MuJoCo(GT pose) | MuJoCo(predicted pose).
    Camera-independent -- no claim of pixel alignment between panels, each
    panel is comparable to the others on its own terms (see module
    docstring's CAMERA CALIBRATION section)."""
    h, w = bg_bgr.shape[:2]
    panels = [bg_bgr.copy(), gt_mesh_bgr.copy(), pred_mesh_bgr.copy()]
    headers = ["real (GT video)", "MuJoCo: GT pose", "MuJoCo: predicted"]
    for p, htext in zip(panels, headers):
        cv2.rectangle(p, (0, 0), (w - 1, 20), (0, 0, 0), -1)
        cv2.putText(p, htext, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    out = np.concatenate(panels, axis=1)
    if label:
        cv2.putText(out, label, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def episode_video_path(dataset_path: Path, info: dict, episode_id: int, video_key: str) -> Path:
    return dataset_path / info["video_path"].format(episode_chunk=0, episode_index=episode_id, video_key=video_key)


def episode_data_path(dataset_path: Path, info: dict, episode_id: int) -> Path:
    return dataset_path / info["data_path"].format(episode_chunk=0, episode_index=episode_id)


def read_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()  # BGR, cv2-native
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_policy(model_path: str | None, ckpt_root: str, use_server: bool, host: str, port: int):
    """(policy, model_tag, model_desc) -- factored out of main() so other
    scripts against this same checkpoint (e.g. overlay_dataset_v2_rollout.py)
    don't reimplement the newest-checkpoint resolution / PolicyClient choice."""
    if use_server:
        return PolicyClient(host=host, port=port), "server", f"server {host}:{port}"
    if model_path is None:
        ckpts = [c for c in Path(ckpt_root).glob("checkpoint-*") if (c / "trainer_state.json").exists()]
        if not ckpts:
            raise SystemExit(f"no complete checkpoint-N/ under {ckpt_root}")
        ckpts.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1)))
        model_path = str(ckpts[-1])
    import torch
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, model_path=model_path,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    return policy, Path(model_path).name, model_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default="/data/project/cosmos3_g1right_lerobot")
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--model-path", default=None,
                     help="checkpoint dir; default: newest complete checkpoint-N/ under --ckpt-root")
    ap.add_argument("--ckpt-root", default="/workspace/checkpoints/cosmos_right13")
    ap.add_argument("--use-server", action="store_true",
                     help="query a running PolicyClient on --host/--port instead of --model-path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--xml-path", default="/workspace/validation_VLA/assets/g1_brainco.xml")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--mode", choices=["panel", "overlay"], default="panel",
                     help="panel (default, camera-independent, RECOMMENDED until RoboHaMeR's real "
                          "camera calibration is available -- see module docstring) or overlay "
                          "(alpha-blend on the real frame -- NOT pixel-calibrated yet)")
    ap.add_argument("--closed-loop", action="store_true",
                     help="Do NOT recondition state.* on the real GT state at each chunk boundary "
                          "-- carry forward the policy's OWN last predicted joint values instead, "
                          "so the hands keep going from wherever the previous chunk actually left "
                          "them (no snap-back to ground truth). Only the very first chunk (step 0) "
                          "still seeds from the real initial state, since there is no prior "
                          "prediction yet. The video/image conditioning still comes from the real "
                          "dataset frame at that index either way -- there is no video generator "
                          "in this loop (see module docstring), so this flag only removes the "
                          "state reset, not the video one.")
    ap.add_argument("--out-dir", default="/workspace/validation_VLA/out")
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    video_key = next(k for k in info["features"] if k.startswith("observation.images."))
    fps = info["fps"]

    policy, model_tag, model_desc = load_policy(args.model_path, args.ckpt_root, args.use_server,
                                                 args.host, args.port)
    print(f"[overlay] policy: {model_desc}", flush=True)

    modality_configs = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(dataset_path), modality_configs=modality_configs,
                                   video_backend="torchcodec", video_backend_kwargs=None)
    traj = loader[args.episode_id]
    actual_steps = min(args.steps, len(traj))
    print(f"[overlay] episode {args.episode_id}: {len(traj)} frames, evaluating {actual_steps}", flush=True)

    video_path = episode_video_path(dataset_path, info, args.episode_id, video_key)
    bg_frames = read_all_frames(video_path)
    if len(bg_frames) < actual_steps:
        print(f"[overlay] WARNING: video has {len(bg_frames)} frames < actual_steps={actual_steps}, clipping")
        actual_steps = min(actual_steps, len(bg_frames))
    h, w = bg_frames[0].shape[:2]

    df = pd.read_parquet(episode_data_path(dataset_path, info, args.episode_id))
    gt_action = np.stack(df["action"].to_numpy())[:actual_steps]

    renderer = MujocoOverlayRenderer(Path(args.xml_path), w, h, alpha=args.alpha)
    bg_frames = [renderer.undistort(f) for f in bg_frames]

    action_keys = loader.modality_configs["action"].modality_keys
    inference_modality_configs = deepcopy(loader.modality_configs)
    inference_modality_configs.pop("action")

    pred_action_across_time = []
    out_frames = []
    carry_state = None  # dict[str, np.ndarray], raw action-space units, set once closed-loop kicks in
    for step_count in range(0, actual_steps, args.action_horizon):
        data_point = extract_step_data(traj, step_count, inference_modality_configs, EmbodimentTag.NEW_EMBODIMENT)
        obs = {}
        for k, v in data_point.states.items():
            if args.closed_loop and carry_state is not None and k in carry_state:
                v = np.tile(carry_state[k], v.shape[:-1] + (1,)) if v.ndim > 1 else carry_state[k]
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        action_chunk_raw, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(action_chunk_raw)

        if args.closed_loop:
            # Seed next chunk's state.* from THIS chunk's last predicted frame (action and
            # state share identical feature layout/units per meta/info.json -- see README) --
            # this is what makes the rollout continue from wherever the hands actually ended
            # up instead of snapping back to the real GT trajectory.
            carry_state = {key: np.atleast_1d(np.asarray(action_chunk[f"action.{key}"])[-1]).astype(np.float32)
                           for key in action_keys}

        for j in range(args.action_horizon):
            frame_idx = step_count + j
            if frame_idx >= actual_steps:
                break
            pred13 = np.concatenate([
                np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j]) for key in action_keys
            ], axis=0)
            pred_action_across_time.append(pred13)
            is_boundary = (j == 0)
            label = f"step {frame_idx}/{actual_steps}"
            if is_boundary:
                label += ("  [NEW CHUNK -- carried from policy]" if args.closed_loop
                          else "  [NEW CHUNK -- real GT reconditioned]")
            if args.mode == "overlay":
                frame = renderer.render_overlay(bg_frames[frame_idx], pred13, label=label)
            else:
                gt_mesh = renderer.render_mesh(gt_action[frame_idx])
                pred_mesh = renderer.render_mesh(pred13)
                frame = make_panel(bg_frames[frame_idx], gt_mesh, pred_mesh, label=label)
            if is_boundary:
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 3)
            out_frames.append(frame)

    renderer.close()

    pred_action_across_time = np.array(pred_action_across_time)
    n = min(len(pred_action_across_time), len(gt_action))
    mse = float(np.mean((gt_action[:n] - pred_action_across_time[:n]) ** 2))
    mae = float(np.mean(np.abs(gt_action[:n] - pred_action_across_time[:n])))
    per_dim_rmse = {name: rmse(gt_action[:n, i], pred_action_across_time[:n, i])
                     for i, name in enumerate(ACTION_JOINT_NAMES_13D)}

    tag_suffix = "_closedloop" if args.closed_loop else ""
    out_mp4 = out_dir / f"{args.mode}_ep{args.episode_id}_{model_tag}{tag_suffix}.mp4"
    out_h, out_w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    for f in out_frames:
        writer.write(f)
    writer.release()

    try:
        import imageio.v2 as imageio
        gif_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out_frames[:: max(1, round(fps / 8))]]
        imageio.mimsave(str(out_mp4.with_suffix(".gif")), gif_frames, fps=8, loop=0)
    except Exception as e:
        print(f"[overlay] gif skipped: {e}")

    metrics = {
        "model": model_desc, "episode_id": args.episode_id, "n_frames": n,
        "closed_loop": args.closed_loop,
        "mse": mse, "mae": mae, "rmse_overall": rmse(gt_action[:n], pred_action_across_time[:n]),
        "per_dim_rmse": per_dim_rmse,
    }
    (out_dir / f"{args.mode}_ep{args.episode_id}_{model_tag}{tag_suffix}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[overlay] wrote {out_mp4} ({len(out_frames)} frames)")
    print(f"[overlay] MSE={mse:.5f} MAE={mae:.5f} RMSE={metrics['rmse_overall']:.5f}")
    for name, v in per_dim_rmse.items():
        print(f"    {name}: rmse={v:.4f}")


if __name__ == "__main__":
    main()
