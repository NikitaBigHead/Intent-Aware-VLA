#!/usr/bin/env python
"""Same overlay rollout as overlay_openloop_rollout.py, but against REAL raw
camera footage (dataset_a_v2) instead of the RoboHaMeR-generated video in
cosmos_right13's own training dataset (cosmos3_g1right_lerobot).

Source: /root/dataset_a/data_v2/<session>/episode_<session>.mp4 (raw robot
camera, 1280x720 @ 30fps -- the TUNED calibration's OWN native resolution,
see overlay_openloop_rollout.py's TUNED_CALIB_WIDTH comment, so this script
needs none of that resize-scaling hack) + data.json (per-frame states/
actions, 26D convention: left_arm(7) + right_arm(7) + left_ee(6) +
right_ee(6) qpos, all RADIANS -- see
src/cosmos_transfer/data/dataset_a_raw_dataset.py's _extract_26d/_STATE_KEYS,
same canonical order as scripts/sim_replay/sim_replay_unified.py's
ACTION_JOINT_NAMES). Episode boundaries come from data_v2/timecodes.txt
("mm:ss-mm:ss" lines, blank-line-separated blocks headed "N:" per session --
sessions 8/9/11 repeat their block ~6x byte-identical in this file per
docs/g1_brainco_datasets.md, only the first copy is used here).

Right-arm+hand slice = indices [7:14] (right_arm) + [20:26] (right_ee) of
the 26D vector -- name-matches cosmos_right13's ACTION_JOINT_NAMES_13D 1:1.

UNIT MISMATCH (the one real adaptation needed): dataset_a_v2's right_ee.qpos is
raw MuJoCo joint RADIANS (direct26d convention), while cosmos_right13's own
training data stores the hand channels as a [0,1] FRACTION (thumb_aux/thumb/
index/middle/ring/pinky -- see overlay_openloop_rollout.py's README section).
The policy's state.joint_positions/action.joint_positions are trained on that
fraction convention, so hand radians pulled from dataset_a_v2 are inverted
through overlay_openloop_rollout.hand_radians_to_fraction (exact inverse of
that script's hand_fraction_to_radians) before being fed to the policy or
treated as "gt_action" -- everything downstream (MuJoCo rendering, metrics)
then stays in the exact same convention overlay_openloop_rollout.py already
uses, unmodified.

CAMERA: video observation fed to the policy is the raw undistorted RGB frame
as-is (no manual resize to 640x392) -- GR00T's own baked-in video transform
resizes to its trained input size regardless of what's handed to it, so a
manual resize here would be redundant, not more correct (the source's 1280x720
aspect differs from the training video's 640x392 aspect, so *some* squish is
unavoidable either way -- flagged here, not silently hidden). The overlay
MESH render, separately, uses the TUNED camera UNSCALED (scale=1.0, since
1280 IS TUNED_CALIB_WIDTH) -- TUNED's own cx/cy (656, 373) sit almost exactly
at this frame's true center (640, 360), which is corroborating evidence this
calibration was fit on frames like this one, not just the resized synthetic
video overlay_openloop_rollout.py normally runs against.

No LeRobotEpisodeLoader / ActionBaseDataset here (this raw format has nothing
to do with LeRobot parquet) -- the policy's own modality_configs (single key
"joint_positions" for state/action, "ego_view" for video, per
gr00t_cosmos_right13_config.py) are built directly into the obs dict, then
run through gr00t.eval.open_loop_eval's parse_observation_gr00t/
parse_action_gr00t exactly as overlay_openloop_rollout.py does.

Usage (inside the tactile-train container):
  /opt/Isaac-GR00T/.venv/bin/python overlay_dataset_v2_rollout.py \
    --session 5 --episode-in-session 0 --closed-loop --mode overlay \
    --action-horizon 16 --out-dir /workspace/validation_VLA/out
  # --model-path defaults to the newest complete checkpoint-N/ under
  #   --ckpt-root /workspace/checkpoints/cosmos_right13 (see load_policy()
  #   in overlay_openloop_rollout.py, reused as-is here)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t

from overlay_openloop_rollout import (
    ACTION_JOINT_NAMES_13D,
    MujocoOverlayRenderer,
    hand_radians_to_fraction,
    load_policy,
    make_panel,
    rmse,
)

# 26D convention (left_arm, right_arm, left_ee, right_ee), matches
# src/cosmos_transfer/data/dataset_a_raw_dataset.py's _STATE_KEYS/_extract_26d.
_STATE_KEYS_26D = ("left_arm", "right_arm", "left_ee", "right_ee")


def extract_right13(frame: dict, field: str) -> np.ndarray:
    """[13] right_arm(7 rad) + right_ee(6 rad) from one dataset_a_v2 data.json
    frame's states/actions block -- hand still in raw RADIANS at this point,
    NOT yet converted to the [0,1] fraction convention (caller's job)."""
    arm = frame[field]["right_arm"]["qpos"]
    hand = frame[field]["right_ee"]["qpos"]
    return np.asarray(list(arm) + list(hand), dtype=np.float64)


def parse_timecodes(path: Path, session: str) -> list[tuple[float, float]]:
    """[(start_sec, end_sec), ...] for the FIRST block belonging to session
    "<session>" in timecodes.txt. The file's blank-line-separated blocks
    alternate header-only ("N:", one line) and timecode-list blocks -- a
    session's header and its own list are NOT in the same block (there's a
    blank line between them too), so each header is matched against the
    block that immediately follows it. Sessions 8/9/11 repeat their
    header+list pair ~6x byte-identical in this file per
    docs/g1_brainco_datasets.md -- only the first copy is used here."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", path.read_text()) if b.strip()]
    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) == 1 and lines[0].rstrip(":").isdigit() and lines[0].rstrip(":") == session:
            episodes = []
            for line in blocks[i + 1].splitlines():
                m = re.match(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)", line.strip())
                if m:
                    m1, s1, m2, s2 = map(int, m.groups())
                    episodes.append((m1 * 60 + s1, m2 * 60 + s2))
            return episodes
    raise ValueError(f"session {session!r} not found in {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_a-root", default="/root/dataset_a/data_v2")
    ap.add_argument("--session", default="5", help="one of 5/6/8/9/11")
    ap.add_argument("--episode-in-session", type=int, default=0,
                     help="index into that session's timecodes.txt block")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--ckpt-root", default="/workspace/checkpoints/cosmos_right13")
    ap.add_argument("--use-server", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None,
                     help="default: whole episode window from timecodes.txt")
    ap.add_argument("--xml-path", default="/workspace/validation_VLA/assets/g1_brainco.xml")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--mode", choices=["panel", "overlay"], default="overlay")
    ap.add_argument("--closed-loop", action="store_true",
                     help="see overlay_openloop_rollout.py's own --closed-loop docs -- same "
                          "carry-forward-the-policy's-own-prediction behavior, here too")
    ap.add_argument("--policy-fps", type=float, default=15.0,
                     help="cosmos_right13's own native decimation rate (POLICY_HZ in "
                          "cosmos_right13_bridge.py, matches training meta/info.json 'fps': 15). "
                          "dataset_a_v2's raw video is recorded at a DIFFERENT fps (30 for session "
                          "5) -- each predicted policy step spans round(video_fps/policy_fps) raw "
                          "frames of real time, not 1 frame. Mapping 1 predicted step -> 1 raw "
                          "frame (the original bug here, found 2026-09-30 from a visible "
                          "forward-then-snap-back artifact matching the live robot's own "
                          "symptom) silently played the video at half the model's assumed rate: "
                          "the render for policy step j showed the state the model predicts "
                          "16/15s into the future while conditioned on video only 16/30s in --"
                          "i.e. it visibly ran ahead of the real footage before each chunk's "
                          "real-GT reconditioning (or, in --closed-loop, before the next chunk's "
                          "carried state) pulled it back. See frame_stride below.")
    ap.add_argument("--out-dir", default="/workspace/validation_VLA/out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_dir = Path(args.dataset_a_root) / args.session

    meta = json.loads((session_dir / "data.json").read_text())
    fps = float(meta["info"]["image"]["fps"])
    frame_stride = max(1, round(fps / args.policy_fps))
    print(f"[overlay] video fps={fps}, policy fps={args.policy_fps} -> frame_stride={frame_stride} "
          f"(1 predicted step = {frame_stride} raw video frame(s) of real time)", flush=True)
    task_text = meta["text"]["goal"]
    frames_meta = meta["data"]
    n = len(frames_meta)

    # [n, 13]: cols 0:7 = right arm radians, cols 7:13 = hand FRACTION
    # (converted from raw radians right here -- see module docstring).
    right13 = np.zeros((n, 13), dtype=np.float64)
    for f in frames_meta:
        i = f["idx"]
        raw = extract_right13(f, "actions")  # "actions" = commanded target, matches
        right13[i, :7] = raw[:7]             # overlay_openloop_rollout.py's gt_action
        right13[i, 7:] = hand_radians_to_fraction(raw[7:])  # (dataset "action" column)

    episodes = parse_timecodes(Path(args.dataset_a_root) / "timecodes.txt", args.session)
    if not (0 <= args.episode_in_session < len(episodes)):
        raise SystemExit(f"session {args.session} has {len(episodes)} episodes "
                          f"(timecodes.txt), got --episode-in-session {args.episode_in_session}")
    start_sec, end_sec = episodes[args.episode_in_session]
    start_frame = round(start_sec * fps)
    end_frame = min(round(end_sec * fps), n)
    window = end_frame - start_frame
    actual_steps = min(args.steps, window) if args.steps else window
    print(f"[overlay] session {args.session} episode {args.episode_in_session}: "
          f"[{start_sec}s-{end_sec}s) -> frames [{start_frame}, {start_frame + actual_steps}), "
          f"task={task_text!r}", flush=True)

    video_path = session_dir / f"episode_{int(args.session):04d}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    bg_frames_bgr = []
    for _ in range(actual_steps):
        ok, bgr = cap.read()
        if not ok:
            break
        bg_frames_bgr.append(bgr)
    cap.release()
    if len(bg_frames_bgr) < actual_steps:
        print(f"[overlay] WARNING: video yielded {len(bg_frames_bgr)} frames < "
              f"actual_steps={actual_steps}, clipping")
        actual_steps = len(bg_frames_bgr)
    h, w = bg_frames_bgr[0].shape[:2]
    gt_action = right13[start_frame:start_frame + actual_steps]

    policy, model_tag, model_desc = load_policy(args.model_path, args.ckpt_root, args.use_server,
                                                 args.host, args.port)
    print(f"[overlay] policy: {model_desc}", flush=True)
    modality_configs = policy.get_modality_config()
    action_keys = modality_configs["action"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    video_keys = modality_configs["video"].modality_keys
    language_keys = modality_configs["language"].modality_keys

    renderer = MujocoOverlayRenderer(Path(args.xml_path), w, h, alpha=args.alpha)
    bg_frames = [renderer.undistort(f) for f in bg_frames_bgr]  # BGR, undistorted (TUNED, scale=1.0 here)

    pred_action_across_time = []   # one entry per PREDICTED policy step (policy_fps cadence)
    gt_action_at_pred_time = []    # matching GT sample at that same real time -- for a fair MSE
    out_frames = []                # one entry per RAW video frame (fps cadence) -- unchanged cadence/length
    carry_state = None
    chunk_span = args.action_horizon * frame_stride  # raw frames of real time one 16-step chunk covers
    for step_count in range(0, actual_steps, chunk_span):
        if args.closed_loop and carry_state is not None:
            state13 = carry_state
        else:
            state13 = gt_action[step_count].astype(np.float32)

        obs = {}
        for k in state_keys:
            obs[f"state.{k}"] = state13.astype(np.float32)[np.newaxis]  # (T=1, D), policy asserts ndim==3 after batching
        frame_rgb = cv2.cvtColor(bg_frames[step_count], cv2.COLOR_BGR2RGB)
        for k in video_keys:
            obs[f"video.{k}"] = frame_rgb[np.newaxis]  # (T=1, H, W, 3), policy asserts ndim==5 after batching
        for k in language_keys:
            obs[k] = task_text
        parsed_obs = parse_observation_gr00t(obs, modality_configs)
        action_chunk_raw, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(action_chunk_raw)

        if args.closed_loop:
            carry_state = np.concatenate([
                np.atleast_1d(np.asarray(action_chunk[f"action.{key}"])[-1]) for key in action_keys
            ], axis=0).astype(np.float32)

        for j in range(args.action_horizon):
            # raw_idx = the video frame whose real timestamp matches this predicted step
            # (j steps of 1/policy_fps each into the chunk) -- NOT step_count+j, which was
            # the original 1:1 bug (see --policy-fps help). pred13 is held zero-order across
            # the frame_stride raw frames that fall within this one predicted step's real-time
            # span, same "hold the last command" idea cosmos_right13_bridge.py's own 15Hz->50Hz
            # interpolation refines further -- not reproduced here, out of scope for this check.
            raw_idx = step_count + j * frame_stride
            if raw_idx >= actual_steps:
                break
            pred13 = np.concatenate([
                np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j]) for key in action_keys
            ], axis=0)
            pred_action_across_time.append(pred13)
            gt_action_at_pred_time.append(gt_action[raw_idx])
            is_boundary = (j == 0)
            label = (f"dataset_a_v2 s{args.session}e{args.episode_in_session} "
                     f"frame {raw_idx}/{actual_steps} (policy step {step_count // frame_stride + j})")
            if is_boundary:
                label += ("  [NEW CHUNK -- carried from policy]" if args.closed_loop
                          else "  [NEW CHUNK -- real GT reconditioned]")
            for k in range(frame_stride):
                frame_idx = raw_idx + k
                if frame_idx >= actual_steps:
                    break
                if args.mode == "overlay":
                    frame = renderer.render_overlay(bg_frames[frame_idx], pred13, label=label)
                else:
                    gt_mesh = renderer.render_mesh(gt_action[frame_idx])
                    pred_mesh = renderer.render_mesh(pred13)
                    frame = make_panel(bg_frames[frame_idx], gt_mesh, pred_mesh, label=label)
                if is_boundary and k == 0:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 3)
                out_frames.append(frame)

    renderer.close()

    pred_action_across_time = np.array(pred_action_across_time)
    gt_action_at_pred_time = np.array(gt_action_at_pred_time)
    n_out = min(len(pred_action_across_time), len(gt_action_at_pred_time))
    mse = float(np.mean((gt_action_at_pred_time[:n_out] - pred_action_across_time[:n_out]) ** 2))
    mae = float(np.mean(np.abs(gt_action_at_pred_time[:n_out] - pred_action_across_time[:n_out])))
    per_dim_rmse = {name: rmse(gt_action_at_pred_time[:n_out, i], pred_action_across_time[:n_out, i])
                     for i, name in enumerate(ACTION_JOINT_NAMES_13D)}

    tag_suffix = "_closedloop" if args.closed_loop else ""
    tag = f"dataset_av2_s{args.session}e{args.episode_in_session}"
    out_mp4 = out_dir / f"{args.mode}_{tag}_{model_tag}{tag_suffix}.mp4"
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
        "model": model_desc, "source": "dataset_a_v2", "session": args.session,
        "episode_in_session": args.episode_in_session, "n_frames": n_out,
        "closed_loop": args.closed_loop,
        "mse": mse, "mae": mae, "rmse_overall": rmse(gt_action_at_pred_time[:n_out], pred_action_across_time[:n_out]),
        "frame_stride": frame_stride, "video_fps": fps, "policy_fps": args.policy_fps,
        "per_dim_rmse": per_dim_rmse,
    }
    (out_dir / f"{args.mode}_{tag}_{model_tag}{tag_suffix}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[overlay] wrote {out_mp4} ({len(out_frames)} frames)")
    print(f"[overlay] MSE={mse:.5f} MAE={mae:.5f} RMSE={metrics['rmse_overall']:.5f}")
    for name, v in per_dim_rmse.items():
        print(f"    {name}: rmse={v:.4f}")


if __name__ == "__main__":
    main()
