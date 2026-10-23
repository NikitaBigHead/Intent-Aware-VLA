# validation_VLA — pre-hardware sanity check for `cosmos_right13` (Lisa's GR00T policy, GPU-02)

Two scripts, both run **inside the `tactile-train` container on GPU-02**
(`/opt/Isaac-GR00T/.venv/bin/python`, `MUJOCO_GL=egl`). Deployed at
`/root/tactile_ws/validation_VLA/` on the host = `/workspace/validation_VLA/`
in the container.

## 1. `run_mse_mae_eval.sh` — numeric GT-vs-predicted action quality

This does **not reimplement** anything: it's a thin wrapper around NVIDIA's
own `/opt/Isaac-GR00T/gr00t/eval/open_loop_eval.py`, already installed in
this container. That script already does the real thing — queries the
policy chunk-by-chunk (`--action_horizon`), compares against GT actions from
the LeRobot dataset, prints unnormalized MSE/MAE, and saves a per-dimension
GT/state/pred plot. Checked before writing anything new here (per the
"if it doesn't already exist on GPU" ask) — it exists, so this is just a
one-liner with cosmos_right13's paths filled in.

```bash
docker exec tactile-train bash /workspace/validation_VLA/run_mse_mae_eval.sh
# or a specific checkpoint + episodes:
docker exec tactile-train bash /workspace/validation_VLA/run_mse_mae_eval.sh \
  /workspace/checkpoints/cosmos_right13/checkpoint-7500 0 1 2
```

Confirmed working 2026-09-29 against checkpoint-7500, episode 0 (96 steps,
action_horizon=16): MSE=0.00403, MAE=0.04208 (dataset-native units: arm
radians + hand [0,1] fraction, mixed in one average).

## 2. `overlay_openloop_rollout.py` — the visual piece `open_loop_eval.py` doesn't have

Rolls the policy out **open-loop** against a real episode: query the policy
for a 16-frame action chunk, render/compare those 16 predicted frames, then
re-query using the **real** next GT frame/state from the dataset (there's no
video generator here, so re-feeding real dataset video is the only way to
keep going past the first chunk — this is a pre-hardware sanity check, not a
closed-loop simulation).

```bash
docker exec -e MUJOCO_GL=egl tactile-train /opt/Isaac-GR00T/.venv/bin/python \
  /workspace/validation_VLA/overlay_openloop_rollout.py \
  --episode-id 0 --steps 96 --action-horizon 16 \
  --out-dir /workspace/validation_VLA/out
# --model-path defaults to the newest complete checkpoint-N/ under
# /workspace/checkpoints/cosmos_right13 (works fine while the watcher +
# training keep rotating checkpoints underneath it -- it re-resolves at
# each invocation, not once at import time)
```

Writes `panel_ep<id>_<checkpoint>.mp4` (+ `.gif`) and a `_metrics.json` with
the same MSE/MAE/RMSE (overall + per-joint) computed independently, as a
cross-check against `run_mse_mae_eval.sh`'s numbers.

### `--closed-loop`: no GT reset at chunk boundaries

By default, every new chunk re-conditions `state.*` on the REAL GT state at
that frame index (see module docstring) -- so even if the policy drifted
during the previous chunk, it gets snapped back to the true trajectory at
every boundary. Pass `--closed-loop` to disable that snap-back: the state fed
into the next chunk is the policy's OWN last predicted joint values (arm rad
+ hand fraction), carried forward chunk-to-chunk, so the hands keep going
from wherever they actually ended up. Only the very first chunk (step 0)
still seeds from the real initial state (there is no prior prediction to
carry at t=0). Output files get a `_closedloop` suffix so they don't
overwrite the default GT-reconditioned run.

Note this only removes the *state* reset -- the video/image conditioning
still comes from the real dataset frame at each index either way, since
there is no video generator in this loop (re-feeding real dataset video is
the only way to keep rolling out past the first chunk at all). So under
`--closed-loop`, visual conditioning and proprioceptive conditioning can
diverge once the policy drifts -- that's expected and is exactly what this
mode is for probing.

```bash
docker exec -e MUJOCO_GL=egl tactile-train /opt/Isaac-GR00T/.venv/bin/python \
  /workspace/validation_VLA/overlay_openloop_rollout.py \
  --episode-id 0 --steps 96 --action-horizon 16 --closed-loop \
  --out-dir /workspace/validation_VLA/out
```

### Two render modes (`--mode panel|overlay`, default `panel`)

**Camera: SOLVED 2026-09-29, uses TUNED (see below).** `--mode overlay`
alpha-blends the MuJoCo render directly onto the real `ego_view` frame (same
segmentation-mask technique as `run_ui_app/live_overlay.py` /
`run_ui_app/video/mesh_overlay.py` in the main repo). The path there, for
whoever touches this next:

1. **g1_brainco.xml's own named cameras** (`cam_right_high`, `cam_right_wrist`,
   `cam_left_wrist`) -- tried all three, ruled out. Rendering the dataset's
   own **ground-truth** pose (not a prediction) through each one did not
   line up with the real frame in any of them.
2. **Assumed the "6grid" pipeline's camera (`TUNED`, from
   `robot_video2pose`'s `render_tuned_overlay.py` / resolved per-episode by
   `resolve_episode_profile()` in `external/scripts_human2robot/pipeline/
   render_all_episodes.py`) was real-hardware-only and therefore
   inapplicable to RoboHaMeR's synthetic render** -- **wrong**, per user
   correction: RoboHaMeR's generation is itself built FROM real
   calibrated-camera footage, so its `ego_view` render uses that same
   calibrated camera. `TUNED` *is* the right calibration.
3. Applying `TUNED`'s raw fx/fy/cx/cy to a 640x392 render still didn't work
   -- wildly over-zoomed (both hands filling the whole frame). Root cause:
   `TUNED` was fit at **1280-wide** source frames (user-confirmed); Lisa's
   `ego_view` is 640x392, a **uniform resize of the whole frame** down from
   that (also user-confirmed) -- exactly half. Fix: scale fx/fy/cx/cy by
   `width / 1280` (`TUNED_CALIB_WIDTH` in the script); extrinsics
   (`offset_x/y/z`, `pitch/yaw/roll`) are a physical mount pose and don't
   change with resolution; distortion coefficients (`k1/k2/p1/p2/k3`) are
   dimensionless in OpenCV's model and stay valid under uniform scaling.
4. Also had to port `build_model()`'s arm-only `geom_group` restriction
   (torso/head/legs hidden, only arm geoms shown) -- without it the
   segmentation mask includes the whole body and throws the mask off even
   with the right camera pose.

With all four pieces (camera pose + resolution-scaled intrinsics + geom
restriction + undistort/shift compositing), the GT-pose overlay lines up
closely with the real frame across the episode (checked frames 0 and 60).
`--mode panel` (GT-video | MuJoCo(GT pose) | MuJoCo(predicted pose), no
claim of pixel alignment) is kept as a camera-independent fallback/cross-
check, but `--mode overlay` is now the meaningful one. MSE/RMSE from both
scripts above are camera-independent regardless and remain the authoritative
number either way.

### Action/joint mapping notes (for anyone touching this script)

- 13D action = 7 right-arm joints (radians, used as-is) + 6 BrainCo hand
  motors (`thumb_aux, thumb, index, middle, ring, pinky`, dataset units
  `[0,1]`) -- exact order from `meta/info.json`'s `action`/`observation.state`
  feature names, confirmed identical for both.
- `ACTION_JOINT_NAMES_13D` in the script = `sim_replay_unified.py`'s
  `ACTION_JOINT_NAMES[7:14] + [20:26]` (the right-arm + right-hand slice of
  the 26D direct-replay convention) -- confirmed to name-match Lisa's 13
  feature names 1:1.
- Hand `[0,1] -> radians`: same `_HAND_RAD_MIN=0` / `_HAND_RAD_MAX=[1.52,
  1.05, 1.47, 1.47, 1.47, 1.47]` table as `run_ui_app/inference/mirror.py`'s
  `_hand_fraction_to_radians` (itself checked there against `g1_brainco.xml`'s
  joint `<range>`s).
- Arm channels get **no** offset correction (unlike `live_overlay.py`'s
  `RIGHT_ARM_OFFSETS`, which corrects a *real* robot's hardware zero-point
  against the URDF -- RoboHaMeR's action space is simulation-native, that
  offset doesn't apply here). If/when camera calibration is solved and the
  arm still looks visibly off in `--mode overlay`, this offset is the first
  thing to check.
- Every other DOF (left arm, legs, waist, head) stays at `qpos=0` -- this is
  the same "freeze everything except what the policy controls" idea from the
  chat about the body-freeze GR00T needs at deploy time (right13 policy gives
  no info about the rest of the body).

### Known limitations

- Camera calibration (above) -- use `--mode panel` until it's resolved.
- `--use-server` (querying the live ZMQ PolicyClient on port 5555) was not
  exercised end-to-end here -- that server was serving an unrelated
  checkpoint (`g1_robohamer_340_train`) during this work, not
  `cosmos_right13`. `--model-path` (direct `Gr00tPolicy` load, the default)
  is what was actually tested, repeatedly, against checkpoint-5000 and
  checkpoint-7500.
- Only episode 0 was exercised. `--episode-id` takes any of the 340.
