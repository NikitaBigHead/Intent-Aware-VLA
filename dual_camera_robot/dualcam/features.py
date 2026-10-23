"""Preprocessing of the dual-camera model, copied verbatim from the training code.

hand_track, hand_features, history  <- artifacts/dual_camera_attention_v1/build_dataset.py
reliability, last_pose              <- artifacts/dual_camera_attention_v1/train_eval.py
Only formatting and the per-window form of last_pose differ; the arithmetic is unchanged.
"""
import numpy as np

from scripts.motion_prediction.maskaware.canonical import features
from scripts.motion_prediction.pose_adapter import Z_SCALE

# Training frame sizes (pixel -> [0,1] normalisation used the recorded resolutions).
TRAIN_SIZE = {'person': (1600, 896), 'hand': (1280, 720)}
HANDS = (91, 112)          # COCO133 left / right hand blocks (21 points each)
TIPS = (4, 8, 12, 16, 20)  # finger tips inside a hand block
FEATURES_PER_CAMERA = 141  # 120 canonical8 v2 (8 joints x 15) + 21 hand features


def hand_track(raw, w, h, thr=.3):
    xy = np.asarray(raw['keypoints_2d'], float); z = np.asarray(raw['keypoints_3d'], float)[..., 2]; s = np.asarray(raw['scores'], float)
    det = np.asarray(raw['person_detected'], bool)
    vis = np.asarray(raw['visibility'], bool) & (s >= thr) & np.isfinite(s) & det[:, None] & np.isfinite(xy).all(-1)
    T = len(xy); center = np.full((T, 2, 3), np.nan); obs = np.zeros((T, 2), bool); conf = np.zeros((T, 2)); opn = np.full((T, 2), np.nan)
    for k, b in enumerate(HANDS):
        ids = np.arange(b, b + 21); v = vis[:, ids]; n = v.sum(1); obs[:, k] = n >= 6
        c = np.concatenate((xy[:, ids] / [w, h], z[:, ids, None] / Z_SCALE), -1)
        with np.errstate(invalid='ignore'):
            center[:, k] = np.where(v[..., None], c, 0).sum(1) / np.maximum(n, 1)[:, None]
            conf[:, k] = np.where(v, s[:, ids], 0).sum(1) / np.maximum(n, 1)
        center[~obs[:, k], k] = np.nan; conf[~obs[:, k], k] = 0
        wr = xy[:, b]; mcp = xy[:, b + 9]; ref = np.linalg.norm(mcp - wr, axis=-1); tips = np.array([b + t for t in TIPS]); tv = vis[:, tips]
        d = np.linalg.norm(xy[:, tips] - wr[:, None], axis=-1) / np.maximum(ref, 1e-6)[:, None]
        ok = vis[:, b] & vis[:, b + 9] & (tv.sum(1) >= 3) & (ref > 2)
        opn[ok, k] = np.clip((np.where(tv, d, 0).sum(1) / np.maximum(tv.sum(1), 1))[ok], 0, 4)
    return dict(center=center, observed=obs, confidence=conf, openness=opn, detected=det)


def hand_features(ht, ix, fresh):
    o = ht['observed'][ix] & fresh[:, None]; x = np.where(o[..., None], ht['center'][ix], 0.); v = np.zeros_like(x)
    vm = np.zeros_like(o); vm[1:] = o[1:] & o[:-1]
    v[1:] = np.diff(x, axis=0) * 10; v = np.where(vm[..., None], v, 0); op = ht['openness'][ix]; ov = np.isfinite(op) & fresh[:, None]
    f = np.concatenate((x, v, o[..., None], np.where(o, ht['confidence'][ix], 0)[..., None]), -1).reshape(len(ix), -1)
    return np.concatenate((f, np.where(ov, op, 0), ov, (ht['detected'][ix] & fresh)[:, None]), -1).astype(np.float32)  # 16+2+2+1=21


def history(a, t, origin):
    grid = origin - 1 + np.arange(11) / 10; ix = np.searchsorted(t, grid + 1e-8, side='right') - 1
    if (ix < 0).any():
        return None  # training asserted this; live buffers can be shorter than 1 s at start-up
    delay = grid - t[ix]; fresh = delay <= .2 + 1e-8
    observed = a['observed'][ix] & fresh[:, None]; age = a['age'][ix] + delay[:, None]
    imputed = a['imputed'][ix] & (age <= .3 + 1e-8) & fresh[:, None]; valid = observed | imputed
    pose = a['input_pose'][ix].copy(); pose[~valid] = np.nan; conf = np.where(observed, a['confidence'][ix], 0)
    rel = a['root_relative_pose'][ix].copy(); rv = a['root_relative_visibility'][ix] & fresh[:, None]; rel[~rv] = np.nan
    seq, _ = features(pose, valid, conf, observed, imputed, age, rel, rv)
    return ix, fresh, seq, dict(pose=pose, valid=valid, conf=conf, observed=observed, imputed=imputed, age=age, person=a['person_detected'][ix] & fresh)


def reliability(x):
    """observed channel (index 11 of each 15-ch joint) + two hand observed flags, times detected."""
    j = x[..., :120].reshape(*x.shape[:-1], 8, 15)[..., 11]; hobs = x[..., [126, 134]]
    return np.concatenate((j, hobs), -1).mean(-1) * (x[..., 140] > 0)


def last_pose(observation_pose):
    """Input of the motion head only (action logits do not depend on it). observation_pose [11,8,3] webcam."""
    last = np.nan_to_num(observation_pose[-1])
    for t in range(9, -1, -1):
        m = ~np.isfinite(observation_pose[-1]).all(-1) & np.isfinite(observation_pose[t]).all(-1) & (last == 0).all(-1)
        last[m] = observation_pose[t][m]
    return last.astype(np.float32)
