"""Causal streaming inference for the dual-camera attention model.

Contract (same as training):
- both cameras are sampled at the SAME instants, 10 Hz; one shared timestamp t per pair;
- RTMW3D COCO133 on BGR frames, keypoint threshold 0.3, primary person = max mean upper-limb confidence;
- a decision every 0.2 s from the last 1 s (11 steps) of history;
- per camera 141 features, normalised with the train mean/std stored in the checkpoint;
- unknown rules: maskaware.decision.decide with threshold from the checkpoint (0.40) and the UNION of the
  two cameras' observed joints as pose quality.
"""
import collections

import numpy as np
import torch

from scripts.motion_prediction.maskaware.canonical import adapt
from scripts.motion_prediction.maskaware.decision import DEFAULT, decide, quality_inputs

from .features import FEATURES_PER_CAMERA, hand_features, hand_track, history, last_pose, reliability
from .model import MultiCam

CLASSES = ['hug', 'handshake', 'fist_bump', 'no_interaction']
RAW_KEYS = ('keypoints_2d', 'keypoints_3d', 'scores', 'visibility', 'person_detected', 'source_frame_index', 'video_time_s')


def raw_sample(person, t, frame_index):
    """One RTMW3D result (scripts.pose3d.infer_video.SelectedPerson) -> the dict stored during training."""
    return dict(keypoints_2d=person.keypoints_2d, keypoints_3d=person.keypoints_3d, scores=person.scores,
                visibility=person.visibility, person_detected=person.person_detected,
                source_frame_index=frame_index, video_time_s=t)


class DualCamEngine:
    def __init__(self, checkpoint, device='cpu', buffer_s=3.0, threshold=None):
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.device = device
        self.model = MultiCam([FEATURES_PER_CAMERA, FEATURES_PER_CAMERA]).to(device)
        self.model.load_state_dict(ck['action_state_dict'])
        self.model.eval()
        self.norm = [(np.asarray(n['mean'], np.float32), np.asarray(n['std'], np.float32)) for n in ck['norm']]
        self.threshold = float(ck['threshold'] if threshold is None else threshold)
        self.cameras = ck.get('cameras', {})
        self.buffer_s = buffer_s
        self.raw = {'person': collections.deque(), 'hand': collections.deque()}
        self.size = {}
        self.probs, self.quals, self.times = (collections.deque(maxlen=10) for _ in range(3))

    def add(self, t, raw_person, raw_hand, size_person, size_hand):
        """Append one synchronised pair (both RTMW3D results share the timestamp t, seconds)."""
        if self.raw['person'] and t <= self.raw['person'][-1]['video_time_s']:
            return False
        self.size = {'person': size_person, 'hand': size_hand}
        for cam, r in (('person', raw_person), ('hand', raw_hand)):
            self.raw[cam].append({**r, 'video_time_s': t})
            while t - self.raw[cam][0]['video_time_s'] > self.buffer_s:
                self.raw[cam].popleft()
        return True

    def reset(self):
        for q in (*self.raw.values(), self.probs, self.quals, self.times):
            q.clear()

    def step(self, origin):
        """Decision for the window ending at `origin` (normally the last added t). None if < 1 s of history."""
        arr = {cam: {k: np.asarray([r[k] for r in self.raw[cam]]) for k in RAW_KEYS} for cam in self.raw}
        aw = adapt(arr['person'], *self.size['person']); ah = adapt(arr['hand'], *self.size['hand'])
        hw = hand_track(arr['person'], *self.size['person']); hh = hand_track(arr['hand'], *self.size['hand'])
        t = aw['video_time_s']
        a = history(aw, t, origin)
        b = history(ah, t, origin)
        if a is None or b is None:
            return None
        ix, fresh, sw, mw = a
        _, _, sh, mh = b
        xw = np.concatenate((sw.reshape(11, -1), hand_features(hw, ix, fresh)), -1)
        xh = np.concatenate((sh.reshape(11, -1), hand_features(hh, ix, fresh)), -1)
        union = mw['observed'] | mh['observed']; uconf = np.maximum(mw['conf'], mh['conf'])
        q = quality_inputs(union, uconf, mw['person'] | mh['person'])
        rel = np.stack([reliability(xw), reliability(xh)], 0)
        xs = [torch.tensor(((x.astype(np.float32) - m) / s)[None], device=self.device) for x, (m, s) in zip((xw, xh), self.norm)]
        with torch.no_grad():
            logits, _, w = self.model(xs, torch.tensor(rel[None], dtype=torch.float32, device=self.device),
                                      torch.tensor(last_pose(mw['pose'])[None], device=self.device))
        p = torch.softmax(logits, -1)[0].cpu().numpy()
        self.probs.append(p); self.quals.append(q); self.times.append(origin)
        labels, reasons, adjusted = decide(np.asarray(self.probs), np.asarray(self.quals), np.zeros(len(self.probs), int),
                                           np.asarray(self.times), self.threshold, CLASSES, DEFAULT)
        k = int(labels[-1])
        return dict(origin_s=float(origin), label=CLASSES[k] if k >= 0 else 'unknown', label_index=k, reason=str(reasons[-1]),
                    confidence=float(adjusted[-1]), probabilities=dict(zip(CLASSES, map(float, p))),
                    quality=dict(person_observed=float(mw['observed'].mean()), hand_observed=float(mh['observed'].mean()),
                                 union_observed=float(q[1]), person_detected_last=bool(mw['person'][-1]),
                                 hand_detected_last=bool(mh['person'][-1]),
                                 person_reliability_last=float(rel[0, -1]), hand_reliability_last=float(rel[1, -1])),
                    hand_attention_last=float(w[0, 1, -1]),
                    _x_webcam=xw, _x_hand=xh, _quality_vector=q, _probabilities=p)
