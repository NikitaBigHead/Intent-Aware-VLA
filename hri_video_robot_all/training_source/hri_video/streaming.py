"""Live, causal inference: one call per incoming (synchronised) set of camera frames.

* Frames are gated to cfg.data.fps (the rate the model was trained at); extra frames
  are skipped, so temporal spacing matches training.
* Each new frame is encoded once; per-view token buffers hold the last
  (T-1)*stride+1 steps. Only past/current frames are ever used.
* Before the buffer is full the oldest frame is replicated (same as training).
* A camera that has delivered nothing for `max_gap_s` is marked missing (view_mask=False);
  shorter gaps reuse the last frame's tokens.

Usage:
    pred = StreamingPredictor(model, cfg, device)
    for t, frames in camera_stream():      # frames = {"head": rgb_uint8 | None, "ego": ...}
        out = pred.step(frames, t)          # None if the frame was skipped by fps gating
        if out: print(out["label"], out["probs"], out["committed"])
"""
from __future__ import annotations

import collections

import cv2
import numpy as np
import torch

from .data import clip_indices


def letterbox(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect ratio and pad to size x size (black). Keeps the full field of view."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = img
    return out


class StreamingPredictor:
    def __init__(self, model, cfg, device, amp: bool = True, max_gap_s: float = 0.5, preprocess: bool = True):
        self.model = model.eval()
        self.cfg, self.device = cfg, device
        self.views = model.views
        self.T, self.stride = int(cfg.data.num_frames), int(cfg.data.frame_stride)
        self.fps, self.size = float(cfg.data.fps), int(cfg.data.img_size)
        self.classes = list(cfg.data.classes)
        self.no_int = self.classes.index("no_interaction")
        self.thr, self.k = float(cfg.eval.threshold), int(cfg.eval.consecutive)
        self.amp = amp and device.type == "cuda"
        self.max_gap_s, self.preprocess = max_gap_s, preprocess
        self.reset()

    def reset(self):
        n = (self.T - 1) * self.stride + 1
        self.buf = {v: collections.deque(maxlen=n) for v in self.views}
        self.valid = {v: collections.deque(maxlen=n) for v in self.views}
        self.last_tok = {v: None for v in self.views}
        self.last_seen = {v: -1e9 for v in self.views}
        self.next_t = None
        self.run_c, self.run = None, 0
        self.committed = None

    def _to_tensor(self, rgb):
        if self.preprocess and (rgb.shape[0] != self.size or rgb.shape[1] != self.size):
            rgb = letterbox(rgb, self.size)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)

    @torch.no_grad()
    def step(self, frames: dict, t: float):
        if self.next_t is not None and t + 1e-6 < self.next_t:
            return None
        self.next_t = (t if self.next_t is None else self.next_t) + 1.0 / self.fps
        if self.next_t < t:                      # fell behind: re-anchor instead of bursting
            self.next_t = t + 1.0 / self.fps

        present = [v for v in self.views if frames.get(v) is not None]
        new_tok = {}
        if present:
            x = torch.stack([self._to_tensor(frames[v]) for v in present]).to(self.device).float().div_(255)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                if "shared" in self.model.encoders:
                    tok = self.model.encoders["shared"](x).float()
                    new_tok = {v: tok[i] for i, v in enumerate(present)}
                else:
                    new_tok = {v: self.model.encoder_for(v)(x[i:i + 1]).float()[0] for i, v in enumerate(present)}

        for v in self.views:
            if v in new_tok:
                self.last_tok[v], self.last_seen[v] = new_tok[v], t
                self.buf[v].append(new_tok[v]); self.valid[v].append(True)
            elif self.last_tok[v] is not None and t - self.last_seen[v] <= self.max_gap_s:
                self.buf[v].append(self.last_tok[v]); self.valid[v].append(True)
            else:
                ref = next((b[-1] for b in self.buf.values() if len(b)), None)
                if ref is None:
                    ref = next(iter(new_tok.values()), None)
                if ref is None:
                    return None                      # nothing seen yet from any camera
                self.buf[v].append(torch.zeros_like(ref)); self.valid[v].append(False)

        win, mask = {}, []
        for v in self.views:
            b = self.buf[v]
            idx = clip_indices(len(b) - 1, self.T, self.stride)
            win[v] = torch.stack([b[i] for i in idx]).unsqueeze(0)    # (1, T, P, C)
            mask.append(bool(self.valid[v][-1]))
        m = torch.tensor([mask], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            o = self.model.head_from_tokens(win, m)
        probs = o["logits"]["action"].float().softmax(-1)[0].cpu().numpy()

        c = int(probs.argmax())
        ok = c != self.no_int and probs[c] >= self.thr
        if ok and c == self.run_c:
            self.run += 1
        elif ok:
            self.run_c, self.run = c, 1
        else:
            self.run_c, self.run = None, 0
        if self.run == self.k and self.committed is None:
            self.committed = (self.classes[c], t)

        out = {"t": t, "probs": probs, "label": self.classes[c], "view_mask": mask,
               "committed": self.committed, "embedding": o["embedding"][0].float().cpu().numpy()}
        if "phase" in o["logits"]:
            out["phase_probs"] = o["logits"]["phase"].float().softmax(-1)[0].cpu().numpy()
        return out
