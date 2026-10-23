"""Episode index + causal clip dataset.

Processed index CSV (written by scripts/extract_frames.py), one row per episode:
    episode_id, person_id, label, t_onset, t_contact, [split], <view>_dir ...
* `<view>_dir`: folder (relative to data.root) with 000000.jpg, 000001.jpg, ...
  frame k is at time k / data.fps seconds from the episode start; all views share
  this time grid. Empty -> that camera is missing for the episode.
* t_onset / t_contact: seconds from episode start; empty for no_interaction.
* split (optional): train / val / test. If absent, persons are split by group k-fold.

A training sample is a *causal* window that ends at frame e and contains
T frames e-(T-1)*s ... e (s = frame_stride). Frames before the episode start are
replaced by the first frame (left replicate padding) — the same rule the live
streaming predictor uses, so training and inference see identical inputs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IGNORE = -100


@dataclass
class Episode:
    episode_id: str
    person_id: str
    label: int
    t_onset: float
    t_contact: float
    view_frames: dict = field(default_factory=dict)  # view -> list[Path] (empty if missing)
    split: str | None = None

    @property
    def length(self) -> int:
        return max((len(f) for f in self.view_frames.values()), default=0)


def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_episodes(cfg) -> list[Episode]:
    root = Path(cfg.data.root)
    df = pd.read_csv(cfg.data.index, dtype={"episode_id": str, "person_id": str})
    classes = list(cfg.data.classes)
    eps = []
    for r in df.itertuples(index=False):
        r = r._asdict()
        if r["label"] not in classes:
            raise ValueError(f"Episode {r['episode_id']}: unknown label '{r['label']}', expected {classes}")
        vf = {}
        for v in cfg.data.views:
            d = r.get(f"{v}_dir")
            if isinstance(d, str) and d:
                vf[v] = sorted((root / d).glob("*.jpg"))
            else:
                vf[v] = []
        sp = r.get("split")
        eps.append(Episode(str(r["episode_id"]), str(r["person_id"]), classes.index(r["label"]),
                           _f(r.get("t_onset")), _f(r.get("t_contact")), vf,
                           sp if isinstance(sp, str) and sp else None))
    return eps


# ----------------------------------------------------------------------------- split
def split_episodes(eps: list[Episode], cfg) -> dict[str, list[Episode]]:
    sc = cfg.split
    if sc.mode in ("auto", "column") and all(e.split for e in eps):
        out = {k: [e for e in eps if e.split == k] for k in ("train", "val", "test")}
    elif sc.mode == "column":
        raise ValueError("split.mode=column but some episodes have no `split` value")
    else:
        persons = sorted({e.person_id for e in eps})
        n = int(sc.n_folds)
        if len(persons) < 3:
            raise ValueError(f"Need >= 3 persons for a person-wise split, got {persons}")
        n = min(n, len(persons))
        rng = random.Random(int(cfg.experiment.seed))
        rng.shuffle(persons)
        folds = [persons[i::n] for i in range(n)]
        k = int(sc.fold) % n
        test_p, val_p = set(folds[k]), set(folds[(k + 1) % n])
        out = {"test": [e for e in eps if e.person_id in test_p],
               "val": [e for e in eps if e.person_id in val_p],
               "train": [e for e in eps if e.person_id not in test_p | val_p]}
    # leakage guard
    pt = {k: {e.person_id for e in v} for k, v in out.items()}
    if pt["train"] & (pt["val"] | pt["test"]):
        print(f"[warn] persons shared between train and val/test: {sorted(pt['train'] & (pt['val'] | pt['test']))}")
    return out


# ----------------------------------------------------------------------------- labels
def frame_labels(ep: Episode, t: float, policy: str, no_int_idx: int = 0) -> tuple[int, int]:
    """(action label, phase label) at time t (end of window)."""
    has_onset = not math.isnan(ep.t_onset)
    has_contact = not math.isnan(ep.t_contact)
    if ep.label == no_int_idx:
        return no_int_idx, 0
    if has_onset and t < ep.t_onset:
        action = {"ignore": IGNORE, "label": ep.label, "no_interaction": no_int_idx}[policy]
        return action, 0
    if not has_onset:   # interaction episode without onset annotation: we can't place phases
        return ep.label, IGNORE
    if has_contact and t >= ep.t_contact:
        return ep.label, 2
    return ep.label, 1


def clip_indices(end: int, num_frames: int, stride: int) -> list[int]:
    return [max(0, end - stride * (num_frames - 1 - k)) for k in range(num_frames)]


def read_rgb(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------------------- dataset
class ClipDataset(Dataset):
    def __init__(self, episodes: list[Episode], cfg, train: bool):
        d = cfg.data
        self.eps, self.cfg, self.train = episodes, cfg, train
        self.views, self.T, self.stride = list(d.views), int(d.num_frames), int(d.frame_stride)
        self.fps, self.size = float(d.fps), int(d.img_size)
        self.policy = d.pre_onset_policy
        self.no_int = list(d.classes).index("no_interaction") if "no_interaction" in d.classes else -1
        step = int(d.train_window_step) if train else 1
        self.items = []
        for i, ep in enumerate(episodes):
            for e in range(0, ep.length, step):
                a, p = frame_labels(ep, e / self.fps, self.policy, self.no_int)
                if a == IGNORE and p == IGNORE:
                    continue
                self.items.append((i, e, a, p))
        aug = cfg.train.get("aug", {}) or {}
        self.crop_scale = aug.get("crop_scale", [1.0, 1.0])
        self.bright, self.contrast = aug.get("brightness", 0.0), aug.get("contrast", 0.0)

    def __len__(self):
        return len(self.items)

    def action_counts(self, n_classes: int) -> np.ndarray:
        c = np.zeros(n_classes)
        for _, _, a, _ in self.items:
            if a != IGNORE:
                c[a] += 1
        return c

    def _load_view(self, frames: list[Path], idx: list[int]) -> np.ndarray:
        n = len(frames)
        imgs = np.stack([read_rgb(frames[min(i, n - 1)], self.size) for i in idx])  # (T, H, W, 3)
        if self.train:
            imgs = self._augment(imgs)
        return imgs

    def _augment(self, imgs: np.ndarray) -> np.ndarray:
        # one set of parameters per clip & view -> temporally consistent
        s = self.size
        lo, hi = self.crop_scale
        sc = random.uniform(lo, hi)
        cs = max(8, int(round(s * math.sqrt(sc))))
        y, x = random.randint(0, s - cs), random.randint(0, s - cs)
        out = np.stack([cv2.resize(im[y:y + cs, x:x + cs], (s, s), interpolation=cv2.INTER_LINEAR)
                        for im in imgs]).astype(np.float32)
        b = 1.0 + random.uniform(-self.bright, self.bright)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        m = out.mean()
        out = ((out - m) * c + m) * b
        return np.clip(out, 0, 255).astype(np.uint8)

    def __getitem__(self, k):
        i, e, a, p = self.items[k]
        ep = self.eps[i]
        idx = clip_indices(e, self.T, self.stride)
        frames, mask = {}, []
        for v in self.views:
            if ep.view_frames[v]:
                arr = self._load_view(ep.view_frames[v], idx)
                frames[v] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # uint8 (T,3,H,W)
                mask.append(True)
            else:
                frames[v] = torch.zeros(self.T, 3, self.size, self.size, dtype=torch.uint8)
                mask.append(False)
        return {"frames": frames, "view_mask": torch.tensor(mask), "action": a, "phase": p,
                "episode": i, "t": e / self.fps}


def collate(batch):
    views = batch[0]["frames"].keys()
    return {
        "frames": {v: torch.stack([b["frames"][v] for b in batch]) for v in views},
        "view_mask": torch.stack([b["view_mask"] for b in batch]),
        "action": torch.tensor([b["action"] for b in batch]),
        "phase": torch.tensor([b["phase"] for b in batch]),
        "episode": torch.tensor([b["episode"] for b in batch]),
        "t": torch.tensor([b["t"] for b in batch]),
    }


def load_episode_frames(ep: Episode, cfg) -> dict:
    """All frames of an episode per view as uint8 (L, 3, H, W) — for streaming evaluation."""
    size, out = int(cfg.data.img_size), {}
    for v in cfg.data.views:
        fr = ep.view_frames[v]
        if fr:
            arr = np.stack([read_rgb(p, size) for p in fr])
            out[v] = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        else:
            out[v] = None
    return out
