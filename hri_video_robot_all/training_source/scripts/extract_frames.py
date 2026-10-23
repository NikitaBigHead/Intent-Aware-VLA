"""Convert raw episode videos into the synchronised frame folders the trainer reads.

Raw index CSV (one row per episode):
    episode_id, person_id, label, t_onset, t_contact, [split], [t0], <view>_video, [<view>_timestamps] ...
* <view>_video        path to the video (mp4/avi/...) of that camera; empty if missing
* <view>_timestamps   optional per-frame timestamps (.npy or one number per line .csv/.txt);
                      without it frame i is at i / video_fps from the video start
* t_onset, t_contact  seconds in the SAME time base as the timestamps (empty for no_interaction)
* t0                  optional episode start; default = latest first-timestamp over cameras

Every camera is resampled to a common grid t0 + k / fps (latest past frame), letterboxed
to img_size x img_size and written as JPEG. t_onset / t_contact are shifted to t0.

    python scripts/extract_frames.py --raw raw_index.csv --out data/frames --views head ego --fps 15 --size 224
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.streaming import letterbox  # noqa: E402


def read_ts(path, n_frames, video_fps, scale):
    if isinstance(path, str) and path:
        p = Path(path)
        if p.suffix == ".npy":
            ts = np.load(p).astype(float).ravel()
        else:
            df = pd.read_csv(p, header=None)
            ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(float)
        ts = ts * scale
        if len(ts) != n_frames:
            print(f"   [warn] {p.name}: {len(ts)} timestamps vs {n_frames} frames; using min")
        return ts[: min(len(ts), n_frames)]
    return np.arange(n_frames) / video_fps


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", nargs="+", default=["head", "ego"])
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--ts-scale", type=float, default=1.0, help="e.g. 1e-9 if timestamps are in ns")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw, dtype={"episode_id": str, "person_id": str})
    out = Path(args.out)
    rows = []
    for r in raw.to_dict("records"):
        eid = str(r["episode_id"])
        print(f"{eid}", flush=True)
        ts, cams = {}, {}
        for v in args.views:
            vp = r.get(f"{v}_video")
            if not isinstance(vp, str) or not vp:
                continue
            n, vfps = video_info(vp)
            ts[v] = read_ts(r.get(f"{v}_timestamps"), n, vfps, args.ts_scale)
            cams[v] = vp
        if not cams:
            print("   [skip] no videos"); continue
        t0 = float(r["t0"]) if "t0" in r and not pd.isna(r.get("t0")) else max(t[0] for t in ts.values())
        t1 = min(t[-1] for t in ts.values())
        grid = t0 + np.arange(int(math.floor((t1 - t0) * args.fps)) + 1) / args.fps
        row = {k: r.get(k) for k in ("episode_id", "person_id", "label", "split") if k in r}
        for k in ("t_onset", "t_contact"):
            row[k] = (float(r[k]) - t0) if k in r and not pd.isna(r.get(k)) else ""
        row["num_frames"] = len(grid)
        for v in args.views:
            if v not in cams:
                row[f"{v}_dir"] = ""
                continue
            # A nearest frame may lie AFTER the prediction timestamp.
            src = np.searchsorted(ts[v], grid, side="right") - 1
            if (src < 0).any():
                raise ValueError(f"{eid}/{v}: no causal frame at grid start")
            need = {}
            for k, s in enumerate(src):
                need.setdefault(int(s), []).append(k)
            d = out / v / eid
            d.mkdir(parents=True, exist_ok=True)
            cap, i, written = cv2.VideoCapture(cams[v]), 0, 0
            last = max(need)
            while i <= last:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i in need:
                    img = cv2.cvtColor(letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.size), cv2.COLOR_RGB2BGR)
                    for k in need[i]:
                        if not cv2.imwrite(str(d / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
                            raise OSError(f"Cannot write {d}/{k:06d}.jpg")
                        written += 1
                i += 1
            cap.release()
            if written != len(grid):
                raise RuntimeError(f"{eid}/{v}: decoded only {written}/{len(grid)} requested frames")
            row[f"{v}_dir"] = f"{v}/{eid}"
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    print(f"wrote {len(rows)} episodes -> {out / 'index.csv'}")


if __name__ == "__main__":
    main()
