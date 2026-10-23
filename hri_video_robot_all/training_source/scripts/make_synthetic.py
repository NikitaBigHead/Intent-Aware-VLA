"""Generate a tiny synthetic dataset (raw videos + raw index) to smoke-test the whole pipeline.

Each class moves a coloured blob differently after t_onset; no_interaction wanders.
It is NOT a benchmark of anything - only proves that extraction -> training ->
streaming evaluation -> benchmark run end to end.

    python scripts/make_synthetic.py --out data/synthetic_raw --persons 6 --per-class 3
    python scripts/extract_frames.py --raw data/synthetic_raw/raw_index.csv --out data/synthetic_frames --size 224
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["no_interaction", "hug", "handshake", "fist_bump"]


def render(label, t, onset, contact, view, rng_off, w=160, h=120):
    img = np.full((h, w, 3), 40 if view == "head" else 70, np.uint8)
    img[:] += np.uint8(rng_off % 20)
    if label == 0 or t < onset:
        x = w / 2 + 25 * np.sin(t * 1.3 + rng_off); y = h / 2 + 10 * np.cos(t + rng_off); r = 10
    else:
        a = min(1.0, (t - onset) / max(1e-3, contact - onset))
        if label == 1:   x, y, r = w / 2, h / 2 - 20 * a, 10 + 25 * a
        elif label == 2: x, y, r = w / 2 + 40 * a, h / 2, 10 + 6 * a
        else:            x, y, r = w / 2, h / 2 + 30 * a, 8 + 4 * a
    if view == "ego":
        y = h - y
    color = (60, 160, 230) if view == "head" else (230, 140, 60)
    cv2.circle(img, (int(x), int(y)), int(r), color, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--persons", type=int, default=6)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = ["episode_id,person_id,label,t_onset,t_contact,head_video,ego_video"]
    for p in range(args.persons):
        for c, name in enumerate(CLASSES):
            for k in range(args.per_class):
                eid = f"p{p}_{name}_{k}"
                dur = rng.uniform(3.0, 4.5)
                onset = rng.uniform(0.8, 1.5) if c else float("nan")
                contact = onset + rng.uniform(1.0, 1.8) if c else float("nan")
                off = rng.uniform(0, 100)
                for view in ("head", "ego"):
                    vw = cv2.VideoWriter(str(out / f"{eid}_{view}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (160, 120))
                    for i in range(int(dur * args.fps)):
                        vw.write(render(c, i / args.fps, onset, contact, view, off))
                    vw.release()
                fmt = lambda x: "" if np.isnan(x) else f"{x:.3f}"  # noqa: E731
                lines.append(f"{eid},person{p},{name},{fmt(onset)},{fmt(contact)},"
                             f"{out / (eid + '_head.mp4')},{out / (eid + '_ego.mp4')}")
    (out / "raw_index.csv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 1} episodes -> {out / 'raw_index.csv'}")


if __name__ == "__main__":
    main()
