"""Sanity check before a long run: can the model memorise a few real windows?

Takes N labelled training windows, encodes them once with the (frozen, pretrained)
encoder and trains temporal+fusion+heads on them with dropout off. With pretrained
weights train accuracy should approach 1.0 within a few hundred steps. If it does
not, the problem is in data/labels/features — fix that before the grid.

    python scripts/sanity_overfit.py --set model.encoder.name=tinyvit model.temporal.name=vtn
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.config import load_config  # noqa: E402
from hri_video.data import IGNORE, ClipDataset, collate, load_episodes, split_episodes  # noqa: E402
from hri_video.model import VideoModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); random.seed(0)

    ds = ClipDataset(split_episodes(load_episodes(cfg), cfg)["train"], cfg, train=False)
    C = len(cfg.data.classes)
    by_c = {c: [i for i, it in enumerate(ds.items) if it[2] == c] for c in range(C)}
    pick = [i for c in range(C) for i in random.sample(by_c[c], min(len(by_c[c]), args.n // C))]
    b = collate([ds[i] for i in pick])
    y, mask = b["action"].to(dev), b["view_mask"].to(dev)
    print(f"{len(pick)} windows, labels per class: {[int((y == c).sum()) for c in range(C)]}")

    model = VideoModel(cfg).to(dev)
    with torch.no_grad():
        tok = {v: model.encode_frames(v, b["frames"][v].to(dev).float() / 255) for v in model.views}
    model.eval()                                  # dropout off; only memorisation is tested
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    for s in range(args.steps):
        o = model.head_from_tokens(tok, mask)
        loss = F.cross_entropy(o["logits"]["action"], y, ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == args.steps - 1:
            acc = (o["logits"]["action"].argmax(-1) == y).float().mean().item()
            print(f"step {s:4d} loss {loss.item():.4f} train-acc {acc:.3f}")
    print("OK" if acc > 0.9 else "WARNING: cannot memorise a few windows - check labels / frames / features")


if __name__ == "__main__":
    main()
