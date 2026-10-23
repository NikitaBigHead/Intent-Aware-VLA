"""Evaluate a checkpoint on a split with the streaming (causal) episode metrics.

    python scripts/evaluate.py --ckpt runs/tinyvit_vtn/best.pt --split test
    python scripts/evaluate.py --ckpt ... --split test --set eval.threshold=0.7   # operating point
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.config import Cfg, load_config  # noqa: E402
from hri_video.data import load_episodes, split_episodes  # noqa: E402
from hri_video.evaluation import compute_metrics, episode_table, episode_traces, summary_line  # noqa: E402
from hri_video.model import VideoModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg_path = Path(args.ckpt).parent / "config.yaml"
    cfg = load_config(cfg_path, args.set) if cfg_path.exists() else Cfg.wrap(ck["config"])
    cfg.model.encoder.pretrained = False          # weights come from the checkpoint
    model = VideoModel(cfg).to(dev)
    model.load_state_dict(ck["model"])
    eps = split_episodes(load_episodes(cfg), cfg)[args.split]
    tr = episode_traces(model, eps, cfg, dev, log_every=20)
    m = compute_metrics(tr, cfg)
    print(summary_line(m))
    print(json.dumps(m, indent=1))
    out = Path(args.ckpt).parent
    (out / f"eval_{args.split}.json").write_text(json.dumps(m, indent=1))
    rows = episode_table(tr, cfg)
    with open(out / f"eval_{args.split}_episodes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
