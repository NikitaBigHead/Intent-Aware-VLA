"""Speed / memory / size benchmark for every encoder x temporal combination.

    python scripts/benchmark.py --config configs/base.yaml
    python scripts/benchmark.py --set benchmark.encoders=[tinyvit] benchmark.temporals=[vtn,videolightformer]

Same T, resolution and views for all combinations (taken from the config).
Weights are random (pretrained=false): speed does not depend on them.

Columns
  params_*                 parameters (M): encoder / temporal / total / trainable (with freeze setting)
  stream_ms_p50/p95        live step: encode ONE new frame per camera + temporal over the cached
                           T-frame token buffer + fusion + heads (what runs on the robot per frame)
  stream_fps               1000 / stream_ms_p50  (max sustainable input rate)
  clip_ms_p50/p95          full recompute of a T-frame clip for all cameras, batch 1
  clip_fps                 clips per second at batch 1
  thr_clips_s, thr_frames_s throughput at benchmark.batch_sizes (frames = clips * T * views)
  vram_*_mb                peak torch-allocated GPU memory incl. weights, per mode
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.config import load_config  # noqa: E402
from hri_video.model import VideoModel, param_report  # noqa: E402


def timeit(fn, warmup, iters, dev):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def peak_mb(fn, dev):
    if dev.type != "cuda":
        return float("nan")
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


@torch.no_grad()
def bench_one(cfg, dev, warmup, iters, batch_sizes, amp):
    model = VideoModel(cfg).to(dev).eval()
    views, T, S = model.views, model.num_frames, int(cfg.data.img_size)
    V = len(views)
    ac = lambda: torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda")  # noqa: E731
    res = dict(param_report(model))

    # --- live streaming step
    enc = model.encoder_for(views[0])
    tok_buf = {v: torch.randn(1, T, enc.num_tokens, enc.out_dim, device=dev) for v in views}
    new_frames = torch.rand(V, 3, S, S, device=dev)
    mask1 = torch.ones(1, V, dtype=torch.bool, device=dev)

    def stream_step():
        with ac():
            if "shared" in model.encoders:
                t = model.encoders["shared"](new_frames)
            else:
                t = torch.stack([model.encoder_for(v)(new_frames[i:i + 1])[0] for i, v in enumerate(views)])
            win = {v: torch.cat([tok_buf[v][:, 1:], t[i][None, None].to(tok_buf[v].dtype)], 1)
                   for i, v in enumerate(views)}
            model.head_from_tokens(win, mask1)

    res["stream_ms_p50"], res["stream_ms_p95"] = timeit(stream_step, warmup, iters, dev)
    res["stream_fps"] = 1000.0 / res["stream_ms_p50"]
    res["vram_stream_mb"] = peak_mb(stream_step, dev)

    # --- full clip recompute, batch 1 and throughput batches
    def make_clip(b):
        x = {v: torch.rand(b, T, 3, S, S, device=dev) for v in views}
        m = torch.ones(b, V, dtype=torch.bool, device=dev)

        def f():
            with ac():
                model({"frames": x, "view_mask": m})
        return f

    f1 = make_clip(1)
    res["clip_ms_p50"], res["clip_ms_p95"] = timeit(f1, warmup, iters, dev)
    res["clip_fps"] = 1000.0 / res["clip_ms_p50"]
    res["vram_clip_mb"] = peak_mb(f1, dev)
    for b in batch_sizes:
        fb = make_clip(b)
        p50, _ = timeit(fb, max(3, warmup // 4), max(5, iters // 4), dev)
        res[f"thr_clips_s_b{b}"] = b * 1000.0 / p50
        res[f"thr_frames_s_b{b}"] = b * T * V * 1000.0 / p50
        res[f"vram_b{b}_mb"] = peak_mb(fb, dev)
        del fb
    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    bc = cfg.benchmark
    dev = torch.device(bc.device if (bc.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    else:
        print("[warn] running on CPU: FPS numbers are not representative of the RTX 4090; VRAM = nan")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    rows = []
    for e, t in itertools.product(bc.encoders, bc.temporals):
        c = load_config(args.config, args.set + [f"model.encoder.name={e}", f"model.temporal.name={t}",
                                                 "model.encoder.pretrained=false"])
        print(f"-> {e} + {t}  (T={c.data.num_frames}, {c.data.img_size}px, views={list(c.data.views)})", flush=True)
        r = bench_one(c, dev, int(bc.warmup), int(bc.iters), list(bc.batch_sizes), bool(bc.amp))
        row = {"encoder": e, "temporal": t, "T": c.data.num_frames, "img": c.data.img_size,
               "views": len(c.data.views), "device": dev.type, "amp": bool(bc.amp) and dev.type == "cuda", **r}
        rows.append(row)
        print("   " + "  ".join(f"{k}={v:.2f}" for k, v in r.items() if isinstance(v, float)), flush=True)

    out = Path(args.out or Path(cfg.experiment.out_dir) / "benchmark")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    cols = ["encoder", "temporal", "params_total_M", "params_trainable_M", "stream_ms_p50", "stream_fps",
            "clip_ms_p50", "clip_fps", f"thr_frames_s_b{bc.batch_sizes[0]}", "vram_stream_mb", "vram_clip_mb"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (out / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"saved {out / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
