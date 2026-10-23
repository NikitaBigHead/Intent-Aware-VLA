"""Train one encoder x temporal combination.

    python scripts/train.py --config configs/base.yaml \
        --set model.encoder.name=tinyvit model.temporal.name=vtn experiment.name=tinyvit_vtn

Model selection on the *validation persons* by episode macro-F1 (tie-break: pre-contact rate).
At the end the best checkpoint is evaluated once on the test persons.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hri_video.config import load_config, save_config  # noqa: E402
from hri_video.data import IGNORE, ClipDataset, collate, load_episodes, split_episodes  # noqa: E402
from hri_video.evaluation import compute_metrics, episode_table, episode_traces, summary_line  # noqa: E402
from hri_video.model import VideoModel, param_report  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def view_dropout(mask: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0 or mask.shape[1] < 2:
        return mask
    drop = torch.rand_like(mask, dtype=torch.float) < p
    new = mask & ~drop
    keep_one = ~new.any(1)                     # never drop every available camera
    new[keep_one] = mask[keep_one]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--skip-test", action="store_true", help="Profiling/development run: never evaluate test")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.experiment.seed))

    dev = torch.device(cfg.train.device if (cfg.train.device != "cuda" or torch.cuda.is_available()) else "cpu")
    out = Path(cfg.experiment.out_dir) / cfg.experiment.name
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    eps = load_episodes(cfg)
    sp = split_episodes(eps, cfg)
    classes = list(cfg.data.classes)
    for k, v in sp.items():
        cnt = {c: sum(e.label == i for e in v) for i, c in enumerate(classes)}
        print(f"{k:5s}: {len(v):4d} episodes, {len({e.person_id for e in v})} persons, {cnt}")
    with open(out / "split.json", "w") as f:
        json.dump({k: [e.episode_id for e in v] for k, v in sp.items()}, f, indent=1)

    train_ds = ClipDataset(sp["train"], cfg, train=True)
    counts = train_ds.action_counts(len(classes))
    print(f"train windows: {len(train_ds)}  per class: {dict(zip(classes, counts.astype(int).tolist()))}")
    dl = DataLoader(train_ds, batch_size=int(cfg.train.batch_size), shuffle=True, drop_last=True,
                    num_workers=int(cfg.data.num_workers), collate_fn=collate, pin_memory=dev.type == "cuda",
                    persistent_workers=int(cfg.data.num_workers) > 0)

    model = VideoModel(cfg).to(dev)
    pr = param_report(model)
    print({k: round(v, 3) for k, v in pr.items()})

    w = None
    if cfg.train.class_weights:
        w = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float, device=dev)
        w = w / w.mean()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    steps_per_epoch = len(dl) if not cfg.train.max_steps_per_epoch else min(len(dl), int(cfg.train.max_steps_per_epoch))
    total = steps_per_epoch * int(cfg.train.epochs)
    warm = steps_per_epoch * float(cfg.train.warmup_epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    use_amp = bool(cfg.train.amp) and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    log_path = out / "log.csv"
    best, best_key = None, (-1.0, -1.0)
    for epoch in range(int(cfg.train.epochs)):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, b in enumerate(dl):
            if step >= steps_per_epoch:
                break
            frames = {v: x.to(dev, non_blocking=True).float().div_(255) for v, x in b["frames"].items()}
            mask = view_dropout(b["view_mask"].to(dev), float(cfg.train.view_dropout))
            ya, yp = b["action"].to(dev), b["phase"].to(dev)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                o = model({"frames": frames, "view_mask": mask})
            la = o["logits"]["action"].float()
            loss = F.cross_entropy(la, ya, weight=w, ignore_index=IGNORE,
                                   label_smoothing=float(cfg.train.label_smoothing)) if (ya != IGNORE).any() else la.sum() * 0
            if "phase" in o["logits"] and (yp != IGNORE).any():
                loss = loss + float(cfg.train.phase_loss_weight) * F.cross_entropy(
                    o["logits"]["phase"].float(), yp, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, float(cfg.train.grad_clip))
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if step == 0 or (step + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {step + 1}/{steps_per_epoch} "
                      f"elapsed={elapsed:.1f}s train_remaining~{elapsed / n * (steps_per_epoch - n):.0f}s", flush=True)
        train_loss = tot / max(1, n)
        train_seconds = time.time() - t0

        tr = episode_traces(model, sp["val"], cfg, dev, amp=use_amp)
        m = compute_metrics(tr, cfg, with_ci=False)
        epoch_seconds = time.time() - t0
        key = (m["episode_macro_f1"], 0.0 if math.isnan(m["pre_contact_correct_rate"]) else m["pre_contact_correct_rate"])
        print(f"ep {epoch:3d} loss={train_loss:.4f} val {summary_line(m)} ({time.time() - t0:.0f}s)")
        row = {"epoch": epoch, "train_loss": train_loss, "train_seconds": train_seconds,
               "val_seconds": epoch_seconds - train_seconds, "epoch_seconds": epoch_seconds,
               **{k: v for k, v in m.items() if isinstance(v, float)}}
        new = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
        if key > best_key:
            best_key, best = key, epoch
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch, "val": m},
                       out / "best.pt")

    print(f"best epoch {best}: val epF1={best_key[0]:.3f}")
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"params": pr, "best_epoch": best}
    for k in ("val", "test"):
        if k == "test" and args.skip_test:
            continue
        if not sp[k]:
            continue
        tr = episode_traces(model, sp[k], cfg, dev, amp=use_amp)
        m = compute_metrics(tr, cfg)
        res[k] = m
        # Preserve causal probabilities for reporting without rerunning test inference.
        np.savez_compressed(out / f"traces_{k}.npz",
                            **{str(t["episode"].episode_id): t["probs"] for t in tr})
        print(f"{k}: {summary_line(m)}")
        rows = episode_table(tr, cfg)
        with open(out / f"episodes_{k}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
    with open(out / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
