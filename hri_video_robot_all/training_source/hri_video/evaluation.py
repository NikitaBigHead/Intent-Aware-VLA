"""Streaming (causal) evaluation over whole episodes + episode-level / pre-contact metrics.

For every frame e of an episode the model sees only frames <= e (the same causal
window as in training / live), giving a probability trace. From the trace:

* commit rule: an interaction class c is *committed* when p(c) >= threshold for
  K consecutive steps (no_interaction is never "committed").
* episode prediction = class of the first commit, or no_interaction if none.
* pre-contact success = first commit is the correct class and happens before t_contact.
* lead time = t_contact - t_first_commit (correct first commits only).
* false alarms = commits on no_interaction episodes (per episode and per minute).
* accuracy vs time-to-contact / vs onset->contact progress (argmax at that frame
  and "correct commit already made by then").
* window macro-F1 over frames with a defined label (after onset for interactions).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from .data import IGNORE, Episode, clip_indices, frame_labels, load_episode_frames


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def episode_traces(model, episodes: list[Episode], cfg, device, amp: bool = True, log_every: int = 0) -> list[dict]:
    model.eval()
    T, stride = int(cfg.data.num_frames), int(cfg.data.frame_stride)
    bs = int(cfg.eval.batch_size)
    views = model.views
    use_amp = amp and device.type == "cuda"
    out = []
    for n, ep in enumerate(episodes):
        frames = load_episode_frames(ep, cfg)
        L = ep.length
        tokens, vmask = {}, []
        for v in views:
            if frames[v] is None:
                tokens[v] = None
                vmask.append(False)
                continue
            vmask.append(True)
            chunks = []
            for s in range(0, frames[v].shape[0], 256):
                x = frames[v][s:s + 256].to(device, non_blocking=True).float().div_(255)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    chunks.append(model.encoder_for(v)(x).float())
            tokens[v] = torch.cat(chunks)                         # (Lv, P, C)
        ref = next(t for t in tokens.values() if t is not None)
        for v in views:
            if tokens[v] is None:
                tokens[v] = torch.zeros_like(ref[:1])
        probs, phase = [], []
        for s in range(0, L, bs):
            ends = list(range(s, min(L, s + bs)))
            win = {}
            for v in views:
                tv = tokens[v]
                idx = torch.tensor([[min(i, tv.shape[0] - 1) for i in clip_indices(e, T, stride)] for e in ends],
                                   device=device)
                win[v] = tv[idx]                                   # (b, T, P, C)
            m = torch.tensor(vmask, device=device).unsqueeze(0).expand(len(ends), -1)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                o = model.head_from_tokens(win, m)
            probs.append(o["logits"]["action"].float().softmax(-1).cpu())
            if "phase" in o["logits"]:
                phase.append(o["logits"]["phase"].float().softmax(-1).cpu())
        out.append({"episode": ep, "probs": torch.cat(probs).numpy(),
                    "phase": torch.cat(phase).numpy() if phase else None})
        if log_every and (n + 1) % log_every == 0:
            print(f"  eval {n + 1}/{len(episodes)}")
    return out


# ----------------------------------------------------------------------------- metrics
def commit_events(probs: np.ndarray, thr: float, k: int, no_int: int) -> list[tuple[int, int]]:
    events, run_c, run = [], None, 0
    for i, p in enumerate(probs):
        c = int(p.argmax())
        ok = c != no_int and p[c] >= thr
        if ok and c == run_c:
            run += 1
        elif ok:
            run_c, run = c, 1
        else:
            run_c, run = None, 0
        if run == k:
            events.append((i, c))
    return events


def _macro_f1(y, p, n):
    return float(f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0))


def episode_table(traces, cfg) -> list[dict]:
    classes = list(cfg.data.classes)
    no_int = classes.index("no_interaction")
    fps = float(cfg.data.fps)
    thr, k = float(cfg.eval.threshold), int(cfg.eval.consecutive)
    rows = []
    for tr in traces:
        ep, pr = tr["episode"], tr["probs"]
        ev = commit_events(pr, thr, k, no_int)
        first = ev[0] if ev else None
        pred = first[1] if first else no_int
        t_first = first[0] / fps if first else float("nan")
        has_c = not math.isnan(ep.t_contact)
        correct_first = first is not None and first[1] == ep.label
        has_o = not math.isnan(ep.t_onset)
        rows.append({
            "episode_id": ep.episode_id, "person_id": ep.person_id,
            "commit_before_onset": bool(first is not None and has_o and t_first < ep.t_onset),
            "label": classes[ep.label], "pred": classes[pred], "correct": pred == ep.label,
            "t_onset": ep.t_onset, "t_contact": ep.t_contact, "t_first_commit": t_first,
            "n_commits": len(ev), "duration_s": len(pr) / fps,
            "pre_contact_correct": bool(correct_first and has_c and t_first < ep.t_contact),
            "lead_time_s": (ep.t_contact - t_first) if (correct_first and has_c) else float("nan"),
        })
    return rows


def compute_metrics(traces, cfg, with_ci: bool = True) -> dict:
    classes = list(cfg.data.classes)
    C, no_int, fps = len(classes), classes.index("no_interaction"), float(cfg.data.fps)
    policy = cfg.data.pre_onset_policy
    rows = episode_table(traces, cfg)
    y = [classes.index(r["label"]) for r in rows]
    p = [classes.index(r["pred"]) for r in rows]
    m = {"n_episodes": len(rows), "n_persons": len({r["person_id"] for r in rows})}
    m["episode_macro_f1"] = _macro_f1(y, p, C)
    per = f1_score(y, p, labels=list(range(C)), average=None, zero_division=0)
    m["episode_f1_per_class"] = {c: float(v) for c, v in zip(classes, per)}
    m["episode_confusion"] = confusion_matrix(y, p, labels=list(range(C))).tolist()

    inter = [r for r in rows if r["label"] != "no_interaction" and not math.isnan(r["t_contact"])]
    m["pre_contact_correct_rate"] = float(np.mean([r["pre_contact_correct"] for r in inter])) if inter else float("nan")
    # commits before the person even started moving = guessing; must be read together with pre_contact
    with_on = [r for r in inter if not math.isnan(r["t_onset"])]
    m["commit_before_onset_rate"] = float(np.mean([r["commit_before_onset"] for r in with_on])) if with_on else float("nan")
    m["pre_contact_correct_after_onset_rate"] = float(np.mean(
        [r["pre_contact_correct"] and not r["commit_before_onset"] for r in with_on])) if with_on else float("nan")
    leads = [r["lead_time_s"] for r in inter if r["pre_contact_correct"]]
    m["lead_time_median_s"] = float(np.median(leads)) if leads else float("nan")
    noi = [r for r in rows if r["label"] == "no_interaction"]
    m["false_alarm_episode_rate"] = float(np.mean([r["n_commits"] > 0 for r in noi])) if noi else float("nan")
    minutes = sum(r["duration_s"] for r in noi) / 60
    m["false_alarms_per_min"] = sum(r["n_commits"] for r in noi) / minutes if minutes > 0 else float("nan")

    # frame/window-level macro-F1 over frames with a defined label
    wy, wp = [], []
    for tr in traces:
        ep = tr["episode"]
        for e, pr in enumerate(tr["probs"]):
            a, _ = frame_labels(ep, e / fps, policy, no_int)
            if a != IGNORE:
                wy.append(a)
                wp.append(int(pr.argmax()))
    m["window_macro_f1"] = _macro_f1(wy, wp, C) if wy else float("nan")

    # accuracy vs time-to-contact and vs onset->contact progress
    def at_time(tr, t):
        e = int(round(t * fps))
        return None if e < 0 or e >= len(tr["probs"]) else e

    by_ttc, by_prog = {}, {}
    rmap = {r["episode_id"]: r for r in rows}
    for d in cfg.eval.ttc_offsets:
        acc, com = [], []
        for tr in traces:
            ep = tr["episode"]
            if ep.label == no_int or math.isnan(ep.t_contact):
                continue
            e = at_time(tr, ep.t_contact - d)
            if e is None:
                continue
            acc.append(int(tr["probs"][e].argmax()) == ep.label)
            r = rmap[ep.episode_id]
            com.append(r["pred"] == classes[ep.label] and r["t_first_commit"] <= e / fps)
        by_ttc[f"-{d:g}s"] = {"argmax_acc": float(np.mean(acc)) if acc else float("nan"),
                              "committed_correct": float(np.mean(com)) if com else float("nan"), "n": len(acc)}
    for q in cfg.eval.progress_points:
        acc = []
        for tr in traces:
            ep = tr["episode"]
            if ep.label == no_int or math.isnan(ep.t_contact) or math.isnan(ep.t_onset):
                continue
            e = at_time(tr, ep.t_onset + q * (ep.t_contact - ep.t_onset))
            if e is not None:
                acc.append(int(tr["probs"][e].argmax()) == ep.label)
        by_prog[f"{q:g}"] = {"argmax_acc": float(np.mean(acc)) if acc else float("nan"), "n": len(acc)}
    m["acc_vs_time_to_contact"] = by_ttc
    m["acc_vs_progress"] = by_prog

    if with_ci and int(cfg.eval.bootstrap) > 0:
        m["episode_macro_f1_ci95"] = person_bootstrap_ci(rows, classes, int(cfg.eval.bootstrap),
                                                         int(cfg.experiment.seed))
    return m


def person_bootstrap_ci(rows, classes, n_boot, seed):
    """95% CI of episode macro-F1, resampling *persons* (episodes of a person stay together)."""
    by_p = defaultdict(list)
    for r in rows:
        by_p[r["person_id"]].append(r)
    persons = list(by_p)
    if len(persons) < 2:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    C, vals = len(classes), []
    for _ in range(n_boot):
        sample = [r for pid in rng.choices(persons, k=len(persons)) for r in by_p[pid]]
        vals.append(_macro_f1([classes.index(r["label"]) for r in sample],
                              [classes.index(r["pred"]) for r in sample], C))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summary_line(m: dict) -> str:
    ci = m.get("episode_macro_f1_ci95")
    ci_s = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
    return (f"epF1={m['episode_macro_f1']:.3f}{ci_s} winF1={m['window_macro_f1']:.3f} "
            f"preContact={m['pre_contact_correct_rate']:.3f} (afterOnset={m['pre_contact_correct_after_onset_rate']:.3f}, "
            f"beforeOnset={m['commit_before_onset_rate']:.3f}) lead={m['lead_time_median_s']:.2f}s "
            f"FA/ep={m['false_alarm_episode_rate']:.3f}")
