"""Collect results.json of all runs (+ benchmark.csv if present) into one comparison table.

    python scripts/summarize.py runs
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

KEYS = ["episode_macro_f1", "window_macro_f1", "pre_contact_correct_rate",
        "pre_contact_correct_after_onset_rate", "commit_before_onset_rate", "lead_time_median_s",
        "false_alarm_episode_rate"]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    bench = {}
    bp = root / "benchmark" / "benchmark.csv"
    if bp.exists():
        for r in csv.DictReader(open(bp)):
            bench[(r["encoder"], r["temporal"])] = r
    rows = []
    for rp in sorted(root.glob("*/results.json")):
        res = json.loads(rp.read_text())
        cfg_txt = (rp.parent / "config.yaml").read_text()
        import yaml
        cfg = yaml.safe_load(cfg_txt)
        enc, tmp = cfg["model"]["encoder"]["name"], cfg["model"]["temporal"]["name"]
        row = {"run": rp.parent.name, "encoder": enc, "temporal": tmp}
        for split in ("val", "test"):
            m = res.get(split, {})
            for k in KEYS:
                row[f"{split}_{k}"] = m.get(k)
            ci = m.get("episode_macro_f1_ci95")
            row[f"{split}_epF1_ci95"] = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        b = bench.get((enc, tmp), {})
        row["stream_fps"] = b.get("stream_fps")
        row["vram_stream_mb"] = b.get("vram_stream_mb")
        row["params_total_M"] = res.get("params", {}).get("params_total_M")
        rows.append(row)
    if not rows:
        print("no results found"); return
    out = root / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cols = ["run", "test_episode_macro_f1", "test_epF1_ci95", "test_pre_contact_correct_after_onset_rate",
            "test_lead_time_median_s", "test_false_alarm_episode_rate", "val_episode_macro_f1", "stream_fps"]
    def fmt(v):
        if v is None or v == "":
            return "-"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
