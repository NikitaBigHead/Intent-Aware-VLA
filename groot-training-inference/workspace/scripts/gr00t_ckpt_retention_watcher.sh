#!/usr/bin/env bash
# Keeps at most TWO checkpoint-<step>/ dirs on disk for a HF-Trainer/GR00T finetune run:
# (a) LATEST complete checkpoint, (b) BEST complete checkpoint by smoothed train loss.
# Everything else is removed as soon as a new checkpoint finishes writing, so the run
# never needs headroom for a 3rd checkpoint mid-save (root cause of the 2026-09-28
# cosmos_right13 disk-full crash at step 7500: two ~23G checkpoints already on disk left
# no room for the next optimizer.pt write).
#
# Completeness marker: HF Trainer's _save_checkpoint() writes model + optimizer.pt first,
# trainer_state.json LAST. A checkpoint dir without trainer_state.json is still being
# written (or died mid-write) -- never treat it as done, and never delete it while it
# could still be the in-progress save. Same "never delete by order of appearance" rule
# as ckpt_retention_watcher.sh (see CLAUDE.md) -- verify, don't assume.
#
# Usage (on the HOST, not inside the container):
#   CKPT_DIR=/root/tactile_ws/checkpoints/cosmos_right13 \
#     nohup bash scripts/gr00t_ckpt_retention_watcher.sh > /root/tactile_ws/logs/ckpt_watcher.log 2>&1 &
set -euo pipefail

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the checkpoints output dir on the HOST}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
STALE_SECS="${STALE_SECS:-300}"  # incomplete checkpoint untouched this long = crash leftover, safe to reclaim

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Prints a smoothed loss for one checkpoint dir (mean of last 5 log_history points at/before
# its global_step), or "nan" if trainer_state.json is missing/unreadable/has no loss entries.
loss_for_checkpoint() {
  python3 - "$1" <<'PYEOF'
import json, sys
d = sys.argv[1]
try:
    with open(f"{d}/trainer_state.json") as f:
        st = json.load(f)
    step = st.get("global_step")
    hist = [h["loss"] for h in st.get("log_history", [])
            if "loss" in h and h.get("step", 0) <= step]
    if not hist:
        print("nan")
    else:
        tail = hist[-5:]
        print(sum(tail) / len(tail))
except Exception:
    print("nan")
PYEOF
}

log "watcher started: CKPT_DIR=$CKPT_DIR poll=${POLL_INTERVAL}s stale=${STALE_SECS}s"

while true; do
  if [ ! -d "$CKPT_DIR" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  mapfile -t dirs < <(find "$CKPT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
                       | sort -t- -k2 -n)

  complete=()
  for d in "${dirs[@]}"; do
    if [ -f "$d/trainer_state.json" ]; then
      complete+=("$d")
    else
      newest_mtime=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
      if [ -n "$newest_mtime" ]; then
        age=$(( $(date +%s) - ${newest_mtime%.*} ))
        if [ "$age" -gt "$STALE_SECS" ]; then
          log "removing stale incomplete checkpoint $d (no write in ${age}s -- crash leftover)"
          rm -rf "$d"
        fi
      fi
    fi
  done

  if [ "${#complete[@]}" -gt 0 ]; then
    latest="${complete[-1]}"
    best="$latest"
    best_loss=""
    for d in "${complete[@]}"; do
      loss=$(loss_for_checkpoint "$d")
      [ "$loss" = "nan" ] && continue
      if [ -z "$best_loss" ] || (( $(echo "$loss < $best_loss" | bc -l) )); then
        best_loss="$loss"
        best="$d"
      fi
    done

    for d in "${complete[@]}"; do
      if [ "$d" != "$latest" ] && [ "$d" != "$best" ]; then
        log "removing superseded checkpoint $d (keeping latest=$(basename "$latest") best=$(basename "$best") loss=$best_loss)"
        rm -rf "$d"
      fi
    done
  fi

  sleep "$POLL_INTERVAL"
done
