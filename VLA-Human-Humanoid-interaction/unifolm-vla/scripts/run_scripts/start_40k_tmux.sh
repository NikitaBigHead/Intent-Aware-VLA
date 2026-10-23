#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla
LOG=/workspace/project/uniformvla_chpts/extended_sep24_interact60_40k.log
SESSION=uniformvla-40k

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists" >&2
  exit 1
fi
cd "$ROOT"
CHECK_ONLY=1 bash scripts/run_scripts/run_40k.sh
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash scripts/run_scripts/run_40k.sh > '$LOG' 2>&1"
echo "Started $SESSION; log: $LOG"
