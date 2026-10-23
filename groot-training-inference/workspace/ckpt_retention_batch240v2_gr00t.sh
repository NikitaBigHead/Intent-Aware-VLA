#!/bin/bash
set -u
CKPT_BASE=/root/tactile_ws/checkpoints/batch240v2_8objects_right13_rejected106_20261002
POLL_SECONDS=30
while true; do
  sleep "$POLL_SECONDS"
  [ -d "$CKPT_BASE" ] || continue
  mapfile -t ALL_DIRS < <(ls -d "$CKPT_BASE"/checkpoint-*/ 2>/dev/null | sort -V)
  [ "${#ALL_DIRS[@]}" -ge 1 ] || continue
  NEWEST_DIR="${ALL_DIRS[${#ALL_DIRS[@]}-1]}"
  NEWEST=$(basename "$NEWEST_DIR")
  SAFE=""
  for ((i=${#ALL_DIRS[@]}-1; i>=0; i--)); do
    d="${ALL_DIRS[$i]}"
    if [ -f "${d}trainer_state.json" ]; then SAFE=$(basename "$d"); break; fi
  done
  for DIR in "${ALL_DIRS[@]}"; do
    ITER=$(basename "$DIR")
    if [ "$ITER" = "$SAFE" ] || [ "$ITER" = "$NEWEST" ]; then continue; fi
    rm -rf -- "$DIR"
    echo "[batch240v2-retention] $(date -u +%FT%TZ) pruned=$ITER kept_safe=$SAFE newest=$NEWEST"
  done
done
