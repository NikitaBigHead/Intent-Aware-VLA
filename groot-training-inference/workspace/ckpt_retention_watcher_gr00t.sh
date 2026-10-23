#!/bin/bash
# Adapted checkpoint-retention watcher for Isaac-GR00T HF-Trainer-style finetune
# checkpoints (checkpoint-N/ dirs with trainer_state.json) -- NOT the DCP/iter_N
# convention that scripts/ckpt_retention_watcher.sh (GPU-03, cosmos_framework)
# targets; different framework, different on-disk layout, so a separate script.
#
# Same proven safety rule as that one, ported to this framework's own
# "is this checkpoint actually complete" signal: a checkpoint-N/ dir is SAFE
# only once its trainer_state.json exists (HF Trainer writes it last, after
# model+optimizer+scheduler+rng_state). Confirmed necessary 2026-09-30 on
# cosmos_right13_rawvideo: a disk-full crash left checkpoint-15000 with model
# weights but no trainer_state.json/optimizer.pt -- unresumable, since
# get_last_checkpoint() only matches the directory-name pattern (ignores
# completeness) and gr00t/experiment/trainer.py unconditionally calls
# TrainerState.load_from_json(checkpoint/trainer_state.json) on resume,
# crashing outright if picked. Keep ONLY:
#   (a) the newest checkpoint-*/ dir that HAS trainer_state.json -- safe to
#       resume from.
#   (b) the single newest checkpoint-*/ dir overall, in case it is mid-write
#       (no-op if identical to (a)).
# Everything else under CKPT_BASE/checkpoint-*/ is deleted.
#
# Usage: CKPT_BASE=/path/to/output_dir bash ckpt_retention_watcher_gr00t.sh
set -u
CKPT_BASE="${CKPT_BASE:?set CKPT_BASE to the training output dir}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "[ckpt-retention-gr00t] started at $(date -u +%FT%TZ), watching $CKPT_BASE (poll ${POLL_SECONDS}s)"

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
        if [ -f "${d}trainer_state.json" ]; then
            SAFE=$(basename "$d")
            break
        fi
    done

    for DIR in "${ALL_DIRS[@]}"; do
        ITER=$(basename "$DIR")
        if [ "$ITER" = "$SAFE" ] || [ "$ITER" = "$NEWEST" ]; then
            continue
        fi
        rm -rf "$DIR"
        echo "[ckpt-retention-gr00t] $(date -u +%FT%TZ) pruned $ITER (kept safe=$SAFE newest=$NEWEST)"
    done
done
