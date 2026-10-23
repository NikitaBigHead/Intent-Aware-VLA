#!/usr/bin/env bash
# Thin wrapper around NVIDIA's OWN open-loop eval script, already installed
# in this container at /opt/Isaac-GR00T/gr00t/eval/open_loop_eval.py -- it
# already computes unnormalized action MSE/MAE (GT vs predicted, chunked by
# --action-horizon) and per-dimension GT/state/pred plots. No need to
# reimplement that; this script only fills in cosmos_right13's paths so it's
# a one-liner to run. See overlay_openloop_rollout.py in this same folder
# for the RGB+MuJoCo visual overlay that open_loop_eval.py does NOT produce.
#
# Usage: bash run_mse_mae_eval.sh [checkpoint-dir] [traj_ids...]
#   (checkpoint-dir defaults to the newest checkpoint-N/ under CKPT_ROOT;
#    traj_ids default to "0 1 2")
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-/workspace/checkpoints/cosmos_right13}"
DATASET_PATH="${DATASET_PATH:-/data/project/cosmos3_g1right_lerobot}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
STEPS="${STEPS:-96}"
OUT_DIR="${OUT_DIR:-/workspace/validation_VLA/out}"

MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' \
               -exec test -e '{}/trainer_state.json' \; -print \
               | sort -t- -k2 -n | tail -1)
  if [ -z "$MODEL_PATH" ]; then
    echo "no complete checkpoint-N/ under $CKPT_ROOT" >&2
    exit 1
  fi
fi
shift || true
TRAJ_IDS="${*:-0 1 2}"

mkdir -p "$OUT_DIR"
echo "[run_mse_mae_eval] model=$MODEL_PATH traj_ids=$TRAJ_IDS"

/opt/Isaac-GR00T/.venv/bin/python /opt/Isaac-GR00T/gr00t/eval/open_loop_eval.py \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --embodiment_tag NEW_EMBODIMENT \
  --steps "$STEPS" \
  --action_horizon "$ACTION_HORIZON" \
  --traj_ids $TRAJ_IDS \
  --save_plot_path "$OUT_DIR/open_loop_eval_$(basename "$MODEL_PATH").jpeg"
