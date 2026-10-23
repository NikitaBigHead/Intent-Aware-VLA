#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="/data/project/beingh-venv/bin:$PATH"
export DATASET_PATH="/root/tactile_ws/datasets/interact_percents/extended-sep-24-with-interact-60"
export PRETRAIN_MODEL="/data/project/beingh_model_zoo/InternVL3_5-2B"
export EXPERT_MODEL="/data/project/beingh_model_zoo/Qwen3-0.6B"
export RESUME_PATH="/data/project/beingh_model_zoo/Being-H05-2B"
export CUDA_VISIBLE_DEVICES=0

for path in "$DATASET_PATH/meta/info.json" "$PRETRAIN_MODEL/model.safetensors" \
            "$EXPERT_MODEL/model.safetensors" "$RESUME_PATH/model.safetensors"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done
command -v torchrun >/dev/null || { echo "torchrun is absent from Being-H environment" >&2; exit 1; }

case "${1:-full}" in
    smoke)
        export OUTPUT_DIR="/data/project/beingh-smoke"
        script="scripts/train/smoketest_g1_sonic_brainco6dof_1xa100.sh"
        ;;
    full)
        export OUTPUT_DIR="/root/tactile_ws/checkpoints/Being-H"
        script="scripts/train/train_g1_sonic_brainco6dof_1xa100.sh"
        ;;
    *) echo "Usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

cd "$BASE_DIR"
exec bash "$script"
