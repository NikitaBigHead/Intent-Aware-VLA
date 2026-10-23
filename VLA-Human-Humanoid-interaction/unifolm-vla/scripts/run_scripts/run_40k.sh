#!/usr/bin/env bash
# Launch the prepared 40k-step UnifoLM run after RLDS and the Python env exist.
set -euo pipefail

TRAIN_VENV="${TRAIN_VENV:-/workspace/project/uniformvla_work/.venv-tfds}"
source "$TRAIN_VENV/bin/activate"

ROOT=/workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla
export OXE_DATA_ROOT="${OXE_DATA_ROOT:-/workspace/project/uniformvla_work/rlds}"
export BASE_VLM="${BASE_VLM:-/workspace/project/uniformvla_work/Unifolm-VLM-Base}"
export RUN_ROOT_DIR=/workspace/project/uniformvla_chpts
export RUN_ID="${RUN_ID:-extended_sep24_interact60_40k}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-6}"

if [[ ! -d "$OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0" ]]; then
  echo "RLDS dataset missing: $OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0" >&2
  exit 1
fi
if ! python -c 'import torch, tensorflow_datasets, unifolm_vla' >/dev/null 2>&1; then
  echo "The UnifoLM training environment is incomplete (torch, tensorflow_datasets, unifolm_vla)." >&2
  exit 1
fi
if ! command -v accelerate >/dev/null; then
  echo "accelerate is missing from the active environment." >&2
  exit 1
fi
if [[ "$BASE_VLM" == /* && ! -d "$BASE_VLM" ]]; then
  echo "Base VLM directory missing: $BASE_VLM" >&2
  exit 1
fi
if [[ "$BASE_VLM" == /* ]]; then
  python - "$BASE_VLM" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if index.is_file():
    names = set(json.loads(index.read_text())["weight_map"].values())
elif (root / "model.safetensors").is_file():
    names = {"model.safetensors"}
else:
    raise SystemExit(f"Base VLM weights are incomplete in {root}")
missing = [name for name in names if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Base VLM weight shards are missing in {root}: {missing}")
PY
fi
free_kb=$(df -Pk "$RUN_ROOT_DIR" | awk 'NR==2 {print $4}')
if (( free_kb < 90 * 1024 * 1024 )); then
  echo "At least 90 GiB free is required at $RUN_ROOT_DIR for two full model checkpoints; found $((free_kb/1024/1024)) GiB." >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU is occupied by another compute process; retry when it is free." >&2
  exit 1
fi

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Training prerequisites are ready."
  exit 0
fi

cd "$ROOT"
exec bash scripts/run_scripts/run_g1_sonic_brainco6dof_1xa100.sh
