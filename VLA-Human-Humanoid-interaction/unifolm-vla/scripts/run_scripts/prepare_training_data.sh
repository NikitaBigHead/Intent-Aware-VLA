#!/usr/bin/env bash
# Stream the interact-60 LeRobot dataset directly into TFDS, without HDF5.
set -euo pipefail

ROOT=/workspace/project/VLA-Human-Humanoid-interaction/unifolm-vla
export LEROBOT_DATASET_ROOT=/workspace/project/extended-sep-24-with-interact-60
WORK_ROOT="${WORK_ROOT:-/workspace/project/uniformvla_work}"
export OXE_DATA_ROOT="${OXE_DATA_ROOT:-$WORK_ROOT/rlds}"
TFDS_BIN="${TFDS_BIN:-$WORK_ROOT/.venv-tfds/bin/tfds}"
export TMPDIR="${TMPDIR:-/dev/shm/uniformvla-tfds-tmp}"
export TFDS_SHUFFLE_DIR="${TFDS_SHUFFLE_DIR:-$TMPDIR/shuffle}"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "$WORK_ROOT" "$OXE_DATA_ROOT" "$TMPDIR" "$TFDS_SHUFFLE_DIR"
free_root_kb=$(df -Pk "$OXE_DATA_ROOT" | awk 'NR==2 {print $4}')
free_temp_kb=$(df -Pk "$TMPDIR" | awk 'NR==2 {print $4}')
if (( free_root_kb < 20 * 1024 * 1024 )); then
  echo "Need at least 20 GiB free at $OXE_DATA_ROOT; found $((free_root_kb/1024/1024)) GiB." >&2
  exit 1
fi
if (( free_temp_kb < 20 * 1024 * 1024 )); then
  echo "Need at least 20 GiB free at $TMPDIR; found $((free_temp_kb/1024/1024)) GiB." >&2
  exit 1
fi
if [[ ! -x "$TFDS_BIN" ]]; then
  echo "TFDS executable missing: $TFDS_BIN" >&2
  exit 1
fi
if [[ -n "${LEROBOT_SAMPLE_EPISODES:-}" ]]; then
  echo "Unset LEROBOT_SAMPLE_EPISODES before a full build." >&2
  exit 1
fi

cd "$ROOT/prepare_data/hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof"
"$TFDS_BIN" build rlds_dataset.py --data_dir "$OXE_DATA_ROOT"
test -d "$OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0"
echo "RLDS ready: $OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0"
