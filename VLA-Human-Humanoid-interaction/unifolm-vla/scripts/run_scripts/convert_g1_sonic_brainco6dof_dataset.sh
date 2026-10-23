#!/bin/bash
# 3-stage data conversion for the Unitree G1 + BrainCo Revo2 SONIC
# motion-token dataset "sep-10-2026-cleaned-merged":
#
#   Stage 1: LeRobot v2.1 dataset  ->  per-episode HDF5 files
#   Stage 2: (implicit -- just a directory of .hdf5 files from stage 1)
#   Stage 3: HDF5  ->  RLDS/TFDS dataset (versioned `1.0.0/` dir), via `tfds build`
#
# This has NOT been run -- there is no GPU and the real dataset was deleted
# from this machine before this adaptation was written. Every path/column
# name below is taken from the fine-tune brief's schema description (itself
# read from meta/info.json + meta/modality.json before those files were also
# removed), not verified against real files. Read
# prepare_data/convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py and
# prepare_data/hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof/rlds_dataset.py's
# module docstrings for the full list of assumptions before running this for
# real -- in particular, verify `action.motion_token` /
# `teleop.left_hand_joints` / `teleop.right_hand_joints` / `observation.state`
# are the actual LeRobot column names in YOUR copy of the dataset (via
# `python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset;
# d = LeRobotDataset(repo_id='sep-10-2026-cleaned-merged', root='/path/to/it');
# print(d.features.keys())"`) before trusting this script.
#
# THIS SCRIPT IS NOT EXECUTED AUTOMATICALLY BY ANYTHING -- run it yourself,
# stage by stage if you want to inspect intermediate output, after setting
# the env vars below. The two stages need DIFFERENT conda environments (see
# prepare_data/hdf5_to_rlds/README.md): stage 1 needs `lerobot` installed
# (python >= 3.10), stage 3 needs `tensorflow_datasets` (the "rlds_dataset"
# env referenced in that README) -- these can conflict, hence two envs.

set -euo pipefail

: "${LEROBOT_DATASET_ROOT:?Set LEROBOT_DATASET_ROOT to the LeRobot v2.1 dataset root, e.g. .../sep-10-2026-cleaned-merged}"
: "${HDF5_OUTPUT_DIR:?Set HDF5_OUTPUT_DIR to where per-episode .hdf5 files should be written}"
: "${OXE_DATA_ROOT:?Set OXE_DATA_ROOT to the RLDS data_dir root (same value you will later pass as OXE_DATA_ROOT to run_g1_sonic_brainco6dof_1xa100.sh)}"

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Stage 1: LeRobot -> HDF5 ==="
echo "(run inside a conda env with lerobot installed, per this repo's own README install instructions)"
CMD1=(python "$BASE_DIR/prepare_data/convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py"
    --data_path "$LEROBOT_DATASET_ROOT"
    --target_path "$HDF5_OUTPUT_DIR")
echo "About to run: ${CMD1[*]}"
"${CMD1[@]}"

echo ""
echo "=== Stage 3: HDF5 -> RLDS/TFDS ==="
echo "(switch to a conda env with tensorflow_datasets installed -- see"
echo " prepare_data/hdf5_to_rlds/README.md. This is a SEPARATE env from stage 1"
echo " in the upstream workflow, not merely a suggestion.)"
echo ""
echo "First, point the builder's train split at your HDF5 output -- edit the glob in"
echo "  $BASE_DIR/prepare_data/hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof/rlds_dataset.py"
echo "  (_split_paths) to: glob.glob(\"$HDF5_OUTPUT_DIR/*.hdf5\")"
echo "(not done automatically by this script -- it's a one-line source edit, see that file's"
echo " _split_paths docstring)."
echo ""
CMD2=(tfds build --data_dir "$OXE_DATA_ROOT")
echo "Then, from $BASE_DIR/prepare_data/hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof/, run:"
echo "  ${CMD2[*]}"
echo ""
echo "After this completes, verify the result exists at:"
echo "  $OXE_DATA_ROOT/g1_sonic_brainco6dof/1.0.0/"
echo "(this is the same OXE_DATA_ROOT you pass to run_g1_sonic_brainco6dof_1xa100.sh --"
echo " that script does a best-effort check for exactly this path)."
echo ""
echo "This script does NOT cd into the builder directory or run tfds build for you --"
echo "the source-file edit above has to happen first, by hand, so auto-running tfds"
echo "build here could silently build against the wrong (placeholder) glob path."
