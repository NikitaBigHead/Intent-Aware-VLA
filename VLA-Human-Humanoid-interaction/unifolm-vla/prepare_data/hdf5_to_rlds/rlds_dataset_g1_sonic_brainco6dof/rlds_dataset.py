"""RLDS/TFDS builder for the Unitree G1 + BrainCo Revo2 (6-DOF hand) SONIC
motion-token dataset "sep-10-2026-cleaned-merged".

This is a dataset-specific fork of
prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py, not a patch of it in
place -- that file is itself already a per-dataset template hardcoded to a
different, unrelated robot's HDF5 layout (cam_left_high/cam_right_high/
cam_left_wrist/cam_right_wrist image keys, a 19-dim joint action, a
pose17->pose23 EE-pose conversion helper -- none of which exists in this
dataset). Every one of the README's 12 open-source G1 datasets was
evidently built the same way: its own private copy of this file, with only
one generic example checked into the repo. See
prepare_data/convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py's module
docstring for the same note on the HDF5-conversion side.

IMPORTANT -- dataset name / directory name contract: TFDS registers a
GeneratorBasedBuilder subclass under
`tensorflow_datasets.core.naming.camelcase_to_snakecase(cls.__name__)`. The
class below is named literally `g1_sonic_brainco6dof` (all-lowercase, like
the upstream template's own `class rlds_dataset(...)`, which is *not*
idiomatic PEP8 but is exactly what makes `tfds build`register the dataset
under the name "rlds_dataset" verbatim) so that it registers under the name
"g1_sonic_brainco6dof" verbatim -- which must exactly match the key used for
this dataset everywhere else it's registered: oxe/configs.py's
OXE_DATASET_CONFIGS key, oxe/transforms.py's OXE_STANDARDIZATION_TRANSFORMS
key, oxe/mixtures.py's OXE_NAMED_MIXTURES key, and the `data_mix` value
passed on the training CLI. This has NOT been verified by actually running
`tfds build` (no GPU/dataset on this machine) -- if the registered name ends
up different from "g1_sonic_brainco6dof" for any reason, every one of those
other registrations needs to be updated to match, or renamed to match this.

Schema written here (see convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py for
where these HDF5 fields come from):
    /observations/qpos                    (T, 41) float32 -- proprio
    /observations/images/depth_camera_rgb (T, H, W, 3) uint8 -- RGB video,
                                           despite the "depth" in the name
                                           (see oxe/configs.py's naming note)
    /observations/images/webcam           (T, H, W, 3) uint8 -- RGB video
    /action                               (T, 76) float32 -- 64 motion_token
                                           + 6 left_hand_joints + 6
                                           right_hand_joints (action.wbc is
                                           NOT in this HDF5 at all -- excluded
                                           upstream at the conversion step)
    /language_raw                         scalar or (1,) string
No `external_view_camera` key exists anywhere in this HDF5 or in the RLDS
features below -- it was excluded at the LeRobot->HDF5 stage (see
ALLOWED_CAMERA_KEYS in the conversion script), so there is nothing to
exclude again here; this builder simply never references that key.
"""

from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from pathlib import Path

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from conversion_utils import MultiThreadedDatasetBuilder

# Action dim (76) and state dim (41) are the same numbers as ACTION_DIM /
# PROPRIO_DIM in src/unifolm_vla/rlds_dataloader/constants.py's
# G1_SONIC_BRAINCO6DOF_CONSTANTS -- kept as plain literals here (rather than
# importing that module) because this builder runs as a standalone `tfds
# build` process, typically in a separate "rlds_dataset" conda environment
# from the one unifolm_vla's training package is installed into (see
# hdf5_to_rlds/README.md). If you change ACTION_DIM/PROPRIO_DIM in
# constants.py, update these two numbers to match by hand.
ACTION_DIM = 76
STATE_DIM = 41
DIRECT_LEROBOT_ROOT = os.environ.get("LEROBOT_DATASET_ROOT")
IMAGE_SHAPE = (224, 224, 3) if DIRECT_LEROBOT_ROOT else (480, 640, 3)


def _parse_lerobot_episode(parquet_path: str) -> Tuple[str, Any]:
    """Decode one LeRobot episode directly, without an HDF5 intermediate."""
    import json
    import cv2
    import pyarrow.parquet as pq

    root = Path(DIRECT_LEROBOT_ROOT)
    path = Path(parquet_path)
    table = pq.read_table(path, columns=[
        "observation.state", "action.motion_token", "teleop.left_hand_joints",
        "teleop.right_hand_joints", "task_index",
    ])
    count = table.num_rows
    task_indices = set(table.column("task_index").to_pylist())
    if len(task_indices) != 1:
        raise ValueError(f"{path}: expected exactly one task_index, got {task_indices}")
    with (root / "meta/tasks.jsonl").open() as stream:
        tasks = {int(record["task_index"]): record["task"] for record in map(json.loads, stream)}
    task_index = next(iter(task_indices))
    if task_index not in tasks:
        raise ValueError(f"{path}: unknown task_index {task_index}")
    arrays = {name: np.asarray(table.column(name).to_pylist(), dtype=np.float32) for name in [
        "observation.state", "action.motion_token", "teleop.left_hand_joints",
        "teleop.right_hand_joints",
    ]}
    states = arrays["observation.state"]
    actions = np.concatenate((arrays["action.motion_token"], arrays["teleop.left_hand_joints"],
                              arrays["teleop.right_hand_joints"]), axis=1)
    if states.shape != (count, STATE_DIM) or actions.shape != (count, ACTION_DIM):
        raise ValueError(f"{path}: invalid state/action shape {states.shape}/{actions.shape}")
    chunk = path.parent.name
    video_paths = [root / "videos" / chunk / key / f"{path.stem}.mp4" for key in (
        "observation.images.depth_camera_rgb", "observation.images.webcam"
    )]
    videos = [cv2.VideoCapture(str(video_path)) for video_path in video_paths]
    try:
        if not all(video.isOpened() for video in videos):
            raise ValueError(f"Could not open episode videos: {video_paths}")
        episode = []
        for i in range(count):
            frames = []
            for video in videos:
                success, frame = video.read()
                if not success:
                    raise ValueError(f"{path}: video ended before parquet row {i}")
                frame = cv2.resize(frame, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            episode.append({
                "observation": {
                    "depth_camera_rgb": frames[0], "webcam": frames[1], "state": states[i],
                },
                "action": actions[i],
                "discount": 1.0,
                "is_first": i == 0,
                "is_last": i == count - 1,
                "is_terminal": i == count - 1,
                "language_instruction": tasks[task_index],
            })
        for video in videos:
            if video.read()[0]:
                raise ValueError(f"{path}: video has more frames than parquet rows")
    finally:
        for video in videos:
            video.release()
    return str(path), {"steps": episode, "episode_metadata": {"file_path": str(path)}}


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for a list of HDF5 file paths."""

    def _parse_example(episode_path):
        with h5py.File(episode_path, "r") as F:
            actions = F["action"][:]
            states = F["observations"]["qpos"][:]
            images_primary = F["observations"]["images"]["depth_camera_rgb"][:]
            images_wrist = F["observations"]["images"]["webcam"][:]

            language_raw_data = F["language_raw"]
            raw_language = language_raw_data[()] if language_raw_data.shape == () else language_raw_data[0]
            language_instruction = raw_language.decode("utf-8") if isinstance(raw_language, bytes) else str(raw_language)

        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"{episode_path}: expected action dim {ACTION_DIM}, got {actions.shape[-1]}. "
                "Did convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py run against the schema "
                "this builder expects?"
            )
        if states.shape[-1] != STATE_DIM:
            raise ValueError(
                f"{episode_path}: expected state dim {STATE_DIM}, got {states.shape[-1]}."
            )

        episode = []
        for i in range(actions.shape[0]):
            episode.append(
                {
                    "observation": {
                        "depth_camera_rgb": images_primary[i],
                        "webcam": images_wrist[i],
                        "state": np.asarray(states[i], np.float32),
                    },
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "discount": 1.0,
                    "is_first": i == 0,
                    "is_last": i == (actions.shape[0] - 1),
                    "is_terminal": i == (actions.shape[0] - 1),
                    "language_instruction": language_instruction,
                }
            )

        sample = {
            "steps": episode,
            "episode_metadata": {"file_path": episode_path},
        }
        return episode_path, sample

    # For a ~535-episode dataset, single-thread parsing per worker is plenty
    # (MultiThreadedDatasetBuilder already parallelizes across episodes via
    # N_WORKERS below).
    for sample in paths:
        yield _parse_lerobot_episode(sample) if DIRECT_LEROBOT_ROOT else _parse_example(sample)


class g1_sonic_brainco6dof(MultiThreadedDatasetBuilder):
    """DatasetBuilder for the G1 + BrainCo Revo2 SONIC motion-token dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release (sep-10-2026-cleaned-merged, 535 episodes).",
    }
    # 535 episodes total is small enough that the upstream template's
    # defaults (N_WORKERS=8, MAX_PATHS_IN_MEMORY=8) are unchanged here --
    # left as-is rather than tuned, since there's no machine here to
    # benchmark conversion throughput on.
    N_WORKERS = 8
    MAX_PATHS_IN_MEMORY = 8
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # Despite the "depth" in the name, this is an RGB
                                    # stream (see the module docstring and
                                    # oxe/configs.py's naming note) -- encoded/decoded
                                    # as a normal 3-channel jpeg, same as `webcam`.
                                    "depth_camera_rgb": tfds.features.Image(
                                        shape=IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc=(
                                            "RGB feed from the depth-capable camera "
                                            "(not a depth map)."
                                        ),
                                    ),
                                    "webcam": tfds.features.Image(
                                        shape=IMAGE_SHAPE,
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="RGB webcam observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(STATE_DIM,),
                                        dtype=np.float32,
                                        doc=(
                                            "41-dim proprio: legs(12) + waist(3) + left "
                                            "arm+hand(13) + right arm+hand(13)."
                                        ),
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(ACTION_DIM,),
                                dtype=np.float32,
                                doc=(
                                    "76-dim: 64 action.motion_token (SONIC continuous "
                                    "latent) + 6 teleop.left_hand_joints + 6 "
                                    "teleop.right_hand_joints. action.wbc is NOT part of "
                                    "this vector (debug/reference field, excluded upstream)."
                                ),
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32, doc="Discount if provided, default to 1."
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original HDF5 file."),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits.

        No validation split -- all 535/535 episodes go to train (per the
        fine-tune recipe). `RLDSDataset`/`make_dataset_from_rlds`
        (rlds_dataloader/datasets/rlds/dataset.py) will still internally slice
        this single "train" TFDS split into "train[:95%]" / "train[95%:]"
        whenever a validation split is requested elsewhere, since this
        builder doesn't define a `val` split -- that 95/5 slicing is shared
        infrastructure used by every dataset in this repo (not something
        introduced here), and this training recipe's launch script never
        constructs a `train=False` dataset, so nothing here ever reads the
        excluded 5%. See the launch script's comments for the same note.

        UPDATE THE GLOB BELOW to the actual directory containing this
        dataset's converted .hdf5 files before running `tfds build` (same
        requirement the upstream README calls out for the template this was
        copied from).
        """
        if DIRECT_LEROBOT_ROOT:
            paths = sorted(glob.glob(os.path.join(DIRECT_LEROBOT_ROOT, "data", "chunk-*", "*.parquet")))
            if not paths:
                raise ValueError(f"No LeRobot parquet episodes found in {DIRECT_LEROBOT_ROOT}")
            sample_limit = os.environ.get("LEROBOT_SAMPLE_EPISODES")
            if sample_limit:
                paths = paths[:int(sample_limit)]
            return {"train": paths}
        hdf5_root = os.environ.get("HDF5_OUTPUT_DIR")
        if not hdf5_root:
            raise ValueError("Set HDF5_OUTPUT_DIR to the converted HDF5 directory")
        paths = sorted(glob.glob(os.path.join(hdf5_root, "*.hdf5")))
        if not paths:
            raise ValueError(f"No HDF5 episodes found in {hdf5_root}")
        return {"train": paths}
