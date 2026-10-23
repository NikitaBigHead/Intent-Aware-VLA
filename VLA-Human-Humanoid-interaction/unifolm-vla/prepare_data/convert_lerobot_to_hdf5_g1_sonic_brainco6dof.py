"""
Script: LeRobot -> HDF5, for the Unitree G1 + BrainCo Revo2 (6-DOF hand)
SONIC motion-token dataset "sep-10-2026-cleaned-merged".

This is a dataset-specific fork of convert_lerobot_to_hdf5.py, not a drop-in
patch of it. That script is itself already hardcoded to a *different*,
unrelated LeRobot schema (it reads `observation.left_arm`, `action.body`,
`observation.left_ee`, etc. -- a dual-arm-with-waist robot, not this G1 +
SONIC dataset), and the same is true of the checked-in
prepare_data/hdf5_to_rlds/rlds_dataset/rlds_dataset.py template it feeds
into (cam_left_high/cam_right_high/cam_left_wrist/cam_right_wrist, pose17->
pose23 EE conversion -- none of that exists in this dataset). Every one of
the README's 12 open-source G1 datasets was evidently converted with its
own private copy of both scripts (only one generic example of each is
checked into this repo); this file and its RLDS-builder counterpart
(hdf5_to_rlds/rlds_dataset_g1_sonic_brainco6dof/rlds_dataset.py) are that
per-dataset copy for sep-10-2026-cleaned-merged, following the same
per-dataset-script pattern the rest of the repo already uses -- not a new
mechanism.

Dataset schema assumed (from meta/info.json + meta/modality.json; the real
dataset has been deleted locally, so none of this has been run or verified
against actual files -- verify column names against your own dataset's
meta/info.json before running):
    observation.state                    41-dim float, proprio (legs 12 +
                                          waist 3 + left arm+hand 13 +
                                          right arm+hand 13)
    action.motion_token                  64-dim float64, SONIC continuous
                                          latent (THE primary action target)
    teleop.left_hand_joints              6-dim float, BrainCo Revo2 hand,
                                          normalized_0_to_1
    teleop.right_hand_joints             6-dim float, same. In the original
                                          535-episode dataset this is all-zero
                                          across every episode. CORRECTED per
                                          the task owner, after the dataset
                                          grew to ~1000 episodes: not purely
                                          an old-task-set artifact -- new
                                          hug/handshake/fist-bump episodes
                                          move the right ARM (via
                                          action.motion_token / proprio
                                          state, not this field), but the
                                          right hand's own DOF range is
                                          hardware-limited and reportedly
                                          can't close into a full fist or
                                          grip even there. So don't assume
                                          this field is still exactly 0.0 in
                                          the new data, but also don't assume
                                          it suddenly spans a full functional
                                          range -- re-profile real min/max
                                          once the new dataset exists. Always
                                          written out regardless; dropping it
                                          silently would desync ACTION_DIM=76
                                          in rlds_dataloader/constants.py
                                          (see that file's comment for the
                                          full writeup)
    action.wbc                           41-dim float, raw joint targets --
                                          DEBUG/REFERENCE FIELD ONLY, deliberately
                                          NOT read or written by this script.
    observation.images.depth_camera_rgb  RGB video (despite the "depth" in
                                          the name -- see the naming note in
                                          oxe/configs.py's g1_sonic_brainco6dof
                                          entry)
    observation.images.webcam            RGB video
    observation.images.external_view_camera  RGB video -- EXCLUDED, see
                                          ALLOWED_CAMERA_KEYS below.

Usage (same CLI contract as convert_lerobot_to_hdf5.py):
    cd prepare_data
    python convert_lerobot_to_hdf5_g1_sonic_brainco6dof.py \\
        --data_path /path/to/lerobot/sep-10-2026-cleaned-merged \\
        --target_path /path/to/save/hdf5/sep-10-2026-cleaned-merged
"""

import os
import h5py
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Only these two video streams are ever read. `external_view_camera` is
# excluded per the fine-tune recipe (not decoded, not converted, no I/O spent
# on it at all).
ALLOWED_CAMERA_KEYS = {"depth_camera_rgb", "webcam"}
TARGET_IMAGE_SIZE = (640, 480)  # width, height; matches the RLDS builder schema

# 76-dim action target = 64 (motion_token) + 6 (left_hand_joints) +
# 6 (right_hand_joints). Order matches ACTION_DIM's documentation in
# rlds_dataloader/constants.py and ActionEncoding.SONIC_LATENT_G1_BRAINCO6DOF
# in oxe/configs.py -- if you change the order here, update both of those
# comments too, since nothing enforces the ordering across files.
ACTION_COLUMNS = ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"]
STATE_COLUMN = "observation.state"


class LeRobotDataProcessor:
    def __init__(self, repo_id: str, root: str = None) -> None:
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")

        # LeRobotDataset.__getitem__ decodes every key in self.dataset.meta.video_keys
        # unconditionally -- there is no constructor argument to restrict which
        # camera/video columns get decoded (verified by reading LeRobotDataset's
        # source at the exact commit this repo's README installs,
        # git+https://github.com/huggingface/lerobot.git@0878c68: `_get_query_timestamps`
        # and `_query_videos` both iterate `self.meta.video_keys` fresh on every
        # call, not a value cached once in __init__). Removing the excluded key from
        # that list here, before any __getitem__ call, is therefore both necessary
        # and sufficient to stop `external_view_camera` mp4s from ever being opened
        # or decoded -- not just dropped after the fact.
        if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "video_keys"):
            before = list(self.dataset.meta.video_keys)
            self.dataset.meta.video_keys = [k for k in before if k in ALLOWED_CAMERA_KEYS]
            dropped = sorted(set(before) - set(self.dataset.meta.video_keys))
            if dropped:
                print(f"[convert_lerobot_to_hdf5_g1_sonic_brainco6dof] Excluding video keys from decode: {dropped}")
        else:
            # Flagged, not silently ignored: if a future lerobot version restructures
            # `meta.video_keys`, this filter silently stops working and
            # external_view_camera would get decoded again. Verify this attribute
            # still exists against your installed lerobot version before trusting
            # the exclusion.
            print(
                "[convert_lerobot_to_hdf5_g1_sonic_brainco6dof] WARNING: could not find "
                "dataset.meta.video_keys on this LeRobotDataset -- external_view_camera "
                "exclusion could NOT be verified/applied. Check your installed lerobot "
                "version before running a real conversion."
            )

    def process_episode(self, episode_index: int) -> dict:
        """Process a single episode to extract camera images, state, and the 76-dim action."""
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()

        episode = defaultdict(list)
        cameras = defaultdict(list)
        task = None

        for step_idx in tqdm(
            range(from_idx, to_idx), desc=f"Episode {episode_index}", position=1, leave=False, dynamic_ncols=True
        ):
            step = self.dataset[step_idx]

            # Belt-and-suspenders: even if meta.video_keys filtering above ever
            # silently stops working, this comprehension still refuses to write
            # anything but the two allowed camera keys into the HDF5 output.
            image_dict = {}
            for key, value in step.items():
                if key.startswith("observation.image") and key.split(".")[-1] in ALLOWED_CAMERA_KEYS:
                    frame = np.transpose((value.numpy() * 255).astype(np.uint8), (1, 2, 0))
                    # The actual cameras are 1280x720 and 1600x896. They must
                    # have one common size for the HDF5 writer and TFDS schema.
                    if (frame.shape[1], frame.shape[0]) != TARGET_IMAGE_SIZE:
                        frame = np.asarray(Image.fromarray(frame).resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR))
                    image_dict[key.split(".")[-1]] = frame
            for key, value in image_dict.items():
                cameras[key].append(value)

            state = step[STATE_COLUMN].numpy().astype(np.float32)

            # action.wbc is intentionally never read here -- debug/reference field only.
            action = np.concatenate(
                [np.asarray(step[col], dtype=np.float32).reshape(-1) for col in ACTION_COLUMNS]
            )
            if action.shape[0] != 76:
                raise ValueError(
                    f"Expected a 76-dim action (64 motion_token + 6 + 6 hand joints), got "
                    f"{action.shape[0]}. Check ACTION_COLUMNS against this dataset's actual "
                    f"meta/info.json column shapes before proceeding."
                )

            episode["state"].append(state)
            episode["action"].append(action)
            task = step["task"]

        if not image_dict:
            raise ValueError(
                f"Episode {episode_index}: no frames found for camera keys {ALLOWED_CAMERA_KEYS}. "
                "Check that depth_camera_rgb/webcam are the real column names in this dataset's "
                "meta/modality.json -- they were assumed from the schema description, not verified "
                "against real files."
            )
        cam_height, cam_width = next(iter(image_dict.values())).shape[:2]

        episode["cameras"] = cameras
        episode["task"] = task
        episode["episode_length"] = to_idx - from_idx
        episode["data_cfg"] = {
            "camera_names": list(image_dict.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": np.squeeze(np.asarray(episode["state"][0]).shape),
            "action_dim": np.squeeze(np.asarray(episode["action"][0]).shape),
        }
        episode["episode_index"] = episode_index

        return episode


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        episode_length = episode["episode_length"]
        episode_index = episode["episode_index"]
        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        data_cfg = episode["data_cfg"]

        data_dict = {
            "/observations/qpos": [state],
            "/action": [action],
            **{f"/observations/images/{k}": [v] for k, v in cameras.items()},
        }

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
            root.attrs["sim"] = False

            obs = root.create_group("observations")
            image = obs.create_group("images")

            for cam_name in cameras.keys():
                image.create_dataset(
                    cam_name,
                    shape=(episode_length, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    dtype="uint8",
                    chunks=(1, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    compression="gzip",
                )

            obs.create_dataset("qpos", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("action", (episode_length, data_cfg["action_dim"]), dtype="float32", compression="gzip")

            root.create_dataset("language_raw", data=task)

            for name, array in data_dict.items():
                root[name][...] = array


def lerobot_to_h5(repo_id: str, output_dir: Path, root: str = None) -> None:
    data_processor = LeRobotDataProcessor(repo_id, root)
    h5_writer = H5Writer(output_dir)

    for episode_index in tqdm(
        range(data_processor.dataset.num_episodes), desc="Episodes", position=0, dynamic_ncols=True
    ):
        if os.path.exists(os.path.join(output_dir, f"episode_{episode_index}.hdf5")):
            print(f"Episode {episode_index} already exists")
            continue
        episode = data_processor.process_episode(episode_index)
        h5_writer.write_to_h5(episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", help="Path to the LeRobot v2.1 dataset root.")
    parser.add_argument("--target_path", type=str, default="", help="Directory to write per-episode .hdf5 files to.")
    args = parser.parse_args()
    repo_id = os.path.basename(args.data_path)
    root_path = args.data_path
    output_dir = args.target_path
    lerobot_to_h5(repo_id, output_dir, root_path)
