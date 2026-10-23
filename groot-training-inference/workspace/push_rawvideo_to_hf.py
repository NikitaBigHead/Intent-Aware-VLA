#!/usr/bin/env python3
"""Push cosmos3_g1right_lerobot_rawvideo (the true-raw-video-fixed 13D
right-arm+BrainCo-hand LeRobot dataset that trained cosmos_right13) to the
org dataset repo, as a new subfolder alongside the existing
g1_robohamer_340_lerobot/ (a DIFFERENT, 46D conversion of the same 340 raw
samples -- do not confuse the two, do not overwrite that folder)."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DATASET_DIR = Path("/data/project/cosmos3_g1right_lerobot_rawvideo")
REPO_ID = os.environ.get("HF_DATASET_REPO")
PATH_IN_REPO = "cosmos3_g1right_lerobot_rawvideo"

token = os.environ.get("HF_TOKEN_WRITE")
if not token:
    sys.exit("HF_TOKEN_WRITE is not set in the environment")
if not REPO_ID:
    sys.exit("HF_DATASET_REPO is not set in the environment")

api = HfApi(token=token)
print(f"[push] uploading {DATASET_DIR} -> {REPO_ID}/{PATH_IN_REPO} (this may take a while)")
api.upload_folder(
    folder_path=str(DATASET_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=PATH_IN_REPO,
    commit_message=(
        "Add cosmos3_g1right_lerobot_rawvideo: true-raw-video fix of the "
        "13D right-arm+BrainCo-hand LeRobot set used to train cosmos_right13 "
        "(banner/2x-upscale from label_video.py removed, genuine 320x176 "
        "Cosmos3 pixels pulled from the source server's surviving pre-label "
        "video_gen/ "
        "copies, not a crop/downscale reconstruction)"
    ),
)
print(f"[push] done: https://huggingface.co/datasets/{REPO_ID}/tree/main/{PATH_IN_REPO}")
