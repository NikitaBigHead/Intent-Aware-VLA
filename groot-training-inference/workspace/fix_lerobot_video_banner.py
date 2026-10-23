"""Rebuilds cosmos3_g1right_lerobot's video files at their TRUE native Cosmos3
resolution, undoing label_video.py's transform (banner_h=40, scale=2 cubic
upscale -- see runs/generate_first_frame/video_generate_from_external/scripts/
label_video.py in the main repo, confirmed math: 640 = 320*2, 392 = 176*2+40).

The raw pre-label frames themselves are gone (the batch dirs under
hf_release_cosmos3_140_200/*/*/ only ever kept the labeled generated_video.mp4
+ qpos), so this is a reconstruction: crop the 40px banner off (exact, no
interpolation), then downscale 640x352 -> 320x176 with INTER_AREA (undoes the
2x cubic upscale as closely as achievable from what's on disk -- NOT itself
an upscale). Parquet/meta are untouched except info.json's video feature
shape. Writes a NEW dataset root, does not modify the original.
"""
import json
import shutil
import subprocess
from pathlib import Path

SRC = Path("/data/project/cosmos3_g1right_lerobot")
DST = Path("/data/project/cosmos3_g1right_lerobot_rawvideo")
VIDEO_KEY = "observation.images.ego_view"

BANNER_H = 40
LABELED_W, LABELED_H = 640, 392
RAW_W, RAW_H = 320, 176
assert LABELED_H == RAW_H * 2 + BANNER_H
assert LABELED_W == RAW_W * 2


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "meta").mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True)

    for name in ("episodes.jsonl", "modality.json", "relative_stats.json", "stats.json", "tasks.jsonl"):
        shutil.copy(SRC / "meta" / name, DST / "meta" / name)

    info = json.loads((SRC / "meta" / "info.json").read_text())
    vf = info["features"][VIDEO_KEY]
    vf["shape"] = [RAW_H, RAW_W, 3]
    vf["info"]["video.height"] = RAW_H
    vf["info"]["video.width"] = RAW_W
    (DST / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    for p in sorted((SRC / "data" / "chunk-000").glob("*.parquet")):
        shutil.copy(p, DST / "data" / "chunk-000" / p.name)

    src_videos = sorted((SRC / "videos" / "chunk-000" / VIDEO_KEY).glob("*.mp4"))
    print(f"[fix] {len(src_videos)} videos to reprocess")
    # label_video.py also burns a per-frame "X.XXs" timestamp bottom-right of
    # the content area (cv2.putText origin (out_w-80, out_h-10), scale 0.5) --
    # a second contamination distinct from the banner. Blank it BEFORE the
    # crop/downscale (drawbox on the original 640x392 canvas, x>=540 y>=355 --
    # comfortably covers the text with margin, confirmed background-only in
    # this rig's framing, not arm/hand content) rather than leave it baked
    # into "raw" frames.
    vf_filter = (
        f"drawbox=x=540:y=355:w={LABELED_W - 540}:h={LABELED_H - 355}:color=black:t=fill,"
        f"crop={LABELED_W}:{LABELED_H - BANNER_H}:0:{BANNER_H},scale={RAW_W}:{RAW_H}:flags=area"
    )
    for i, src in enumerate(src_videos):
        dst = DST / "videos" / "chunk-000" / VIDEO_KEY / src.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-vf", vf_filter, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-an", str(dst),
        ]
        subprocess.run(cmd, check=True)
        if (i + 1) % 50 == 0 or i == len(src_videos) - 1:
            print(f"  [{i + 1}/{len(src_videos)}] {src.name} -> {dst.name}")

    print(f"[fix] done -> {DST}")


if __name__ == "__main__":
    main()
