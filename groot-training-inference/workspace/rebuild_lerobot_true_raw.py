"""Supersedes fix_lerobot_video_banner.py's crop/downscale reconstruction:
the TRUE pre-label raw Cosmos3 output is still on GPU-03 (never deleted),
found via video_generation.py's own `run_dir / "generated_video.mp4"`
(labeled copy = `out_dir / f"03_video_{run_name}_seed{seed}.mp4"`, see that
file's `generate_video_for_frame()`). Pulls the genuine raw pixels instead
of reconstructing them, then reproduces convert_cosmos_to_lerobot.py's own
decimate_video() so frame timing/count stays identical to what the existing
parquet (qpos-derived state/action) already assumes -- only pixels change.

Run INSIDE tactile-train (needs python+ffmpeg+/data mount). GPU-03 is
reachable from here via the existing relay key (same one pull_140_videos.sh
uses), NOT from tactile-train directly -- this script shells out to a host
scp helper for each file (see _pull_raw()) which SSHes host->host, then reads
the result from the shared /data mount... actually simpler: run the pull
step on the HOST (not in-container) since that's where the relay key lives,
then hand off to this script (in-container) for the ffmpeg decimation. See
the two-stage split below: RAW_STAGE is populated by a host-side scp loop
BEFORE this script runs; this script only decimates+assembles.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

HF_RELEASE = Path("/data/skurchev/workspace/datasets/hf_release_cosmos3_140_200")
RAW_STAGE = Path("/data/skurchev/workspace/datasets/raw_regen_340")  # populated by host pull step
SRC_DS = Path("/data/project/cosmos3_g1right_lerobot")          # original (banner) dataset, for meta/data
DST_DS = Path("/data/project/cosmos3_g1right_lerobot_rawvideo")  # overwritten with TRUE raw video
VIDEO_KEY = "observation.images.ego_view"
FPS = 15


def discover_samples():
    """Exact same sort order as convert_cosmos_to_lerobot.py's discover_samples()
    -- this IS what assigned episode_index originally, so reusing it here
    keeps every episode_index meaning the same sample."""
    items = []
    b140 = HF_RELEASE / "batch_140_sam3d_objects"
    for d in sorted(b140.iterdir()):
        if (d / "training_data_g1brainco_robohamer.npz").exists():
            items.append(d.name)
    b200 = HF_RELEASE / "batch_200_oreo_grasp"
    for d in sorted(b200.iterdir()):
        if (d / "qpos_robohamer.npz").exists():
            items.append(d.name)
    return items


def decimate_video(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"select='not(mod(n\\,2))',setpts=N/{FPS}/TB",
        "-r", str(FPS), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(dst)],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip().split(",")[0])


def main() -> None:
    samples = discover_samples()
    print(f"[rebuild] {len(samples)} samples (expect 340)")
    assert len(samples) == 340, f"expected 340, got {len(samples)} -- discover_samples() drifted from the original converter"

    if DST_DS.exists():
        shutil.rmtree(DST_DS)
    (DST_DS / "meta").mkdir(parents=True)
    (DST_DS / "data" / "chunk-000").mkdir(parents=True)
    (DST_DS / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True)

    for name in ("episodes.jsonl", "modality.json", "relative_stats.json", "stats.json", "tasks.jsonl"):
        shutil.copy(SRC_DS / "meta" / name, DST_DS / "meta" / name)
    for p in sorted((SRC_DS / "data" / "chunk-000").glob("*.parquet")):
        shutil.copy(p, DST_DS / "data" / "chunk-000" / p.name)

    info = json.loads((SRC_DS / "meta" / "info.json").read_text())
    vf = info["features"][VIDEO_KEY]
    vf["shape"] = [176, 320, 3]
    vf["info"]["video.height"] = 176
    vf["info"]["video.width"] = 320
    (DST_DS / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    mismatches = []
    for ep, sample_id in enumerate(samples):
        raw_src = RAW_STAGE / f"{sample_id}.mp4"
        if not raw_src.exists():
            raise SystemExit(f"missing staged raw video for episode {ep} ({sample_id}): {raw_src}")

        old_video = SRC_DS / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{ep:06d}.mp4"
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(old_video)],
            capture_output=True, text=True,
        )
        expected_n = int(r.stdout.strip().split(",")[0])

        dst = DST_DS / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{ep:06d}.mp4"
        n = decimate_video(raw_src, dst)
        if n != expected_n:
            mismatches.append((ep, sample_id, expected_n, n))

        if (ep + 1) % 50 == 0 or ep == len(samples) - 1:
            print(f"  [{ep + 1}/340] {sample_id} -> episode_{ep:06d}.mp4 ({n} frames, expected {expected_n})")

    if mismatches:
        print(f"[rebuild] WARNING: {len(mismatches)} frame-count mismatches (old banner-video vs new raw-video):")
        for ep, sid, exp, got in mismatches[:20]:
            print(f"   ep {ep} ({sid}): expected {exp}, got {got}")
    else:
        print("[rebuild] all 340 episodes: raw-video frame count matches the original banner-video exactly")

    print(f"[rebuild] done -> {DST_DS}")


if __name__ == "__main__":
    main()
