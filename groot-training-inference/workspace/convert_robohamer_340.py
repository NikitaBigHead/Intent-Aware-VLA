#!/usr/bin/env python3
"""Build a separate LeRobot v2.1 dataset from Cosmos3 video + RoboHaMeR qpos.

Run inside the existing tactile-train container, where pyarrow/numpy exist.
Never changes the source release. Use --limit 2 for the first validation.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 15
HW = (392, 640)
JOINTS = [f"left_arm_{i}" for i in range(7)] + [f"left_hand_{i}" for i in range(16)] + [f"right_arm_{i}" for i in range(7)] + [f"right_hand_{i}" for i in range(16)]


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def put_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in entries))


def probe(video):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,r_frame_rate,width,height", "-of", "json", str(video),
    ]))["streams"][0]
    assert result["r_frame_rate"] == "30/1" and (result["height"], result["width"]) == HW, result
    return int(result["nb_frames"])


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": np.mean(a, axis=0).tolist(),
            "std": np.std(a, axis=0).tolist(),
            "min": np.min(a, axis=0).tolist(),
            "max": np.max(a, axis=0).tolist(),
            "q01": np.quantile(a, .01, axis=0).tolist(),
            "q99": np.quantile(a, .99, axis=0).tolist()}


def task_for(batch, sample):
    obj = "oreo package" if batch.startswith("batch_200") else sample.name.rsplit("_", 2)[0].replace("redball", "red ball")
    return f"pick up the {obj} and place it down"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("/data/skurchev/workspace/datasets/hf_release_cosmos3_140_200"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert args.limit >= 0
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    specs = (("batch_140_sam3d_objects", "video_L.mp4", "training_data_g1brainco_robohamer.npz"),
             ("batch_200_oreo_grasp", "generated_video.mp4", "qpos_robohamer.npz"))
    samples = [(batch, p, video, qpos) for batch, video, qpos in specs
               for p in sorted((args.source / batch).iterdir()) if p.is_dir()]
    assert len(samples) == 340, len(samples)
    if args.limit:
        samples = samples[:args.limit]
    output = args.output
    output.mkdir(parents=True)
    episodes, episode_stats, tasks, all_vectors, all_timestamps = [], [], {}, [], []
    offset = 0
    try:
        for episode, (batch, sample, video_name, qpos_name) in enumerate(samples):
            src_video = sample / video_name
            src_frames = probe(src_video)
            with np.load(sample / qpos_name, allow_pickle=False) as d:
                qpos = np.asarray(d["qpos"], dtype=np.float64)
                assert float(d["fps"]) == FPS
                assert tuple(d["active_sides"].tolist()) == ("left", "right")
            assert qpos.ndim == 2 and qpos.shape[1] == 61 and len(qpos) == (src_frames + 1) // 2
            joints = qpos[:, 15:61].copy()
            assert np.isfinite(joints).all()
            length = len(joints)
            task = task_for(batch, sample)
            task_index = tasks.setdefault(task, len(tasks))
            chunk = episode // 1000
            stem = f"episode_{episode:06d}"
            dest_video = output / f"videos/chunk-{chunk:03d}/observation.images.ego_view/{stem}.mp4"
            dest_video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src_video),
                            "-vf", "select=not(mod(n\\,2))", "-vsync", "vfr", "-r", str(FPS),
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dest_video)], check=True)
            assert probe_15(dest_video) == length, (sample.name, length)
            timestamps = np.arange(length, dtype=np.float64) / FPS
            rows = {
                "observation.state": pa.array(joints.tolist(), type=pa.list_(pa.float64(), 46)),
                "action": pa.array(joints.tolist(), type=pa.list_(pa.float64(), 46)),
                "timestamp": pa.array(timestamps, type=pa.float64()),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode), type=pa.int64()),
                "index": pa.array(np.arange(offset, offset + length), type=pa.int64()),
                "task_index": pa.array(np.full(length, task_index), type=pa.int64()),
            }
            dest_parquet = output / f"data/chunk-{chunk:03d}/{stem}.parquet"
            dest_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(rows), dest_parquet, compression="zstd")
            episodes.append({"episode_index": episode, "tasks": [task], "length": length})
            episode_stats.append({"episode_index": episode, "stats": {
                "observation.state": statistics(joints), "action": statistics(joints),
                "timestamp": statistics(timestamps),
            }})
            all_vectors.append(joints)
            all_timestamps.append(timestamps)
            offset += length
            print(f"DONE {episode + 1}/{len(samples)} {sample.name} {length}", flush=True)

        feature = lambda dim, names: {"dtype": "float64", "shape": [dim], "names": names}
        info = {
            "codebase_version": "v2.1", "robot_type": "g1_brainco_robohamer",
            "total_episodes": len(samples), "total_frames": offset, "total_tasks": len(tasks),
            "total_videos": len(samples), "total_chunks": 1, "chunks_size": 1000,
            "fps": FPS, "splits": {"train": f"0:{len(samples)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.ego_view": {
                    "dtype": "video", "shape": [*HW, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.height": HW[0], "video.width": HW[1], "video.codec": "h264",
                             "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                             "video.fps": FPS, "video.channels": 3, "has_audio": False}},
                "observation.state": feature(46, JOINTS), "action": feature(46, JOINTS),
                "timestamp": feature(1, None), "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        modality = {
            "state": {"joint_positions": {"start": 0, "end": 46}},
            "action": {"joint_positions": {"start": 0, "end": 46, "original_key": "action"}},
            "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        }
        meta = output / "meta"
        put_json(meta / "info.json", info)
        put_json(meta / "modality.json", modality)
        put_json(meta / "stats.json", {
            "observation.state": statistics(np.concatenate(all_vectors)),
            "action": statistics(np.concatenate(all_vectors)),
            "timestamp": statistics(np.concatenate(all_timestamps)),
        })
        put_json(meta / "relative_stats.json", {})
        put_jsonl(meta / "tasks.jsonl", [
            {"task_index": index, "task": task} for task, index in tasks.items()])
        put_jsonl(meta / "episodes.jsonl", episodes)
        put_jsonl(meta / "episodes_stats.jsonl", episode_stats)
        print(f"READY {output}: {len(samples)} episodes, {offset} frames", flush=True)
    except Exception:
        print(f"INCOMPLETE {output}: inspect before retrying", flush=True)
        raise


def probe_15(video):
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,r_frame_rate", "-of", "json", str(video),
    ]))["streams"][0]
    assert result["r_frame_rate"] == "15/1", result
    return int(result["nb_frames"])


if __name__ == "__main__":
    main()
