#!/usr/bin/env python3
"""Copy a LeRobot dataset and relabel a percentage of each active task as interact."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    content = "".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in records)
    replace_text(path, content)


def replace_text(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".interact-", mode="w", encoding="utf-8", delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_parquet(path: Path, task_index: int) -> None:
    table = pq.read_table(path)
    col = table.schema.get_field_index("task_index")
    if col < 0:
        raise ValueError(f"Missing task_index: {path}")
    replacement = pa.array([task_index] * table.num_rows, type=table.column(col).type)
    table = table.set_column(col, "task_index", replacement)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".interact-", suffix=".parquet", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--percent", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.dataset.resolve()
    output = args.output_path.resolve()
    if not 0 <= args.percent <= 100 or not math.isfinite(args.percent):
        parser.error("--percent must be between 0 and 100")
    if source == output or source in output.parents or output.exists():
        parser.error("Output must be a new directory outside the source dataset")
    meta = source / "meta"
    info = json.loads((meta / "info.json").read_text())
    tasks = read_jsonl(meta / "tasks.jsonl")
    episodes = read_jsonl(meta / "episodes.jsonl")
    if info["total_episodes"] != len(episodes):
        raise ValueError("info.json episode count mismatch")
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode in episodes:
        labels = episode["tasks"]
        if len(labels) != 1:
            raise ValueError(f"Episode {episode['episode_index']} has multiple tasks")
        by_task[labels[0]].append(episode["episode_index"])
    if set(by_task) != {record["task"] for record in tasks}:
        raise ValueError("tasks.jsonl and episodes.jsonl disagree")
    if "interact" in by_task:
        raise ValueError("Source already has interact task")
    rng = random.Random(args.seed)
    selected: set[int] = set()
    for task, indices in sorted(by_task.items()):
        if task == "none":
            continue
        candidates = sorted(indices)
        rng.shuffle(candidates)
        count = math.floor(len(candidates) * args.percent / 100 + 0.5)
        selected.update(candidates[:count])
        print(f"{task}: {count}/{len(candidates)} -> interact")
    print(f"Total: {len(selected)}/{len(episodes)} -> interact; none: {len(by_task.get('none', []))} unchanged")
    if args.dry_run:
        return
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, copy_function=os.link)
        dstmeta = staging / "meta"
        interact_index = max(int(record["task_index"]) for record in tasks) + 1
        tasks.append({"task_index": interact_index, "task": "interact"})
        write_jsonl(dstmeta / "tasks.jsonl", tasks)
        info["total_tasks"] = len(tasks)
        replace_text(dstmeta / "info.json", json.dumps(info, ensure_ascii=False, indent=4) + "\n")
        for episode in episodes:
            if episode["episode_index"] in selected:
                episode["tasks"] = ["interact"]
        write_jsonl(dstmeta / "episodes.jsonl", episodes)
        interactions = read_jsonl(meta / "interaction_metadata.jsonl")
        for record in interactions:
            if record["episode_index"] in selected:
                record["original_interaction_class"] = record["interaction_class"]
                record["interaction_class"] = "interact"
        write_jsonl(dstmeta / "interaction_metadata.jsonl", interactions)
        stats = read_jsonl(meta / "episodes_stats.jsonl")
        for record in stats:
            if record["episode_index"] in selected:
                task_stats = record["stats"]["task_index"]
                task_stats.update(min=[interact_index], max=[interact_index], mean=[float(interact_index)], std=[0.0])
        write_jsonl(dstmeta / "episodes_stats.jsonl", stats)
        parquet_paths = sorted((staging / "data").glob("chunk-*/*.parquet"))
        if len(parquet_paths) != len(episodes):
            raise ValueError(f"Expected {len(episodes)} parquet files, got {len(parquet_paths)}")
        count = 0
        for path in parquet_paths:
            index = int(path.stem.split("_")[-1])
            if index in selected:
                replace_parquet(path, interact_index)
                count += 1
                if count % 100 == 0:
                    print(f"Rewrote {count}/{len(selected)} selected parquet files", flush=True)
        if count != len(selected):
            raise ValueError(f"Selected {len(selected)} but rewrote {count} parquet files")
        staging.replace(output)
        print(f"Created {output}")
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
