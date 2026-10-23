#!/usr/bin/env python3
"""Create a dataset with sampled active episodes relabelled as ``interact``.

Each --interact-percent TASK=PERCENT moves that percentage of episodes from
TASK to interact. The none class is always preserved unchanged.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile

import pandas as pd

from _dataset_utils import (
    commit_staging,
    ensure_safe_paths,
    episode_index_from_path,
    indexed_records,
    load_dataset,
    make_staging_directory,
    read_jsonl,
    remove_staging,
    validate_asset_indices,
    write_json,
    write_jsonl,
)

INTERACT = "interact"
NONE = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Source LeRobot dataset; it is never modified")
    parser.add_argument(
        "--interact-percent", action="append", required=True, metavar="TASK=PERCENT",
        help="Percentage of episodes from TASK to relabel as interact; may be repeated",
    )
    parser.add_argument("--output-path", type=Path, help="Default: <dataset>-with-interact")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output folder")
    return parser.parse_args()


def parse_percentages(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected TASK=PERCENT, got {value!r}")
        task, raw = (part.strip() for part in value.split("=", 1))
        if not task or task in {NONE, INTERACT} or task in result:
            raise ValueError(f"Invalid task in {value!r}; {NONE!r} and {INTERACT!r} cannot be selected")
        try:
            percent = float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid percentage in {value!r}") from exc
        if not math.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError(f"Percentage for {task!r} must be between 0 and 100")
        result[task] = percent
    return result


def get_task_map(root: Path) -> dict[int, str]:
    records = read_jsonl(root / "meta" / "tasks.jsonl")
    mapping = {int(record["task_index"]): str(record["task"]) for record in records}
    if not mapping or len(mapping) != len(records) or len(set(mapping.values())) != len(mapping):
        raise ValueError("tasks.jsonl has duplicate or invalid task definitions")
    return mapping


def choose(episodes: list[dict], percentages: dict[str, float], seed: int) -> tuple[set[int], dict[str, tuple[int, int]]]:
    by_task: dict[str, list[int]] = {}
    for episode in episodes:
        labels = episode.get("tasks")
        if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
            raise ValueError(f"Episode {episode.get('episode_index')} must have exactly one task")
        by_task.setdefault(labels[0], []).append(int(episode["episode_index"]))
    unknown = set(percentages) - set(by_task)
    if unknown:
        raise ValueError(f"Requested tasks absent from dataset: {sorted(unknown)}")
    rng = random.Random(seed)
    selected: set[int] = set()
    counts = {}
    for task, percent in percentages.items():
        candidates = sorted(by_task[task])
        amount = math.floor(len(candidates) * percent / 100 + 0.5)
        rng.shuffle(candidates)
        selected.update(candidates[:amount])
        counts[task] = (len(candidates), amount)
    return selected, counts


def replace_parquet(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    source = args.dataset.expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise ValueError(f"Not a LeRobot dataset: {source}")
    output = args.output_path.expanduser().resolve() if args.output_path else source.with_name(f"{source.name}-with-interact")
    ensure_safe_paths(source, output)
    if output in source.parents:
        raise ValueError("Output must not contain the source dataset")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")

    percentages = parse_percentages(args.interact_percent)
    info, episodes = load_dataset(source)
    indices = {int(episode["episode_index"]) for episode in episodes}
    validate_asset_indices(source, indices)
    task_map = get_task_map(source)
    if INTERACT in task_map.values():
        raise ValueError(f"Source already contains task {INTERACT!r}")
    episode_task = {int(episode["episode_index"]): episode["tasks"][0] for episode in episodes}
    if set(episode_task.values()) - set(task_map.values()):
        raise ValueError("episodes.jsonl has a task absent from tasks.jsonl")
    selected, counts = choose(episodes, percentages, args.seed)

    print(f"Source: {source}\nOutput: {output}\nSeed: {args.seed}")
    for task, (available, moved) in counts.items():
        print(f"{task}: {moved}/{available} episodes ({percentages[task]:g}%) -> {INTERACT}")
    print(f"Total moved: {len(selected)}; {NONE} remains unchanged")
    if args.dry_run:
        print("Dry run complete; nothing was written.")
        return 0

    interact_index = max(task_map) + 1
    staging = make_staging_directory(output)
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        parquet_paths = {}
        for path in (staging / "data").rglob("*.parquet"):
            index = episode_index_from_path(path.relative_to(staging))
            if index in selected:
                if index in parquet_paths:
                    raise ValueError(f"Multiple parquet files for episode {index}")
                parquet_paths[index] = path
        if set(parquet_paths) != selected:
            raise ValueError(f"Selected episodes missing parquet files: {sorted(selected - set(parquet_paths))[:10]}")

        task_index_by_name = {name: index for index, name in task_map.items()}
        for number, episode_index in enumerate(sorted(selected), 1):
            frame = pd.read_parquet(parquet_paths[episode_index])
            old_task = episode_task[episode_index]
            old_index = task_index_by_name[old_task]
            values = {int(value) for value in frame["task_index"].dropna().unique()}
            if values != {old_index}:
                raise ValueError(f"{parquet_paths[episode_index]} has task indices {sorted(values)}, expected {old_index}")
            frame["task_index"] = interact_index
            replace_parquet(parquet_paths[episode_index], frame)
            if number % 50 == 0 or number == len(selected):
                print(f"Rewrote {number}/{len(selected)} selected parquet files")

        output_episodes = []
        for episode in episodes:
            record = deepcopy(episode)
            if int(record["episode_index"]) in selected:
                record["tasks"] = [INTERACT]
            output_episodes.append(record)
        output_tasks = read_jsonl(source / "meta" / "tasks.jsonl")
        output_tasks.append({"task_index": interact_index, "task": INTERACT})
        info = deepcopy(info)
        info["total_tasks"] = len(output_tasks)
        info["interact_relabeling"] = {
            "seed": args.seed, "percent_by_source_task": percentages,
            "moved_episode_count_by_source_task": {task: moved for task, (_, moved) in counts.items()},
        }

        provenance_path = staging / "meta" / "episode_provenance.jsonl"
        if provenance_path.exists():
            provenance = indexed_records(provenance_path)
            if set(provenance) != indices:
                raise ValueError("episode_provenance.jsonl does not match episodes.jsonl")
            changed = []
            for index in sorted(provenance):
                record = deepcopy(provenance[index])
                if index in selected:
                    record["task_before_relabel"] = episode_task[index]
                    record["task_after_relabel"] = INTERACT
                changed.append(record)
            write_jsonl(provenance_path, changed)

        write_jsonl(staging / "meta" / "episodes.jsonl", output_episodes)
        write_jsonl(staging / "meta" / "tasks.jsonl", output_tasks)
        write_json(staging / "meta" / "info.json", info)
        commit_staging(staging, output, args.overwrite)
    except Exception:
        remove_staging(staging)
        raise
    print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
