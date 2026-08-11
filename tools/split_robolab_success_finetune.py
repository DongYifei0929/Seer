#!/usr/bin/env python3
"""Create episode-level train/validation indexes for RoboLab fine-tuning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-info", type=Path, default=Path("data_info/robolab_success_all.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_info/robolab_success_all_manifest.json"),
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data_info/robolab_success_all_train.json"),
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=Path("data_info/robolab_success_all_val.json"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data_info/robolab_success_all_split.json"),
    )
    parser.add_argument(
        "--min-episodes-for-validation",
        type=int,
        default=2,
        help="Tasks below this episode count remain train-only.",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def task_name(source_dir: str) -> str:
    episode_name = Path(source_dir).name
    return episode_name.rsplit("_env_", 1)[0]


def split_summary(
    entries: list[list[Any]], records_by_key: dict[str, dict[str, Any]], window_size: int
) -> dict[str, Any]:
    tasks = sorted({task_name(records_by_key[entry[0]]["source_dir"]) for entry in entries})
    return {
        "episodes": len(entries),
        "tasks": len(tasks),
        "steps": sum(int(records_by_key[entry[0]]["steps"]) for entry in entries),
        "windows": sum(len(entry) - 2 for entry in entries),
        "indexed_windows": sum(int(entry[1]) - window_size for entry in entries),
        "task_names": tasks,
        "episode_keys": [entry[0] for entry in entries],
    }


def write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    full_index = json.loads(args.data_info.read_text(encoding="utf-8"))
    source_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_by_key = {
        record["episode_key"]: record for record in source_manifest["records"]
    }
    index_by_key = {entry[0]: entry for entry in full_index}
    if set(index_by_key) != set(records_by_key):
        missing_index = sorted(set(records_by_key) - set(index_by_key))
        missing_manifest = sorted(set(index_by_key) - set(records_by_key))
        raise ValueError(
            f"index/manifest mismatch: missing_index={missing_index}, "
            f"missing_manifest={missing_manifest}"
        )

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_manifest["records"]:
        by_task[task_name(record["source_dir"])].append(record)

    train_keys: set[str] = set()
    val_keys: set[str] = set()
    train_only_tasks: list[str] = []
    validation_episode_by_task: dict[str, str] = {}
    for task, records in sorted(by_task.items()):
        records = sorted(records, key=lambda record: Path(record["source_dir"]).name)
        if len(records) < args.min_episodes_for_validation:
            train_keys.update(record["episode_key"] for record in records)
            train_only_tasks.append(task)
            continue

        val_record = records[-1]
        val_keys.add(val_record["episode_key"])
        validation_episode_by_task[task] = Path(val_record["source_dir"]).name
        train_keys.update(record["episode_key"] for record in records[:-1])

    if train_keys & val_keys:
        raise AssertionError("train and validation episode sets overlap")
    if train_keys | val_keys != set(index_by_key):
        raise AssertionError("split does not cover the complete episode index")

    train_index = [entry for entry in full_index if entry[0] in train_keys]
    val_index = [entry for entry in full_index if entry[0] in val_keys]
    split_manifest = {
        "source_data_info": str(args.data_info.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "policy": (
            "For each task with at least min_episodes_for_validation, hold out "
            "the lexicographically last environment episode. Singleton tasks are train-only."
        ),
        "min_episodes_for_validation": args.min_episodes_for_validation,
        "window_size": args.window_size,
        "train": split_summary(train_index, records_by_key, args.window_size),
        "validation": split_summary(val_index, records_by_key, args.window_size),
        "train_only_tasks": train_only_tasks,
        "validation_episode_by_task": validation_episode_by_task,
    }

    write_json(args.train_output, train_index, args.overwrite)
    write_json(args.val_output, val_index, args.overwrite)
    write_json(args.split_manifest, split_manifest, args.overwrite)
    print(json.dumps({
        "train": split_manifest["train"],
        "validation": split_manifest["validation"],
        "train_only_tasks": train_only_tasks,
    }, indent=2))


if __name__ == "__main__":
    main()
