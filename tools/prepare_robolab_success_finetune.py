#!/usr/bin/env python3
"""Prepare all valid RoboLab successful rollouts for Seer real fine-tuning.

This intentionally writes a new dataset name and does not modify the existing
single-episode converter or any previously prepared dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# The single-episode converter contains the camera-frame and action semantics
# used by this repository.  Import it from the sibling tools directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_robolab_to_seer_real import (  # noqa: E402
    build_data_info,
    build_steps,
    load_camera_extrinsics,
    read_frame,
    save_jpeg,
    validate_against_source_hdf5,
    video_info,
    write_step_npz,
)


REQUIRED_FILES = ("data.parquet", "rgb.mp4", "wrist.mp4", "extrinsics.json")
REQUIRED_COLUMNS = {
    "steps/observation/joint_position",
    "steps/observation/cartesian_position",
    "steps/observation/gripper_position",
    "steps/language_instruction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert successful RoboLab episodes into a Seer real dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/mnt/afs/dongyifei/DreamFlyWheel/RoboLab/output/robolab_with_depth"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("real_data"))
    parser.add_argument("--data-info-dir", type=Path, default=Path("data_info"))
    parser.add_argument("--dataset-name", default="robolab_success_all")
    parser.add_argument("--manifest", type=Path, default=Path("data_info/robolab_success_all_manifest.json"))
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--replicate-gripper-steps", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-steps", type=int, default=10)
    parser.add_argument("--language-column", default="steps/language_instruction")
    parser.add_argument("--language-column-2", default="steps/language_instruction_2")
    parser.add_argument("--language-column-3", default="steps/language_instruction_3")
    parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-is-high", action="store_true")
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--skip-source-hdf5-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def episode_record(source_dir: Path, args: argparse.Namespace, episode_index: int) -> tuple[dict, list]:
    missing = [name for name in REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing files: {missing}")

    df = pd.read_parquet(source_dir / "data.parquet")
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"missing parquet columns: {sorted(missing_columns)}")

    primary_info = video_info(source_dir / "rgb.mp4")
    wrist_info = video_info(source_dir / "wrist.mp4")
    usable_frames = min(
        len(df), int(primary_info["frames"]), int(wrist_info["frames"])
    )
    usable_steps = usable_frames - 1
    if usable_steps < args.min_steps:
        raise ValueError(f"only {usable_steps} usable steps")

    extrinsics = load_camera_extrinsics(source_dir, "extrinsics.json")
    validation = None
    if not args.skip_source_hdf5_validation:
        validation = validate_against_source_hdf5(
            source_dir, df, extrinsics, args, usable_frames
        )
    steps = build_steps(df, extrinsics, args, usable_frames)
    episode_key = f"0000/{episode_index:06d}"
    data_info = build_data_info(
        steps,
        episode_key,
        args.sequence_length,
        args.action_pred_steps,
        args.replicate_gripper_steps,
        plain_index=False,
    )
    record = {
        "source_dir": str(source_dir),
        "episode_key": episode_key,
        "steps": len(steps),
        "usable_frames": usable_frames,
        "primary_video": primary_info,
        "wrist_video": wrist_info,
        "camera_extrinsics_source": extrinsics.source,
        "validation": validation,
    }
    return record, (steps, data_info[0])


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    dataset_dir = (args.output_root / args.dataset_name).resolve()
    data_info_path = (args.data_info_dir / f"{args.dataset_name}.json").resolve()
    manifest_path = args.manifest.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if dataset_dir.exists() or data_info_path.exists() or manifest_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                "output already exists; pass --overwrite only for this explicitly named dataset"
            )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        for path in (data_info_path, manifest_path):
            if path.exists():
                path.unlink()

    dataset_dir.mkdir(parents=True, exist_ok=False)
    data_info_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    skipped: list[dict] = []
    all_data_info: list[list] = []
    for source_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        try:
            record, payload = episode_record(source_dir, args, len(records))
        except Exception as exc:
            skipped.append({"source_dir": str(source_dir), "reason": f"{type(exc).__name__}: {exc}"})
            continue

        steps, info_entry = payload
        episode_dir = dataset_dir / "0000" / f"{len(records):06d}" / "steps"
        episode_dir.mkdir(parents=True, exist_ok=True)
        primary_cap = cv2.VideoCapture(str(source_dir / "rgb.mp4"))
        wrist_cap = cv2.VideoCapture(str(source_dir / "wrist.mp4"))
        try:
            for step_idx, step in enumerate(steps):
                step_dir = episode_dir / f"{step_idx:04d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_jpeg(step_dir / "image_primary.jpg", read_frame(primary_cap, source_dir / "rgb.mp4", step_idx), args.jpeg_quality)
                save_jpeg(step_dir / "image_wrist.jpg", read_frame(wrist_cap, source_dir / "wrist.mp4", step_idx), args.jpeg_quality)
                write_step_npz(step_dir, step)
        finally:
            primary_cap.release()
            wrist_cap.release()

        records.append(record)
        all_data_info.append(info_entry)
        print(f"[{len(records):03d}] {source_dir.name}: {record['steps']} steps", flush=True)

    if not records:
        raise RuntimeError("No valid episodes were converted")
    data_info_path.write_text(json.dumps(all_data_info, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "source_root": str(source_root),
        "dataset_dir": str(dataset_dir),
        "data_info": str(data_info_path),
        "sequence_length": args.sequence_length,
        "action_pred_steps": args.action_pred_steps,
        "window_size": args.sequence_length + args.action_pred_steps,
        "episodes": len(records),
        "converted_steps": int(sum(r["steps"] for r in records)),
        "skipped": skipped,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dataset_dir", "data_info", "episodes", "converted_steps", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
