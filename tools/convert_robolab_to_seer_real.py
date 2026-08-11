#!/usr/bin/env python3
"""Convert RoboLab camera-frame rollout data to Seer real fine-tune format.

Recommended from the Seer repo root:

    UV_CACHE_DIR=/tmp/seer-uv-cache uv run python tools/convert_robolab_to_seer_real.py --overwrite

Fallback interpreters:

    .venv/bin/python tools/convert_robolab_to_seer_real.py --overwrite
    /mnt/afs/dongyifei/DreamFlyWheel/RoboLab/.venv/bin/python tools/convert_robolab_to_seer_real.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


DEFAULT_SOURCE_DIR = Path(
    "/mnt/afs/dongyifei/DreamFlyWheel/RoboLab/output/robolab_with_depth/BananaInBowlTask_env_000"
)
DEFAULT_OUTPUT_ROOT = Path("real_data")
DEFAULT_DATA_INFO_DIR = Path("data_info")
DEFAULT_DATASET_NAME = "robolab_banana"
AXIS_CORRECTION = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraExtrinsics:
    t_base: np.ndarray
    r_camera_to_base: np.ndarray
    source: str
    env_origin: np.ndarray | None


@dataclass(frozen=True)
class SeerStep:
    joints: np.ndarray
    gripper_pose: np.ndarray
    gripper_open_state: float
    action_gripper_pose: np.ndarray
    delta_cur_2_last_action: np.ndarray
    language_instruction: str
    language_instruction_2: str
    language_instruction_3: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a RoboLab output/robolab_with_depth episode directory into "
            "Seer's real_data/<dataset>/<exp>/<episode>/steps layout."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-info-dir", type=Path, default=DEFAULT_DATA_INFO_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--exp-id", default="0000")
    parser.add_argument("--episode-id", default="000000")
    parser.add_argument("--primary-video", default="rgb.mp4")
    parser.add_argument("--wrist-video", default="wrist.mp4")
    parser.add_argument("--parquet", default="data.parquet")
    parser.add_argument("--extrinsics", default="extrinsics.json")
    parser.add_argument("--language-column", default="steps/language_instruction")
    parser.add_argument("--language-column-2", default="steps/language_instruction_2")
    parser.add_argument("--language-column-3", default="steps/language_instruction_3")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--action-pred-steps", type=int, default=3)
    parser.add_argument("--replicate-gripper-steps", type=int, default=10)
    parser.add_argument(
        "--plain-index",
        action="store_true",
        help="Write a simple [[episode, length]] data_info file instead of the default --use_aug_data index.",
    )
    parser.add_argument(
        "--gripper-open-threshold",
        type=float,
        default=0.5,
        help="RoboLab gripper values <= this threshold are treated as open by default.",
    )
    parser.add_argument(
        "--gripper-open-is-high",
        action="store_true",
        help="Use this if larger RoboLab gripper values mean open. Default assumes 0=open, 1=closed.",
    )
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-source-hdf5-validation",
        action="store_true",
        help="Skip optional validation against meta.json provenance.hdf5_file.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_vector(value: Any, column: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    elif isinstance(value, str):
        arr = np.asarray(json.loads(value), dtype=float)
    else:
        raise TypeError(f"{column}: cannot parse {type(value).__name__} as a vector")
    if arr.ndim != 1:
        raise ValueError(f"{column}: expected a 1-D vector, got shape {arr.shape}")
    return arr


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required parquet columns: {missing}")


def camera_rotation_from_raw_params(raw_6d: list[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", raw_6d[3:]).as_matrix() @ AXIS_CORRECTION


def load_camera_extrinsics(source_dir: Path, extrinsics_name: str) -> CameraExtrinsics:
    extrinsics_path = source_dir / extrinsics_name
    payload = load_json(extrinsics_path)
    camera = payload.get("camera", payload)

    env_origin = camera.get("env_origin")
    env_origin_arr = np.asarray(env_origin, dtype=float) if env_origin is not None else None

    if "params_cam2base_extrinsics_6d" in camera:
        raw_6d = [float(v) for v in camera["params_cam2base_extrinsics_6d"]]
        return CameraExtrinsics(
            t_base=np.asarray(raw_6d[:3], dtype=float),
            r_camera_to_base=camera_rotation_from_raw_params(raw_6d),
            source="params_cam2base_extrinsics_6d + OpenGL axis correction",
            env_origin=env_origin_arr,
        )

    for key in ("camera_pose_in_base_6d", "cam2base_extrinsics_6d"):
        if key in camera:
            corrected_6d = [float(v) for v in camera[key]]
            return CameraExtrinsics(
                t_base=np.asarray(corrected_6d[:3], dtype=float),
                r_camera_to_base=Rotation.from_euler("xyz", corrected_6d[3:]).as_matrix(),
                source=f"{key} as corrected camera-to-base pose",
                env_origin=env_origin_arr,
            )

    raise KeyError(f"{extrinsics_path}: no usable camera-to-base extrinsics found")


def pose6d_to_matrix(pose6d: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = pose6d[:3]
    mat[:3, :3] = Rotation.from_euler("xyz", pose6d[3:6]).as_matrix()
    return mat


def matrix_to_pose6d(mat: np.ndarray) -> np.ndarray:
    pose6d = np.zeros(6, dtype=float)
    pose6d[:3] = mat[:3, 3]
    pose6d[3:6] = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz")
    return pose6d


def pose_camera_to_base(pose_camera: np.ndarray, extrinsics: CameraExtrinsics) -> np.ndarray:
    if pose_camera.shape[0] < 6:
        raise ValueError(f"Expected 6-D camera pose, got {pose_camera.shape}")
    position_base = extrinsics.r_camera_to_base @ pose_camera[:3] + extrinsics.t_base
    r_ee_camera = Rotation.from_euler("xyz", pose_camera[3:6]).as_matrix()
    r_ee_base = extrinsics.r_camera_to_base @ r_ee_camera
    euler_base = Rotation.from_matrix(r_ee_base).as_euler("xyz")
    return np.concatenate([position_base, euler_base]).astype(np.float32)


def delta_pose_action(last_pose6d: np.ndarray, target_pose6d: np.ndarray) -> np.ndarray:
    last_to_world = pose6d_to_matrix(last_pose6d)
    target_to_world = pose6d_to_matrix(target_pose6d)
    target_to_last = np.linalg.inv(last_to_world) @ target_to_world
    return matrix_to_pose6d(target_to_last).astype(np.float32)


def gripper_to_seer(value: float, threshold: float, open_is_high: bool) -> float:
    if open_is_high:
        return 1.0 if value >= threshold else -1.0
    return 1.0 if value <= threshold else -1.0


def bytes_scalar(text: str) -> np.ndarray:
    return np.asarray(text.encode("utf-8"))


def video_info(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frame(cap: cv2.VideoCapture, path: Path, frame_idx: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"{path}: failed to read frame {frame_idx}")
    return frame


def save_jpeg(path: Path, frame_bgr: np.ndarray, quality: int) -> None:
    ok = cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def build_steps(df: pd.DataFrame, extrinsics: CameraExtrinsics, args: argparse.Namespace, usable_frames: int) -> list[SeerStep]:
    require_columns(
        df,
        [
            "steps/observation/joint_position",
            "steps/observation/cartesian_position",
            "steps/observation/gripper_position",
            args.language_column,
        ],
    )

    action_gripper_column = (
        "steps/action_dict/gripper_position"
        if "steps/action_dict/gripper_position" in df.columns
        else "steps/observation/gripper_position"
    )

    base_poses = [
        pose_camera_to_base(parse_vector(df["steps/observation/cartesian_position"].iloc[i], "cartesian_position"), extrinsics)
        for i in range(usable_frames)
    ]

    steps: list[SeerStep] = []
    for i in range(usable_frames - 1):
        joints = parse_vector(df["steps/observation/joint_position"].iloc[i], "joint_position")[:7].astype(np.float32)
        current_gripper_raw = float(df["steps/observation/gripper_position"].iloc[i])
        target_gripper_raw = float(df[action_gripper_column].iloc[i + 1])
        current_gripper = gripper_to_seer(
            current_gripper_raw, args.gripper_open_threshold, args.gripper_open_is_high
        )
        target_gripper = gripper_to_seer(
            target_gripper_raw, args.gripper_open_threshold, args.gripper_open_is_high
        )

        current_pose = base_poses[i]
        target_pose = base_poses[i + 1]
        delta = np.zeros(7, dtype=np.float32)
        delta[:6] = delta_pose_action(current_pose, target_pose)
        delta[-1] = target_gripper

        action_gripper_pose = np.zeros(7, dtype=np.float32)
        action_gripper_pose[:6] = target_pose
        action_gripper_pose[-1] = target_gripper

        steps.append(
            SeerStep(
                joints=joints,
                gripper_pose=current_pose.astype(np.float32),
                gripper_open_state=current_gripper,
                action_gripper_pose=action_gripper_pose,
                delta_cur_2_last_action=delta,
                language_instruction=str(df[args.language_column].iloc[i]),
                language_instruction_2=str(df[args.language_column_2].iloc[i])
                if args.language_column_2 in df.columns
                else "",
                language_instruction_3=str(df[args.language_column_3].iloc[i])
                if args.language_column_3 in df.columns
                else "",
            )
        )

    return steps


def build_data_info(
    steps: list[SeerStep],
    episode_key: str,
    sequence_length: int,
    action_pred_steps: int,
    replicate_gripper_steps: int,
    plain_index: bool,
) -> list[Any]:
    num_steps = len(steps)
    if plain_index:
        return [[episode_key, num_steps]]

    window_size = sequence_length + action_pred_steps
    windows: list[list[int]] = []
    prev_gripper = None
    for step_id, step in enumerate(steps):
        if step_id >= window_size:
            windows.append([step_id - window_size, step_id])

        curr_gripper = float(step.delta_cur_2_last_action[-1])
        if prev_gripper is not None and curr_gripper != prev_gripper:
            for _ in range(replicate_gripper_steps):
                for k in range(action_pred_steps):
                    start = step_id - window_size + k
                    end = step_id + k
                    if start >= 0 and end < num_steps:
                        windows.append([start, end])
        prev_gripper = curr_gripper

    if not windows:
        raise ValueError(
            f"Episode has {num_steps} steps, which is too short for augmented window_size={window_size}"
        )
    return [[episode_key, len(windows) + window_size, *windows]]


def write_step_npz(step_dir: Path, step: SeerStep) -> None:
    np.savez_compressed(
        step_dir / "other.npz",
        joints=step.joints,
        gripper_pose=step.gripper_pose,
        gripper_open_state=np.asarray([step.gripper_open_state], dtype=np.float32),
        action_gripper_pose=step.action_gripper_pose,
        delta_cur_2_last_action=step.delta_cur_2_last_action,
        language_instruction=bytes_scalar(step.language_instruction),
        language_instruction_2=bytes_scalar(step.language_instruction_2),
        language_instruction_3=bytes_scalar(step.language_instruction_3),
    )


def write_dataset(steps: list[SeerStep], source_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_dir = args.output_root / args.dataset_name
    episode_dir = dataset_dir / args.exp_id / args.episode_id
    data_info_path = args.data_info_dir / f"{args.dataset_name}.json"

    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(dataset_dir)
    if data_info_path.exists() and not args.overwrite:
        raise FileExistsError(f"{data_info_path} exists. Re-run with --overwrite to replace it.")

    steps_dir = episode_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    args.data_info_dir.mkdir(parents=True, exist_ok=True)

    primary_path = source_dir / args.primary_video
    wrist_path = source_dir / args.wrist_video
    primary_cap = cv2.VideoCapture(str(primary_path))
    wrist_cap = cv2.VideoCapture(str(wrist_path))
    if not primary_cap.isOpened():
        raise FileNotFoundError(f"Could not open primary video: {primary_path}")
    if not wrist_cap.isOpened():
        raise FileNotFoundError(f"Could not open wrist video: {wrist_path}")

    try:
        for step_idx, step in enumerate(steps):
            step_dir = steps_dir / f"{step_idx:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            primary_frame = read_frame(primary_cap, primary_path, step_idx)
            wrist_frame = read_frame(wrist_cap, wrist_path, step_idx)
            save_jpeg(step_dir / "image_primary.jpg", primary_frame, args.jpeg_quality)
            save_jpeg(step_dir / "image_wrist.jpg", wrist_frame, args.jpeg_quality)
            write_step_npz(step_dir, step)
    finally:
        primary_cap.release()
        wrist_cap.release()

    episode_key = f"{args.exp_id}/{args.episode_id}"
    data_info = build_data_info(
        steps=steps,
        episode_key=episode_key,
        sequence_length=args.sequence_length,
        action_pred_steps=args.action_pred_steps,
        replicate_gripper_steps=args.replicate_gripper_steps,
        plain_index=args.plain_index,
    )
    data_info_path.write_text(json.dumps(data_info, indent=1) + "\n", encoding="utf-8")

    return dataset_dir, data_info_path


def quat_to_rotation(quat: np.ndarray, quat_order: str) -> Rotation:
    if quat_order == "wxyz":
        xyzw = [quat[1], quat[2], quat[3], quat[0]]
    else:
        xyzw = [quat[0], quat[1], quat[2], quat[3]]
    return Rotation.from_quat(xyzw)


def validate_against_source_hdf5(
    source_dir: Path,
    df: pd.DataFrame,
    extrinsics: CameraExtrinsics,
    args: argparse.Namespace,
    usable_frames: int,
) -> dict[str, float | str] | None:
    meta_path = source_dir / "meta.json"
    if not meta_path.exists() or extrinsics.env_origin is None:
        return None
    meta = load_json(meta_path)
    hdf5_file = meta.get("provenance", {}).get("hdf5_file")
    if not hdf5_file:
        return None
    hdf5_path = Path(hdf5_file)
    if not hdf5_path.exists():
        return {"status": f"skipped missing source HDF5: {hdf5_path}"}

    try:
        import h5py
    except ImportError:
        return {"status": "skipped because h5py is not installed"}

    env_id = int(meta.get("env_id", 0))
    demo = f"data/demo_{env_id}"
    with h5py.File(hdf5_path, "r") as h5:
        source_positions = np.asarray(h5[f"{demo}/ee_pose/position"][:usable_frames], dtype=float)
        source_quats = np.asarray(h5[f"{demo}/ee_pose/orientation"][:usable_frames], dtype=float)

    source_positions_base = source_positions - extrinsics.env_origin
    converted_poses = [
        pose_camera_to_base(parse_vector(df["steps/observation/cartesian_position"].iloc[i], "cartesian_position"), extrinsics)
        for i in range(usable_frames)
    ]
    converted_positions = np.stack([pose[:3] for pose in converted_poses])
    max_position_error = float(np.max(np.abs(converted_positions - source_positions_base)))

    rotation_errors: list[float] = []
    for i, pose in enumerate(converted_poses):
        converted_rot = Rotation.from_euler("xyz", pose[3:6])
        source_rot = quat_to_rotation(source_quats[i], args.quat_order)
        rotation_errors.append(float((source_rot.inv() * converted_rot).magnitude()))
    max_rotation_error = float(max(rotation_errors))

    if max_position_error > 1e-4 or max_rotation_error > 1e-4:
        raise RuntimeError(
            "camera->base validation failed: "
            f"max_position_error={max_position_error:.6g}, "
            f"max_rotation_error={max_rotation_error:.6g}"
        )

    return {
        "status": "ok",
        "hdf5": str(hdf5_path),
        "max_position_error": max_position_error,
        "max_rotation_error": max_rotation_error,
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    parquet_path = source_dir / args.parquet
    primary_path = source_dir / args.primary_video
    wrist_path = source_dir / args.wrist_video

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    if not wrist_path.exists():
        raise FileNotFoundError(wrist_path)

    df = pd.read_parquet(parquet_path)
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)

    primary_info = video_info(primary_path)
    wrist_info = video_info(wrist_path)
    usable_frames = min(len(df), int(primary_info["frames"]), int(wrist_info["frames"]))
    if usable_frames < 2:
        raise ValueError(f"Need at least two aligned frames, got {usable_frames}")

    extrinsics = load_camera_extrinsics(source_dir, args.extrinsics)
    if not args.skip_source_hdf5_validation:
        validation = validate_against_source_hdf5(source_dir, df, extrinsics, args, usable_frames)
    else:
        validation = {"status": "skipped by flag"}

    steps = build_steps(df, extrinsics, args, usable_frames)
    dataset_dir, data_info_path = write_dataset(steps, source_dir, args)

    summary = {
        "source_dir": str(source_dir),
        "dataset_dir": str(dataset_dir),
        "data_info": str(data_info_path),
        "steps": len(steps),
        "usable_frames": usable_frames,
        "primary_video": primary_info,
        "wrist_video": wrist_info,
        "camera_extrinsics_source": extrinsics.source,
        "index_type": "plain" if args.plain_index else "augmented_for_use_aug_data",
        "validation": validation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
