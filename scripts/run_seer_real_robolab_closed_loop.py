#!/usr/bin/env python3
"""Run Seer real-world controller closed-loop inside RoboLab rel_ik envs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2  # noqa: F401 -- import before isaaclab
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOLAB_ROOT = Path("/mnt/afs/dongyifei/DreamFlyWheel/RoboLab")
DEFAULT_SEER_CHECKPOINT = Path("checkpoints/robolab_success_all_ft/best.pth")
DEFAULT_VIT_CHECKPOINT = Path("checkpoints/vit_mae/mae_pretrain_vit_base.pth")
POLICY = "seer_real_closed_loop"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if DEFAULT_ROBOLAB_ROOT.exists() and str(DEFAULT_ROBOLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROBOLAB_ROOT))

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402


def resolve_repo_path(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


parser = argparse.ArgumentParser(
    description="Evaluate Seer-real closed-loop in RoboLab relative-IK environments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--robolab-root", type=Path, default=DEFAULT_ROBOLAB_ROOT)
parser.add_argument("--seer-root", type=Path, default=REPO_ROOT)
parser.add_argument("--resume-from-checkpoint", type=Path, default=DEFAULT_SEER_CHECKPOINT)
parser.add_argument("--vit-checkpoint-path", type=Path, default=DEFAULT_VIT_CHECKPOINT)
parser.add_argument("--reference-dataset-dir", type=Path, default=None)
parser.add_argument("--reference-parquet", default="data.parquet")
parser.add_argument("--extrinsics", default="extrinsics.json")
parser.add_argument("--max-rel-pos", type=float, default=0.02)
parser.add_argument("--max-rel-orn", type=float, default=0.05)
parser.add_argument(
    "--controller-scale",
    type=float,
    default=1.0,
    help="Uniform multiplier on xyz and rotation-vector actions before env.step().",
)
parser.add_argument("--gripper-open-threshold", type=float, default=0.5)
parser.add_argument(
    "--gripper-open-is-high",
    action="store_true",
    help="Use this if larger RoboLab gripper values mean open. Default assumes 0=open, 1=closed.",
)
parser.add_argument("--gripper-mode", choices=("zero_one", "minus_one_one"), default="zero_one")
parser.add_argument("--binarize-gripper", action="store_true")
parser.add_argument("--sequence-length", type=int, default=None)
parser.add_argument("--action-pred-steps", type=int, default=None)
parser.add_argument("--real-eval-max-steps", type=int, default=600)
parser.add_argument("--precision", default="fp32")
parser.add_argument("--num-resampler-query", type=int, default=None)
parser.add_argument("--transformer-layers", type=int, default=None)
parser.add_argument("--hidden-dim", type=int, default=None)
parser.add_argument("--transformer-heads", type=int, default=12)
parser.add_argument("--calvin-input-image-size", type=int, default=224)
parser.add_argument("--phase", default="evaluate")
parser.add_argument("--finetune-type", default="real")
parser.add_argument("--bf16-module", default="vision_encoder")
parser.add_argument("--eval-libero-ensembling", action="store_true", default=True)
parser.add_argument("--no-ensembling", dest="eval_libero_ensembling", action="store_false")
parser.add_argument("--ensembling-temp", type=float, default=0.01)
parser.add_argument("--dist-backend", default="nccl")
parser.add_argument(
    "--dist-url",
    default=None,
    help="Torch distributed init URL passed into Seer. Defaults to a local file:// rendezvous.",
)
parser.add_argument("--master-addr", default="127.0.0.1")
parser.add_argument("--master-port", default="29592")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true")

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.task is None:
    args_cli.task = ["BananaInBowlTask"]
if args_cli.num_envs != 1:
    parser.error(
        "SeerController keeps one temporal history queue; "
        "this runner currently supports --num-envs 1 only."
    )

args_cli.seer_root = args_cli.seer_root.resolve()
args_cli.robolab_root = args_cli.robolab_root.resolve()
args_cli.resume_from_checkpoint = resolve_repo_path(args_cli.seer_root, args_cli.resume_from_checkpoint)
args_cli.vit_checkpoint_path = resolve_repo_path(args_cli.seer_root, args_cli.vit_checkpoint_path)
if args_cli.reference_dataset_dir is not None:
    args_cli.reference_dataset_dir = args_cli.reference_dataset_dir.resolve()
if str(args_cli.seer_root) not in sys.path:
    sys.path.insert(0, str(args_cli.seer_root))
if str(args_cli.robolab_root) not in sys.path:
    sys.path.insert(0, str(args_cli.robolab_root))

if not getattr(args_cli, "headless", False) and not os.environ.get("DISPLAY"):
    print("[RoboLab] No DISPLAY detected; forcing --headless for Isaac Sim.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import robolab.constants  # noqa: E402
import torch  # noqa: E402
from robolab.constants import TASK_DIR  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs  # noqa: E402
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg  # noqa: E402
from robolab.eval.base_client import InferenceClient  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa: E402
from robolab.robots.droid import (  # noqa: E402
    DroidCfg,
    DroidRelIKActionCfg,
    ProprioceptionObservationCfg,
    WristCameraCfg,
    contact_gripper,
)
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # noqa: E402
from robolab.variations.camera import EgocentricMirroredCameraCfg  # noqa: E402
from robolab.variations.lighting import SphereLightCfg  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from scripts.export_seer_actions_for_robolab import (  # noqa: E402
    build_reference_rel_ik_actions,
    compute_action_mae,
    convert_gripper,
    infer_model_args_from_checkpoint,
    import_seer_controller,
    load_camera_extrinsics,
)


robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug


def register_rel_ik_envs(args: argparse.Namespace) -> None:
    ImageObsCfg = generate_image_obs_from_cameras(WRIST_LEFT)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg(
        {
            "image_obs": ImageObsCfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
            "viewport_cam": ViewportCameraCfg(),
        }
    )
    scene_cameras = [camera for camera in WRIST_LEFT if camera is not WristCameraCfg]
    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=args.task_dirs,
        tasks=args.task,
        pattern="*.py",
        env_prefix="",
        env_postfix="RelIK",
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidRelIKActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )


def quat_wxyz_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    xyzw = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    return Rotation.from_quat(xyzw).as_euler("xyz").astype(np.float32)


def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    out[top : top + resized_h, left : left + resized_w] = resized
    return out


class SeerRealRoboLabClient(InferenceClient):
    open_loop_horizon = 1

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.controller = import_seer_controller(args)
        if hasattr(self.controller, "reset"):
            self.controller.reset()
        self.timestep = 0
        self.episode_index = 0
        self.raw_trace: list[np.ndarray] = []
        self.action_trace: list[np.ndarray] = []
        self.reference_actions = self._load_reference_actions(args)

    def _load_reference_actions(self, args: argparse.Namespace) -> np.ndarray | None:
        if args.reference_dataset_dir is None:
            return None
        import pandas as pd

        dataset_dir = args.reference_dataset_dir
        parquet_path = dataset_dir / args.reference_parquet
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        if "t" in df.columns:
            df = df.sort_values("t").reset_index(drop=True)
        extrinsics = load_camera_extrinsics(dataset_dir, args.extrinsics)
        return build_reference_rel_ik_actions(df, extrinsics, args, len(df))

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        if env_id != 0:
            raise RuntimeError("Seer real closed-loop runner currently supports only env_id=0.")
        image_obs = raw_obs["image_obs"]
        proprio = raw_obs["proprio_obs"]

        primary = image_obs["over_shoulder_left_camera"][env_id].detach().cpu().numpy()
        wrist = image_obs["wrist_cam"][env_id].detach().cpu().numpy()
        ee_pos = proprio["ee_pos"][env_id].detach().cpu().numpy().astype(np.float32)
        ee_quat = proprio["ee_quat"][env_id].detach().cpu().numpy()
        gripper_raw = float(proprio["gripper_pos"][env_id].detach().cpu().numpy().reshape(-1)[0])
        if self.args.gripper_open_is_high:
            is_open = gripper_raw >= self.args.gripper_open_threshold
        else:
            is_open = gripper_raw <= self.args.gripper_open_threshold
        gripper_model_state = np.asarray([1.0 if is_open else 0.0], dtype=np.float32)
        pose6d = np.concatenate([ee_pos, quat_wxyz_to_euler_xyz(ee_quat)]).astype(np.float32)

        return {
            "primary": primary,
            "wrist": wrist,
            "pose6d": pose6d,
            "gripper_model_state": gripper_model_state,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "robot_state": {
                "pose6d": extracted_obs["pose6d"],
                "gripper_open_state": extracted_obs["gripper_model_state"],
                "gripper_position": extracted_obs["gripper_model_state"],
            },
            "color_image": [extracted_obs["primary"], extracted_obs["wrist"]],
            "language_instruction": instruction,
        }

    def _query_server(self, request: dict) -> np.ndarray:
        target_pos, target_euler, target_gripper, _ = self.controller.forward(
            request,
            include_info=True,
            timestep=self.timestep,
        )
        self.timestep += 1
        return np.concatenate(
            [
                np.asarray(target_pos, dtype=np.float32).reshape(3),
                np.asarray(target_euler, dtype=np.float32).reshape(3),
                np.asarray([target_gripper], dtype=np.float32),
            ]
        )

    def _unpack_response(self, response: np.ndarray) -> np.ndarray:
        return np.asarray(response, dtype=np.float32).reshape(1, 7)

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        raw = np.asarray(chunk[0], dtype=np.float32)
        action = np.zeros((7,), dtype=np.float32)
        action[:3] = raw[:3] * float(self.args.max_rel_pos)
        rpy_delta = raw[3:6] * float(self.args.max_rel_orn)
        action[3:6] = Rotation.from_euler("xyz", rpy_delta).as_rotvec()
        action[:6] *= float(self.args.controller_scale)
        action[6] = convert_gripper(raw[6], self.args.gripper_mode, self.args.binarize_gripper)
        self.raw_trace.append(raw.copy())
        self.action_trace.append(action.copy())
        return action.reshape(1, 7)

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        primary = resize_with_pad(extracted_obs["primary"])
        wrist = resize_with_pad(extracted_obs["wrist"])
        return np.concatenate([primary, wrist], axis=1)

    def _write_action_trace(self) -> None:
        if not self.action_trace:
            return
        from robolab.constants import get_output_dir

        output_dir = Path(get_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        actions = np.stack(self.action_trace).astype(np.float32)
        raw_actions = np.stack(self.raw_trace).astype(np.float32)
        metrics = (
            compute_action_mae(actions, self.reference_actions)
            if self.reference_actions is not None
            else None
        )
        metadata: dict[str, Any] = {
            "generator": "InternRobotics/Seer",
            "policy": POLICY,
            "mode": "closed_loop_robolab_rel_ik",
            "seer_root": str(self.args.seer_root),
            "num_steps": int(actions.shape[0]),
            "max_rel_pos": float(self.args.max_rel_pos),
            "max_rel_orn": float(self.args.max_rel_orn),
            "controller_scale": float(self.args.controller_scale),
            "gripper_mode": self.args.gripper_mode,
            "binarize_gripper": bool(self.args.binarize_gripper),
            "resume_from_checkpoint": str(self.args.resume_from_checkpoint),
            "vit_checkpoint_path": str(self.args.vit_checkpoint_path),
            "reference_dataset_dir": (
                str(self.args.reference_dataset_dir) if self.args.reference_dataset_dir else None
            ),
            "reference_metrics": metrics,
        }
        trace_path = output_dir / f"seer_real_closed_loop_actions_{self.episode_index:03d}.npz"
        payload: dict[str, Any] = {
            "actions": actions,
            "raw_seer_actions": raw_actions,
            "metadata": json.dumps(metadata, indent=2),
        }
        if self.reference_actions is not None:
            payload["reference_actions_rel_ik"] = self.reference_actions[: len(actions)]
        np.savez_compressed(trace_path, **payload)
        print(f"[Seer RoboLab] Wrote closed-loop action trace: {trace_path}")
        if metrics is not None:
            metrics_path = trace_path.with_name(f"{trace_path.stem}_metrics.json")
            metrics_path.write_text(
                json.dumps(
                    {
                        "action_file": str(trace_path),
                        "reference_dataset_dir": str(self.args.reference_dataset_dir),
                        "metrics": metrics,
                        "units": {
                            "xyz_mae": "meters",
                            "rpy_mae": "radians",
                            "rotvec_mae": "radians",
                            "gripper_mae": "RoboLab binary command units",
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "[Seer RoboLab] "
                f"xyz_mae={metrics['xyz_mae']:.6f} m "
                f"rpy_mae={metrics['rpy_mae']:.6f} rad "
                f"metrics={metrics_path}"
            )
        self.episode_index += 1

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._write_action_trace()
            self.raw_trace.clear()
            self.action_trace.clear()
            self.timestep = 0
            if hasattr(self.controller, "reset"):
                self.controller.reset()
        super().reset(env_id=env_id)

    def close(self) -> None:
        self._write_action_trace()


register_rel_ik_envs(args_cli)


def make_client(args: argparse.Namespace) -> SeerRealRoboLabClient:
    infer_model_args_from_checkpoint(args)
    return SeerRealRoboLabClient(args)


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Seer RoboLab] Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
