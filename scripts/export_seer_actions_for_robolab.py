#!/usr/bin/env python3
"""Export fine-tuned Seer real-policy actions for RoboLab relative-IK replay.

This script does not use GR00T. It runs the causal attention layout used by
Seer's real-world fine-tuning recipe on a RoboLab episode directory containing:

    data.parquet
    rgb.mp4
    wrist.mp4
    meta.json

At timestep t, the model receives only observations through t. Startup history
is padded by repeating the current observation, matching Seer's real
controller. The output is a .npz bundle compatible with RoboLab's
scripts/idm_deploy/deploy_idm_actions.py when used with --controller rel_ik.
It also integrates the predicted local EEF deltas and projects the resulting
trajectory onto the scene image under outputs/pictures by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


AXIS_CORRECTION = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraExtrinsics:
    t_base: np.ndarray
    r_camera_to_base: np.ndarray
    source: str


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: np.ndarray
    source: str


def _patch_seer_controller_forward() -> None:
    """Patch Seer's released controller forward unpacking bug in-place.

    The upstream real_controller/controller.py currently unpacks five values
    from SeerAgent.forward(), while SeerAgent.forward() returns six. Keep the
    patch local to this bridge so the downloaded Seer checkout stays untouched.
    """

    import numpy as np
    import torch
    from PIL import Image as PILImage

    from real_controller import controller as controller_module

    if getattr(controller_module.SeerController.forward, "_robolab_patched", False):
        return

    def forward(self, obs_dict, include_info=False, timestep=0):
        image_x = obs_dict["color_image"][0]
        image_x = PILImage.fromarray(image_x).convert("RGB")
        image_x = self.image_process_fn([image_x])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_dtype)

        gripper_x = obs_dict["color_image"][1]
        gripper_x = PILImage.fromarray(gripper_x).convert("RGB")
        gripper_x = self.image_process_fn([gripper_x])
        gripper_x = gripper_x.unsqueeze(1).to(dtype=self.cast_dtype)

        text_x = self.text_process_fn([obs_dict["language_instruction"]])
        text_x = text_x.unsqueeze(1)

        gripper_xyzeuler = obs_dict["robot_state"]["pose6d"]
        gripper_state = obs_dict["robot_state"]["gripper_open_state"]
        gripper_position = obs_dict["robot_state"]["gripper_position"]
        if not self.gripper_width:
            state_x = torch.from_numpy(np.concatenate([gripper_xyzeuler, gripper_state])).to(dtype=self.cast_dtype)
        else:
            state_x = torch.from_numpy(
                np.concatenate([gripper_xyzeuler, gripper_position, gripper_position])
            ).to(dtype=self.cast_dtype)
        state_x = state_x.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            image_x = image_x.to(self.device_id)
            gripper_x = gripper_x.to(self.device_id)
            text_x = text_x.to(self.device_id)
            state_x = state_x.to(self.device_id)
            self.img_queue.append(image_x)
            self.gripper_queue.append(gripper_x)
            self.state_queue.append(state_x)
            if len(self.text_queue) == 0 and text_x is not None:
                self.text_queue.append(text_x)
                for _ in range(self.args.sequence_length - 1):
                    self.text_queue.append(text_x)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary.shape[1]
            if num_step < self.history_len:
                input_image_primary = torch.cat(
                    [image_primary, image_primary[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)],
                    dim=1,
                )
                input_image_wrist = torch.cat(
                    [image_wrist, image_wrist[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)],
                    dim=1,
                )
                input_state = torch.cat(
                    [state, state[:, -1].repeat(1, self.history_len - num_step, 1)],
                    dim=1,
                )
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state

            arm_action, gripper_action, _, _, _, _ = self.ddp_model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
            )
            if not self.use_ensembling:
                action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2
                action = action.cpu().detach().numpy()
                action = action[num_step - 1] if num_step < self.history_len else action[-1]
            else:
                selected_step = num_step - 1 if num_step < self.history_len else -1
                action = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
                self.all_time_actions[timestep : timestep + 1, timestep : timestep + self.action_pred_steps] = action
                actions_for_curr_step = self.all_time_actions[:, timestep]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = self.ensembling_temp
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device_id).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2
                action = action.detach().cpu().numpy()[-1]

        target_pos = action[:3]
        target_euler = action[3:6]
        target_gripper = action[6]
        is_terminal = -1.0
        return target_pos, target_euler, target_gripper, is_terminal

    forward._robolab_patched = True
    controller_module.SeerController.forward = forward


DEFAULT_DATASET_DIR = Path(
    "/mnt/afs/dongyifei/DreamFlyWheel/RoboLab/output/robolab_with_depth/BananaInBowlTask_env_000"
)
DEFAULT_SEER_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fine-tuned Seer causal policy over a RoboLab episode and export 7D rel_ik actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Where to write xyz/rpy MAE metrics JSON. Defaults next to --output.",
    )
    parser.add_argument(
        "--skip-reference-mae",
        action="store_true",
        help="Skip GT rel_ik construction and xyz/rpy MAE reporting.",
    )
    parser.add_argument("--seer-root", type=Path, default=DEFAULT_SEER_ROOT)
    parser.add_argument("--resume-from-checkpoint", type=Path, required=True)
    parser.add_argument("--vit-checkpoint-path", type=Path, required=True)
    parser.add_argument("--primary-video", default="rgb.mp4")
    parser.add_argument("--wrist-video", default="wrist.mp4")
    parser.add_argument("--extrinsics", default="extrinsics.json")
    parser.add_argument("--intrinsics", default="intrinsics.json")
    parser.add_argument(
        "--trajectory-picture-dir",
        type=Path,
        default=Path("outputs") / "pictures",
        help="Directory for the predicted EEF trajectory overlay on the scene image.",
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of repeated first observations to seed causal history before timestep 0.",
    )
    parser.add_argument("--max-rel-pos", type=float, default=0.02)
    parser.add_argument("--max-rel-orn", type=float, default=0.05)
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
    parser.add_argument("--gripper-mode", choices=("zero_one", "minus_one_one"), default="zero_one")
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--action-pred-steps", type=int, default=None)
    parser.add_argument("--real-eval-max-steps", type=int, default=600)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--num-resampler-query", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--transformer-heads", type=int, default=12)
    parser.add_argument("--calvin-input-image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--cached-feature-inference", action="store_true", default=True)
    parser.add_argument(
        "--no-cached-feature-inference",
        dest="cached_feature_inference",
        action="store_false",
    )
    parser.add_argument("--phase", default="evaluate")
    parser.add_argument("--finetune-type", default="real")
    parser.add_argument("--bf16-module", default="vision_encoder")
    parser.add_argument(
        "--eval-libero-ensembling",
        "--temporal-ensembling",
        action="store_true",
        default=False,
        help="Temporally ensemble the three causal action horizons as in Seer's deploy script.",
    )
    parser.add_argument("--no-ensembling", dest="eval_libero_ensembling", action="store_false")
    parser.add_argument("--ensembling-temp", type=float, default=0.01)
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument(
        "--dist-url",
        default=None,
        help="Torch distributed init URL passed into Seer. Defaults to a local file:// rendezvous.",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29591")
    return parser.parse_args()


def infer_model_args_from_checkpoint(args: argparse.Namespace) -> None:
    import torch

    checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    pos_key = "module.transformer_backbone_position_embedding"
    if args.sequence_length is None and pos_key in state_dict:
        args.sequence_length = int(state_dict[pos_key].shape[1])

    action_token_key = "module.action_pred_token"
    if args.action_pred_steps is None and action_token_key in state_dict:
        args.action_pred_steps = int(state_dict[action_token_key].shape[2])

    resampler_key = "module.perceiver_resampler.latents"
    if args.num_resampler_query is None and resampler_key in state_dict:
        args.num_resampler_query = int(state_dict[resampler_key].shape[0])

    text_projector_key = "module.text_projector.weight"
    if args.hidden_dim is None and text_projector_key in state_dict:
        args.hidden_dim = int(state_dict[text_projector_key].shape[0])

    if args.transformer_layers is None:
        layers = []
        for key in state_dict:
            if key.startswith("module.transformer_backbone.h."):
                try:
                    layers.append(int(key.split(".")[3]))
                except (IndexError, ValueError):
                    pass
        if layers:
            args.transformer_layers = max(layers) + 1

    obs_tokens_key = "module.obs_tokens"
    obs_tokens = state_dict.get(obs_tokens_key)
    args.obs_pred = obs_tokens is not None
    args.num_obs_token_per_image = int(obs_tokens.shape[2] // 2) if obs_tokens is not None else 9
    args.gripper_width = "module.gripper_width" in state_dict

    # Match scripts/REAL/robolab_success_ft.sh and Seer's real fine-tune/deploy
    # recipe. Future frames are labels during fine-tuning, not model inputs.
    args.atten_only_obs = False
    args.attn_robot_proprio_state = False
    args.atten_goal = 0
    args.atten_goal_state = False
    args.mask_l_obs_ratio = 0.0

    args.sequence_length = args.sequence_length or 11
    args.action_pred_steps = args.action_pred_steps or 3
    args.num_resampler_query = args.num_resampler_query or 6
    args.transformer_layers = args.transformer_layers or 24
    args.hidden_dim = args.hidden_dim or 384


def parse_vector(value: Any, expected_len: int, name: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < expected_len:
        raise ValueError(f"{name} expected 1-D length >= {expected_len}, got {arr.shape}")
    return arr[:expected_len]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def camera_rotation_from_raw_params(raw_6d: list[float]) -> np.ndarray:
    return Rotation.from_euler("xyz", raw_6d[3:]).as_matrix() @ AXIS_CORRECTION


def load_camera_extrinsics(dataset_dir: Path, extrinsics_name: str) -> CameraExtrinsics:
    extrinsics_path = dataset_dir / extrinsics_name
    payload = load_json(extrinsics_path)
    camera = payload.get("camera", payload)

    if "params_cam2base_extrinsics_6d" in camera:
        raw_6d = [float(v) for v in camera["params_cam2base_extrinsics_6d"]]
        return CameraExtrinsics(
            t_base=np.asarray(raw_6d[:3], dtype=np.float32),
            r_camera_to_base=camera_rotation_from_raw_params(raw_6d).astype(np.float32),
            source=f"{extrinsics_path}: params_cam2base_extrinsics_6d + OpenGL axis correction",
        )

    for key in ("camera_pose_in_base_6d", "cam2base_extrinsics_6d"):
        if key in camera:
            corrected_6d = [float(v) for v in camera[key]]
            return CameraExtrinsics(
                t_base=np.asarray(corrected_6d[:3], dtype=np.float32),
                r_camera_to_base=Rotation.from_euler("xyz", corrected_6d[3:]).as_matrix().astype(np.float32),
                source=f"{extrinsics_path}: {key}",
            )

    raise KeyError(f"{extrinsics_path}: no usable camera-to-base extrinsics found")


def load_camera_intrinsics(
    dataset_dir: Path,
    intrinsics_name: str,
    extrinsics_name: str,
    camera_name: str,
    image_size: tuple[int, int],
) -> CameraIntrinsics:
    """Load and scale scene-camera intrinsics to the decoded video size."""

    intrinsics_path = dataset_dir / intrinsics_name
    if intrinsics_path.exists():
        payload = load_json(intrinsics_path)
        source_path = intrinsics_path
    else:
        source_path = dataset_dir / extrinsics_name
        payload = load_json(source_path)

    camera: Any = payload.get(dataset_dir.name, payload)
    if isinstance(camera, dict):
        camera = camera.get(camera_name, camera.get("rgb", camera.get("camera", camera)))
    if not isinstance(camera, dict):
        raise ValueError(f"{source_path}: camera intrinsics entry must be a JSON object")

    if "intrinsic_matrix" in camera:
        intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    elif "cameraMatrix" in camera:
        values = np.asarray(camera["cameraMatrix"], dtype=np.float64)
        if values.shape == (3, 3):
            intrinsic = values
        elif values.shape == (4,):
            fx, cx, fy, cy = values
            intrinsic = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        else:
            raise ValueError(f"{source_path}: unsupported cameraMatrix shape {values.shape}")
    else:
        raise KeyError(f"{source_path}: no intrinsic_matrix or cameraMatrix found for {camera_name}")

    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError(f"{source_path}: expected finite 3x3 intrinsic matrix, got {intrinsic.shape}")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError(f"{source_path}: focal lengths must be positive")

    image_width, image_height = image_size
    source_width = float(camera.get("width", 0.0))
    source_height = float(camera.get("height", 0.0))
    if source_width <= 0.0 and intrinsic[0, 2] >= image_width:
        source_width = 2.0 * intrinsic[0, 2]
    if source_height <= 0.0 and intrinsic[1, 2] >= image_height:
        source_height = 2.0 * intrinsic[1, 2]

    scaled = intrinsic.copy()
    scale_x = image_width / source_width if source_width > 0.0 else 1.0
    scale_y = image_height / source_height if source_height > 0.0 else 1.0
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    source = f"{source_path}: {camera_name}, scaled by ({scale_x:.6g}, {scale_y:.6g})"
    return CameraIntrinsics(matrix=scaled.astype(np.float32), source=source)


def pose_camera_to_robot(pose_camera: np.ndarray, extrinsics: CameraExtrinsics) -> np.ndarray:
    position_base = extrinsics.r_camera_to_base @ pose_camera[:3] + extrinsics.t_base
    rotation_camera = Rotation.from_euler("xyz", pose_camera[3:6]).as_matrix()
    rotation_base = extrinsics.r_camera_to_base @ rotation_camera
    euler_base = Rotation.from_matrix(rotation_base).as_euler("xyz")
    return np.concatenate([position_base, euler_base]).astype(np.float32)


def pose6d_to_matrix(pose6d: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = pose6d[:3]
    mat[:3, :3] = Rotation.from_euler("xyz", pose6d[3:6]).as_matrix()
    return mat


def matrix_to_pose6d(mat: np.ndarray) -> np.ndarray:
    pose6d = np.zeros(6, dtype=np.float32)
    pose6d[:3] = mat[:3, 3]
    pose6d[3:6] = Rotation.from_matrix(mat[:3, :3]).as_euler("xyz")
    return pose6d


def delta_pose_action(current_pose6d: np.ndarray, target_pose6d: np.ndarray) -> np.ndarray:
    current_to_world = pose6d_to_matrix(current_pose6d)
    target_to_world = pose6d_to_matrix(target_pose6d)
    target_to_current = np.linalg.inv(current_to_world) @ target_to_world
    return matrix_to_pose6d(target_to_current)


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def gripper_to_seer_model_state(value: float, threshold: float, open_is_high: bool) -> float:
    is_open = value >= threshold if open_is_high else value <= threshold
    return 1.0 if is_open else -1.0


def gripper_to_robolab_action(value: float, threshold: float, open_is_high: bool) -> float:
    is_open = value >= threshold if open_is_high else value <= threshold
    return 0.0 if is_open else 1.0


def build_reference_rel_ik_actions(
    df: Any,
    extrinsics: CameraExtrinsics,
    args: argparse.Namespace,
    n_steps: int,
) -> np.ndarray:
    """Build RoboLab rel_ik GT actions from adjacent recorded poses.

    The returned rotation channels are RoboLab/IsaacLab rotation vectors.
    """

    metric_steps = min(max(0, n_steps - 1), len(df) - 1)
    reference = np.zeros((metric_steps, 7), dtype=np.float32)
    if metric_steps == 0:
        return reference

    action_gripper_column = (
        "steps/action_dict/gripper_position"
        if "steps/action_dict/gripper_position" in df.columns
        else "steps/observation/gripper_position"
    )
    base_poses = [
        pose_camera_to_robot(
            parse_vector(
                df["steps/observation/cartesian_position"].iloc[i],
                6,
                "steps/observation/cartesian_position",
            ),
            extrinsics,
        )
        for i in range(metric_steps + 1)
    ]
    for i in range(metric_steps):
        delta = delta_pose_action(base_poses[i], base_poses[i + 1])
        reference[i, :3] = delta[:3]
        reference[i, 3:6] = Rotation.from_euler("xyz", delta[3:6]).as_rotvec()
        target_gripper_raw = float(df[action_gripper_column].iloc[i + 1])
        reference[i, 6] = gripper_to_robolab_action(
            target_gripper_raw,
            args.gripper_open_threshold,
            args.gripper_open_is_high,
        )
    return reference


def compute_action_mae(pred_actions: np.ndarray, reference_actions: np.ndarray) -> dict[str, Any]:
    steps = min(len(pred_actions), len(reference_actions))
    if steps <= 0:
        return {
            "steps": 0,
            "xyz_mae": float("nan"),
            "rpy_mae": float("nan"),
            "rotvec_mae": float("nan"),
            "gripper_mae": float("nan"),
            "xyz_mae_per_axis": [float("nan")] * 3,
            "rpy_mae_per_axis": [float("nan")] * 3,
            "rotvec_mae_per_axis": [float("nan")] * 3,
        }

    pred = np.asarray(pred_actions[:steps], dtype=np.float64)
    ref = np.asarray(reference_actions[:steps], dtype=np.float64)
    pred_rpy = Rotation.from_rotvec(pred[:, 3:6]).as_euler("xyz")
    ref_rpy = Rotation.from_rotvec(ref[:, 3:6]).as_euler("xyz")
    pred_rotvec = pred[:, 3:6]
    ref_rotvec = ref[:, 3:6]

    xyz_abs = np.abs(pred[:, :3] - ref[:, :3])
    rpy_abs = np.abs(wrap_radians(pred_rpy - ref_rpy))
    rotvec_abs = np.abs(pred_rotvec - ref_rotvec)
    gripper_abs = np.abs(pred[:, 6] - ref[:, 6])

    return {
        "steps": int(steps),
        "xyz_mae": float(np.mean(xyz_abs)),
        "rpy_mae": float(np.mean(rpy_abs)),
        "rotvec_mae": float(np.mean(rotvec_abs)),
        "gripper_mae": float(np.mean(gripper_abs)),
        "xyz_mae_per_axis": [float(v) for v in np.mean(xyz_abs, axis=0)],
        "rpy_mae_per_axis": [float(v) for v in np.mean(rpy_abs, axis=0)],
        "rotvec_mae_per_axis": [float(v) for v in np.mean(rotvec_abs, axis=0)],
    }


def metrics_output_path(action_output: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    return action_output.with_name(f"{action_output.stem}_metrics.json")


def load_video(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Video has no frames: {path}")
    return frames


def integrate_predicted_eef_trajectory(initial_pose6d: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Integrate Seer's local-frame EEF pose deltas into robot-base positions."""

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected actions with shape (N, >=6), got {actions.shape}")

    current_to_base = pose6d_to_matrix(np.asarray(initial_pose6d, dtype=np.float64))
    positions = np.empty((len(actions) + 1, 3), dtype=np.float64)
    positions[0] = current_to_base[:3, 3]
    for action_index, action in enumerate(actions):
        target_to_current = np.eye(4, dtype=np.float64)
        target_to_current[:3, 3] = action[:3]
        target_to_current[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix()
        current_to_base = current_to_base @ target_to_current
        positions[action_index + 1] = current_to_base[:3, 3]
    return positions


def project_base_points_to_scene(
    points_base: np.ndarray,
    extrinsics: CameraExtrinsics,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project robot-base points into the scene image using a pinhole camera.

    RoboLab stores the converted EEF pose in an OpenGL camera frame (+Y up,
    -Z forward), while the RGB image uses the usual pinhole (+Y down, +Z
    forward) convention.
    """

    points_camera_opengl = (points_base - extrinsics.t_base) @ extrinsics.r_camera_to_base
    points_camera = points_camera_opengl @ AXIS_CORRECTION
    depth = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (depth > 1e-6)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv[:, 0] = intrinsics.matrix[0, 0] * points_camera[:, 0] / depth + intrinsics.matrix[0, 2]
        uv[:, 1] = intrinsics.matrix[1, 1] * points_camera[:, 1] / depth + intrinsics.matrix[1, 2]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def render_predicted_eef_trajectory(
    scene_frames: list[np.ndarray],
    df: Any,
    actions: np.ndarray,
    target_indices: np.ndarray,
    extrinsics: CameraExtrinsics,
    intrinsics: CameraIntrinsics,
    output_path: Path,
) -> dict[str, Any]:
    """Draw the integrated predicted EEF trajectory over the first scene frame."""

    import cv2

    target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if len(actions) == 0 or len(target_indices) == 0:
        raise ValueError("Cannot render an empty predicted EEF trajectory")
    if len(actions) != len(target_indices):
        raise ValueError("Action and target index counts do not match for trajectory rendering")
    if np.any(np.diff(target_indices) != 1):
        raise ValueError("Trajectory rendering requires consecutive prediction target indices")

    scene_index = int(target_indices[0])
    if scene_index < 0 or scene_index >= len(scene_frames):
        raise IndexError(f"Scene frame index {scene_index} is outside [0, {len(scene_frames)})")
    pose_camera = parse_vector(
        df["steps/observation/cartesian_position"].iloc[scene_index],
        6,
        "steps/observation/cartesian_position",
    )
    initial_pose_base = pose_camera_to_robot(pose_camera, extrinsics)
    points_base = integrate_predicted_eef_trajectory(initial_pose_base, actions)
    uv, valid = project_base_points_to_scene(points_base, extrinsics, intrinsics)

    image_bgr = cv2.cvtColor(scene_frames[scene_index], cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    segment_values = np.linspace(0, 255, max(len(actions), 1), dtype=np.uint8).reshape(-1, 1)
    segment_colors = cv2.applyColorMap(segment_values, cv2.COLORMAP_TURBO).reshape(-1, 3)
    drawn_segments = 0
    last_segment: tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]] | None = None
    for segment_index in range(len(actions)):
        if not (valid[segment_index] and valid[segment_index + 1]):
            continue
        p0 = tuple(int(value) for value in np.rint(uv[segment_index]))
        p1 = tuple(int(value) for value in np.rint(uv[segment_index + 1]))
        inside, clipped_p0, clipped_p1 = cv2.clipLine((0, 0, width, height), p0, p1)
        if not inside:
            continue
        color = tuple(int(value) for value in segment_colors[segment_index])
        cv2.line(image_bgr, clipped_p0, clipped_p1, (16, 16, 16), 6, cv2.LINE_AA)
        cv2.line(image_bgr, clipped_p0, clipped_p1, color, 3, cv2.LINE_AA)
        drawn_segments += 1
        last_segment = (clipped_p0, clipped_p1, color)

    if last_segment is not None and last_segment[0] != last_segment[1]:
        p0, p1, color = last_segment
        cv2.arrowedLine(image_bgr, p0, p1, (16, 16, 16), 7, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(image_bgr, p0, p1, color, 4, cv2.LINE_AA, tipLength=0.35)

    marker_specs = ((0, (255, 255, 255), "START"), (-1, (0, 215, 255), "END"))
    for point_index, color, label in marker_specs:
        if not valid[point_index]:
            continue
        point = tuple(int(value) for value in np.rint(uv[point_index]))
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            continue
        cv2.circle(image_bgr, point, 7, (16, 16, 16), -1, cv2.LINE_AA)
        cv2.circle(image_bgr, point, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            label,
            (point[0] + 9, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (16, 16, 16),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            label,
            (point[0] + 9, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    caption = f"Seer predicted EEF trajectory | {len(actions)} local action deltas"
    cv2.rectangle(image_bgr, (8, 8), (min(width - 8, 470), 36), (16, 16, 16), -1)
    cv2.putText(
        image_bgr,
        caption,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Failed to write trajectory picture: {output_path}")
    return {
        "path": str(output_path),
        "scene_frame_index": scene_index,
        "trajectory_points": int(len(points_base)),
        "positive_depth_points": int(valid.sum()),
        "drawn_segments": int(drawn_segments),
        "intrinsics_source": intrinsics.source,
        "camera_projection": "OpenGL camera points converted with diag(1,-1,-1) before pinhole projection",
        "delta_frame": "local EEF; poses integrated by right-multiplying SE(3) deltas",
    }


def infer_language(df: Any, meta: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for column in (
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ):
        if column in df.columns:
            for value in df[column].tolist():
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for key in ("summary", "task", "clip_id"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "perform the task"


def build_seer_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "seer_robolab_export",
        "--traj_cons",
        "--rgb_pad",
        "10",
        "--gripper_pad",
        "4",
        "--gradient_accumulation_steps",
        "1",
        "--bf16_module",
        args.bf16_module,
        "--vit_checkpoint_path",
        str(args.vit_checkpoint_path),
        "--workers",
        "1",
        "--dist-backend",
        args.dist_backend,
        "--dist-url",
        args.dist_url,
        "--calvin_dataset",
        "",
        "--lr_scheduler",
        "cosine",
        "--save_every_iter",
        "50000",
        "--num_epochs",
        "20",
        "--seed",
        "42",
        "--batch_size",
        "1",
        "--precision",
        args.precision,
        "--weight_decay",
        "1e-4",
        "--num_resampler_query",
        str(args.num_resampler_query),
        "--run_name",
        "robolab_seer_export",
        "--save_checkpoint_path",
        str(Path("outputs") / "seer_robolab_unused"),
        "--transformer_layers",
        str(args.transformer_layers),
        "--hidden_dim",
        str(args.hidden_dim),
        "--transformer_heads",
        str(args.transformer_heads),
        "--calvin_input_image_size",
        str(args.calvin_input_image_size),
        "--phase",
        args.phase,
        "--finetune_type",
        args.finetune_type,
        "--action_pred_steps",
        str(args.action_pred_steps),
        "--future_steps",
        str(args.action_pred_steps),
        "--sequence_length",
        str(args.sequence_length),
        "--resume_from_checkpoint",
        str(args.resume_from_checkpoint),
        "--num_obs_token_per_image",
        str(args.num_obs_token_per_image),
        "--atten_goal",
        str(args.atten_goal),
        "--mask_l_obs_ratio",
        str(args.mask_l_obs_ratio),
        "--real_eval_max_steps",
        str(args.real_eval_max_steps),
        "--max_rel_pos",
        str(args.max_rel_pos),
        "--max_rel_orn",
        str(args.max_rel_orn),
    ]
    if args.obs_pred:
        argv.append("--obs_pred")
    if args.atten_only_obs:
        argv.append("--atten_only_obs")
    if args.attn_robot_proprio_state:
        argv.append("--attn_robot_proprio_state")
    if args.atten_goal_state:
        argv.append("--atten_goal_state")
    if args.gripper_width:
        argv.append("--gripper_width")
    if args.eval_libero_ensembling:
        argv.extend(["--eval_libero_ensembling", "--ensembling_temp", str(args.ensembling_temp)])
    return argv


def import_seer_controller(args: argparse.Namespace):
    seer_root = args.seer_root.resolve()
    if not seer_root.exists():
        raise FileNotFoundError(f"Missing Seer checkout: {seer_root}")
    if args.dist_url is None:
        rendezvous = Path("/tmp") / f"seer_robolab_dist_{os.getpid()}"
        try:
            rendezvous.unlink()
        except FileNotFoundError:
            pass
        args.dist_url = f"file://{rendezvous}"
    sys.path.insert(0, str(seer_root))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", args.master_port)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = build_seer_argv(args)
    try:
        os.chdir(seer_root)
        from real_controller.controller import SeerController

        _patch_seer_controller_forward()
        return SeerController()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def make_obs(
    primary_frame: np.ndarray,
    wrist_frame: np.ndarray,
    row: pd.Series,
    language: str,
    extrinsics: CameraExtrinsics,
    args: argparse.Namespace,
) -> dict[str, Any]:
    pose_camera = parse_vector(
        row["steps/observation/cartesian_position"],
        6,
        "steps/observation/cartesian_position",
    )
    pose6d = pose_camera_to_robot(pose_camera, extrinsics)
    gripper_raw = float(row["steps/observation/gripper_position"])
    gripper_model_state = gripper_to_seer_model_state(
        gripper_raw,
        args.gripper_open_threshold,
        args.gripper_open_is_high,
    )
    gripper_state = np.asarray([gripper_model_state], dtype=np.float32)
    gripper_position = np.asarray([gripper_raw], dtype=np.float32)
    return {
        "robot_state": {
            "pose6d": pose6d.astype(np.float32),
            "gripper_open_state": gripper_state,
            "gripper_position": gripper_position,
        },
        "color_image": [primary_frame, wrist_frame],
        "language_instruction": language,
    }


def convert_gripper(value: float, mode: str, binarize: bool) -> float:
    open_score = float(value)
    if mode == "minus_one_one":
        open_score = (open_score + 1.0) * 0.5
    if binarize:
        open_score = 1.0 if open_score > 0.5 else 0.0
    open_score = float(np.clip(open_score, 0.0, 1.0))
    return 1.0 - open_score


def build_robolab_model_episode(
    primary_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    df: Any,
    language: str,
    extrinsics: CameraExtrinsics,
    args: argparse.Namespace,
    n_steps: int,
) -> dict[str, Any]:
    """Build the observation arrays consumed by OfflineSeerRealRunner."""

    pose6d = []
    gripper_open_state = []
    gripper_positions = []
    for step_idx in range(n_steps):
        obs = make_obs(
            primary_frames[step_idx],
            wrist_frames[step_idx],
            df.iloc[step_idx],
            language,
            extrinsics,
            args,
        )
        robot_state = obs["robot_state"]
        pose6d.append(robot_state["pose6d"])
        gripper_open_state.append(robot_state["gripper_open_state"])
        gripper_positions.append(robot_state["gripper_position"])

    return {
        "language_instruction": language,
        "primary_images": np.stack(primary_frames[:n_steps]).astype(np.uint8, copy=False),
        "wrist_images": np.stack(wrist_frames[:n_steps]).astype(np.uint8, copy=False),
        "gripper_pose6d": np.stack(pose6d).astype(np.float32, copy=False),
        "gripper_open_state": np.stack(gripper_open_state).astype(np.float32, copy=False),
        "gripper_positions": np.stack(gripper_positions).astype(np.float32, copy=False),
        # OfflineSeerRealRunner uses this array only to determine episode length
        # in inference. Reference actions are constructed separately below.
        "action_delta_wrist_pose": np.zeros((n_steps, 7), dtype=np.float32),
    }


def convert_raw_actions_to_robolab(raw_actions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    actions = np.zeros_like(raw_actions, dtype=np.float32)
    actions[:, :3] = raw_actions[:, :3] * float(args.max_rel_pos)
    rpy_delta = raw_actions[:, 3:6] * float(args.max_rel_orn)
    actions[:, 3:6] = Rotation.from_euler("xyz", rpy_delta).as_rotvec().astype(np.float32)
    actions[:, 6] = np.asarray(
        [convert_gripper(value, args.gripper_mode, args.binarize_gripper) for value in raw_actions[:, 6]],
        dtype=np.float32,
    )
    return actions


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd

    args.eval_mode = "causal"
    infer_model_args_from_checkpoint(args)
    print(
        "Using Seer model args: "
        f"sequence_length={args.sequence_length}, "
        f"action_pred_steps={args.action_pred_steps}, "
        f"num_resampler_query={args.num_resampler_query}, "
        f"num_obs_token_per_image={args.num_obs_token_per_image}, "
        f"obs_pred={args.obs_pred}, "
        f"gripper_width={args.gripper_width}, "
        f"transformer_layers={args.transformer_layers}, "
        f"hidden_dim={args.hidden_dim}, "
        f"atten_goal={args.atten_goal}, "
        f"atten_only_obs={args.atten_only_obs}, "
        f"atten_goal_state={args.atten_goal_state}, "
        f"attn_robot_proprio_state={args.attn_robot_proprio_state}"
    )

    dataset_dir = args.dataset_dir.resolve()
    data_path = dataset_dir / "data.parquet"
    primary_path = dataset_dir / args.primary_video
    wrist_path = dataset_dir / args.wrist_video
    extrinsics_path = dataset_dir / args.extrinsics
    meta_path = dataset_dir / "meta.json"
    for path in (data_path, primary_path, wrist_path, extrinsics_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    extrinsics = load_camera_extrinsics(dataset_dir, args.extrinsics)
    df = pd.read_parquet(data_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    language = infer_language(df, meta, args.language)
    primary_frames = load_video(primary_path)
    wrist_frames = load_video(wrist_path)
    scene_height, scene_width = primary_frames[0].shape[:2]
    intrinsics = load_camera_intrinsics(
        dataset_dir=dataset_dir,
        intrinsics_name=args.intrinsics,
        extrinsics_name=args.extrinsics,
        camera_name=primary_path.stem,
        image_size=(scene_width, scene_height),
    )
    n_steps = min(len(df), len(primary_frames), len(wrist_frames))
    if args.max_steps is not None:
        n_steps = min(n_steps, args.max_steps)
    if n_steps < 2:
        raise ValueError("Causal action export needs at least two synchronized observations")
    if args.eval_libero_ensembling:
        args.real_eval_max_steps = max(
            args.real_eval_max_steps,
            n_steps + args.action_pred_steps,
        )

    seer_root = args.seer_root.resolve()
    if str(seer_root) not in sys.path:
        sys.path.insert(0, str(seer_root))
    from scripts.eval_seer_real_on_droid import OfflineSeerRealRunner

    episode = build_robolab_model_episode(
        primary_frames=primary_frames,
        wrist_frames=wrist_frames,
        df=df,
        language=language,
        extrinsics=extrinsics,
        args=args,
        n_steps=n_steps,
    )
    runner = OfflineSeerRealRunner(args)
    runtime_model_args = runner.model_args
    requested_model_args = {
        "sequence_length": args.sequence_length,
        "action_pred_steps": args.action_pred_steps,
        "num_resampler_query": args.num_resampler_query,
        "transformer_layers": args.transformer_layers,
        "hidden_dim": args.hidden_dim,
    }
    for name, requested_value in requested_model_args.items():
        runtime_value = getattr(runtime_model_args, name)
        if requested_value != runtime_value:
            raise ValueError(
                f"--{name.replace('_', '-')}={requested_value} does not match checkpoint value {runtime_value}; "
                "causal fine-tune inference requires checkpoint-compatible dimensions"
            )

    prediction = runner.run_episode(
        episode=episode,
        language=language,
        warmup_steps=args.warmup_steps,
        # RoboLab records N observations and therefore N-1 adjacent actions.
        # The conversion used for fine-tuning also drops the terminal frame.
        max_steps=n_steps - 1,
    )

    raw_actions = np.asarray(prediction.actions, dtype=np.float32)
    target_indices = np.asarray(prediction.target_indices, dtype=np.int64)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
        raise ValueError(f"Expected causal predictions with shape (N, 7), got {raw_actions.shape}")
    if len(raw_actions) != len(target_indices):
        raise ValueError("Prediction and target index counts do not match")
    actions = convert_raw_actions_to_robolab(raw_actions, args)

    reference_actions = None
    reference_metrics = None
    if not args.skip_reference_mae:
        all_reference_actions = build_reference_rel_ik_actions(df, extrinsics, args, n_steps)
        if target_indices[-1] >= len(all_reference_actions):
            raise ValueError(
                f"Prediction target index {target_indices[-1]} has no reference action; "
                f"only {len(all_reference_actions)} reference actions are available"
            )
        reference_actions = all_reference_actions[target_indices]
        reference_metrics = compute_action_mae(actions, reference_actions)

    output = args.output
    if output is None:
        output = Path("outputs") / "seer_causal" / dataset_dir.name / "robolab_actions_rel_ik.npz"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory_picture_path = (
        args.trajectory_picture_dir.resolve() / f"{dataset_dir.name}_seer_predicted_eef_trajectory.png"
    )
    trajectory_picture = render_predicted_eef_trajectory(
        scene_frames=primary_frames,
        df=df,
        actions=actions,
        target_indices=target_indices,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        output_path=trajectory_picture_path,
    )

    metadata = {
        "generator": "InternRobotics/Seer",
        "seer_root": str(args.seer_root.resolve()),
        "dataset_dir": str(dataset_dir),
        "primary_video": str(primary_path),
        "wrist_video": str(wrist_path),
        "language_instruction": language,
        "num_steps": int(len(actions)),
        "source_num_steps": int(n_steps),
        "prediction_target_index_min": int(target_indices[0]),
        "prediction_target_index_max": int(target_indices[-1]),
        "prediction_action_horizon": (
            "temporal_ensemble_0_to_action_pred_steps_minus_1"
            if args.eval_libero_ensembling
            else 0
        ),
        "prediction_aggregation": (
            "Seer exponential temporal ensembling"
            if args.eval_libero_ensembling
            else "one causal horizon-0 prediction per source timestep"
        ),
        "source_action_steps": int(n_steps - 1),
        "terminal_observation_steps": 1,
        "unpredicted_action_steps": int((n_steps - 1) - len(actions)),
        "inference_mode": "real_finetune_causal_policy",
        "future_proprioception_offset": 0,
        "future_visual_observation_attention": False,
        "causal_startup_padding": "repeat current observation on the right",
        "warmup_steps": int(args.warmup_steps),
        "input_cartesian_position_source_frame": "camera",
        "input_cartesian_position_model_frame": "robot_base",
        "camera_extrinsics_source": extrinsics.source,
        "predicted_eef_trajectory_picture": trajectory_picture,
        "gripper_observation_source": "steps/observation/gripper_position",
        "gripper_model_state": "+1=open, -1=closed (DROID real-mode encoding)",
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "gripper_open_is_high": bool(args.gripper_open_is_high),
        "gripper_action_output": "RoboLab 0/1 with open=0, close=1",
        "action_mode": "rel_ik",
        "action_dim": 7,
        "rel_ik_delta_frame": "local_eef",
        "rel_ik_rotation_format": "rotation_vector",
        "rel_ik_integration": "T_next = T_current @ DeltaT_local_eef",
        "controller_scale": 1.0,
        "max_rel_pos": float(args.max_rel_pos),
        "max_rel_orn": float(args.max_rel_orn),
        "gripper_mode": args.gripper_mode,
        "binarize_gripper": bool(args.binarize_gripper),
        "resume_from_checkpoint": str(args.resume_from_checkpoint),
        "vit_checkpoint_path": str(args.vit_checkpoint_path),
        "sequence_length": int(args.sequence_length),
        "action_pred_steps": int(args.action_pred_steps),
        "num_resampler_query": int(args.num_resampler_query),
        "num_obs_token_per_image": int(args.num_obs_token_per_image),
        "obs_pred": bool(args.obs_pred),
        "gripper_width": bool(args.gripper_width),
        "transformer_layers": int(args.transformer_layers),
        "hidden_dim": int(args.hidden_dim),
        "atten_only_obs": bool(args.atten_only_obs),
        "attn_robot_proprio_state": bool(args.attn_robot_proprio_state),
        "atten_goal": int(args.atten_goal),
        "atten_goal_state": bool(args.atten_goal_state),
        "mask_l_obs_ratio": float(args.mask_l_obs_ratio),
        "cached_feature_inference": bool(args.cached_feature_inference),
        "feature_batch_size": int(args.feature_batch_size),
        "device": str(runner.device),
    }
    if reference_metrics is not None:
        metadata["reference_action_source"] = "adjacent RoboLab observation/cartesian_position deltas"
        metadata["reference_metrics"] = reference_metrics

    save_payload: dict[str, Any] = {
        "actions": actions,
        "raw_seer_actions": raw_actions,
        "target_indices": target_indices,
        "metadata": json.dumps(metadata, indent=2),
    }
    if reference_actions is not None:
        save_payload["reference_actions_rel_ik"] = reference_actions
    np.savez_compressed(
        output,
        **save_payload,
    )
    metrics_path = None
    if reference_metrics is not None:
        metrics_path = metrics_output_path(output, args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "action_file": str(output),
            "dataset_dir": str(dataset_dir),
            "metrics": reference_metrics,
            "units": {
                "xyz_mae": "meters",
                "rpy_mae": "radians",
                "rotvec_mae": "radians",
                "gripper_mae": "RoboLab binary command units",
            },
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Seer RoboLab actions: {output}")
    print(f"  shape: {actions.shape}")
    print(f"  source_steps: {n_steps}")
    print(f"  target_indices: {target_indices[0]}..{target_indices[-1]}")
    print(f"  language: {language}")
    print(f"  mean_abs_xyz: {np.mean(np.abs(actions[:, :3])):.6f}")
    print(f"  mean_abs_rpy: {np.mean(np.abs(actions[:, 3:6])):.6f}")
    print(f"  gripper_min_max: {actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}")
    if reference_metrics is not None:
        print(f"  xyz_mae: {reference_metrics['xyz_mae']:.6f} m")
        print(f"  rpy_mae: {reference_metrics['rpy_mae']:.6f} rad")
        print(f"  rotvec_mae: {reference_metrics['rotvec_mae']:.6f} rad")
        print(f"  gripper_mae: {reference_metrics['gripper_mae']:.6f}")
        print(f"Wrote Seer RoboLab MAE metrics: {metrics_path}")
    print(f"Wrote predicted EEF trajectory picture: {trajectory_picture_path}")


if __name__ == "__main__":
    main()
