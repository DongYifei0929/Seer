#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation


DEFAULT_TFDS_DIR = Path("/mnt/afs/dongyifei/DreamFlyWheel/GR00T-Dreams/dataset/droid/droid_100/1.0.0")
DEFAULT_CACHE_DIR = Path("outputs/seer_droid_real_eval/cache")
DEFAULT_SUMMARY_DIR = Path("outputs/seer_droid_real_eval")
DEFAULT_SEER_CHECKPOINT = Path("checkpoints/real_world_droid/seer.pth")
DEFAULT_VIT_CHECKPOINT = Path("checkpoints/vit_mae/mae_pretrain_vit_base.pth")
DEFAULT_CLIP_PATH = Path("/mnt/afs/dongyifei/.cache/clip/ViT-B-32.pt")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_tfds_runtime_version_compat() -> None:
    try:
        from google.protobuf import runtime_version  # noqa: F401

        return
    except ImportError:
        pass

    import google.protobuf as protobuf_pkg

    runtime_version = types.ModuleType("google.protobuf.runtime_version")

    class _Domain:
        PUBLIC = "PUBLIC"

    def _validate_protobuf_runtime_version(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    runtime_version.Domain = _Domain
    runtime_version.ValidateProtobufRuntimeVersion = _validate_protobuf_runtime_version
    sys.modules["google.protobuf.runtime_version"] = runtime_version
    setattr(protobuf_pkg, "runtime_version", runtime_version)


def load_tfds_dataset(tfds_dir: Path, split: str):
    ensure_tfds_runtime_version_compat()
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(str(tfds_dir))
    return builder.as_dataset(split=split, shuffle_files=False)


def decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").strip()
        return str(item).strip()
    return str(value).strip()


def to_float_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] != expected_dim:
        raise ValueError(f"{name} has {vector.shape[0]} dims; expected {expected_dim}")
    return vector


def _6d_to_pose(pose6d: np.ndarray, degrees: bool = False) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = Rotation.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose


def pose_to_6d(pose: np.ndarray, degrees: bool = False) -> np.ndarray:
    pose6d = np.zeros(6, dtype=np.float32)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def rotation_geodesic_error(pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    rel_rot = pred_pose[:3, :3].T @ gt_pose[:3, :3]
    return float(Rotation.from_matrix(rel_rot).magnitude())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda":
            torch.empty(1, device=device)
        return device

    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda:0")
            return torch.device("cuda:0")
        except Exception as exc:  # noqa: BLE001
            print(f"[device] CUDA is visible but failed a tensor allocation probe; falling back to CPU: {exc}")
    return torch.device("cpu")


def get_fk_solution(joint_angles: np.ndarray) -> np.ndarray:
    def get_tf_mat(i: int, dh: list[list[float]]) -> np.ndarray:
        a = dh[i][0]
        d = dh[i][1]
        alpha = dh[i][2]
        theta = dh[i][3]
        q = theta
        return np.array(
            [
                [np.cos(q), -np.sin(q), 0, a],
                [
                    np.sin(q) * np.cos(alpha),
                    np.cos(q) * np.cos(alpha),
                    -np.sin(alpha),
                    -np.sin(alpha) * d,
                ],
                [
                    np.sin(q) * np.sin(alpha),
                    np.cos(q) * np.sin(alpha),
                    np.cos(alpha),
                    np.cos(alpha) * d,
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    dh_params = [
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        [0, 0.107, 0, 0],
        [0, 0, 0, -np.pi / 4],
        [0.0, 0.1034, 0, 0],
    ]
    T = np.eye(4, dtype=np.float32)
    for i in range(8):
        T = T @ get_tf_mat(i, dh_params)
    return T


def compute_gripper_commands(gripper_positions: np.ndarray) -> np.ndarray:
    commands = np.zeros((gripper_positions.shape[0], 1), dtype=np.float32)
    prev = None
    current = 1.0
    for i, value in enumerate(gripper_positions[:, -1]):
        if i == 0:
            current = 1.0
        elif value > prev:
            current = -1.0
        elif value < prev:
            current = 1.0
        commands[i, 0] = current
        prev = value
    return commands


def build_episode_cache_from_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        raise ValueError("empty episode")

    cartesian_positions = []
    gripper_positions = []
    joint_positions = []
    primary_images = []
    wrist_images = []
    language_candidates = []

    for step_index, step in enumerate(steps):
        obs = step["observation"]
        cartesian_positions.append(
            to_float_vector(
                obs["cartesian_position"],
                6,
                f"steps[{step_index}].observation.cartesian_position",
            )
        )
        gripper_positions.append(
            to_float_vector(
                obs["gripper_position"],
                1,
                f"steps[{step_index}].observation.gripper_position",
            )
        )
        joint_positions.append(
            to_float_vector(
                obs["joint_position"],
                7,
                f"steps[{step_index}].observation.joint_position",
            )
        )
        primary_images.append(np.asarray(obs["exterior_image_1_left"], dtype=np.uint8))
        wrist_images.append(np.asarray(obs["wrist_image_left"], dtype=np.uint8))
        for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
            text = decode_text(step.get(key))
            if text:
                language_candidates.append(text)

    cartesian_positions = np.stack(cartesian_positions).astype(np.float32, copy=False)
    gripper_positions = np.stack(gripper_positions).astype(np.float32, copy=False)
    joint_positions = np.stack(joint_positions).astype(np.float32, copy=False)
    primary_images = np.stack(primary_images).astype(np.uint8, copy=False)
    wrist_images = np.stack(wrist_images).astype(np.uint8, copy=False)

    gripper_open_state = np.ones_like(gripper_positions, dtype=np.float32) * (-1.0)
    gripper_open_state[gripper_positions < (13.0 / 255.0)] = 1.0
    gripper_pose6d = np.stack([pose_to_6d(get_fk_solution(q)) for q in joint_positions]).astype(
        np.float32,
        copy=False,
    )

    action_wrist_pose = np.stack([step["action_dict"]["cartesian_position"] for step in steps]).astype(
        np.float32,
        copy=False,
    )
    action_wrist_pose = action_wrist_pose.reshape(-1, 6)
    action_delta_wrist_pose = np.zeros((len(steps), 7), dtype=np.float32)
    gripper_commands = compute_gripper_commands(gripper_positions)
    action_delta_wrist_pose[:, -1] = gripper_commands[:, 0]

    for step_index in range(len(steps)):
        if step_index == 0:
            last2world = get_fk_solution(joint_positions[0])
        else:
            last2world = _6d_to_pose(action_wrist_pose[step_index - 1], degrees=False)
        cur2world = _6d_to_pose(action_wrist_pose[step_index], degrees=False)
        cur2last = np.linalg.inv(last2world) @ cur2world
        action_delta_wrist_pose[step_index, :6] = pose_to_6d(cur2last)

    language = language_candidates[0] if language_candidates else "perform the task"

    return {
        "language_instruction": np.array(language, dtype=np.unicode_),
        "primary_images": primary_images,
        "wrist_images": wrist_images,
        "joint_positions": joint_positions,
        "gripper_positions": gripper_positions,
        "gripper_open_state": gripper_open_state,
        "gripper_pose6d": gripper_pose6d,
        "action_wrist_pose": action_wrist_pose,
        "action_delta_wrist_pose": action_delta_wrist_pose,
    }


def save_episode_cache(cache_dir: Path, episode_idx: int, payload: dict[str, Any], manifest_row: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"episode_{episode_idx:06d}.npz", **payload)
    manifest_path = cache_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")


def extract_raw_droid(
    tfds_dir: Path,
    cache_dir: Path,
    split: str,
    start_episode: int,
    max_episodes: int,
    filter_substring: str,
    overwrite: bool,
) -> None:
    if overwrite and cache_dir.exists():
        for path in cache_dir.glob("episode_*.npz"):
            path.unlink()
        manifest_path = cache_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

    dataset = load_tfds_dataset(tfds_dir, split)
    selected = 0
    source_index = -1

    for episode in dataset:
        source_index += 1
        if source_index < start_episode:
            continue

        meta_path = decode_text(episode["episode_metadata"]["file_path"].numpy())
        if filter_substring and filter_substring not in meta_path:
            continue

        steps = list(episode["steps"].as_numpy_iterator())
        payload = build_episode_cache_from_steps(steps)
        episode_idx = selected
        manifest_row = {
            "episode_index": episode_idx,
            "source_episode_index": source_index,
            "source_file_path": meta_path,
            "num_steps": int(len(steps)),
            "language_instruction": decode_text(payload["language_instruction"]),
        }
        save_episode_cache(cache_dir, episode_idx, payload, manifest_row)
        print(
            f"[extract] source_episode={source_index:06d} -> episode_{episode_idx:06d} "
            f"steps={len(steps)}"
        )
        selected += 1
        if selected >= max_episodes:
            break

    if selected == 0:
        raise RuntimeError("No episodes were extracted")


@dataclass
class ModelArgs:
    sequence_length: int
    action_pred_steps: int
    num_resampler_query: int
    transformer_layers: int
    hidden_dim: int
    num_obs_token_per_image: int
    obs_pred: bool
    gripper_width: bool


@dataclass
class EpisodePredictions:
    """Normalized predictions paired with their source indices in an episode."""

    actions: np.ndarray
    target_indices: np.ndarray


def infer_model_args(checkpoint_path: Path) -> ModelArgs:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    def maybe_shape(name: str) -> tuple[int, ...] | None:
        tensor = state_dict.get(name)
        return tuple(tensor.shape) if tensor is not None else None

    sequence_length = maybe_shape("module.transformer_backbone_position_embedding")
    action_pred_steps = maybe_shape("module.action_pred_token")
    num_resampler_query = maybe_shape("module.perceiver_resampler.latents")
    hidden_dim = maybe_shape("module.text_projector.weight")

    layers = []
    for key in state_dict:
        if key.startswith("module.transformer_backbone.h."):
            try:
                layers.append(int(key.split(".")[3]))
            except (IndexError, ValueError):
                pass

    obs_tokens = maybe_shape("module.obs_tokens")
    return ModelArgs(
        sequence_length=int(sequence_length[1]) if sequence_length else 11,
        action_pred_steps=int(action_pred_steps[2]) if action_pred_steps else 3,
        num_resampler_query=int(num_resampler_query[0]) if num_resampler_query else 6,
        transformer_layers=max(layers) + 1 if layers else 24,
        hidden_dim=int(hidden_dim[0]) if hidden_dim else 384,
        num_obs_token_per_image=int(obs_tokens[2] // 2) if obs_tokens else 9,
        obs_pred=obs_tokens is not None,
        gripper_width="module.gripper_width" in state_dict,
    )


class OfflineSeerRealRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        import clip

        from models.seer_model import SeerAgent
        from utils.data_utils import preprocess_image, preprocess_text_calvin
        from utils.train_utils import get_cast_dtype

        self.args = args
        self.device = resolve_device(args.device)
        print(f"[device] using {self.device}")
        self.model_args = infer_model_args(args.resume_from_checkpoint)
        self.idm_mode = args.eval_mode == "droid-idm"
        if self.idm_mode:
            if args.idm_goal_steps <= 0:
                raise ValueError("--idm-goal-steps must be positive in droid-idm mode")
            if self.model_args.sequence_length <= args.idm_goal_steps:
                raise ValueError(
                    "sequence_length must exceed --idm-goal-steps in droid-idm mode "
                    f"({self.model_args.sequence_length} <= {args.idm_goal_steps})"
                )
            self.idm_supervised_positions = self.model_args.sequence_length - args.idm_goal_steps
        else:
            self.idm_supervised_positions = None

        if not args.vit_checkpoint_path.exists():
            raise FileNotFoundError(f"Missing ViT checkpoint: {args.vit_checkpoint_path}")
        if not args.resume_from_checkpoint.exists():
            raise FileNotFoundError(f"Missing Seer checkpoint: {args.resume_from_checkpoint}")

        self.model = SeerAgent(
            finetune_type="real",
            clip_device=str(self.device),
            vit_checkpoint_path=str(args.vit_checkpoint_path),
            sequence_length=self.model_args.sequence_length,
            num_resampler_query=self.model_args.num_resampler_query,
            num_obs_token_per_image=self.model_args.num_obs_token_per_image,
            calvin_input_image_size=args.calvin_input_image_size,
            patch_size=args.patch_size,
            action_pred_steps=self.model_args.action_pred_steps,
            obs_pred=self.model_args.obs_pred,
            # The released DROID pre-training recipe uses these three settings.
            # The goal state is supplied at t + idm_goal_steps by run_episode().
            atten_only_obs=self.idm_mode,
            attn_robot_proprio_state=False,
            atten_goal=args.idm_goal_steps if self.idm_mode else 0,
            atten_goal_state=self.idm_mode,
            mask_l_obs_ratio=0.0,
            transformer_layers=self.model_args.transformer_layers,
            hidden_dim=self.model_args.hidden_dim,
            transformer_heads=args.transformer_heads,
            phase="evaluate",
            gripper_width=self.model_args.gripper_width,
        )

        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"[model] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        if "vision_encoder" in getattr(args, "bf16_module", ""):
            self.model.vision_encoder.bfloat16()
        self.model.to(self.device)
        self.model._init_model_type()
        self.model.eval()

        self.cast_dtype = get_cast_dtype(args.precision)
        self.text_process_fn = partial(preprocess_text_calvin, tokenizer=clip)
        self.image_process_fn = partial(preprocess_image, image_processor=self.model.image_processor)
        self.history_len = self.model_args.sequence_length
        self.action_pred_steps = self.model_args.action_pred_steps
        self.use_ensembling = bool(args.eval_libero_ensembling)
        self.ensembling_temp = args.ensembling_temp
        self.real_eval_max_steps = args.real_eval_max_steps
        self.img_queue = deque(maxlen=self.history_len)
        self.wrist_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [self.real_eval_max_steps, self.real_eval_max_steps + self.action_pred_steps, 7],
                device=self.device,
            )

        if self.idm_mode:
            print(
                "[eval] DROID IDM mode: "
                f"window={self.history_len}, supervised_positions={self.idm_supervised_positions}, "
                f"action_pred_steps={self.action_pred_steps}, goal_offset={args.idm_goal_steps}"
            )

    def reset(self) -> None:
        self.img_queue.clear()
        self.wrist_queue.clear()
        self.state_queue.clear()
        self.text_queue.clear()
        if self.use_ensembling:
            self.all_time_actions.zero_()

    @staticmethod
    def _to_module(tensor: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
        param = next(module.parameters())
        return tensor.to(device=param.device, dtype=param.dtype)

    def _make_obs(self, episode: dict[str, Any], step_idx: int, language: str) -> dict[str, Any]:
        primary = PILImage.fromarray(episode["primary_images"][step_idx]).convert("RGB")
        wrist = PILImage.fromarray(episode["wrist_images"][step_idx]).convert("RGB")
        primary = self.image_process_fn([primary]).unsqueeze(1).to(dtype=self.cast_dtype)
        wrist = self.image_process_fn([wrist]).unsqueeze(1).to(dtype=self.cast_dtype)
        text_x = self.text_process_fn([language]).unsqueeze(1)
        pose6d = episode["gripper_pose6d"][step_idx]
        gripper_state = episode["gripper_open_state"][step_idx]
        gripper_position = episode["gripper_positions"][step_idx]
        if not self.model_args.gripper_width:
            state = torch.from_numpy(np.concatenate([pose6d, gripper_state])).to(dtype=self.cast_dtype)
        else:
            state = torch.from_numpy(np.concatenate([pose6d, gripper_position, gripper_position])).to(
                dtype=self.cast_dtype
            )
        state = state.unsqueeze(0).unsqueeze(0)
        return {
            "primary": primary.to(self.device),
            "wrist": wrist.to(self.device),
            "text": text_x.to(self.device),
            "state": state.to(self.device),
        }

    def _preprocess_frames(self, frames: np.ndarray, start: int, end: int) -> torch.Tensor:
        pil_images = [PILImage.fromarray(frames[i]).convert("RGB") for i in range(start, end)]
        return self.image_process_fn(pil_images).to(device=self.device, dtype=self.cast_dtype)

    def _precompute_image_tokens(self, frames: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        cls_embeddings = []
        batch_size = int(self.args.feature_batch_size)
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                end = min(start + batch_size, len(frames))
                image = self._preprocess_frames(frames, start, end)
                image = self._to_module(image, self.model.vision_encoder)
                image_feature, _, _ = self.model.vision_encoder.forward_encoder(image, mask_ratio=0.0)
                image_feature = self._to_module(image_feature, self.model.perceiver_resampler)
                cls_token = image_feature[:, :1, :]
                patch_tokens = image_feature[:, 1:, :]
                resampled = self.model.perceiver_resampler(patch_tokens.unsqueeze(1).unsqueeze(1))
                resampled = self._to_module(resampled, self.model.image_primary_projector)
                cls_token = self._to_module(cls_token, self.model.cls_token_primary_projector)
                projected = self.model.image_primary_projector(resampled.flatten(0, 2)).view(
                    end - start,
                    -1,
                    self.model_args.hidden_dim,
                )
                cls_projected = self.model.cls_token_primary_projector(cls_token).view(
                    end - start,
                    -1,
                    self.model_args.hidden_dim,
                )
                embeddings.append(projected)
                cls_embeddings.append(cls_projected)
        return torch.cat(embeddings, dim=0), torch.cat(cls_embeddings, dim=0)

    def _precompute_wrist_tokens(self, frames: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        cls_embeddings = []
        batch_size = int(self.args.feature_batch_size)
        with torch.inference_mode():
            for start in range(0, len(frames), batch_size):
                end = min(start + batch_size, len(frames))
                image = self._preprocess_frames(frames, start, end)
                image = self._to_module(image, self.model.vision_encoder)
                image_feature, _, _ = self.model.vision_encoder.forward_encoder(image, mask_ratio=0.0)
                image_feature = self._to_module(image_feature, self.model.perceiver_resampler)
                cls_token = image_feature[:, :1, :]
                patch_tokens = image_feature[:, 1:, :]
                resampled = self.model.perceiver_resampler(patch_tokens.unsqueeze(1).unsqueeze(1))
                resampled = self._to_module(resampled, self.model.image_wrist_projector)
                cls_token = self._to_module(cls_token, self.model.cls_token_wrist_projector)
                projected = self.model.image_wrist_projector(resampled.flatten(0, 2)).view(
                    end - start,
                    -1,
                    self.model_args.hidden_dim,
                )
                cls_projected = self.model.cls_token_wrist_projector(cls_token).view(
                    end - start,
                    -1,
                    self.model_args.hidden_dim,
                )
                embeddings.append(projected)
                cls_embeddings.append(cls_projected)
        return torch.cat(embeddings, dim=0), torch.cat(cls_embeddings, dim=0)

    def _precompute_episode_tokens(
        self,
        episode: dict[str, Any],
        language: str,
        n_steps: int,
    ) -> dict[str, torch.Tensor]:
        primary_emb, primary_cls = self._precompute_image_tokens(episode["primary_images"][:n_steps])
        wrist_emb, wrist_cls = self._precompute_wrist_tokens(episode["wrist_images"][:n_steps])

        pose6d = torch.from_numpy(episode["gripper_pose6d"][:n_steps]).to(device=self.device, dtype=self.cast_dtype)
        gripper_state = torch.from_numpy(episode["gripper_open_state"][:n_steps]).to(
            device=self.device,
            dtype=self.cast_dtype,
        )
        if not self.model_args.gripper_width:
            state = torch.cat((pose6d, gripper_state), dim=1)
        else:
            gripper_position = torch.from_numpy(episode["gripper_positions"][:n_steps]).to(
                device=self.device,
                dtype=self.cast_dtype,
            )
            state = torch.cat((pose6d, gripper_position, gripper_position), dim=1)

        with torch.inference_mode():
            arm_state_feature = self.model.arm_state_encoder(state[:, :6])
            if not self.model_args.gripper_width:
                zero = torch.tensor(0, device=self.device)
                one = torch.tensor(1, device=self.device)
                gripper_state_one_hot = torch.nn.functional.one_hot(
                    torch.where(state[:, 6:].flatten() < 1, zero, one),
                    num_classes=2,
                )
                gripper_state_feature = self.model.gripper_state_encoder(gripper_state_one_hot.type_as(state))
            else:
                gripper_state_feature = self.model.gripper_state_encoder(state[:, 6:])
            state_embedding = self.model.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
            state_embedding = state_embedding.view(n_steps, 1, self.model_args.hidden_dim)

            text_token = self.text_process_fn([language]).to(self.device)
            text_feature = self.model.clip_model.encode_text(text_token)
            text_feature = text_feature.type_as(state_embedding)
            text_embedding = self.model.text_projector(text_feature).view(1, 1, self.model_args.hidden_dim)
            text_embedding = text_embedding.repeat(n_steps, 1, 1)

        return {
            "text": text_embedding,
            "state": state_embedding,
            "image": torch.cat((primary_emb, wrist_emb), dim=1),
            "image_cls": torch.cat((primary_cls, wrist_cls), dim=1),
        }

    def _actions_from_cached_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        input_indices: list[int],
    ) -> np.ndarray:
        idx = torch.tensor(input_indices, device=self.device, dtype=torch.long)
        text_embedding = tokens["text"].index_select(0, idx).unsqueeze(0)
        state_embedding = tokens["state"].index_select(0, idx).unsqueeze(0)
        image_embedding = tokens["image"].index_select(0, idx).unsqueeze(0)
        image_cls_token_embedding = tokens["image_cls"].index_select(0, idx).unsqueeze(0)

        with torch.inference_mode():
            embeddings = torch.cat((text_embedding, state_embedding, image_embedding, image_cls_token_embedding), dim=2)
            pred_token_start_idx = embeddings.shape[2]
            transformer_input_list = [embeddings]
            if self.model.obs_pred:
                transformer_input_list.append(self.model.obs_tokens.repeat(1, self.history_len, 1, 1))
            transformer_input_list.append(self.model.action_pred_token.repeat(1, self.history_len, 1, 1))
            transformer_input = torch.cat(transformer_input_list, dim=2)
            transformer_input = transformer_input + self.model.transformer_backbone_position_embedding.repeat(
                1,
                1,
                transformer_input.shape[-2],
                1,
            )
            transformer_input = transformer_input.flatten(1, 2)
            transformer_input = self._to_module(transformer_input, self.model.transformer_backbone)
            transformer_input = self.model.embedding_layer_norm(transformer_input)
            transformer_output = self.model.transformer_backbone(
                inputs_embeds=transformer_input,
                attention_mask=self.model.attention_mask,
            )
            transformer_output = transformer_output.view(1, self.history_len, -1, self.model_args.hidden_dim)
            obs_tokens = self.model.NUM_OBS_TOKEN if self.model.obs_pred else 0
            action_pred_feature = transformer_output[
                :,
                :,
                pred_token_start_idx + obs_tokens : pred_token_start_idx + obs_tokens + self.action_pred_steps,
                :,
            ]
            action_pred_feature = self.model.action_decoder(action_pred_feature)
            arm_action = self.model.arm_action_decoder(action_pred_feature)
            gripper_action = self.model.gripper_action_decoder(action_pred_feature)
            action = torch.cat((arm_action[0], (gripper_action[0] > 0.5)), dim=-1)
            action[..., -1] = (action[..., -1] - 0.5) * 2
        return action.detach().cpu().numpy()

    def _actions_from_idm_window(
        self,
        episode: dict[str, Any],
        language: str,
        input_indices: list[int],
    ) -> np.ndarray:
        """Return all action heads from one full training-style IDM window."""
        if len(input_indices) != self.history_len:
            raise ValueError(f"Expected {self.history_len} IDM inputs, got {len(input_indices)}")
        observations = [self._make_obs(episode, index, language) for index in input_indices]
        image_primary = torch.cat([obs["primary"] for obs in observations], dim=1)
        image_wrist = torch.cat([obs["wrist"] for obs in observations], dim=1)
        state = torch.cat([obs["state"] for obs in observations], dim=1)
        text_token = torch.cat([obs["text"] for obs in observations], dim=1)
        with torch.inference_mode():
            arm_action, gripper_action, _, _, _, _ = self.model(
                image_primary=image_primary,
                image_wrist=image_wrist,
                state=state,
                text_token=text_token,
                action=torch.zeros(1, self.history_len, 7, device=self.device),
            )
        action = torch.cat((arm_action[0], gripper_action[0] > 0.5), dim=-1)
        action[..., -1] = (action[..., -1] - 0.5) * 2
        return action.detach().cpu().numpy()

    def _idm_actions_from_cached_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        input_indices: list[int],
    ) -> np.ndarray:
        """Cached-feature equivalent of _actions_from_idm_window()."""
        return self._actions_from_cached_tokens(
            tokens=tokens,
            input_indices=input_indices,
        )

    def run_episode(self, episode: dict[str, Any], language: str, warmup_steps: int, max_steps: int | None) -> EpisodePredictions:
        if self.idm_mode:
            return self._run_idm_episode(episode, language, max_steps)

        if self.use_ensembling or not self.args.cached_feature_inference:
            self.reset()
            for _ in range(max(0, warmup_steps)):
                self.forward(episode, 0, language, warmup=True)
            num_steps = len(episode["action_delta_wrist_pose"])
            if max_steps is not None:
                num_steps = min(num_steps, max_steps)
            preds = []
            for step_idx in range(num_steps):
                preds.append(self.forward(episode, step_idx, language))
            return EpisodePredictions(
                actions=np.asarray(preds, dtype=np.float32),
                target_indices=np.arange(num_steps, dtype=np.int64),
            )

        num_steps = len(episode["action_delta_wrist_pose"])
        if max_steps is not None:
            num_steps = min(num_steps, max_steps)
        tokens = self._precompute_episode_tokens(episode, language, num_steps)
        history = deque(maxlen=self.history_len)
        for _ in range(max(0, warmup_steps)):
            history.append(0)
        preds = []
        for step_idx in range(num_steps):
            history.append(step_idx)
            selected_position = len(history) - 1 if len(history) < self.history_len else -1
            input_indices = list(history)
            if len(input_indices) < self.history_len:
                input_indices += [input_indices[-1]] * (self.history_len - len(input_indices))
            preds.append(self._actions_from_cached_tokens(tokens, input_indices)[selected_position, 0])
        return EpisodePredictions(
            actions=np.asarray(preds, dtype=np.float32),
            target_indices=np.arange(num_steps, dtype=np.int64),
        )

    def _run_idm_episode(
        self,
        episode: dict[str, Any],
        language: str,
        max_steps: int | None,
    ) -> EpisodePredictions:
        """Reproduce the action labels and attention inputs used during pre-training."""
        num_episode_steps = len(episode["action_delta_wrist_pose"])
        num_windows = num_episode_steps - self.history_len + 1
        window_starts = np.arange(max(0, num_windows), dtype=np.int64)
        if max_steps is not None:
            window_starts = window_starts[:max_steps]
        if window_starts.size == 0:
            raise ValueError(
                f"Episode has {num_episode_steps} steps, but DROID IDM evaluation needs at least "
                f"{self.history_len} steps"
            )

        predictions = []
        target_indices = []
        if self.args.cached_feature_inference:
            tokens = self._precompute_episode_tokens(episode, language, num_episode_steps)
            get_actions = lambda indices: self._idm_actions_from_cached_tokens(tokens, indices)
        else:
            get_actions = lambda indices: self._actions_from_idm_window(episode, language, indices)

        position_offsets = np.arange(self.idm_supervised_positions, dtype=np.int64)[:, None]
        horizon_offsets = np.arange(self.action_pred_steps, dtype=np.int64)[None, :]
        for window_start in window_starts:
            indices = list(range(int(window_start), int(window_start) + self.history_len))
            actions = get_actions(indices)
            predictions.append(actions[: self.idm_supervised_positions].reshape(-1, 7))
            target_indices.append((window_start + position_offsets + horizon_offsets).reshape(-1))
        return EpisodePredictions(
            actions=np.concatenate(predictions, axis=0).astype(np.float32, copy=False),
            target_indices=np.concatenate(target_indices, axis=0),
        )

    def forward(self, episode: dict[str, Any], step_idx: int, language: str, warmup: bool = False):
        obs = self._make_obs(episode, step_idx, language)
        self.img_queue.append(obs["primary"])
        self.wrist_queue.append(obs["wrist"])
        self.state_queue.append(obs["state"])
        if len(self.text_queue) == 0:
            for _ in range(self.history_len):
                self.text_queue.append(obs["text"])

        image_primary = torch.cat(list(self.img_queue), dim=1)
        image_wrist = torch.cat(list(self.wrist_queue), dim=1)
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
            input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len - num_step, 1)], dim=1)
        else:
            input_image_primary = image_primary
            input_image_wrist = image_wrist
            input_state = state

        with torch.inference_mode():
            arm_action, gripper_action, _, _, _, _ = self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7, device=self.device),
            )

        if not self.use_ensembling:
            action = torch.cat((arm_action[0, :, 0, :], (gripper_action[0, :, 0, :] > 0.5)), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2
            action = action.detach().cpu().numpy()
            action = action[num_step - 1] if num_step < self.history_len else action[-1]
        else:
            selected_step = num_step - 1 if num_step < self.history_len else -1
            action = torch.cat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
            self.all_time_actions[step_idx : step_idx + 1, step_idx : step_idx + self.action_pred_steps] = action
            actions_for_curr_step = self.all_time_actions[:, step_idx]
            actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            weights = np.exp(-self.ensembling_temp * np.arange(len(actions_for_curr_step)))
            weights = weights / weights.sum()
            weights = torch.from_numpy(weights).to(self.device).unsqueeze(1)
            action = (actions_for_curr_step * weights).sum(dim=0, keepdim=True)
            action = torch.cat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2
            action = action.detach().cpu().numpy()[-1]

        if warmup:
            return None
        return action


def load_episode_cache(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as data:
        episode = {key: data[key] for key in data.files}
    if isinstance(episode["language_instruction"], np.ndarray):
        episode["language_instruction"] = decode_text(episode["language_instruction"])
    return episode


def summarize(diff: np.ndarray) -> dict[str, float]:
    if diff.size == 0:
        return {k: float("nan") for k in ("mean_error", "mae", "rmse", "max_abs")}
    abs_diff = np.abs(diff)
    return {
        "mean_error": float(np.mean(diff)),
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(abs_diff)),
    }


def summarize_scalar_errors(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if errors.size == 0:
        return {k: float("nan") for k in ("mean", "median", "rmse", "p90", "max")}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "p90": float(np.percentile(errors, 90)),
        "max": float(np.max(errors)),
    }


def compute_cumulative_trajectory_errors(
    pred_delta_wrist_pose: np.ndarray,
    gt_action_wrist_pose: np.ndarray,
    start_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = int(pred_delta_wrist_pose.shape[0])
    pred_pose = np.array(start_pose, dtype=np.float64, copy=True)
    pos_errors = np.zeros(num_steps, dtype=np.float64)
    rot_errors = np.zeros(num_steps, dtype=np.float64)

    for step_idx in range(num_steps):
        pred_pose = pred_pose @ _6d_to_pose(pred_delta_wrist_pose[step_idx, :6]).astype(np.float64)
        gt_pose = _6d_to_pose(gt_action_wrist_pose[step_idx, :6]).astype(np.float64)
        pos_errors[step_idx] = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
        rot_errors[step_idx] = rotation_geodesic_error(pred_pose, gt_pose)

    return pos_errors, rot_errors


def run_eval(args: argparse.Namespace) -> None:
    runner = OfflineSeerRealRunner(args)
    cache_paths = sorted(args.cache_dir.glob("episode_*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No cached episodes found under {args.cache_dir}")

    per_episode_rows = []
    all_gt = []
    all_pred = []
    all_gt_norm = []
    all_pred_norm = []
    all_traj_pos_errors = []
    all_traj_rot_errors = []
    final_traj_pos_errors = []
    final_traj_rot_errors = []
    trajectory_step_rows = []

    for episode_idx, cache_path in enumerate(cache_paths[: args.max_episodes]):
        episode = load_episode_cache(cache_path)
        language = decode_text(episode["language_instruction"])
        prediction = runner.run_episode(
            episode=episode,
            language=language,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps_per_episode,
        )
        pred_norm = prediction.actions
        target_indices = prediction.target_indices
        if pred_norm.shape[0] != target_indices.shape[0]:
            raise RuntimeError("Prediction and target index counts do not match")
        num_steps = pred_norm.shape[0]
        pred_phys = pred_norm.copy()
        pred_phys[:, :3] *= float(args.max_rel_pos)
        pred_phys[:, 3:6] *= float(args.max_rel_orn)
        gt = np.asarray(episode["action_delta_wrist_pose"][target_indices], dtype=np.float32)
        gt_norm = gt.copy()
        gt_norm[:, :3] /= float(args.max_rel_pos)
        gt_norm[:, 3:6] /= float(args.max_rel_orn)

        diff = pred_phys - gt
        diff_norm = pred_norm - gt_norm
        row = {
            "episode_index": episode_idx,
            "source_cache": cache_path.name,
            "action_labels": int(len(gt)),
            "phys_mae": summarize(diff)["mae"],
            "phys_rmse": summarize(diff)["rmse"],
            "phys_max_abs": summarize(diff)["max_abs"],
            "norm_mae": summarize(diff_norm)["mae"],
            "norm_rmse": summarize(diff_norm)["rmse"],
            "norm_max_abs": summarize(diff_norm)["max_abs"],
            "pos_mae": summarize(diff[:, :3])["mae"],
            "rot_mae": summarize(diff[:, 3:6])["mae"],
            "gripper_mae": summarize(diff[:, 6:7])["mae"],
            "gripper_acc": float(np.mean(np.sign(pred_phys[:, 6]) == np.sign(gt[:, 6]))),
            "target_step_min": int(target_indices.min()),
            "target_step_max": int(target_indices.max()),
            "language_instruction": language,
        }
        if not runner.idm_mode:
            start_pose = get_fk_solution(np.asarray(episode["joint_positions"][0], dtype=np.float32))
            gt_action_wrist_pose = np.asarray(episode["action_wrist_pose"][:num_steps], dtype=np.float32)
            traj_pos_errors, traj_rot_errors = compute_cumulative_trajectory_errors(
                pred_delta_wrist_pose=pred_phys,
                gt_action_wrist_pose=gt_action_wrist_pose,
                start_pose=start_pose,
            )
            traj_pos_summary = summarize_scalar_errors(traj_pos_errors)
            traj_rot_summary = summarize_scalar_errors(traj_rot_errors)
            row.update(
                {
                    "traj_pos_mean": traj_pos_summary["mean"],
                    "traj_pos_rmse": traj_pos_summary["rmse"],
                    "traj_pos_p90": traj_pos_summary["p90"],
                    "traj_pos_final": float(traj_pos_errors[-1]),
                    "traj_pos_max": traj_pos_summary["max"],
                    "traj_rot_mean": traj_rot_summary["mean"],
                    "traj_rot_rmse": traj_rot_summary["rmse"],
                    "traj_rot_p90": traj_rot_summary["p90"],
                    "traj_rot_final": float(traj_rot_errors[-1]),
                    "traj_rot_max": traj_rot_summary["max"],
                }
            )
            all_traj_pos_errors.append(traj_pos_errors)
            all_traj_rot_errors.append(traj_rot_errors)
            final_traj_pos_errors.append(float(traj_pos_errors[-1]))
            final_traj_rot_errors.append(float(traj_rot_errors[-1]))
            for step_idx, (pos_error, rot_error) in enumerate(zip(traj_pos_errors, traj_rot_errors, strict=True)):
                trajectory_step_rows.append(
                    {
                        "episode_index": episode_idx,
                        "source_cache": cache_path.name,
                        "step": step_idx,
                        "traj_pos_error": float(pos_error),
                        "traj_rot_error": float(rot_error),
                    }
                )
        per_episode_rows.append(row)
        all_gt.append(gt)
        all_pred.append(pred_phys)
        all_gt_norm.append(gt_norm)
        all_pred_norm.append(pred_norm)
        message = (
            f"[eval] episode={episode_idx:03d} action_labels={len(gt)} "
            f"phys_mae={row['phys_mae']:.6g} phys_rmse={row['phys_rmse']:.6g}"
        )
        if not runner.idm_mode:
            message += f" traj_pos_final={row['traj_pos_final']:.6g} traj_rot_final={row['traj_rot_final']:.6g}"
        print(message)

    gt = np.concatenate(all_gt, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    gt_norm = np.concatenate(all_gt_norm, axis=0)
    pred_norm = np.concatenate(all_pred_norm, axis=0)
    diff = pred - gt
    diff_norm = pred_norm - gt_norm

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    per_episode_csv = args.summary_dir / "seer_droid_real_per_episode.csv"
    with per_episode_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_episode_rows)
    trajectory_steps_csv = None
    if not runner.idm_mode:
        trajectory_steps_csv = args.summary_dir / "seer_droid_real_trajectory_per_step.csv"
        with trajectory_steps_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trajectory_step_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_step_rows)

    summary = {
        "episodes": len(per_episode_rows),
        "eval_mode": args.eval_mode,
        "action_labels": int(diff.shape[0]),
        "physical": summarize(diff),
        "normalized": summarize(diff_norm),
        "position_physical": summarize(diff[:, :3]),
        "rotation_physical": summarize(diff[:, 3:6]),
        "gripper_physical": summarize(diff[:, 6:7]),
        "gripper_accuracy": float(np.mean(np.sign(pred[:, 6]) == np.sign(gt[:, 6]))),
        "max_rel_pos": float(args.max_rel_pos),
        "max_rel_orn": float(args.max_rel_orn),
        "cache_dir": str(args.cache_dir),
        "resume_from_checkpoint": str(args.resume_from_checkpoint),
        "device": str(runner.device),
    }
    if runner.idm_mode:
        summary["idm"] = {
            "sequence_length": runner.history_len,
            "atten_goal": int(args.idm_goal_steps),
            "atten_goal_state": True,
            "atten_only_obs": True,
            "supervised_positions_per_window": int(runner.idm_supervised_positions),
            "action_pred_steps": runner.action_pred_steps,
            "labels_per_window": int(runner.idm_supervised_positions * runner.action_pred_steps),
        }
    else:
        traj_pos_errors = np.concatenate(all_traj_pos_errors, axis=0)
        traj_rot_errors = np.concatenate(all_traj_rot_errors, axis=0)
        summary.update(
            {
                "cumulative_trajectory_position_error_m": summarize_scalar_errors(traj_pos_errors),
                "cumulative_trajectory_rotation_error_rad": summarize_scalar_errors(traj_rot_errors),
                "cumulative_trajectory_final_position_error_m": summarize_scalar_errors(
                    np.asarray(final_traj_pos_errors)
                ),
                "cumulative_trajectory_final_rotation_error_rad": summarize_scalar_errors(
                    np.asarray(final_traj_rot_errors)
                ),
            }
        )
    with (args.summary_dir / "seer_droid_real_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Compared Seer {args.eval_mode} predictions against raw DROID GT")
    print(f"Episodes: {summary['episodes']}")
    print(f"Action labels: {summary['action_labels']}")
    print(f"Physical MAE: {summary['physical']['mae']:.6g}")
    print(f"Physical RMSE: {summary['physical']['rmse']:.6g}")
    print(f"Normalized MAE: {summary['normalized']['mae']:.6g}")
    print(f"Position MAE: {summary['position_physical']['mae']:.6g}")
    print(f"Rotation MAE:  {summary['rotation_physical']['mae']:.6g}")
    print(f"Gripper acc:   {summary['gripper_accuracy']:.6g}")
    if not runner.idm_mode:
        print(f"Trajectory position mean/final: {summary['cumulative_trajectory_position_error_m']['mean']:.6g} / {summary['cumulative_trajectory_final_position_error_m']['mean']:.6g}")
        print(f"Trajectory rotation mean/final: {summary['cumulative_trajectory_rotation_error_rad']['mean']:.6g} / {summary['cumulative_trajectory_final_rotation_error_rad']['mean']:.6g}")
    print(f"Wrote {per_episode_csv}")
    if trajectory_steps_csv is not None:
        print(f"Wrote {trajectory_steps_csv}")
    print(f"Wrote {args.summary_dir / 'seer_droid_real_summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Seer real-mode inference on raw DROID TFDS data.")
    parser.add_argument("--stage", choices=("extract", "eval"), required=True)
    parser.add_argument("--tfds-dir", type=Path, default=DEFAULT_TFDS_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=DEFAULT_SEER_CHECKPOINT)
    parser.add_argument("--vit-checkpoint-path", type=Path, default=DEFAULT_VIT_CHECKPOINT)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--filter-substring", default="success")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument(
        "--eval-mode",
        choices=("droid-idm", "causal"),
        default="droid-idm",
        help="droid-idm reproduces DROID pre-training labels and goal-state attention.",
    )
    parser.add_argument(
        "--idm-goal-steps",
        type=int,
        default=4,
        help="Future state offset used by the released DROID pre-training recipe.",
    )
    parser.add_argument("--max-rel-pos", type=float, default=0.02)
    parser.add_argument("--max-rel-orn", type=float, default=0.05)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device", default="auto", help="Torch device for Seer inference: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--eval-libero-ensembling", action="store_true", default=False)
    parser.add_argument("--ensembling-temp", type=float, default=0.01)
    parser.add_argument("--real-eval-max-steps", type=int, default=600)
    parser.add_argument("--calvin-input-image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--transformer-heads", type=int, default=12)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--cached-feature-inference", action="store_true", default=True)
    parser.add_argument("--no-cached-feature-inference", dest="cached_feature_inference", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "extract":
        extract_raw_droid(
            tfds_dir=args.tfds_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            start_episode=args.start_episode,
            max_episodes=args.max_episodes,
            filter_substring=args.filter_substring,
            overwrite=args.overwrite,
        )
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
