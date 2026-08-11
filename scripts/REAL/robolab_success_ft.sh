#!/usr/bin/env bash
set -euo pipefail

SEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${SEER_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="${SEER_ROOT}/real_data"
DATASET_NAME="robolab_success_all"
TRAIN_DATA_INFO="robolab_success_all_train"
VAL_DATA_INFO="robolab_success_all_val"
CHECKPOINT_ROOT="${SEER_ROOT}/checkpoints"
SAVE_CHECKPOINT_ROOT="${SAVE_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT}}"
PRETRAINED_CHECKPOINT="${CHECKPOINT_ROOT}/real_world_droid/seer.pth"
VIT_CHECKPOINT="${CHECKPOINT_ROOT}/vit_mae/mae_pretrain_vit_base.pth"
RUN_NAME="${RUN_NAME:-robolab_success_all_ft}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${SEER_ROOT}"

# The 7-frame causal input and 10-frame data window match the real fine-tune
# recipe. Future frames are targets for action/image losses, not inputs.
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/seer-uv-cache}" uv run --python "${SEER_ROOT}/.venv/bin/python" "${SEER_ROOT}/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --bf16_module vision_encoder \
    --vit_checkpoint_path "${VIT_CHECKPOINT}" \
    --calvin_dataset "" \
    --workers "${WORKERS:-8}" \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs "${NUM_EPOCHS:-20}" \
    --seed 42 \
    --batch_size "${BATCH_SIZE:-16}" \
    --precision fp32 \
    --learning_rate 1e-3 \
    --save_checkpoint \
    --finetune_type real \
    --root_dir "${DATA_ROOT}" \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name "${RUN_NAME}" \
    --save_checkpoint_path "${SAVE_CHECKPOINT_ROOT}" \
    --transformer_layers 24 \
    --phase finetune \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --small_size "${SMALL_SIZE:-0}" \
    --obs_pred \
    --loss_action \
    --loss_image \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 15 \
    --warmup_epochs 5 \
    --real_dataset_names "${DATASET_NAME}" \
    --real_dataset_info "${TRAIN_DATA_INFO}" \
    --real_val_dataset_info "${VAL_DATA_INFO}" \
    --reset_action_token \
    --reset_obs_token \
    --use_aug_data \
    --validation \
    --validation_every "${VALIDATION_EVERY:-1}" \
    --validation_max_batches "${VALIDATION_MAX_BATCHES:--1}" \
    --offline \
    --finetune_from_pretrained_ckpt "${PRETRAINED_CHECKPOINT}"
