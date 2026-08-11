#!/usr/bin/env bash
# Evaluate a LIBERO-10 checkpoint from any working directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/LIBERO/.venv/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-${REPO_ROOT}/LIBERO}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/libero_scratch}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints}"
CKPT_IDS="${CKPT_IDS:-38}"
MASTER_PORT="${MASTER_PORT:-10133}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the Python executable with LIBERO and PyTorch installed." >&2
    exit 1
fi

for required_path in \
    "${LIBERO_PATH}/libero/libero/bddl_files" \
    "${LIBERO_PATH}/libero/libero/init_files" \
    "${CHECKPOINT_DIR}" \
    "${VIT_CHECKPOINT_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required evaluation resource not found: ${required_path}" >&2
        exit 1
    fi
done

GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < 1 )); then
    echo "No CUDA GPU is available to PyTorch in ${PYTHON_BIN}." >&2
    echo "Run this script in a CUDA-enabled environment, or expose GPUs with CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
if (( NPROC_PER_NODE < 1 || NPROC_PER_NODE > GPU_COUNT )); then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must be between 1 and the visible GPU count (${GPU_COUNT})." >&2
    exit 1
fi
if (( 200 % NPROC_PER_NODE != 0 )); then
    echo "NPROC_PER_NODE (${NPROC_PER_NODE}) must divide the 200 LIBERO-10 episodes." >&2
    echo "Use one of: 1, 2, 4, 5, 8, or 10 (for example, NPROC_PER_NODE=2)." >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/$(basename "${CHECKPOINT_DIR}")}"
mkdir -p "${LOG_DIR}"
read -r -a pthlist <<< "${CKPT_IDS}"

for ckpt_id in "${pthlist[@]}"; do
    this_resume_from_checkpoint="${CHECKPOINT_DIR}/${ckpt_id}.pth"
    logfile="${LOG_DIR}/${ckpt_id}.log"

    if [[ ! -f "${this_resume_from_checkpoint}" ]]; then
        echo "Checkpoint not found: ${this_resume_from_checkpoint}" >&2
        exit 1
    fi

    "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        eval_libero.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module vision_encoder \
        --vit_checkpoint_path "${VIT_CHECKPOINT_PATH}" \
        --calvin_dataset "" \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name test \
        --transformer_layers 24 \
        --phase evaluate \
        --finetune_type libero_10 \
        --libero_path "${LIBERO_PATH}" \
        --save_checkpoint_path "${SAVE_CHECKPOINT_PATH}" \
        --action_pred_steps 3 \
        --future_steps 3 \
        --sequence_length 7 \
        --obs_pred \
        --gripper_width \
        --eval_libero_ensembling \
        --resume_from_checkpoint "${this_resume_from_checkpoint}" \
        2>&1 | tee "${logfile}"
done
