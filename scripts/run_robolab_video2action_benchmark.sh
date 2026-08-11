#!/usr/bin/env bash
set -euo pipefail

SEER_ROOT="${SEER_ROOT:-/mnt/afs/dongyifei/DreamFlyWheel/Seer}"
ROBOLAB_ROOT="${ROBOLAB_ROOT:-/mnt/afs/dongyifei/DreamFlyWheel/RoboLab}"
DATA_ROOT="${DATA_ROOT:-$ROBOLAB_ROOT/output/robolab_with_depth}"
OUT_ROOT="${OUT_ROOT:-$SEER_ROOT/outputs/robolab_video2action_causal}"

GPU="${GPU:-7}"
ENV_COUNT="${ENV_COUNT:-4}"
ROBOLAB_NUM_ENVS="${ROBOLAB_NUM_ENVS:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_DEPLOY="${RUN_DEPLOY:-1}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_seer_robolab_video2action}"
ACTION_SCALE="${ACTION_SCALE:-1.0}"
VIDEO_MODE="${VIDEO_MODE:-none}"
ENABLE_SUBTASK="${ENABLE_SUBTASK:-1}"
HEADLESS="${HEADLESS:-1}"
SEER_PRECISION="${SEER_PRECISION:-fp32}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${USER:-seer}-uv-cache}"
ROBOLAB_OUTPUT_FOLDER="${ROBOLAB_OUTPUT_FOLDER:-$RUN_NAME}"

SEER_CHECKPOINT="${SEER_CHECKPOINT:-$SEER_ROOT/checkpoints/robolab_success_all_ft/best.pth}"
VIT_CHECKPOINT="${VIT_CHECKPOINT:-$SEER_ROOT/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"

TASKS=(
  BananaInBowlTask
  RubiksCubeOrBananaTask
  BananasInCrateTask
  MustardInLeftBinTask
  SpoonInMugTask
  StackYellowOnRedTask
  BananaThenRubiksCubeTask
  BananasOutOfBinTask
)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--allow-partial] [--export-only] [--deploy-only] [--force-export] [--dry-run]

Environment variables:
  GPU=7
  ENV_COUNT=4          # number of source demonstration env dirs selected per task
  ROBOLAB_NUM_ENVS=1   # deploy_idm_actions.py --num-envs
  SEER_ROOT=$SEER_ROOT
  ROBOLAB_ROOT=$ROBOLAB_ROOT
  DATA_ROOT=$DATA_ROOT
  OUT_ROOT=$OUT_ROOT
  RUN_NAME=$RUN_NAME
  VIDEO_MODE=none        # set to all/sensor/viewport if replay videos are needed
  ACTION_SCALE=1.0       # Seer export already writes physical rel_ik deltas
  ENABLE_SUBTASK=1
  HEADLESS=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-partial)
      ALLOW_PARTIAL=1
      shift
      ;;
    --export-only)
      RUN_EXPORT=1
      RUN_DEPLOY=0
      shift
      ;;
    --deploy-only)
      RUN_EXPORT=0
      RUN_DEPLOY=1
      shift
      ;;
    --force-export)
      FORCE_EXPORT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      RUN_EXPORT=0
      RUN_DEPLOY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$ENV_COUNT" -lt 1 ]]; then
  echo "ENV_COUNT must be >= 1, got $ENV_COUNT" >&2
  exit 2
fi

for path in "$SEER_ROOT" "$ROBOLAB_ROOT" "$DATA_ROOT" "$SEER_ROOT/.venv" "$ROBOLAB_ROOT/.venv"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

for path in "$SEER_CHECKPOINT" "$VIT_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing checkpoint: $path" >&2
    exit 1
  fi
done

seer_uv_python() {
  (
    cd "$SEER_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    VIRTUAL_ENV="$SEER_ROOT/.venv" \
    PATH="$SEER_ROOT/.venv/bin:$PATH" \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
      uv run --active --no-project python "$@"
  )
}

robolab_uv_python() {
  (
    cd "$ROBOLAB_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    UV_CACHE_DIR="$UV_CACHE_DIR" \
      uv run --frozen --no-sync python "$@"
  )
}

discover_envs() {
  local task="$1"
  find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "${task}_env_*" -printf '%f\n' \
    | sed -E "s/^${task}_env_//" \
    | sort -n
}

declare -A TASK_ENVS
validation_failed=0
for task in "${TASKS[@]}"; do
  mapfile -t envs < <(discover_envs "$task")
  if [[ "${#envs[@]}" -lt "$ENV_COUNT" ]]; then
    echo "Task $task has ${#envs[@]} available env(s), requested $ENV_COUNT: ${envs[*]:-(none)}" >&2
    validation_failed=1
  fi
  selected=("${envs[@]:0:$ENV_COUNT}")
  TASK_ENVS["$task"]="${selected[*]}"
done

if [[ "$validation_failed" -ne 0 && "$ALLOW_PARTIAL" -ne 1 ]]; then
  cat >&2 <<EOF

Strict validation failed. Some selected tasks have fewer than ENV_COUNT=$ENV_COUNT
episodes in:
  $DATA_ROOT

Generate more RoboLab exports for those tasks, reduce ENV_COUNT, or rerun with:
  $0 --allow-partial
EOF
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run selection:"
  for task in "${TASKS[@]}"; do
    echo "  $task: ${TASK_ENVS[$task]:-(none)}"
  done
  exit 0
fi

mkdir -p "$OUT_ROOT/$RUN_NAME" "$UV_CACHE_DIR"

summary_tsv="$OUT_ROOT/$RUN_NAME/run_manifest.tsv"
printf "task\tdata_env\tclip_id\taction_file\tmetrics_file\tdeploy_output_folder\tstatus\n" > "$summary_tsv"

global_episode_index=0
for task in "${TASKS[@]}"; do
  read -r -a envs <<< "${TASK_ENVS[$task]}"
  if [[ "${#envs[@]}" -eq 0 ]]; then
    echo "Skipping $task: no available envs"
    continue
  fi

  echo
  echo "==== $task (${#envs[@]} envs) ===="
  for env_id in "${envs[@]}"; do
    clip_id="${task}_env_${env_id}"
    dataset_dir="$DATA_ROOT/$clip_id"
    action_dir="$OUT_ROOT/$RUN_NAME/$task/$clip_id"
    action_file="$action_dir/robolab_actions_rel_ik.npz"
    metrics_file="$action_dir/robolab_actions_rel_ik_metrics.json"
    log_file="$action_dir/run.log"
    mkdir -p "$action_dir"

    status="ok"
    echo "[$task env_$env_id] dataset=$dataset_dir"

    if [[ "$RUN_EXPORT" == "1" ]]; then
      if [[ -f "$action_file" && "$FORCE_EXPORT" != "1" ]]; then
        echo "[$task env_$env_id] export exists, skipping: $action_file"
      else
        echo "[$task env_$env_id] exporting Seer actions on GPU $GPU"
        seer_uv_python "$SEER_ROOT/scripts/export_seer_actions_for_robolab.py" \
          --seer-root "$SEER_ROOT" \
          --dataset-dir "$dataset_dir" \
          --resume-from-checkpoint "$SEER_CHECKPOINT" \
          --vit-checkpoint-path "$VIT_CHECKPOINT" \
          --output "$action_file" \
          --metrics-output "$metrics_file" \
          --precision "$SEER_PRECISION" \
          > "$log_file" 2>&1 || status="export_failed"
      fi
    fi

    if [[ "$status" == "ok" && "$RUN_DEPLOY" == "1" ]]; then
      if [[ ! -f "$action_file" ]]; then
        echo "[$task env_$env_id] missing action file for deploy: $action_file" >&2
        status="missing_action"
      else
        deploy_args=(
          "$ROBOLAB_ROOT/scripts/idm_deploy/deploy_idm_actions.py"
          --actions "$action_file"
          --task "$task"
          --controller rel_ik
          --episode-index "$global_episode_index"
          --num-envs "$ROBOLAB_NUM_ENVS"
          --action-scale "$ACTION_SCALE"
          --video-mode "$VIDEO_MODE"
          --output-folder-name "$ROBOLAB_OUTPUT_FOLDER"
          --source-dataset "$dataset_dir"
          --reference-video "$dataset_dir/rgb.mp4"
        )
        if [[ "$ENABLE_SUBTASK" == "1" ]]; then
          deploy_args+=(--enable-subtask)
        fi
        if [[ "$HEADLESS" == "1" ]]; then
          deploy_args+=(--headless)
        fi

        echo "[$task env_$env_id] replaying actions in RoboLab on GPU $GPU"
        robolab_uv_python "${deploy_args[@]}" \
          >> "$log_file" 2>&1 || status="deploy_failed"
      fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$task" "$env_id" "$clip_id" "$action_file" "$metrics_file" "$ROBOLAB_OUTPUT_FOLDER" "$status" \
      >> "$summary_tsv"

    if [[ "$status" != "ok" ]]; then
      echo "[$task env_$env_id] status=$status, see $log_file" >&2
      exit 1
    fi
    global_episode_index=$((global_episode_index + 1))
  done
done

echo
echo "Benchmark run complete."
echo "Manifest: $summary_tsv"
echo "Seer outputs: $OUT_ROOT/$RUN_NAME"
echo "RoboLab outputs: $ROBOLAB_ROOT/output/$ROBOLAB_OUTPUT_FOLDER"
