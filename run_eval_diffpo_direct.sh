#!/bin/bash
set -euo pipefail

echo "[run_eval_diffpo_direct] Starting direct-control evaluation inside container..."

CONTAINER=isaaclab-test
DST=/workspace

# Required positional: artifact ref
ARTIFACT_REF=${1:-}
OUT_LOCAL=${2:-"$(pwd)/videos"}

if [[ -z "$ARTIFACT_REF" ]]; then
  echo "Error: missing artifact ref."
  echo "Usage: $0 ENTITY/PROJECT/ARTIFACT:VERSION [OUT_LOCAL] [--flags...]"
  exit 1
fi

# Defaults
TASK="Isaac-Open-Drawer-Franka-Camera-v0"
MAX_STEPS=300
NUM_ENVS=1
SEED=123
CAMERA_H=120
CAMERA_W=160
ACTIONS_PER_INFERENCE=16
HEADLESS=1

shift || true
shift || true

# Parse remaining flags (optional)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --camera_h) CAMERA_H="$2"; shift 2 ;;
    --camera_w) CAMERA_W="$2"; shift 2 ;;
    --actions_per_inference) ACTIONS_PER_INFERENCE="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift 1 ;;
    --with-ui) HEADLESS=0; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUT_IN_CONTAINER="${DST}/videos"

echo "[run_eval_diffpo_direct] Config: task=${TASK} max_steps=${MAX_STEPS} num_envs=${NUM_ENVS} ${CAMERA_H}x${CAMERA_W} actions_per_inference=${ACTIONS_PER_INFERENCE}"
echo "[run_eval_diffpo_direct] Artifact: ${ARTIFACT_REF}"

# Copy direct eval script into the container
docker cp /home/agundogdu_theaiinstitute_com/test/eval_diffpo_direct.py ${CONTAINER}:${DST}/

# Run inside Isaac Lab container
docker exec -e WANDB_API_KEY="${WANDB_API_KEY:-}" ${CONTAINER} bash -lc "
set -e
rm -rf \"${OUT_IN_CONTAINER}\"
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/eval_diffpo_direct.py \
  --artifact \"${ARTIFACT_REF}\" \
  --task \"${TASK}\" \
  --max_steps \"${MAX_STEPS}\" \
  --num_envs \"${NUM_ENVS}\" \
  --seed \"${SEED}\" \
  --camera_h \"${CAMERA_H}\" \
  --camera_w \"${CAMERA_W}\" \
  --output_dir \"${OUT_IN_CONTAINER}\" \
  --actions_per_inference \"${ACTIONS_PER_INFERENCE}\" \
  $([[ \"${HEADLESS}\" == \"1\" ]] && echo --headless || true)
"

# Copy results back
rm -rf "${OUT_LOCAL}"
mkdir -p "${OUT_LOCAL}"
docker cp ${CONTAINER}:"${OUT_IN_CONTAINER}"/. "${OUT_LOCAL}"

echo "[run_eval_diffpo_direct] Done. Videos saved to ${OUT_LOCAL}"


