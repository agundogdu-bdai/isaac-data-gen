#!/bin/bash
set -euo pipefail

echo "[run_eval_diffpo] Starting evaluation inside container..."

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

# Defaults (override via CLI flags below)
# Set Lift camera task by default; tune horizons and resolution for clearer videos
TASK="Isaac-Lift-Cube-Franka-Camera-Play-v0"
MAX_STEPS=120
NUM_ENVS=1
SEED=123
CAMERA_H=240
CAMERA_W=320
POLICY_ENTRY="inference:load_policy"
# Lift task exposes two cameras: front and wrist. Default accordingly.
CAMS="front,wrist"
STREAM=1
HEADLESS=1
LOADER_PATH=""
ALLOW_UNPICKLE=0
# Evaluate DiffPO trajectories by executing multiple actions per inference
ACTIONS_PER_INFERENCE=1

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
    --policy_entry) POLICY_ENTRY="$2"; shift 2 ;;
    --cams) CAMS="$2"; shift 2 ;;
    --stream) STREAM=1; shift 1 ;;
    --no-stream) STREAM=0; shift 1 ;;
    --loader_path) LOADER_PATH="$2"; shift 2 ;;
    --allow_unpickle) ALLOW_UNPICKLE=1; shift 1 ;;
    --actions_per_inference) ACTIONS_PER_INFERENCE="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift 1 ;;
    --with-ui) HEADLESS=0; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUT_IN_CONTAINER="${DST}/videos"

# If loader path provided, copy into container and set container path
LOADER_IN_CONT=""
if [[ -n "$LOADER_PATH" ]]; then
  if [[ -f "$LOADER_PATH" ]]; then
    docker cp "$LOADER_PATH" ${CONTAINER}:${DST}/ext_loader.py
    LOADER_IN_CONT="${DST}/ext_loader.py"
    echo "[run_eval_diffpo] Copied loader to container: $LOADER_IN_CONT"
  else
    echo "[run_eval_diffpo] WARNING: loader_path not found on host: $LOADER_PATH"
  fi
fi

echo "[run_eval_diffpo] Config: task=${TASK} max_steps=${MAX_STEPS} num_envs=${NUM_ENVS} cams=${CAMS} ${CAMERA_H}x${CAMERA_W} stream=${STREAM} headless=${HEADLESS} actions_per_inference=${ACTIONS_PER_INFERENCE} loader_path=${LOADER_PATH} allow_unpickle=${ALLOW_UNPICKLE}"
echo "[run_eval_diffpo] Artifact: ${ARTIFACT_REF}"

# Copy eval script into the container
docker cp /home/agundogdu_theaiinstitute_com/test/eval_diffpo.py ${CONTAINER}:${DST}/

# No additional dependency installation needed; REGISTRY handles artifact loading

# Run inside Isaac Lab container
docker exec -e WANDB_API_KEY="${WANDB_API_KEY:-}" ${CONTAINER} bash -lc "
set -e
rm -rf \"${OUT_IN_CONTAINER}\"
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/eval_diffpo.py \
  --artifact \"${ARTIFACT_REF}\" \
  --task \"${TASK}\" \
  --max_steps \"${MAX_STEPS}\" \
  --num_envs \"${NUM_ENVS}\" \
  --seed \"${SEED}\" \
  --camera_h \"${CAMERA_H}\" \
  --camera_w \"${CAMERA_W}\" \
  --cams \"${CAMS}\" \
  $([[ \"${STREAM}\" == \"1\" ]] && echo --stream || true) \
  $([[ \"${ALLOW_UNPICKLE}\" == \"1\" ]] && echo --allow_unpickle || true) \
  --output_dir \"${OUT_IN_CONTAINER}\" \
  --policy_entry \"${POLICY_ENTRY}\" \
  --loader_path \"${LOADER_IN_CONT}\" \
  --actions_per_inference \"${ACTIONS_PER_INFERENCE}\" \
  $([[ \"${HEADLESS}\" == \"1\" ]] && echo --headless || true)
"

# Copy results back
rm -rf "${OUT_LOCAL}"
mkdir -p "${OUT_LOCAL}"
docker cp ${CONTAINER}:"${OUT_IN_CONTAINER}"/. "${OUT_LOCAL}"

echo "[run_eval_diffpo] Done. Videos saved to ${OUT_LOCAL}"


