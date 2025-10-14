#!/bin/bash
set -euo pipefail

echo "[collect_flat] Starting data collection..."

CONTAINER=isaaclab-test
DST=/workspace

# Defaults (override via CLI flags below)
TASK="Isaac-Open-Drawer-Franka-Camera-v0"
CHECKPOINT="${DST}/model_trained.pt"
OUT_IN_CONTAINER="/workspace/datasets/vpl_data_flat"
NUM_ENVS=1
NUM_EPISODES=3
STEPS=70
SEED=123
CAMERA_H=120
CAMERA_W=160

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output) OUT_IN_CONTAINER="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --num_episodes) NUM_EPISODES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --camera_h) CAMERA_H="$2"; shift 2 ;;
    --camera_w) CAMERA_W="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[collect_flat] Config: task=${TASK} num_envs=${NUM_ENVS} num_episodes=${NUM_EPISODES} steps=${STEPS}"
echo "[collect_flat] Cameras: ${CAMERA_H}x${CAMERA_W} | output=${OUT_IN_CONTAINER}"

# Copy scripts into the container
docker cp collect_flat.py  ${CONTAINER}:${DST}/
docker cp replay_flat.py   ${CONTAINER}:${DST}/

# Run collection (clear dataset dir inside container to refresh metadata)
docker exec ${CONTAINER} bash -lc "
set -e
rm -rf \"${OUT_IN_CONTAINER}\"
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/collect_flat.py \
  --task \"${TASK}\" \
  --checkpoint \"${CHECKPOINT}\" \
  --num_envs \"${NUM_ENVS}\" \
  --num_episodes \"${NUM_EPISODES}\" \
  --steps \"${STEPS}\" \
  --seed \"${SEED}\" \
  --output \"${OUT_IN_CONTAINER}\" \
  --camera_h \"${CAMERA_H}\" --camera_w \"${CAMERA_W}\" \
  --headless
"

# Copy results back
rm -rf vpl_collected_flat
docker cp ${CONTAINER}:"${OUT_IN_CONTAINER}" ./vpl_collected_flat
echo "[collect_flat] Done! Data saved to ./vpl_collected_flat/"


