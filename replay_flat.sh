#!/bin/bash
set -euo pipefail

EPISODE=${1:-vpl_collected_flat/episode_000}
CAMERA=${2:-top}  # top | wrist | side
EPISODE_NAME=$(basename "$EPISODE")
CONTAINER=isaaclab-test

FULL_PATH=$(find vpl_collected_flat -name "$EPISODE_NAME" -type d | head -1)
[ -z "$FULL_PATH" ] && FULL_PATH="$EPISODE"
echo "[replay_flat] Replaying: $FULL_PATH (camera: $CAMERA)"

docker cp replay_flat.py ${CONTAINER}:/workspace/
docker cp "$FULL_PATH" ${CONTAINER}:/workspace/episode_flat/

docker exec ${CONTAINER} bash -lc "
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/replay_flat.py \
  --episode /workspace/episode_flat \
  --camera $CAMERA \
  --output /workspace/replay_${EPISODE_NAME}_${CAMERA}_flat.mp4 \
  --headless
"

docker cp ${CONTAINER}:/workspace/replay_${EPISODE_NAME}_${CAMERA}_flat.mp4 .
echo "[replay_flat] Saved: replay_${EPISODE_NAME}_${CAMERA}_flat.mp4"


