#!/bin/bash
# Replay episode and save video locally

EPISODE=${1:-vpl_collected/episode_000}
CAMERA=${2:-top}  # top, wrist, or side
EPISODE_NAME=$(basename $EPISODE)

# Find full path
FULL_PATH=$(find vpl_collected -name "$EPISODE_NAME" -type d | head -1)
[ -z "$FULL_PATH" ] && FULL_PATH=$EPISODE

echo "Replaying: $FULL_PATH (camera: $CAMERA)"

# Copy to container
docker cp replay.py isaaclab-test:/workspace/
docker cp "$FULL_PATH" isaaclab-test:/workspace/episode/

# Run replay
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/replay.py \
  --episode /workspace/episode \
  --camera $CAMERA \
  --output /workspace/replay_${EPISODE_NAME}_${CAMERA}.mp4 \
  --headless"

# Copy video back
docker cp isaaclab-test:/workspace/replay_${EPISODE_NAME}_${CAMERA}.mp4 .
echo "Saved: replay_${EPISODE_NAME}_${CAMERA}.mp4"
