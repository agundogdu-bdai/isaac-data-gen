#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="isaaclab-test"
EPISODE_PATH=${1:-}
OUTPUT_DIR=${2:-./replay_videos}
VIDEO_DIR_IN="/workspace/replay_videos"

if [ -z "$EPISODE_PATH" ]; then
  echo "Usage: $0 <episode_path_in_container> [output_dir]"
  echo ""
  echo "Example:"
  echo "  $0 /workspace/data/episode_0/episode_0.h5"
  echo "  $0 /workspace/data/episode_0/episode_0.h5 ./my_replay_videos"
  echo ""
  echo "Available episodes in container:"
  docker exec ${CONTAINER_NAME} bash -c "ls /workspace/data/episode_*/episode_*.h5 2>/dev/null | head -10"
  exit 1
fi

# Determine TTY flags
if [ -t 1 ]; then
  TTY_FLAG="-it"
else
  TTY_FLAG="-i"
fi

echo "========================================="
echo "IsaacLab Episode Replay + Video Export"
echo "========================================="
echo "Container: $CONTAINER_NAME"
echo "Episode: $EPISODE_PATH"
echo "Container output: $VIDEO_DIR_IN"
echo "Host output: $OUTPUT_DIR"
echo "========================================="

# Step 0: Optionally clear previous replay videos in the container
if [ "${CLEAR_CONTAINER_VIDEOS:-1}" -eq 1 ]; then
  echo ""
  echo "[0/2] Clearing previous replay videos in container at ${VIDEO_DIR_IN} ..."
  docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "rm -rf ${VIDEO_DIR_IN} && mkdir -p ${VIDEO_DIR_IN}"
  if [ $? -ne 0 ]; then
      echo "❌ Failed to clear ${VIDEO_DIR_IN} in container!"
      exit 1
  fi
  echo "✓ Cleared ${VIDEO_DIR_IN}"
else
  echo ""
  echo "[0/2] Skipping container replay video clear (set CLEAR_CONTAINER_VIDEOS=0)"
fi

# Step 1: Replay episode with video recording
echo ""
echo "[1/2] Replaying episode with video recording..."
docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "cd /workspace && \
  mkdir -p ${VIDEO_DIR_IN} && \
  export REPLAY_VIDEO_PATH=${VIDEO_DIR_IN} && \
  export REPLAY_VIDEO_FPS=${FPS:-30} && \
  /workspace/isaaclab/isaaclab.sh -p /workspace/vpl_tools/replay.py \
    --episode_path ${EPISODE_PATH} \
    --device cuda:0 \
    --enable_cameras \
    --task Isaac-Lift-Cube-Franka-Camera-Play-v0"

if [ $? -ne 0 ]; then
    echo "❌ Replay failed!"
    exit 1
fi

echo ""
echo "✓ Replay completed!"

# Step 2: Copy videos to host
echo ""
echo "[2/2] Copying replay video to host: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
docker cp ${CONTAINER_NAME}:${VIDEO_DIR_IN}/. "$OUTPUT_DIR/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy videos from container!"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
echo "✓ Copied $VIDEO_COUNT video file(s) to $OUTPUT_DIR"

# Step 3: Show summary
echo ""
echo "========================================="
echo "📹 Replay video export complete!"
echo "========================================="
echo "Videos saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  - View videos: ls $OUTPUT_DIR"
echo "  - Play video: vlc $OUTPUT_DIR/*.mp4"
echo "========================================="

