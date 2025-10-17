#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="isaaclab-test"
DATA_DIR_IN="/workspace/data"
OUT_DIR_IN="/workspace/videos"
FPS=${FPS:-20}
OUTPUT_DIR=${1:-./videos}

# Determine TTY flags
if [ -t 1 ]; then
  TTY_FLAG="-it"
else
  TTY_FLAG="-i"
fi

echo "========================================="
echo "IsaacLab Video Rendering Pipeline"
echo "========================================="
echo "Container: $CONTAINER_NAME"
echo "Input data: $DATA_DIR_IN"
echo "Container output: $OUT_DIR_IN"
echo "Host output: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "========================================="

# Step 0: Optionally clear previous videos in the container
if [ "${CLEAR_CONTAINER_VIDEOS:-1}" -eq 1 ]; then
  echo ""
  echo "[0/2] Clearing previous videos in container at ${OUT_DIR_IN} ..."
  docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "rm -rf ${OUT_DIR_IN} && mkdir -p ${OUT_DIR_IN}"
  if [ $? -ne 0 ]; then
      echo "❌ Failed to clear ${OUT_DIR_IN} in container!"
      exit 1
  fi
  echo "✓ Cleared ${OUT_DIR_IN}"
else
  echo ""
  echo "[0/2] Skipping container video clear (set CLEAR_CONTAINER_VIDEOS=0)"
fi

# Step 1: Render videos in container
echo ""
echo "[1/2] Rendering videos from HDF5 episodes..."
docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "/isaac-sim/python.sh /workspace/vpl_tools/render_videos.py \
  --base_dir ${DATA_DIR_IN} \
  --out_dir ${OUT_DIR_IN} \
  --fps ${FPS}"

if [ $? -ne 0 ]; then
    echo "❌ Video rendering failed!"
    exit 1
fi

echo ""
echo "✓ Video rendering completed!"

# Step 2: Copy videos to host
echo ""
echo "[2/2] Copying videos to host: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
docker cp ${CONTAINER_NAME}:${OUT_DIR_IN}/. "$OUTPUT_DIR/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy videos from container!"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
echo "✓ Copied $VIDEO_COUNT video files to $OUTPUT_DIR"

# Step 3: Show summary
echo ""
echo "========================================="
echo "📹 Video rendering complete!"
echo "========================================="
echo "Videos saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  - View videos: ls $OUTPUT_DIR"
echo "  - Play a video: vlc $OUTPUT_DIR/episode_0_camera_0.mp4"
echo "========================================="

