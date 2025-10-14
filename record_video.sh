#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Isaac Lab Policy Video Recorder"
echo "=========================================="
echo ""

# Step 1: Run policy with video recording
echo "[1/2] Running policy and recording video..."
echo "This will record 500 steps (~17 seconds at 30fps)"
echo "NOTE: This may take a few minutes..."
echo ""

docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 25 \
  --checkpoint /workspace/model_trained.pt \
  --video \
  --video_length 500 \
  --headless"

echo ""
echo "✓ Recording complete"
echo ""

# Step 2: Find and copy the video
echo "[2/2] Copying video to local machine..."

# The play.py script saves videos to logs directory
# Find the most recent video file
VIDEO_FILE=$(docker exec isaaclab-test bash -c "find /workspace/isaaclab/logs -name '*.mp4' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-")

if [ -z "$VIDEO_FILE" ]; then
    echo "❌ Error: No video file found in container"
    echo ""
    echo "The video recording may have failed."
    echo "Try running without --headless flag on a machine with display:"
    echo ""
    echo "docker exec -it isaaclab-test bash -c \"cd /workspace/isaaclab && \\"
    echo "./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "  --task Isaac-Open-Drawer-Franka-v0 \\"
    echo "  --num_envs 25 \\"
    echo "  --checkpoint /workspace/model_trained.pt\""
    echo ""
    echo "Then use external screen recording software."
    exit 1
fi

echo "Found video: $VIDEO_FILE"

# Create local videos directory
mkdir -p "${SCRIPT_DIR}/videos"

# Copy video to local
LOCAL_VIDEO="${SCRIPT_DIR}/videos/demo.mp4"
docker cp "isaaclab-test:${VIDEO_FILE}" "${LOCAL_VIDEO}"

echo "✓ Video copied to: ${LOCAL_VIDEO}"
echo ""
echo "=========================================="
echo "Recording Complete!"
echo "=========================================="
echo ""
echo "Video location: ${LOCAL_VIDEO}"
echo ""
echo "To view:"
echo "  vlc ${LOCAL_VIDEO}"
echo "  # or"
echo "  xdg-open ${LOCAL_VIDEO}"
echo ""
