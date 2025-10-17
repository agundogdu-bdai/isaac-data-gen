#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="isaaclab-test"
TASK="Isaac-Lift-Cube-Franka-Camera-Play-v0"
NUM_ENVS=${NUM_ENVS:-4}  # Reduced from 16 to avoid OOM
NUM_EPISODES=${NUM_EPISODES:-10}
CHECKPOINT=${1:-}
OUTPUT_DIR=${2:-./vpl_data}

# Optional checkpoint: if not provided, default to pretrained checkpoint in save.py
if [ $# -ge 3 ]; then CONTAINER_NAME=$3; fi

# Determine TTY flags based on whether stdout is a TTY
if [ -t 1 ]; then
  TTY_FLAG="-it"
else
  TTY_FLAG="-i"
fi

echo "========================================="
echo "IsaacLab Data Collection Pipeline"
echo "========================================="
echo "Container: $CONTAINER_NAME"
echo "Task: $TASK"
echo "Parallel Environments: $NUM_ENVS"
echo "Episodes per env: $NUM_EPISODES"
echo "Expected total: $((NUM_ENVS * NUM_EPISODES)) episodes"
echo "Output: $OUTPUT_DIR"
if [ -n "${HYDRA_TIMEOUT_OVERRIDE:-}" ]; then
  echo "Hydra override: $HYDRA_TIMEOUT_OVERRIDE"
fi
echo "========================================="

# Step -1: Sync updated scripts into container to ensure latest code runs
echo ""
echo "[0/3] Syncing updated scripts into container..."
docker cp ./vpl_tools/save.py ${CONTAINER_NAME}:/workspace/vpl_tools/save.py
docker cp ./vpl_tools/vpl_saver.py ${CONTAINER_NAME}:/workspace/vpl_tools/vpl_saver.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to sync scripts into container!"
    exit 1
fi
docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "echo '--- verify vpl_saver.py ---'; head -n 40 /workspace/vpl_tools/vpl_saver.py | sed -n '1,40p'; echo '--- grep MAX_STEPS ---'; grep -n 'MAX_STEPS' /workspace/vpl_tools/vpl_saver.py || true"
echo "✓ Synced scripts"

# Step 0: Optionally clear previous data in the container
if [ "${CLEAR_CONTAINER_DATA:-1}" -eq 1 ]; then
  echo ""
  echo "[0/3] Clearing previous data in container at /workspace/data ..."
  docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "rm -rf /workspace/data && mkdir -p /workspace/data"
  if [ $? -ne 0 ]; then
      echo "❌ Failed to clear /workspace/data in container!"
      exit 1
  fi
  echo "✓ Cleared /workspace/data"
else
  echo ""
  echo "[0/3] Skipping container data clear (set CLEAR_CONTAINER_DATA=0)"
fi

# Step 1: Collect data
echo ""
echo "[1/3] Collecting episodes in headless mode..."
docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "cd /workspace && \
  MAX_EPISODE_STEPS=\"${MAX_EPISODE_STEPS:-84}\" ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
  /workspace/isaaclab/isaaclab.sh -p /workspace/vpl_tools/save.py \
    --task ${TASK} \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    $( [ -n \"$CHECKPOINT\" ] && echo --checkpoint \"$CHECKPOINT\" ) \
    --device cuda:0 \
    ${HYDRA_TIMEOUT_OVERRIDE:-}"

if [ $? -ne 0 ]; then
    echo "❌ Data collection failed!"
    exit 1
fi

echo ""
echo "✓ Data collection completed!"

# Step 2: Copy data to host
echo ""
echo "[2/3] Copying episode data to host: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
docker cp ${CONTAINER_NAME}:/workspace/data/. "$OUTPUT_DIR/"

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy data from container!"
    exit 1
fi

# Count episodes
EPISODE_COUNT=$(ls -d "$OUTPUT_DIR"/episode_* 2>/dev/null | wc -l)
echo "✓ Copied $EPISODE_COUNT episodes to $OUTPUT_DIR"

# Show data summary
if [ -f "$OUTPUT_DIR/metadata.json" ]; then
    echo ""
    echo "Metadata:"
    cat "$OUTPUT_DIR/metadata.json"
fi

# Step 3: Show next steps
echo ""
echo "[3/3] Data collection complete!"
echo "========================================="
echo "📁 Episodes saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Render videos (optional):"
echo "     ./render_videos.sh /workspace/data /workspace/videos 20"
echo "     docker cp ${CONTAINER_NAME}:/workspace/videos ./videos"
echo ""
echo "  2. Replay an episode (optional):"
echo "     ./replay_episode.sh /workspace/data/episode_0/episode_0.h5"
echo ""
echo "  3. Train a visuomotor policy with this data"
echo "========================================="

