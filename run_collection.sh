#!/bin/bash
# Quick script to run data collection with pre-installed camera environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Data Collection Runner (Camera-Enabled)"
echo "=========================================="
echo ""
echo "Using pre-installed camera environment:"
echo "  • Isaac-Open-Drawer-Franka-Camera-v0"
echo ""

# Step 1: Clean up old data
echo "[1/4] Cleaning up old data in Docker container..."
docker exec isaaclab-test bash -c "rm -rf /workspace/datasets/vpl_tiled"
echo "✓ Old data cleaned"
echo ""

# Step 2: Copy scripts
echo "[2/4] Copying collection scripts to Docker container..."
docker cp "${SCRIPT_DIR}/collect_with_camera_env.py" isaaclab-test:/workspace/
docker cp "${SCRIPT_DIR}/collect_tiled_with_checkpoint.py" isaaclab-test:/workspace/
docker cp "${SCRIPT_DIR}/vpl_saver.py" isaaclab-test:/workspace/
echo "✓ Files copied"
echo ""

# Step 3: Run collection with pre-installed camera-enabled environment
echo "[3/4] Running data collection with camera-enabled environment..."
echo "Task: Isaac-Open-Drawer-Franka-Camera-v0"
echo "Cameras: wrist (auto-follow) + top-down"
echo "Settings: 20 envs, 20 episodes, 60 steps"
echo "Expected time: ~10-15 minutes"
echo ""
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_with_camera_env.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --num_envs 25 \
  --env_spacing 6.0 \
  --steps 200 \
  --num_episodes 40 \
  --checkpoint /workspace/model_trained.pt \
  --data_root /workspace/datasets/vpl_tiled \
  --robot_name franka \
  --sim_or_real sim \
  --fps 30 \
  --headless"

# Step 4: Copy results
echo ""
echo "[4/4] Copying results to local machine..."
# Clean up local directory first
rm -rf "${SCRIPT_DIR}/vpl_tiled"
docker cp isaaclab-test:/workspace/datasets/vpl_tiled "${SCRIPT_DIR}/"
echo "✓ Results copied to: ${SCRIPT_DIR}/vpl_tiled"
echo ""

echo "=========================================="
echo "✓ Data Collection Complete!"
echo "=========================================="
echo ""
echo "Features of collected data:"
echo "  • Wrist camera: auto-follows robot hand"
echo "  • Top camera: bird's eye view"
echo "  • No coordinate system clutter"
echo "  • Total episodes: 400 (20 envs × 20 rounds)"
echo ""
echo "To inspect the dataset, run:"
echo "  python3 /home/agundogdu_theaiinstitute_com/test/inspect_dataset.py"
echo ""

