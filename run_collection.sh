#!/bin/bash
# Quick script to copy files and run data collection

set -e

echo "=========================================="
echo "Data Collection Runner"
echo "=========================================="
echo ""

# Step 1: Clean up old data and copy files
echo "[1/4] Cleaning up old data in Docker container..."
docker exec isaaclab-test bash -c "rm -rf /workspace/datasets/vpl_tiled"
echo "✓ Old data cleaned"
echo ""

echo "[2/4] Copying files to Docker container..."
docker cp /home/agundogdu_theaiinstitute_com/test/collect_tiled_with_checkpoint.py isaaclab-test:/workspace/
docker cp /home/agundogdu_theaiinstitute_com/test/vpl_saver.py isaaclab-test:/workspace/
echo "✓ Files copied"
echo ""

# Step 3: Run collection
echo "[3/4] Running data collection (2 envs, 2 episodes, 60 steps)..."
echo "Collecting with 3 cameras: overview, wrist, and top-view"
echo "This should take ~2-3 minutes total"
echo ""
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_tiled_with_checkpoint.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 20 \
  --env_spacing 6.0 \
  --steps 60 \
  --num_episodes 20 \
  --width 320 \
  --height 240 \
  --checkpoint /workspace/model_trained.pt \
  --data_root /workspace/datasets/vpl_tiled \
  --robot_name franka \
  --sim_or_real sim \
  --fps 30 \
  --enable_wrist_camera \
  --enable_top_camera \
  --headless"

# Step 4: Copy results
echo ""
echo "[4/4] Copying results to local machine..."
# Clean up local directory first
rm -rf /home/agundogdu_theaiinstitute_com/test/vpl_tiled
docker cp isaaclab-test:/workspace/datasets/vpl_tiled /home/agundogdu_theaiinstitute_com/test/
echo "✓ Results copied to: /home/agundogdu_theaiinstitute_com/test/vpl_tiled"
echo ""

echo "=========================================="
echo "✓ Data Collection Complete!"
echo "=========================================="
echo ""
echo "To inspect the dataset, run:"
echo "  python3 /home/agundogdu_theaiinstitute_com/test/inspect_dataset.py"
echo ""

