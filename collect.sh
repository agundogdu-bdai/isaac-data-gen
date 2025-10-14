#!/bin/bash
# Simple data collection script

echo "Starting data collection..."

# Copy scripts to container
docker cp collect.py isaaclab-test:/workspace/
docker cp vpl_saver.py isaaclab-test:/workspace/

# Run collection
docker exec isaaclab-test bash -c "
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/collect.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --checkpoint /workspace/model_trained.pt \
  --num_envs 1 \
  --num_episodes 1 \
  --steps 12 \
  --output /workspace/datasets/vpl_data \
  --headless
"

# Copy results back
rm -rf vpl_collected
docker cp isaaclab-test:/workspace/datasets/vpl_data vpl_collected
echo "Done! Data saved to vpl_collected/"

