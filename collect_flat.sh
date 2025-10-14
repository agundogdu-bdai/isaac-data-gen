#!/bin/bash
set -euo pipefail

echo "[collect_flat] Starting data collection..."

CONTAINER=isaaclab-test
DST=/workspace
OUT_IN_CONTAINER=/workspace/datasets/vpl_data_flat

# Copy scripts into the container
docker cp collect_flat.py  ${CONTAINER}:${DST}/
docker cp replay_flat.py   ${CONTAINER}:${DST}/

# Run collection (edit task/checkpoint as needed)
docker exec ${CONTAINER} bash -lc "
set -e
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/collect_flat.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --checkpoint ${DST}/model_trained.pt \
  --num_envs 25 \
  --num_episodes 40 \
  --steps 70 \
  --seed 123 \
  --output ${OUT_IN_CONTAINER} \
  --camera_h 120 --camera_w 160 \
  --headless
"

# Copy results back
rm -rf vpl_collected_flat
docker cp ${CONTAINER}:${OUT_IN_CONTAINER} ./vpl_collected_flat
echo "[collect_flat] Done! Data saved to ./vpl_collected_flat/"


