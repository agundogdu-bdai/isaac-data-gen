#!/bin/bash
set -euo pipefail

echo "[collect_v2] Starting data collection..."

# Paths inside container
CONTAINER=isaaclab-test
DST=/workspace
OUT_IN_CONTAINER=/workspace/datasets/vpl_data_v2

# Copy scripts into the container
docker cp collect_v2.py  ${CONTAINER}:${DST}/
docker cp replay_v2.py   ${CONTAINER}:${DST}/
docker cp vpl_saver_v2.py ${CONTAINER}:${DST}/

# Run collection (edit task/checkpoint as needed)
docker exec ${CONTAINER} bash -lc "
set -e
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/collect_v2.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --checkpoint ${DST}/model_trained.pt \
  --num_envs 1 \
  --num_episodes 3 \
  --steps 70 \
  --seed 123 \
  --output ${OUT_IN_CONTAINER} \
  --headless
"

# Copy results back
rm -rf vpl_collected_v2
docker cp ${CONTAINER}:${OUT_IN_CONTAINER} ./vpl_collected_v2
echo "[collect_v2] Done! Data saved to ./vpl_collected_v2/"


