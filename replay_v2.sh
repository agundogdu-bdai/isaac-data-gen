#!/bin/bash
set -euo pipefail

echo "[replay_v2] Starting replay..."

# Container and paths
CONTAINER=isaaclab-test
DST=/workspace

if [ ${#} -lt 2 ]; then
  echo "Usage: $0 <episode_dir_or_h5_on_host> <camera:{top|wrist|side}> [output_mp4_on_host]"
  exit 1
fi

EP_HOST="$1"
CAM="$2"

if [ ! -d "${EP_HOST}" ] && [[ "${EP_HOST}" != *.h5 ]]; then
  echo "Error: episode path '${EP_HOST}' must be a directory containing .h5 or a .h5 file"
  exit 1
fi

EP_BASE="$(basename "${EP_HOST%/}")"
OUT_HOST="${3:-replay_${EP_BASE}_${CAM}_v2.mp4}"

# Paths inside container
EP_IN_CONTAINER="${DST}/tmp_episode_v2"
OUT_IN_CONTAINER="${DST}/$(basename "${OUT_HOST}")"

# Copy script and episode into the container
docker cp replay_v2.py  ${CONTAINER}:${DST}/
docker cp "${EP_HOST}"    ${CONTAINER}:${EP_IN_CONTAINER}

# Run replay inside container
docker exec ${CONTAINER} bash -lc "
set -e
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p ${DST}/replay_v2.py \
  --episode ${EP_IN_CONTAINER} \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --camera ${CAM} \
  --output ${OUT_IN_CONTAINER} \
  --headless
"

# Copy result out
docker cp ${CONTAINER}:${OUT_IN_CONTAINER} "${OUT_HOST}"

# Optional cleanup inside container (comment out if you want to keep files)
docker exec ${CONTAINER} bash -lc "rm -rf ${EP_IN_CONTAINER} ${OUT_IN_CONTAINER} || true"

echo "[replay_v2] Saved video to ${OUT_HOST}"


