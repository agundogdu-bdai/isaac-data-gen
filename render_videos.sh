#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="isaaclab-test"
DATA_DIR_IN="/workspace/vpl_tools/data"
OUT_DIR_IN="/workspace/videos"
FPS=${FPS:-20}
CAM_LIST=${CAMS:-}
if [ $# -ge 1 ]; then DATA_DIR_IN=$1; fi
if [ $# -ge 2 ]; then OUT_DIR_IN=$2; fi
if [ $# -ge 3 ]; then FPS=$3; fi
if [ $# -ge 4 ]; then CAM_LIST=$4; fi

# Determine TTY flags
if [ -t 1 ]; then
  TTY_FLAG="-it"
else
  TTY_FLAG="-i"
fi

 # Step 0: Optionally clear previous videos in the container
if [ "${CLEAR_CONTAINER_VIDEOS:-1}" -eq 1 ]; then
  docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "rm -rf ${OUT_DIR_IN} && mkdir -p ${OUT_DIR_IN}"
fi

EXTRA=""
if [ -n "$CAM_LIST" ]; then EXTRA="--cams $CAM_LIST"; fi

docker exec ${TTY_FLAG} ${CONTAINER_NAME} bash -lc "/isaac-sim/python.sh /workspace/vpl_tools/render_videos.py \
  --base_dir ${DATA_DIR_IN} \
  --out_dir ${OUT_DIR_IN} \
  --fps ${FPS} ${EXTRA}"


