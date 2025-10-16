#!/bin/bash
set -euo pipefail

# Defaults (override via CLI flags)
EPISODE=${1:-vpl_collected_flat/episode_000}
CAMERA=${2:-top}  # top | wrist | side
TASK="Isaac-Open-Drawer-Franka-Camera-v0"
CONTAINER=isaaclab-test
OUTPUT=""         # host output path; default computed below

# Parse additional flags after the first two positional args
shift $(( $# > 0 ? 1 : 0 )) || true
shift $(( $# > 0 ? 1 : 0 )) || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera) CAMERA="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

EPISODE_NAME=$(basename "$EPISODE")

# Resolve episode path (allow directory or .h5)
FULL_PATH=$(find vpl_collected_flat -name "$EPISODE_NAME" -type d | head -1)
[ -z "$FULL_PATH" ] && FULL_PATH="$EPISODE"

echo "[replay_flat] Replaying: $FULL_PATH (camera: $CAMERA, task: $TASK)"

# Decide output file on host and inside container
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="replay_${EPISODE_NAME}_${CAMERA}_actions_flat.mp4"
fi
OUT_BASENAME=$(basename "$OUTPUT")
OUT_IN_CONTAINER="/workspace/${OUT_BASENAME}"

# Copy script and episode into the container
docker cp replay_flat.py ${CONTAINER}:/workspace/
docker cp "$FULL_PATH" ${CONTAINER}:/workspace/episode_flat/

# Run replay inside container
docker exec ${CONTAINER} bash -lc "
cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/replay_flat.py \
  --episode /workspace/episode_flat \
  --camera \"$CAMERA\" \
  --task \"$TASK\" \
  --output \"$OUT_IN_CONTAINER\" \
  --headless
"

# Copy result back to host
docker cp ${CONTAINER}:"${OUT_IN_CONTAINER}" "$OUTPUT"
echo "[replay_flat] Saved: $OUTPUT"


