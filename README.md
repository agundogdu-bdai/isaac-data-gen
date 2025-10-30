## Quickstart

End-to-end workflow for IsaacLab Visuomotor Data Collection using camera-enabled environments.

### 1) Build the image
```bash
cd /home/agundogdu_theaiinstitute_com/test
docker build -f Dockerfile.isaaclab-visuomotor -t isaaclab-visuomotor:latest .
```

### 2) Start a container
```bash
# Remove existing container if rebuilding
docker rm -f isaaclab-test

# Make scripts executable
chmod +x *.sh

docker run --gpus all -d --name isaaclab-test \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --entrypoint bash isaaclab-visuomotor:latest -c "sleep infinity"
```

### 3) Available Camera Tasks (pre-registered in image)
- Isaac-Open-Drawer-Franka-Camera-v0
- Isaac-Open-Drawer-Franka-Camera-Play-v0
- Isaac-Lift-Cube-Franka-Camera-v0
- Isaac-Lift-Cube-Franka-Camera-Play-v0

### 4) Collect episodes and export to host
```bash
# Collects episodes AND automatically copies to host
# With 4 parallel envs × 10 episodes = 40 total episodes
./collect_and_export.sh                              # → ./vpl_data (40 episodes)
./collect_and_export.sh "" ./my_data                 # custom output dir
./collect_and_export.sh /workspace/checkpoint.pt     # custom checkpoint

# Collect more episodes
NUM_EPISODES=20 ./collect_and_export.sh              # 4 envs × 20 = 80 episodes

# Use more/fewer parallel environments
NUM_ENVS=25 NUM_EPISODES=40 ./collect_and_export.sh   # 2 envs × 2 = 4 episodes
```

### 5) Render videos and export to host (optional)
```bash
# Renders per-camera MP4s AND automatically copies to host
./render_and_export.sh                               # → ./videos
./render_and_export.sh ./my_videos                   # custom output dir

# Custom FPS
FPS=30 ./render_and_export.sh                        # 30 FPS videos
```

### 6) Replay an episode and export video (optional)
```bash
# Replay a specific episode AND automatically export video to host
FPS=20 ./replay_episode_in_container.sh /workspace/data/episode_0/episode_0.h5 ./replay_videos_20fps                 # → ./replay_videos
./replay_episode_in_container.sh /workspace/data/episode_0/episode_0.h5 ./my_replays       # custom output dir

# List available episodes first
docker exec isaaclab-test ls /workspace/data/episode_*/episode_*.h5
```

## Complete Workflow Summary

```bash
# 1. Build image (one time)
docker build -f Dockerfile.isaaclab-visuomotor -t isaaclab-visuomotor:latest .

# 2. Start container (remove existing if rebuilding)
docker rm -f isaaclab-test
chmod +x *.sh
docker run --gpus all -d --name isaaclab-test \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 \
  --entrypoint bash isaaclab-visuomotor:latest -c "sleep infinity"

# 3. Collect data (creates ./vpl_data with 40 episodes: 4 envs × 10 each)
./collect_and_export.sh

# 4. Optional: Render videos from HDF5 data (creates ./videos)
./render_and_export.sh

# 5. Optional: Replay episode with video export (creates ./replay_videos)
./replay_episode_in_container.sh /workspace/data/episode_0/episode_0.h5
```

## Push the data:

```bash
gsutil -m rsync -r new_data gs://bdai-common-storage/visuomotor/datasets/ahmet/Isaac-Open-Drawer-Franka-v0_sim_franka_20251016_trunced
```

## Tag the dockername us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest and push to the artifactory

```bash
# Variables
PROJECT_ID=engineering-380817
REPO=bdai
IMAGE_NAME=isaaclab-data-gen
REGION=us
LOCAL_IMAGE=isaaclab-visuomotor:latest

# Optional: use a versioned tag instead of "latest"
TAG=latest    # e.g., TAG=$(date +%Y%m%d_%H%M%S)

# Build full remote image reference
REMOTE_IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}

# Tag local image to remote
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE}

# Push to Artifact Registry
docker push ${REMOTE_IMAGE}

# Verify (optional)
docker pull ${REMOTE_IMAGE}
```