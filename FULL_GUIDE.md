# Isaac Data Gen - Command Reference

---

## Quick Start

### 1. Build Docker Image
```bash
docker build -t isaaclab:2.2-ray \
  -f isaaclab-tiledcam-starter/Dockerfile1.isaaclab-ray .
```

### 2. Run Container
```bash
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g \
  -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  isaaclab:2.2-ray -lc "sleep infinity"
```

### 2.5. Clean GPU Processes (if needed)
```bash
# Check GPU usage (run on host)
nvtop  # or nvidia-smi

# Option 1: Restart container (cleanest)
docker restart isaaclab-test

# Option 2: Kill Python processes in container
docker exec isaaclab-test bash -c "pkill -9 python"

# Option 3: Stop and remove old container, start fresh
docker stop isaaclab-test
docker rm isaaclab-test
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g \
  -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  isaaclab:2.2-ray -lc "sleep infinity"
```

### 3. Train Policy (if needed)
```bash
# Fast training for testing (~4 min, no video recording)
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --headless"

# Copy checkpoint (find the most recent run and copy last model)
docker exec isaaclab-test bash -c "
  LATEST_RUN=\$(ls -td /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/2025-* | head -1)
  LATEST_MODEL=\$(ls -t \${LATEST_RUN}/model_*.pt | head -1)
  cp \${LATEST_MODEL} /workspace/model_trained.pt
  echo 'Copied:' \${LATEST_MODEL}
"

# Or manually specify the checkpoint
docker exec isaaclab-test bash -c "cp /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/2025-10-12_21-23-43/model_19.pt /workspace/model_trained.pt"
```

### 4. Collect Dataset
```bash
# Quick method
./run_collection.sh
```

---

## Push Docker Image

```bash

# Tag image
docker tag isaaclab:2.2-ray \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# Push
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# Verify
gcloud artifacts docker images list \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen
```

---

## Run Container

```bash
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g \
  -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  isaaclab:2.2-ray -lc "sleep infinity"
```

---

## Train RL Policy (Optional)

**Note:** Training uses the standard environment (no cameras) for faster training.

### Option 1: Full Training (Recommended)
```bash
# Fast version (no video recording, ~45-60 min)
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --headless"

# With WandB logging (optional)
docker exec -e WANDB_API_KEY=$WANDB_API_KEY isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --logger wandb \
  --headless"

# With video recording (slower, ~2-3 hours, useful for monitoring)
docker exec -e WANDB_API_KEY=$WANDB_API_KEY isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --logger wandb \
  --video \
  --video_interval 100 \
  --headless"
```
Copy checkpoint to collection location:
```bash
# Find and copy the latest checkpoint
docker exec isaaclab-test bash -c "
  LATEST_RUN=\$(ls -td /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/2025-* | head -1)
  LATEST_MODEL=\$(ls -t \${LATEST_RUN}/model_*.pt | head -1)
  cp \${LATEST_MODEL} /workspace/model_trained.pt
  echo 'Copied checkpoint:' \${LATEST_MODEL}
"

# Or specify iteration explicitly (e.g., model_3000.pt for 3000 iterations)
docker exec isaaclab-test bash -c "cp /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/*/model_3000.pt /workspace/model_trained.pt 2>/dev/null || echo 'Checkpoint not found'"
```

### Option 2: Quick Training (For Testing)
```bash
# Fast version (no video, ~4 min, 15k-17k steps/s)
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --headless"

# With WandB logging (optional)
docker exec -e WANDB_API_KEY=$WANDB_API_KEY isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --logger wandb \
  --headless"
```
Copy checkpoint:
```bash
# Find and copy the latest checkpoint automatically
docker exec isaaclab-test bash -c "
  LATEST_RUN=\$(ls -td /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/2025-* | head -1)
  LATEST_MODEL=\$(ls -t \${LATEST_RUN}/model_*.pt | head -1)
  cp \${LATEST_MODEL} /workspace/model_trained.pt
  echo 'Copied checkpoint:' \${LATEST_MODEL}
"
```

### Option 3: Use Existing Checkpoint
```bash
# Copy your pre-trained checkpoint from host
docker cp /path/to/your/model.pt isaaclab-test:/workspace/model_trained.pt
```

---

## Prepare Data Collection

**Note:** Collection scripts are now pre-installed in the Docker image!

If you're using an old image, rebuild with:
```bash
docker build -t isaaclab:2.2-ray \
  -f isaaclab-tiledcam-starter/Dockerfile1.isaaclab-ray .
```

Or manually copy scripts (if needed):
```bash
docker cp collect_with_camera_env.py isaaclab-test:/workspace/
docker cp vpl_saver.py isaaclab-test:/workspace/
```

---

## Collect Dataset

### Quick Method (Recommended)
```bash
./run_collection.sh
```

### Manual Collection

**Method 1: Camera-Enabled Environment**
```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_with_camera_env.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --num_envs 20 \
  --env_spacing 5.0 \
  --steps 60 \
  --num_episodes 2 \
  --checkpoint /workspace/model_trained.pt \
  --data_root /workspace/datasets/vpl_tiled \
  --robot_name franka \
  --sim_or_real sim \
  --fps 30 \
  --headless"
```

**Method 2: Legacy Environment**
```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_tiled_with_checkpoint.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 20 \
  --env_spacing 5.0 \
  --steps 60 \
  --num_episodes 2 \
  --width 320 \
  --height 240 \
  --checkpoint /workspace/model_trained.pt \
  --data_root /workspace/datasets/vpl_tiled \
  --robot_name franka \
  --sim_or_real sim \
  --fps 30 \
  --headless"
```

Dual camera:
```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_tiled_with_checkpoint.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 20 \
  --env_spacing 5.0 \
  --steps 60 \
  --num_episodes 2 \
  --width 320 \
  --height 240 \
  --checkpoint /workspace/model_trained.pt \
  --data_root /workspace/datasets/vpl_tiled \
  --robot_name franka \
  --sim_or_real sim \
  --fps 30 \
  --enable_wrist_camera \
  --wrist_cam_offset -0.05 0.0 0.08 \
  --wrist_cam_look_offset 0.25 0.0 0.0 \
  --enable_top_camera \
  --top_cam_offset 0.0 0.0 3.0 \
  --top_tgt_offset 0.4 0.0 0.5 \
  --headless"
```

---

## Copy Dataset to Host

```bash
docker cp isaaclab-test:/workspace/datasets/vpl_tiled ./vpl_tiled
```

---

## Customization

```bash
# More episodes
--num_envs 20 --num_episodes 10  # 200 total

# Longer episodes
--steps 100

# Higher resolution
--width 640 --height 480

# Frame sampling
--video_interval 2

# Wrist camera position (EE frame: X=forward, Y=side, Z=up)
--wrist_cam_offset -0.10 0.0 0.10
--wrist_cam_look_offset 0.30 0.05 0.0
```

---

## View Results

```bash
# View videos
vlc vpl_tiled/*/episode_000/camera_0/episode_000.mp4  # Overview
vlc vpl_tiled/*/episode_000/camera_1/episode_000.mp4  # Wrist
vlc vpl_tiled/*/episode_000/camera_2/episode_000.mp4  # Top
vlc vpl_tiled/*/episode_000/camera_*/episode_000.mp4  # All cameras

# Inspect HDF5
python3 -c "import h5py; f = h5py.File('vpl_tiled/*/episode_000/episode_000.h5'); print(list(f.keys())); print('color shape:', f['color'].shape)"
```

---

## Dataset Format

```
vpl_tiled/
└── Isaac-Open-Drawer-Franka-v0_sim_franka_YYYYMMDD_HHMMSS/
    ├── episode_000/
    │   ├── camera_0/episode_000.mp4
    │   ├── camera_1/episode_000.mp4
    │   ├── camera_2/episode_000.mp4
    │   └── episode_000.h5
    └── metadata.json
```

**HDF5 Contents:**
- `color`: `(T, N_cams, H, W, 3)` - RGB frames
- `action`: `(T, action_dim)` - Actions
- `proprio`: `(T, obs_dim)` - Proprioception
- `intrinsics`: `(N_cams, 3, 3)` - Camera intrinsics
- `extrinsics`: `(T, N_cams, 4, 4)` - Camera extrinsics
- `joint_pos`, `joint_vel`: Joint states
- `ee_position`, `ee_rotation`: End-effector pose
- `timestamps`: Timestep indices

---

## Troubleshooting

### GPU Cleanup (Stale Processes)
```bash
# Check GPU usage on host
nvtop  # or nvidia-smi

# Quick fix: Restart container
docker restart isaaclab-test
# Wait 5-10 seconds for clean restart
sleep 10

# Force kill all Python processes in container
docker exec isaaclab-test bash -c "pkill -9 python"
docker exec isaaclab-test bash -c "pkill -9 isaaclab.sh"

# Nuclear option: Fresh container
docker stop isaaclab-test
docker rm isaaclab-test
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  isaaclab:2.2-ray -lc "sleep infinity"
```

### Common Issues
```bash
# GPU out of memory
--num_envs 12 --width 256 --height 192

# No videos created - check env vars
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1

# Training stuck or slow - kill stale processes
docker restart isaaclab-test

# Check logs
docker logs isaaclab-test --tail 100

# Check running processes in container
docker exec isaaclab-test bash -c "ps aux | grep python"
```

### Checkpoint Issues
```bash
# Can't find checkpoint - list all available checkpoints
docker exec isaaclab-test bash -c "find /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/ -name 'model_*.pt' | sort"

# Find most recent checkpoint
docker exec isaaclab-test bash -c "
  LATEST_RUN=\$(ls -td /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/2025-* | head -1)
  echo 'Latest run:' \${LATEST_RUN}
  ls -lh \${LATEST_RUN}/model_*.pt
"

# Copy specific checkpoint manually
docker exec isaaclab-test bash -c "cp /workspace/isaaclab/logs/rsl_rl/franka_open_drawer/YYYY-MM-DD_HH-MM-SS/model_N.pt /workspace/model_trained.pt"

# Checkpoint naming: model_0.pt, model_50.pt, model_100.pt, etc.
# Saved every 50 iterations by default (configurable in PPO config)
```

---

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | `Isaac-Open-Drawer-Franka-v0` | Isaac Lab task |
| `--num_envs` | 8 | Parallel environments |
| `--num_episodes` | 2 | Episodes per environment |
| `--steps` | 60 | Steps per episode |
| `--env_spacing` | 5.0 | Meters between environments |
| `--width` | 320 | Camera width |
| `--height` | 240 | Camera height |
| `--fps` | 30 | Video FPS |
| `--video_interval` | 1 | Frame sampling interval |
| `--enable_wrist_camera` | False | Enable wrist camera |
| `--wrist_cam_offset` | `[-0.05, 0.0, 0.08]` | Wrist camera position in EE frame |
| `--wrist_cam_look_offset` | `[0.25, 0.0, 0.0]` | Wrist camera look-at in EE frame |
| `--enable_top_camera` | False | Enable top-view camera |
| `--top_cam_offset` | `[0.0, 0.0, 3.0]` | Top camera position from env origin |
| `--top_tgt_offset` | `[0.4, 0.0, 0.5]` | Top camera look-at from env origin |

---

## Key Files

- `collect_with_camera_env.py` - Optimized for camera-enabled environments
- `collect_tiled_with_checkpoint.py` - Legacy collector (manual camera setup)
- `vpl_saver.py` - Dataset writer
- `joint_pos_env_camera_cfg.py` - Camera environment config (pre-installed)

---

## Verify Camera Environment

```bash
# List camera environments
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
  ./isaaclab.sh -p -c 'import gymnasium as gym; \
  print([e for e in gym.envs.registry.keys() if \"Camera\" in e])'"

# Test environment
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Open-Drawer-Franka-Camera-Play-v0 \
  --enable_cameras --num_envs 4"
```

---

## Notes

- Total episodes = `num_envs × num_episodes`
- Camera offsets: EE frame (wrist) or env origin (top)
  - EE frame: X=forward, Y=side, Z=up
  - Env frame: X=forward, Y=side, Z=up

### Camera Environment Setup
The custom camera environment (`Isaac-Open-Drawer-Franka-Camera-v0`) includes:
- **wrist_camera**: Attached to robot's end-effector (auto-follows)
- **top_camera**: Static overhead view
- **ee_frame**: End-effector frame sensor
- **cabinet_frame**: Cabinet frame sensor

**Pre-installed in Docker image** - no manual installation needed!

To verify cameras are available:
```bash
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
  ./isaaclab.sh -p -c 'import gymnasium as gym; \
  print([e for e in gym.envs.registry.keys() if \"Camera\" in e])'"
```

---

## Performance Tips

### Before Training - Clean GPU Processes
Always check for stale processes before starting new training:
```bash
# On host - check GPU usage
nvtop  # or nvidia-smi

# If GPU is in use, restart container
docker restart isaaclab-test && sleep 10

# Or kill stale processes
docker exec isaaclab-test bash -c "pkill -9 python"
```

### Training Speed
**Fast (No Video):**
- 15,000-17,000 steps/s on L4 GPU
- ~3 seconds per iteration (512 envs)
- 100 iterations: ~4 minutes
- **Recommended for: Quick testing, fast iteration**

**With Video Recording:**
- 3,000-6,000 steps/s (4-5x slower)
- 8-18 seconds per iteration (inconsistent due to encoding)
- 100 iterations: ~15-25 minutes
- **Recommended for: Monitoring training, debugging behavior**

### Video Recording Best Practices
- **For fast training:** Remove `--video` flag entirely
- **For monitoring:** Use high `--video_interval` (e.g., 100-200)
- **Video encoding overhead:** ~5-10 seconds per video generation
- **Note:** Videos are saved only every N iterations, causing performance spikes

### WandB Logging
- Minimal performance impact (<5%)
- Requires `WANDB_API_KEY` environment variable
- Use `wandb offline` mode for maximum speed if needed

### Hardware Notes
- ECC enabled on GPU adds ~5-10% overhead (normal for cloud GPUs)
- Increase `--num_envs` for better GPU utilization (512-2048)
- Reduce if running out of memory

---

## GPU Optimization

### Maximizing GPU Utilization

**Problem:** Low GPU utilization (~25%) means unused compute capacity.

**Solution:** Increase parallel environments to saturate GPU.

```bash
# Test different environment counts
# 512 envs: ~25% GPU, 15k-17k steps/s
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --headless"

# 1024 envs: Better GPU utilization, higher throughput
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 100 \
  --headless"

# 2048 envs: Maximum GPU saturation (if memory allows)
docker exec isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 2048 \
  --max_iterations 100 \
  --headless"
```

### Finding Optimal Configuration

**Monitor GPU utilization on host:**
```bash
# Real-time monitoring
nvtop  # or nvidia-smi -l 1

# Target: 70-95% GPU utilization for optimal performance
```

**Guidelines:**
- **< 30% GPU util:** Increase `--num_envs` (try doubling)
- **70-95% GPU util:** Optimal! ✅
- **> 95% GPU util:** May be GPU-limited, consider reducing if unstable
- **OOM (Out of Memory):** Reduce `--num_envs` or task complexity

### Performance Scaling (L4 GPU)

| Environments | GPU Util | Memory | Steps/s | Time (100 iter) | Best For |
|--------------|----------|--------|---------|-----------------|----------|
| 512 | ~25% | 4 GB | 15k-17k | ~4 min | Quick tests |
| 1024 | ~50-60% | 8 GB | 25k-35k | ~2-3 min | **Recommended** |
| 2048 | ~80-90% | 15 GB | 40k-50k | ~1.5-2 min | Maximum speed |

**Note:** Isaac Sim can be CPU-bound for physics simulation, so GPU utilization < 100% is normal.

### Why Low GPU Utilization Happens
1. **Physics simulation on CPU** - Isaac Sim physics runs on CPU
2. **Data transfer overhead** - Moving data between CPU/GPU
3. **Small batch size** - Not enough parallel work for GPU
4. **Memory bandwidth** - Limited by data movement speed

**Best practice:** Start with 1024 environments, monitor GPU util, adjust up/down as needed.
