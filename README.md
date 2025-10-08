# Isaac Data Gen

Train RL policies and collect VPL datasets with dual-camera support (overview + wrist-mounted) using Isaac Lab.

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

### 3. Train Policy (Optional)
```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --headless"
```

Copy checkpoint:
```bash
docker cp model_trained.pt isaaclab-test:/workspace/model_trained.pt
```

### 4. Collect Dataset
```bash
./run_collection.sh
```

Or manually:
```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p /workspace/collect_tiled_with_checkpoint.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 20 \
  --env_spacing 5.0 \
  --steps 60 \
  --num_episodes 2 \
  --checkpoint /workspace/model_trained.pt \
  --enable_wrist_camera \
  --headless"
```

---

## Features

- **Dual Camera Support**: Overview camera + wrist-mounted camera following end-effector
- **MP4-Only Storage**: No intermediate PNGs (memory efficient)
- **Unified HDF5 Format**: Single `color` dataset with shape `(T, N_cams, H, W, 3)`
- **Per-Episode Resets**: Different initial conditions via `env.reset()`
- **Vectorized Collection**: Parallel environments for faster data generation

---

## Dataset Structure

```
vpl_tiled/
└── Isaac-Open-Drawer-Franka-v0_sim_franka_YYYYMMDD_HHMMSS/
    ├── episode_000/
    │   ├── camera_0/episode_000.mp4  # Overview camera
    │   ├── camera_1/episode_000.mp4  # Wrist camera (if enabled)
    │   └── episode_000.h5            # All data: actions, proprio, color (T,2,H,W,3)
    ├── episode_001/
    └── metadata.json
```

**HDF5 Contents:**
- `color`: `(T, N_cams, H, W, 3)` - RGB from all cameras (N_cams=2 with wrist)
- `action`: `(T, action_dim)` - Actions
- `proprio`: `(T, obs_dim)` - Proprioceptive observations
- `intrinsics`: `(N_cams, 3, 3)` - Camera intrinsics
- `extrinsics`: `(T, N_cams, 4, 4)` - Camera extrinsics (if available)
- `joint_pos`, `joint_vel`, `ee_position`, `ee_rotation` - Robot states

---

## Documentation

- **FULL_GUIDE.md** - Complete command reference with minimal explanations
- **WHAT_IT_DOES.md** - Detailed explanations of how everything works

---

## Files

- `collect_tiled_with_checkpoint.py` - Policy-driven data collector with dual cameras
- `vpl_saver.py` - VPL dataset writer (MP4 + HDF5)
- `run_collection.sh` - Quick runner script
- `isaaclab-tiledcam-starter/Dockerfile1.isaaclab-ray` - Docker image definition

---

## Customization

```bash
# More episodes
--num_envs 20 --num_episodes 10  # 200 total episodes

# Higher resolution
--width 640 --height 480

# Frame sampling (reduce IO)
--video_interval 2  # Save every 2nd frame, playback FPS auto-adjusts

# Adjust wrist camera position
--wrist_cam_offset -0.05 0.0 0.08       # Behind and above gripper
--wrist_cam_look_offset 0.25 0.0 0.0    # Look forward along gripper axis
```

---

## Requirements

- NVIDIA GPU with recent drivers
- Docker with NVIDIA runtime
- Access to `nvcr.io/nvidia/isaac-lab:2.2.0` base image
