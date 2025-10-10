# Isaac Data Gen - Full Command Guide

Complete terminal commands for building, training, and collecting datasets.

---

## Build Docker Image

```bash
docker build -t isaaclab:2.2-ray \
  -f isaaclab-tiledcam-starter/Dockerfile1.isaaclab-ray .
```

---

## Push Docker Image

```bash
# 1. Authenticate (if needed)
gcloud auth configure-docker us-docker.pkg.dev

# 2. Tag your image
docker tag isaaclab:2.2-ray \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# 3. Push
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# 4. Verify
gcloud artifacts docker images list \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen
```

**Notes:**
- Image size: ~18GB (push takes 10-20 min on first upload)
- Includes: Isaac Lab 2.2.0, Ray 2.31.0, TorchRL 0.8.1, triple camera support
- Registry: `us-docker.pkg.dev/engineering-380817/bdai/`

---

## Run Container

```bash
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g \
  -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  isaaclab:2.2-ray -lc "sleep infinity"
```

Verify:
```bash
docker exec -it isaaclab-test python -c "import sys; print(sys.version)"
```

---

## Train RL Policy

```bash
docker exec -it isaaclab-test bash -c "cd /workspace/isaaclab && \
ENABLE_CAMERAS=0 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 \
  --num_envs 1024 \
  --max_iterations 3000 \
  --logger wandb \
  --headless"
```

Copy checkpoint:
```bash
docker cp model_trained.pt isaaclab-test:/workspace/model_trained.pt
```

---

## Prepare Data Collection

Install MP4 writer:
```bash
docker exec -it isaaclab-test bash -c "/isaac-sim/python.sh -m pip install 'imageio[ffmpeg]'"
```

Copy scripts:
```bash
docker cp collect_tiled_with_checkpoint.py isaaclab-test:/workspace/
docker cp vpl_saver.py isaaclab-test:/workspace/
```

---

## Collect Dataset (Quick Method)

```bash
./run_collection.sh
```

What it does:
- Cleans old data in container
- Copies scripts
- Runs collection with 2 envs, 2 episodes, 60 steps, dual cameras
- Copies results back to host

---

## Collect Dataset (Manual - Full Control)

Single camera (overview only):
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

Dual camera (overview + wrist):
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
  --headless"
```

Triple camera (overview + wrist + top):
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

## Customization Examples

### More Episodes
```bash
--num_envs 20 --num_episodes 5   # 100 total episodes
--num_envs 20 --num_episodes 10  # 200 total episodes
```

### Longer Episodes
```bash
--steps 100
```

### Higher Resolution
```bash
--width 640 --height 480
```

### Frame Sampling (reduce IO)
```bash
--video_interval 2  # Store every 2nd frame, playback FPS auto-adjusts
```

### Adjust Wrist Camera View
```bash
# Position: [X, Y, Z] in EE frame (X=forward, Y=side, Z=up)
--wrist_cam_offset -0.10 0.0 0.10     # Further back and higher
--wrist_cam_look_offset 0.30 0.05 0.0 # Look forward and slightly right
```

---

## View Results

```bash
# View a sample video (overview)
vlc vpl_tiled/*/episode_000/camera_0/episode_000.mp4

# View wrist camera
vlc vpl_tiled/*/episode_000/camera_1/episode_000.mp4

# View top camera
vlc vpl_tiled/*/episode_000/camera_2/episode_000.mp4

# View all 3 cameras together
vlc vpl_tiled/*/episode_000/camera_*/episode_000.mp4

# Inspect HDF5
python3 -c "import h5py; f = h5py.File('vpl_tiled/*/episode_000/episode_000.h5'); print(list(f.keys())); print('color shape:', f['color'].shape)"
```

---

## Dataset Format

**Directory Structure:**
```
vpl_tiled/
└── Isaac-Open-Drawer-Franka-v0_sim_franka_YYYYMMDD_HHMMSS/
    ├── episode_000/
    │   ├── camera_0/episode_000.mp4  # Overview camera
    │   ├── camera_1/episode_000.mp4  # Wrist camera (if enabled)
    │   ├── camera_2/episode_000.mp4  # Top camera (if enabled)
    │   └── episode_000.h5
    └── metadata.json
```

**HDF5 Datasets:**
- `color`: `(T, N_cams, H, W, 3)` - RGB frames (N_cams=1-3 depending on flags)
- `action`: `(T, action_dim)` - Actions
- `proprio`: `(T, obs_dim)` - Proprioception
- `intrinsics`: `(N_cams, 3, 3)` - Camera intrinsics
- `extrinsics`: `(T, N_cams, 4, 4)` - Camera extrinsics
- `joint_pos`: `(T, n_joints)` - Joint positions
- `joint_vel`: `(T, n_joints)` - Joint velocities
- `ee_position`: `(T, 3)` - End-effector position
- `ee_rotation`: `(T, 4)` - End-effector quaternion (x,y,z,w)
- `timestamps`: `(T,)` - Timestep indices

**HDF5 Attributes:**
- `env_idx`, `success`, `terminated`, `num_frames`, `fps`

---

## Troubleshooting

### GPU OOM
```bash
--num_envs 12 --width 256 --height 192
```

### No Videos Created
Check environment variables are set:
```bash
ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1
```

### Restart Container
```bash
docker restart isaaclab-test
```

### Check Container Logs
```bash
docker logs isaaclab-test --tail 100
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

## Documentation

- **FULL_GUIDE.md** - All terminal commands organized by workflow step
- **WHAT_IT_DOES.md** - Detailed explanations of implementation and design decisions
- **TOP_CAMERA_SETUP.md** - Third camera setup guide and configuration

---

## Key Files

- `collect_tiled_with_checkpoint.py` - Main data collector (369 lines)
- `vpl_saver.py` - Dataset writer with multi-camera support (452 lines)
- `run_collection.sh` - Quick runner script (62 lines)
- `isaaclab-tiledcam-starter/Dockerfile1.isaaclab-ray` - Docker image

---

## Notes

- Each parallel environment = separate episode with randomized initial state
- Total episodes = `num_envs × num_episodes`
- **Wrist camera**: Positioned behind/above gripper to avoid clipping with robot body
- **Top camera**: Static bird's-eye view from above the environment
- Camera offsets are in end-effector frame (wrist) or env origin (top)
  - EE frame: X=forward (fingers), Y=side, Z=up
  - Env frame: X=forward, Y=side, Z=up
