## Quickstart

Minimal commands to run the full workflow end-to-end.

### 1) Build the image (keep other Dockerfiles as-is)
```bash
cd /home/agundogdu_theaiinstitute_com/test
docker build -f Dockerfile.isaaclab-visuomotor -t isaaclab-visuomotor:latest .
```

### 2) Start a container
```bash
# GPU recommended; expose X if using UI
docker run --gpus all -d --name isaaclab-test \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  isaaclab-visuomotor:latest sleep infinity
```

### 3) Train RL for the drawer task
```bash
# Copy your checkpoint out when done (example path below uses /workspace/model_trained.pt)
docker exec -it isaaclab-test bash -lc "cd /workspace/isaaclab && \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Open-Drawer-Franka-v0 --headless"
```

### 4) Ensure camera-enabled env (3 cameras) is installed (done by Dockerfile)
Task ID used: `Isaac-Open-Drawer-Franka-Camera-v0`.

### 5) Collect flat data
```bash
cd /home/agundogdu_theaiinstitute_com/test
./collect_flat.sh --checkpoint /workspace/model_trained.pt \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --num_envs 1 --num_episodes 3 --steps 70 --camera_h 120 --camera_w 160
```

### 6) Replay a collected episode to MP4
```bash
./replay_flat.sh /home/agundogdu_theaiinstitute_com/test/vpl_collected_flat/episode_000 \
  --task Isaac-Open-Drawer-Franka-Camera-v0 --camera top
```

### 7) Evaluate DiffPO policy artifact and save videos
```bash
# Example: replace ARTIFACT with your W&B artifact path
./run_eval_diffpo.sh ENTITY/PROJECT/ARTIFACT:VERSION /home/agundogdu_theaiinstitute_com/test/videos \
  --task Isaac-Open-Drawer-Franka-Camera-v0 --num_envs 1 --cams top,wrist,side --headless
```

Outputs:
- Collected dataset: `./vpl_collected_flat/`
- Eval videos: `./videos/`


