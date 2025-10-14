# Visuomotor DiffPo Policy Evaluation Setup

Complete instructions to build Docker image with visuomotor and run your DiffPo policy.

## Step 1: Build Docker Image with Visuomotor

```bash
cd /home/agundogdu_theaiinstitute_com/test

# Build the image (takes ~10-15 minutes)
docker build \
  -f Dockerfile.isaaclab-visuomotor \
  -t isaaclab-visuomotor:latest \
  --build-arg VISUOMOTOR_TAG=0.9.7-all \
  .
```

## Step 2: Stop and Remove Old Container

```bash
# Stop existing container
docker stop isaaclab-test || true

# Remove it
docker rm isaaclab-test || true
```

## Step 3: Start New Container with Visuomotor

```bash
docker run -d \
  --name isaaclab-test \
  --gpus all \
  --network host \
  -v /home/agundogdu_theaiinstitute_com/test:/workspace/host \
  isaaclab-visuomotor:latest \
  tail -f /dev/null
```

## Step 4: Verify Visuomotor Installation

```bash
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -c 'import visuomotor; print(\"visuomotor version:\", visuomotor.__version__)'"
```

Expected output:
```
visuomotor version: 0.9.7
```

## Step 5: Run Policy Evaluation

```bash
export WANDB_API_KEY="8c04cd703eb0fea969e4eb4f38af3f05897851f8"

/home/agundogdu_theaiinstitute_com/test/run_eval_diffpo.sh \
  bdaii/Isaac-Open-Drawer-Franka-v0_sim_franka_20251014-agundogdu/diffpo-bqc7v7jo-kb4j741o:v1 \
  /home/agundogdu_theaiinstitute_com/test/videos \
  --num_envs 1 \
  --cams top,wrist,side \
  --stream \
  --camera_h 160 \
  --camera_w 160 \
  --headless \
  --loader_path /home/agundogdu_theaiinstitute_com/test/visuomotor_loader.py \
  --policy_entry "" \
  --max_steps 300
```

## Step 6: Check Results

```bash
# List generated videos
ls -lh /home/agundogdu_theaiinstitute_com/test/videos/

# Play a video (if you have a display)
mpv /home/agundogdu_theaiinstitute_com/test/videos/eval_diffpo_top.mp4
```

## Troubleshooting

### If visuomotor import fails:
```bash
# Check if visuomotor is installed
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -m pip list | grep visuomotor"

# Reinstall if needed
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -m pip install --no-cache-dir 'git+https://github.com/bdaiinstitute/visuomotor.git@0.9.7-all'"
```

### If policy loading fails:
```bash
# Check the checkpoint structure inside container
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -c \"
import torch
ckpt = torch.load('/workspace/isaaclab/artifacts/diffpo-bqc7v7jo-kb4j741o:v1/model.ckpt', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
if 'hyper_parameters' in ckpt:
    print('Hyper-parameters:', list(ckpt['hyper_parameters'].keys())[:10])
\""
```

### Memory issues:
```bash
# Reduce resolution and episode length
/home/agundogdu_theaiinstitute_com/test/run_eval_diffpo.sh \
  bdaii/Isaac-Open-Drawer-Franka-v0_sim_franka_20251014-agundogdu/diffpo-bqc7v7jo-kb4j741o:v1 \
  /home/agundogdu_theaiinstitute_com/test/videos \
  --num_envs 1 \
  --cams top \
  --stream \
  --camera_h 120 \
  --camera_w 120 \
  --headless \
  --loader_path /home/agundogdu_theaiinstitute_com/test/visuomotor_loader.py \
  --policy_entry "" \
  --max_steps 100
```

## Quick Test (50 steps)

```bash
export WANDB_API_KEY="8c04cd703eb0fea969e4eb4f38af3f05897851f8"

/home/agundogdu_theaiinstitute_com/test/run_eval_diffpo.sh \
  bdaii/Isaac-Open-Drawer-Franka-v0_sim_franka_20251014-agundogdu/diffpo-bqc7v7jo-kb4j741o:v1 \
  /home/agundogdu_theaiinstitute_com/test/videos \
  --num_envs 1 \
  --cams top,wrist,side \
  --stream \
  --camera_h 160 \
  --camera_w 160 \
  --headless \
  --loader_path /home/agundogdu_theaiinstitute_com/test/visuomotor_loader.py \
  --policy_entry "" \
  --max_steps 50
```

## Files Created

- `Dockerfile.isaaclab-visuomotor` - Docker image with visuomotor installed
- `visuomotor_loader.py` - Loader using visuomotor registry
- `eval_diffpo.py` - Evaluation script (already created)
- `run_eval_diffpo.sh` - Runner script (already created)
- `VISUOMOTOR_SETUP.md` - This file

## Summary

1. Build Docker image with visuomotor
2. Start new container
3. Run evaluation with visuomotor_loader.py
4. Videos saved to `/home/agundogdu_theaiinstitute_com/test/videos/`

