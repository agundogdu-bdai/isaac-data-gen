# Manual Visuomotor Installation (Private Repo Workaround)

Since the visuomotor repository is private, you need to download the wheel manually first.

## Option 1: Download Wheel Manually (Recommended)

### Step 1: Download the wheel with GitHub CLI
```bash
cd /home/agundogdu_theaiinstitute_com/test

# Login to GitHub CLI if not already
gh auth login

# Download the wheel from the release
gh release download 0.9.7-all \
  --repo bdaiinstitute/visuomotor \
  --pattern "*.whl" \
  --dir .
```

### Step 2: Build Docker with local wheel
```bash
# The Dockerfile will now copy the local wheel
docker build -f Dockerfile.isaaclab-visuomotor -t isaaclab-visuomotor:latest .
```

## Option 2: Install visuomotor After Container Starts

Skip visuomotor in Docker build and install it in the running container:

### Step 1: Build without visuomotor
Comment out the visuomotor installation in Dockerfile, then:
```bash
docker build -f Dockerfile.isaaclab-visuomotor -t isaaclab-visuomotor:latest .
```

### Step 2: Start container
```bash
docker stop isaaclab-test && docker rm isaaclab-test
docker run -d --name isaaclab-test --gpus all --network host \
  -v /home/agundogdu_theaiinstitute_com/test:/workspace/host \
  isaaclab-visuomotor:latest tail -f /dev/null
```

### Step 3: Install visuomotor with GitHub token
```bash
# Get your GitHub token
GH_TOKEN=$(gh auth token)

# Install visuomotor in the container
docker exec -e GH_TOKEN="$GH_TOKEN" isaaclab-test bash -c \
  "GH_USER=\$(gh api user --jq .login 2>/dev/null || echo 'token') && \
   /isaac-sim/python.sh -m pip install --no-cache-dir \
   'git+https://\${GH_USER}:\${GH_TOKEN}@github.com/bdaiinstitute/visuomotor.git@0.9.7-all'"
```

### Step 4: Verify
```bash
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -c 'import visuomotor; print(visuomotor.__version__)'"
```

## Option 3: Use Existing Container and Install Manually

If you already have the isaaclab-test container:

```bash
# Download wheel on host
cd /home/agundogdu_theaiinstitute_com/test
gh release download 0.9.7-all \
  --repo bdaiinstitute/visuomotor \
  --pattern "*.whl"

# Copy into container
docker cp visuomotor-0.9.7-py3-none-any.whl isaaclab-test:/tmp/

# Install in container
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -m pip install --no-cache-dir /tmp/visuomotor-0.9.7-py3-none-any.whl"

# Verify
docker exec isaaclab-test bash -c \
  "/isaac-sim/python.sh -c 'import visuomotor; print(visuomotor.__version__)'"
```

## After Installation - Run Evaluation

Once visuomotor is installed (via any option above):

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

