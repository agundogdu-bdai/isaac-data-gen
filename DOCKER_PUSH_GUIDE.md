# Push Docker Image to Google Artifact Registry

Guide for pushing the Isaac Lab image to `us-docker.pkg.dev/engineering-380817/bdai/`

---

## Prerequisites

Authenticate with Google Artifact Registry (if not already done):

```bash
gcloud auth configure-docker us-docker.pkg.dev
```

---

## Step 1: Tag Your Local Image

Choose a descriptive tag for your image:

```bash
# Option A: Tag with version (recommended)
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

# Option B: Tag as latest
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# Option C: Both (recommended for production)
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest
```

**Suggested naming**: `isaaclab-data-gen` (matches your GitHub repo name)

---

## Step 2: Push to Artifact Registry

```bash
# Push versioned tag
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

# Push latest tag (optional)
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest
```

This will take ~10-20 minutes depending on your network (image is ~18GB).

---

## Step 3: Verify Push

```bash
# List images in the registry
gcloud artifacts docker images list us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen

# Or check in web console
# https://console.cloud.google.com/artifacts/docker/engineering-380817/us/bdai
```

---

## Alternative: Use Existing Image Name

If you want to keep the same naming convention as your existing images:

```bash
# Tag as isaaclab-ray with new version
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-ray:2.2.0-triple-cam

# Push
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-ray:2.2.0-triple-cam
```

---

## Image Features

This image includes:

- **Base**: Isaac Lab 2.2.0 + Isaac Sim
- **Ray**: 2.31.0 for distributed training
- **TorchRL**: 0.8.1
- **Triple Camera Support**: Overview + Wrist + Top cameras
- **VPL Dataset Collection**: `collect_tiled_with_checkpoint.py` and `vpl_saver.py`
- **Custom Kit**: `isaaclab.python.headless.rendering.tiled.kit`

---

## Usage After Push

Pull and run on any machine:

```bash
# Pull from registry
docker pull us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

# Run container
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g \
  -v $(pwd):$(pwd) \
  --entrypoint /bin/bash \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam \
  -lc "sleep infinity"

# Copy scripts and collect data
docker cp collect_tiled_with_checkpoint.py isaaclab-test:/workspace/
docker cp vpl_saver.py isaaclab-test:/workspace/
# ... continue with data collection
```

---

## Quick Commands Summary

```bash
# 1. Authenticate (if needed)
gcloud auth configure-docker us-docker.pkg.dev

# 2. Tag your image
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

# 3. Push to registry
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

# 4. Verify
gcloud artifacts docker images list \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen
```

---

## Troubleshooting

### Authentication Error

```bash
# Re-authenticate
gcloud auth login
gcloud auth configure-docker us-docker.pkg.dev
```

### Permission Denied

```bash
# Check your GCP project permissions
gcloud projects get-iam-policy engineering-380817

# You need "Artifact Registry Writer" role
```

### Image Not Found Locally

```bash
# Rebuild if needed
docker build -t isaaclab-tiledcam:latest \
  -f isaaclab-tiledcam-starter/Dockerfile.isaaclab-ray .
```

---

## Version Tagging Strategy

Recommended format: `<isaac-lab-version>-<feature-name>`

Examples:
- `2.2.0-triple-cam` - Current version with triple camera
- `2.2.0-ray2.31.0` - Isaac Lab 2.2.0 + Ray 2.31.0
- `latest` - Always points to most recent stable build

---

## Notes

- Push time: ~10-20 minutes for 18GB image
- Storage cost: ~$0.10/GB/month in GCP Artifact Registry
- Image includes all dependencies pre-installed
- No need to rebuild on remote machines, just pull and run

