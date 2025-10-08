# Push Docker Image to Google Artifact Registry

## Quick Commands

```bash
# 1. Authenticate (if needed)
gcloud auth configure-docker us-docker.pkg.dev

# 2. Tag your image
docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam

docker tag isaaclab-tiledcam:latest \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# 3. Push
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:2.2.0-triple-cam
docker push us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# 4. Verify
gcloud artifacts docker images list \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen
```

---

## Pull and Use

```bash
# Pull from registry
docker pull us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest

# Run container
docker run -d --gpus all --name isaaclab-test \
  --shm-size 16g -v $(pwd):$(pwd) --entrypoint /bin/bash \
  us-docker.pkg.dev/engineering-380817/bdai/isaaclab-data-gen:latest \
  -lc "sleep infinity"
```

---

## Notes

- Image size: ~18GB (push takes 10-20 min on first upload)
- Includes: Isaac Lab 2.2.0, Ray 2.31.0, TorchRL 0.8.1, triple camera support
- Registry: `us-docker.pkg.dev/engineering-380817/bdai/`
