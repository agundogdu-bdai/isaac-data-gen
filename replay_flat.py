#!/usr/bin/env python3
# flake8: noqa
# -*- coding: utf-8 -*-
"""
Replay for flat HDF5 (minimal actions-only):
- Build env for --task, reset once, no scene restore
- Replay recorded actions step-by-step and capture camera frames
"""
import argparse
import io
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, required=True)  # dir or .h5
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--camera", type=str, default="top", choices=["top", "wrist", "side"])
parser.add_argument("--output", type=str, default="/workspace/replay_flat.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

# Import after Kit is up (these may transitively import omni.*)
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore


# Resolve H5
ep_path = Path(args.episode)
if ep_path.is_file() and ep_path.suffix == ".h5":
    h5_path = ep_path
else:
    h5s = list(ep_path.glob("*.h5"))
    assert h5s, f"No .h5 found in {ep_path}"
    h5_path = h5s[0]
print(f"[replay_flat] Loading: {h5_path}")

with h5py.File(h5_path, "r") as f:
    if "action" not in f:
        raise RuntimeError("Action dataset not found in H5; actions-only replay requires it.")
    cmd = f["action"][()]
    physics_dt = float(f.attrs["physics_dt"]) if "physics_dt" in f.attrs else 0.01
    step_dt = float(f.attrs.get("step_dt", physics_dt))
    decimation = max(1, int(round(step_dt / physics_dt))) if physics_dt > 0 else 1
    H = int(f.attrs.get("camera_h", 240))
    W = int(f.attrs.get("camera_w", 320))

print(
    f"[replay_flat] T={len(cmd)}, physics_dt={physics_dt}, step_dt={step_dt}, decimation={decimation}"
)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.scene.env_spacing = 6.0
env = gym.make(args.task, cfg=env_cfg)
env.reset()

scene = env.unwrapped.scene


def move_nested_state(obj, device):
    if isinstance(obj, dict):
        return {k: move_nested_state(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj



# Acquire sensors and robot AFTER potential reset_to (to avoid stale handles)
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
cam_key = f"{args.camera}_camera"
assert cam_key in sensors, f"Camera {cam_key} not found. Available: {list(sensors.keys())}"
cam = sensors[cam_key]

# Robot articulation
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)
assert robot is not None, "Robot articulation 'robot' not found."

# Replay
frames = []
for t in range(cmd.shape[0]):
    act = torch.from_numpy(cmd[t][None, ...]).to(env.device)
    _obs, _rew, _terminated, _truncated, _info = env.step(act)
    cam.update(dt=step_dt)
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if (rgb.shape[0], rgb.shape[1]) != (H, W):
        x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None] / 255.0
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        rgb = (x.clamp(0, 1) * 255.0).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    frames.append(rgb)
    if (t % 20) == 0:
        print(f"  step {t}/{cmd.shape[0]}")

fps = int(round(1.0 / step_dt)) if step_dt > 0 else 30
iio.imwrite(args.output, frames, fps=fps)
print(f"[replay_flat] Saved: {args.output}")

env.close()
app.close()


