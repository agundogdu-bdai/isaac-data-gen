#!/usr/bin/env python3
"""Replay episode actions and save video."""
import argparse
import h5py
import numpy as np
import torch
import gymnasium as gym
import imageio.v3 as iio
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--output", type=str, default="/workspace/replay.mp4")
parser.add_argument("--camera", type=str, default="top", choices=["top", "wrist", "side"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

from isaaclab_tasks.utils import parse_env_cfg

# Load actions (and applied targets if present)
h5_path = list(Path(args.episode).glob("*.h5"))[0]
with h5py.File(h5_path, 'r') as f:
    actions = np.array(f['action'])
    applied_targets = np.array(f['applied_joint_target']) if 'applied_joint_target' in f else None
print(f"Loaded {len(actions)} steps from {h5_path.name} (applied_targets={'yes' if applied_targets is not None else 'no'})")

# Create env (same as collection)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.scene.env_spacing = 6.0

# Make ActionManager identity in replay too
try:
    if hasattr(env_cfg, 'actions') and hasattr(env_cfg.actions, 'arm_action'):
        env_cfg.actions.arm_action.normalize = False
        env_cfg.actions.arm_action.use_delta = False
        env_cfg.actions.arm_action.scale = 1.0
        env_cfg.actions.arm_action.offset = 0.0
        # leave clip as default (do not override)
        if hasattr(env_cfg.actions.arm_action, 'rate_limit'):
            env_cfg.actions.arm_action.rate_limit = 0.0
    if hasattr(env_cfg, 'actions') and hasattr(env_cfg.actions, 'gripper_action'):
        env_cfg.actions.gripper_action.normalize = False
        env_cfg.actions.gripper_action.scale = 1.0
        env_cfg.actions.gripper_action.offset = 0.0
        # leave clip as default (do not override)
        if hasattr(env_cfg.actions.gripper_action, 'rate_limit'):
            env_cfg.actions.gripper_action.rate_limit = 0.0
except Exception:
    pass
env = gym.make(args.task, cfg=env_cfg)
env.reset()

# Get the camera from scene
scene = env.unwrapped.scene
sensors = scene.sensors if hasattr(scene, 'sensors') and isinstance(scene.sensors, dict) else {}

# Select camera
camera_name = f"{args.camera}_camera"
if camera_name not in sensors:
    print(f"Error: {camera_name} not found. Available: {list(sensors.keys())}")
    exit(1)

cam = sensors[camera_name]
print(f"Using {camera_name} for replay")

# Replay and capture
frames = []
robot = None
try:
    robot = scene._articulations.get("robot")
except Exception:
    robot = None

for i in range(len(actions)):
    # Either apply recorded joint targets directly, or step with raw action
    did_apply = False
    if applied_targets is not None and robot is not None:
        try:
            tgt = torch.from_numpy(applied_targets[i]).to(env.device)
            if tgt.ndim == 1:
                tgt = tgt.unsqueeze(0)
            if hasattr(robot, 'set_joint_position_target'):
                robot.set_joint_position_target(tgt)
                did_apply = True
            elif hasattr(robot, 'set_dof_position_target'):
                robot.set_dof_position_target(tgt)
                did_apply = True
        except Exception:
            did_apply = False
    if did_apply:
        # Advance physics with zero action; drives should pull to targets
        zero_action = torch.zeros((1, actions.shape[1]), device=env.device)
        env.step(zero_action)
    else:
        action_tensor = torch.from_numpy(actions[i]).unsqueeze(0).to(env.device)
        env.step(action_tensor)
    
    # Capture frame (same as collection)
    cam.update(dt=scene.physics_dt)
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    
    # Process frame
    if rgb.shape[0] in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    frames.append(rgb)
    
    if i % 20 == 0:
        print(f"  Step {i}/{len(actions)}")

# Save video
iio.imwrite(args.output, frames, fps=30)
print(f"Saved replay: {args.output}")

env.close()
app.close()
