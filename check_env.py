#!/usr/bin/env python3
"""Check what cameras are in the environment."""
import gymnasium as gym
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
env.reset()

scene = env.unwrapped.scene
print(f"\nChecking scene for cameras...")
print(f"Scene attributes: {[x for x in dir(scene) if 'camera' in x.lower()]}")

if hasattr(scene, 'sensors'):
    print(f"Scene has sensors")
    if isinstance(scene.sensors, dict):
        print(f"Sensors (dict): {list(scene.sensors.keys())}")
        for name, sensor in scene.sensors.items():
            if 'camera' in name.lower():
                print(f"  {name}: {sensor.cfg.prim_path}")
    else:
        print(f"Sensors (object): {[x for x in dir(scene.sensors) if 'camera' in x.lower()]}")

if hasattr(scene, 'wrist_camera'):
    print(f"Found scene.wrist_camera: {scene.wrist_camera.cfg.prim_path}")
if hasattr(scene, 'top_camera'):
    print(f"Found scene.top_camera: {scene.top_camera.cfg.prim_path}")
if hasattr(scene, 'tiled_camera'):
    print(f"Found scene.tiled_camera: {scene.tiled_camera.cfg.prim_path}")

env.close()
app.close()
print("\nDone!")

