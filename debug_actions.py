#!/usr/bin/env python3
"""Debug action processing."""
import torch
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

# Check action manager
action_mgr = env.unwrapped.action_manager
print("Action Manager:")
print(f"  Action terms: {list(action_mgr._terms.keys())}")

for name, term in action_mgr._terms.items():
    print(f"\n  {name}:")
    print(f"    Type: {type(term).__name__}")
    if hasattr(term, 'cfg'):
        print(f"    Config: {term.cfg}")
    if hasattr(term, '_scale'):
        print(f"    Scale: {term._scale}")
    if hasattr(term, '_offset'):
        print(f"    Offset: {term._offset}")

# Test action processing
test_action = torch.randn(1, env.action_space.shape[0], device=args.device)
print(f"\nTest action shape: {test_action.shape}")
print(f"Test action: {test_action}")

env.close()
app.close()

