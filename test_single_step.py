#!/usr/bin/env python3
"""Test single step to compare actions."""
import torch
import gymnasium as gym
import h5py
import numpy as np
from pathlib import Path
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.modules import ActorCritic

# Load saved action from h5
h5_path = "/workspace/episode/episode_000.h5"
with h5py.File(h5_path, 'r') as f:
    saved_action = f['action'][0]  # First action
    print(f"Saved action from h5: {saved_action}")

# Create env and load policy
env_cfg = parse_env_cfg("Isaac-Open-Drawer-Franka-Camera-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-Open-Drawer-Franka-Camera-v0", cfg=env_cfg)
obs, _ = env.reset()

# Load policy and get action
checkpoint = torch.load("/workspace/model_trained.pt", map_location="cuda:0")
state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
num_obs = sum(v.numel() // v.shape[0] for v in obs.values())
num_actions = env.action_space.shape[0]

policy = ActorCritic(num_obs, num_obs, num_actions, [256, 128, 64], [256, 128, 64]).to("cuda:0")
if "_std" in state_dict:
    state_dict["std"] = state_dict.pop("_std")
policy.load_state_dict(state_dict)
policy.eval()

# Get policy action
with torch.inference_mode():
    obs_tensor = torch.cat([v.reshape(v.shape[0], -1) for v in obs.values()], dim=-1)
    policy_action = policy.act_inference(obs_tensor)
    print(f"Policy action (fresh): {policy_action[0]}")

print(f"\nDifference: {np.abs(saved_action - policy_action[0].cpu().numpy()).max()}")

env.close()
app.close()

