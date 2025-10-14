#!/usr/bin/env python3
"""Simple data collection with cameras."""
import argparse
import os
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--num_episodes", type=int, default=40)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--output", type=str, default="/workspace/datasets/vpl_data")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.modules import ActorCritic
from vpl_saver import VPLSaver
import sys

# Create env
print(f"Creating environment: {args.task}")
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.scene.env_spacing = 6.0

# Make ActionManager identity: no normalization/scale/offset/clip/rate-limit
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
obs, _ = env.reset()

# Check cameras
scene = env.unwrapped.scene
has_wrist = False
has_top = False
has_side = False

if hasattr(scene, 'sensors') and isinstance(scene.sensors, dict):
    has_wrist = 'wrist_camera' in scene.sensors
    has_top = 'top_camera' in scene.sensors
    has_side = 'side_camera' in scene.sensors
else:
    has_wrist = hasattr(scene, 'wrist_camera')
    has_top = hasattr(scene, 'top_camera')
    has_side = hasattr(scene, 'side_camera')

print(f"Cameras found: wrist={has_wrist}, top={has_top}, side={has_side}")

# Load policy
print(f"Loading policy from: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=args.device)
if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
else:
    state_dict = checkpoint

num_obs = obs.shape[-1] if not isinstance(obs, dict) else sum(v.numel() // v.shape[0] for v in obs.values())
num_actions = env.action_space.shape[-1]

policy = ActorCritic(num_obs, num_obs, num_actions, [256, 128, 64], [256, 128, 64]).to(args.device)
if "_std" in state_dict:
    state_dict["std"] = state_dict.pop("_std")
policy.load_state_dict(state_dict)
policy.eval()
print("Policy loaded")

# Create saver
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)
saver = VPLSaver(
    base_dir=str(output_dir),
    fps=30,
    keep_terminated=False,
    enable_wrist_camera=has_wrist,
    enable_top_camera=has_top,
    enable_side_camera=has_side,
    wrist_auto_follow=True,
)
print(f"Saving to: {output_dir}")

# Debug log file (inside container path expected by collect.sh)
debug_log_path = "/workspace/collect_debug.log"
try:
    debug_f = open(debug_log_path, "w", buffering=1)
except Exception:
    debug_f = None

# Collect data
print(f"\nCollecting {args.num_episodes} rounds × {args.num_envs} envs = {args.num_episodes * args.num_envs} episodes")
for round_idx in range(args.num_episodes):
    print(f"\nRound {round_idx+1}/{args.num_episodes}")
    obs, _ = env.reset()
    
    for step in range(args.steps):
        # Get action from policy
        with torch.inference_mode():
            obs_tensor = torch.cat([v.reshape(v.shape[0], -1) for v in obs.values()], dim=-1) if isinstance(obs, dict) else obs
            actions = policy.act_inference(obs_tensor)
        
        # Clip actions to env.action_space bounds (if provided)
        try:
            low = torch.tensor(env.action_space.low, device=actions.device, dtype=actions.dtype)
            high = torch.tensor(env.action_space.high, device=actions.device, dtype=actions.dtype)
            actions_clipped = torch.max(torch.min(actions, high), low)
        except Exception:
            actions_clipped = actions

        # Clone BEFORE stepping (env.step might modify)
        actions_to_save = actions_clipped.clone()
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions_clipped)
        
        # Store data AFTER stepping (to capture resulting state)
        saver.store(actions=actions_to_save, env=env, store_frame=True)
        
        if step < 10 or (step % 10 == 0):
            a0 = actions_to_save[0].detach().cpu().numpy()
            # Try to fetch effective applied joint targets from robot articulation
            applied = None
            try:
                robot = env.unwrapped.scene._articulations.get("robot")
                rd = getattr(robot, 'data', None)
                if rd is not None:
                    for key in [
                        'joint_pos_target', 'dof_pos_target', 'joint_targets',
                        'drive_target', 'actuated_dof_pos_target'
                    ]:
                        if hasattr(rd, key):
                            val = getattr(rd, key)
                            applied = val[0].detach().cpu().numpy() if hasattr(val, 'detach') else np.array(val[0])
                            break
                    if applied is None and hasattr(rd, 'joint_pos'):
                        # Fallback: current joint positions (post-step)
                        jp = rd.joint_pos[0].detach().cpu().numpy() if hasattr(rd.joint_pos, 'detach') else np.array(rd.joint_pos[0])
                        applied = jp
            except Exception:
                applied = None

            applied_str = np.round(applied, 3).tolist() if applied is not None else None
            line = f"step={step} action0={np.round(a0, 3).tolist()} applied0={applied_str} reward0={float(rewards[0])}\n"
            print(line.strip())
            if debug_f:
                try:
                    debug_f.write(line)
                except Exception:
                    pass
    
    # Write episodes
    dones = np.ones(args.num_envs, dtype=bool)
    terminated = np.zeros(args.num_envs, dtype=bool)
    successes = np.ones(args.num_envs, dtype=bool)
    saver.write(dones=dones, terminated=terminated, successes=successes, save_to_video=True)
    print(f"  Saved {args.num_envs} episodes")

print(f"\nDone! Collected {args.num_episodes * args.num_envs} episodes")
env.close()
app.close()
if debug_f:
    debug_f.close()
