#!/usr/bin/env python3
# flake8: noqa
"""
Data collection script optimized for camera-enabled environments.
Works with Isaac-Open-Drawer-Franka-Camera-v0 which has cameras pre-configured.
"""

import argparse
import os
from isaaclab.app import AppLauncher

# ----------------- CLI -----------------
parser = argparse.ArgumentParser("Collect VPL dataset with camera-enabled environment.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0",
                    help="Task name. Use Isaac-Open-Drawer-Franka-Camera-v0 for camera-enabled env")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--env_spacing", type=float, default=5.0)
parser.add_argument("--steps", type=int, default=60, help="Steps per episode")
parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to collect per environment")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt or .pth)")

# dataset/video
parser.add_argument("--data_root", type=str, default="/workspace/datasets/vpl_tiled")
parser.add_argument("--robot_name", type=str, default="franka")
parser.add_argument("--sim_or_real", type=str, default="sim", choices=["sim", "real"])
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--video_interval", type=int, default=1, help="Interval between video recordings (in steps).")

# Camera flags (cameras are pre-configured in Camera-enabled environments)
parser.add_argument("--disable_wrist_camera", action="store_true", help="Disable wrist camera even if env supports it")
parser.add_argument("--disable_top_camera", action="store_true", help="Disable top camera even if env supports it")

# Let Kit parse --headless / --renderer
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# Launch the app first
app = AppLauncher(args).app

# ------------ Now Isaac Lab imports -------------
import time
import datetime as dt
import numpy as np
import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

# RSL-RL helper + runner
from rsl_rl.modules import ActorCritic

# Your dataset helper
from vpl_saver import VPLSaver


def _make_out_dir(base_root: str, task: str, sim_or_real: str, robot_name: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(base_root, f"{task}_{sim_or_real}_{robot_name}_{stamp}"))
    os.makedirs(path, exist_ok=True)
    return path


def create_env(args):
    """Create environment - cameras are already configured in Camera-enabled tasks."""
    print(f"[INFO] Creating environment: {args.task}")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.scene.env_spacing = args.env_spacing
    
    # Check if this is a camera-enabled environment
    is_camera_env = "Camera" in args.task
    
    if is_camera_env:
        print("[INFO] Using camera-enabled environment (cameras pre-configured)")
        print("[INFO] Wrist camera: attached to panda_hand (auto-follows gripper)")
        print("[INFO] Top camera: positioned at (0.8, 0.0, 3.5) looking down")
        print("[INFO] Debug visualization: disabled")
    else:
        print("[WARNING] Not using a camera-enabled environment!")
        print("[WARNING] Use --task Isaac-Open-Drawer-Franka-Camera-v0 for best results")
    
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device
    obs, _ = env.reset()
    return env, device, obs, is_camera_env


def verify_cameras(scene, is_camera_env):
    """Verify that cameras are available in the scene."""
    cameras_found = {}
    
    # Check for wrist camera
    if hasattr(scene, 'wrist_camera'):
        cameras_found['wrist'] = scene.wrist_camera
        print("[INFO] ✓ Wrist camera found at scene.wrist_camera")
    elif hasattr(scene, 'sensors'):
        if isinstance(scene.sensors, dict) and 'wrist_camera' in scene.sensors:
            cameras_found['wrist'] = scene.sensors['wrist_camera']
            print("[INFO] ✓ Wrist camera found at scene.sensors['wrist_camera']")
    
    # Check for top camera
    if hasattr(scene, 'top_camera'):
        cameras_found['top'] = scene.top_camera
        print("[INFO] ✓ Top camera found at scene.top_camera")
    elif hasattr(scene, 'sensors'):
        if isinstance(scene.sensors, dict) and 'top_camera' in scene.sensors:
            cameras_found['top'] = scene.sensors['top_camera']
            print("[INFO] ✓ Top camera found at scene.sensors['top_camera']")
    
    # Check for side camera
    if hasattr(scene, 'side_camera'):
        cameras_found['side'] = scene.side_camera
        print("[INFO] ✓ Side camera found at scene.side_camera")
    elif hasattr(scene, 'sensors'):
        if isinstance(scene.sensors, dict) and 'side_camera' in scene.sensors:
            cameras_found['side'] = scene.sensors['side_camera']
            print("[INFO] ✓ Side camera found at scene.sensors['side_camera']")
    
    # Check if wrist camera is attached to robot hand
    if 'wrist' in cameras_found:
        wrist_cam = cameras_found['wrist']
        prim_path = wrist_cam.cfg.prim_path
        if 'panda_hand' in prim_path:
            print("[INFO] ✓ Wrist camera is attached to robot hand (will auto-follow)")
            cameras_found['wrist_auto_follow'] = True
        else:
            print("[INFO] ℹ Wrist camera requires manual positioning")
            cameras_found['wrist_auto_follow'] = False
    
    return cameras_found


def compute_dimensions(obs, env):
    obs_dict = obs if isinstance(obs, dict) else {"policy": obs}
    if isinstance(obs_dict, dict):
        obs_flat = torch.cat([v.view(v.shape[0], -1) for v in obs_dict.values()], dim=-1)
        num_obs = obs_flat.shape[-1]
    else:
        num_obs = obs.shape[-1]
    num_actions = env.action_space.shape[-1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
    num_envs = env.unwrapped.num_envs
    return num_obs, num_actions, num_envs


def load_policy_from_checkpoint(checkpoint_path, num_obs, num_actions, device):
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint
    policy = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    ).to(device)
    if "_std" in model_state and "std" not in model_state:
        model_state["std"] = model_state.pop("_std")
    policy.load_state_dict(model_state)
    policy.eval()
    print(f"[INFO] Policy loaded successfully")
    return policy


def make_saver(args, cameras_found):
    base_dir = _make_out_dir(args.data_root, args.task, args.sim_or_real, args.robot_name)
    effective_fps = max(1, int(round(args.fps / max(1, args.video_interval))))
    
    # Determine which cameras to enable
    enable_wrist = 'wrist' in cameras_found and not args.disable_wrist_camera
    enable_top = 'top' in cameras_found and not args.disable_top_camera
    enable_side = 'side' in cameras_found and not getattr(args, 'disable_side_camera', False)
    wrist_auto_follow = cameras_found.get('wrist_auto_follow', False)
    
    saver = VPLSaver(
        base_dir=base_dir,
        keep_terminated=False,
        fps=effective_fps,
        initial_timesteps_to_discard=0,
        enable_wrist_camera=enable_wrist,
        enable_top_camera=enable_top,
        enable_side_camera=enable_side,
        wrist_auto_follow=wrist_auto_follow,  # New parameter
        # Offsets not used when cameras are pre-configured
        wrist_cam_offset=[0.06, 0.0, 0.0],
        wrist_cam_look_offset=[0.25, 0.0, 0.0],
        top_cam_offset=[0.8, 0.0, 3.5],
        top_tgt_offset=[0.4, 0.0, 0.5],
    )
    print(f"[INFO] Writing VPL dataset to: {base_dir}")
    print(f"[INFO] Video FPS (effective): {effective_fps}")
    print(f"[INFO] Cameras enabled: wrist={enable_wrist}, top={enable_top}, side={enable_side}")
    if enable_wrist and wrist_auto_follow:
        print(f"[INFO] Wrist camera mode: auto-follow (attached to robot)")
    elif enable_wrist:
        print(f"[INFO] Wrist camera mode: manual positioning")
    return saver, base_dir


def run_collection(env, policy, saver, args, num_envs):
    print(f"\n[INFO] Collection Setup:")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Rollout rounds: {args.num_episodes}")
    print(f"  Steps per episode: {args.steps}")
    print(f"  Video interval (steps): {args.video_interval}")
    print(f"  TOTAL EPISODES: {args.num_episodes * args.num_envs}")
    print(f"\nNote: Each parallel env is a separate episode with different initial conditions")
    
    for round_idx in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Rollout Round {round_idx + 1}/{args.num_episodes}")
        print(f"Collecting {args.num_envs} episodes in parallel...")
        print(f"{'='*60}")
        obs, _ = env.reset()
        episode_terminated = np.zeros(num_envs, dtype=bool)
        episode_truncated = np.zeros(num_envs, dtype=bool)
        for t in range(args.steps):
            if t % 20 == 0:
                print(f"  Step {t}/{args.steps}...")
            with torch.inference_mode():
                if isinstance(obs, dict):
                    obs_tensor = torch.cat([v.view(v.shape[0], -1) for v in obs.values()], dim=-1)
                else:
                    obs_tensor = obs
                act = policy.act_inference(obs_tensor)
            obs, rewards, terminated, truncated, info = env.step(act)
            episode_terminated = episode_terminated | terminated.cpu().numpy()
            episode_truncated = episode_truncated | truncated.cpu().numpy()
            store_frame = (args.video_interval <= 1) or (t % args.video_interval == 0)
            saver.store(actions=act, env=env, store_frame=store_frame)
        episode_success = np.ones(num_envs, dtype=bool)
        dones = np.ones(num_envs, dtype=bool)
        saver.write(
            dones=dones,
            terminated=episode_terminated,
            successes=episode_success,
            point_cloud_only=False,
            save_to_video=True
        )
        print(f"\n  ✓ Round {round_idx + 1} complete: saved {num_envs} episodes")


def main():
    # --------- Build env and verify cameras ----------
    env, device, obs, is_camera_env = create_env(args)
    scene = env.unwrapped.scene
    cameras_found = verify_cameras(scene, is_camera_env)
    
    if not cameras_found:
        print("\n[ERROR] No cameras found in environment!")
        print("[ERROR] Make sure you're using Isaac-Open-Drawer-Franka-Camera-v0")
        print("[ERROR] and that the custom environment is installed (run install_camera_env.sh)")
        env.close()
        return

    # --------- Dimensions & info ----------
    num_obs, num_actions, num_envs = compute_dimensions(obs, env)
    print(f"\n[INFO] Environment: {args.task}")
    print(f"[INFO] Num envs: {args.num_envs}, spacing: {args.env_spacing}m")
    print(f"[INFO] Observation dimension: {num_obs}")
    print(f"[INFO] Action dimension: {num_actions}")

    # --------- Policy ----------
    policy = load_policy_from_checkpoint(args.checkpoint, num_obs, num_actions, device)

    # --------- Saver ----------
    saver, base_dir = make_saver(args, cameras_found)

    # --------- Rollout ----------
    run_collection(env, policy, saver, args, num_envs)

    # --------- Final Summary ----------
    total_episodes = args.num_episodes * num_envs
    num_cams = sum([
        1,  # Always have tiled camera
        'wrist' in cameras_found and not args.disable_wrist_camera,
        'top' in cameras_found and not args.disable_top_camera
    ])
    print(f"\n{'='*60}")
    print(f"[OK] Data Collection Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {base_dir}")
    print(f"")
    print(f"TOTAL EPISODES COLLECTED: {total_episodes}")
    print(f"  Rollout rounds: {args.num_episodes}")
    print(f"  Parallel envs per round: {num_envs}")
    print(f"  Steps per episode: {args.steps}")
    print(f"  Cameras per episode: {num_cams}")
    print(f"")
    env.close()


if __name__ == "__main__":
    main()
    app.close()

