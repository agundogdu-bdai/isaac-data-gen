# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay saved episode data in Isaac Lab."""

import argparse
import h5py
import numpy as np
import os
import sys
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay saved episode data.")
parser.add_argument("--episode_path", type=str, default="vpl_data", help="Path to the episode HDF5 file or directory (default: vpl_data).")
parser.add_argument("--episode_idx", type=int, default=None, help="Episode index to replay (when episode_path is a directory).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (should be 1 for replay).")
parser.add_argument("--video_path", type=str, default=None, help="Directory to save replay videos (overrides REPLAY_VIDEO_PATH).")
parser.add_argument("--video_fps", type=int, default=None, help="FPS for replay videos (overrides REPLAY_VIDEO_FPS, default 30).")
parser.add_argument("--camera_name", type=str, default=None, help="Name of camera sensor to capture (e.g., front_camera, wrist_camera).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments (separate from hydra args)
args_cli, hydra_args = parser.parse_known_args()

# Resolve video settings: CLI flags take precedence over environment variables
env_video_path = os.environ.get("REPLAY_VIDEO_PATH", None)
env_video_fps = os.environ.get("REPLAY_VIDEO_FPS", None)

if args_cli.video_path is None:
    args_cli.video_path = env_video_path

if args_cli.video_fps is None:
    args_cli.video_fps = int(env_video_fps) if env_video_fps is not None else 30

# Debug: show resolved video settings before launching app
print(f"[DEBUG] video_path={args_cli.video_path}, video_fps={args_cli.video_fps}")
print(f"[DEBUG] env video_path={env_video_path}, env video_fps={env_video_fps}")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Force single environment for replay
args_cli.num_envs = 1

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import imageio

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


def load_episode_data(episode_path: str, episode_idx: int = None) -> dict:
    """Load episode data from HDF5 file.
    
    Args:
        episode_path: Path to the HDF5 file or directory containing episode data.
                     If directory, will use the first episode (or episode_idx if provided).
        episode_idx: Specific episode index to load (only used when episode_path is a directory).
        
    Returns:
        Dictionary containing episode data (actions, observations, etc.)
    """
    # Check if path exists
    if not os.path.exists(episode_path):
        raise FileNotFoundError(f"Path not found: {episode_path}")
    
    # If it's a directory, try to find an episode file
    if os.path.isdir(episode_path):
        print(f"Directory provided: {episode_path}")
        print("Searching for episode files...")
        
        # Look for episode directories
        episode_dirs = sorted([d for d in os.listdir(episode_path) if d.startswith("episode_")])
        
        if not episode_dirs:
            raise FileNotFoundError(f"No episode directories found in: {episode_path}")
        
        print(f"Found {len(episode_dirs)} episode(s)")
        
        # Select episode
        if episode_idx is not None:
            # Use specified episode index
            target_dir = f"episode_{episode_idx}"
            if target_dir not in episode_dirs:
                raise FileNotFoundError(
                    f"Episode {episode_idx} not found. Available episodes: {[d.split('_')[1] for d in episode_dirs]}"
                )
            selected_episode = target_dir
            print(f"Using specified episode: {episode_idx}")
        else:
            # Use the first episode
            selected_episode = episode_dirs[0]
            episode_idx = selected_episode.split("_")[1]
            print(f"Using first episode: {episode_idx}")
        
        episode_path = os.path.join(episode_path, selected_episode, f"episode_{episode_idx}.h5")
    
    # Now episode_path should be a file
    if not os.path.isfile(episode_path):
        raise FileNotFoundError(f"Episode file not found: {episode_path}")
    
    data = {}
    with h5py.File(episode_path, "r") as f:
        print(f"\nLoading episode from: {episode_path}")
        print(f"Available keys: {list(f.keys())}")
        
        for key in f.keys():
            data[key] = torch.from_numpy(f[key][:]).float()
            print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
    
    return data


def get_all_episodes(base_dir: str) -> list[str]:
    """Get list of all episode files in a directory.
    
    Args:
        base_dir: Directory containing episode subdirectories
        
    Returns:
        List of episode file paths sorted by episode number
    """
    episode_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("episode_")])
    episode_files = []
    
    for episode_dir in episode_dirs:
        episode_idx = episode_dir.split("_")[1]
        episode_path = os.path.join(base_dir, episode_dir, f"episode_{episode_idx}.h5")
        if os.path.isfile(episode_path):
            episode_files.append(episode_path)
    
    return episode_files


def render_video_from_hdf5(h5_path: str, out_path: str, fps: int = 30, cam_idx: int = 0):
    """Render video from saved camera images in HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 episode file
        out_path: Output video file path
        fps: Frames per second for the video
        cam_idx: Camera index to render (default: 0 for first camera)
    """
    with h5py.File(h5_path, "r") as f:
        if "color" not in f:
            print(f"  Warning: No 'color' dataset in {h5_path}, cannot create video")
            return False
        
        color = f["color"][:]  # shape: (T, C, H, W, 3)
        if color.ndim != 5:
            print(f"  Warning: Unexpected 'color' shape {color.shape}, cannot create video")
            return False
        
        T, C, H, W, _ = color.shape
        
        if cam_idx >= C:
            print(f"  Warning: Camera index {cam_idx} >= number of cameras {C}")
            return False
        
        # Extract frames for the specified camera
        frames = color[:, cam_idx]  # (T, H, W, 3)
        
        # Normalize if needed (0-1 to 0-255)
        if frames.dtype in (np.float32, np.float64):
            frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Write video
        imageio.mimwrite(out_path, frames, fps=fps, codec="libx264")
        print(f"  Video saved: {out_path} [{T} frames @ {fps} FPS]")
        return True


def replay_single_episode(env, episode_data: dict, episode_num: int) -> dict:
    """Replay a single episode.
    
    Args:
        env: The Isaac Lab environment
        episode_data: Dictionary containing episode data
        episode_num: Episode number for display purposes
        video_writer: Optional cv2.VideoWriter for recording replay
        
    Returns:
        Dictionary with replay statistics
    """
    # Extract data
    actions = episode_data["action"]
    num_timesteps = actions.shape[0]
    
    # Check if we have joint_pos to set initial state
    if "joint_pos" in episode_data:
        initial_joint_pos = episode_data["joint_pos"][0:1]  # First timestep, shape (1, num_joints)
    else:
        initial_joint_pos = None
        print("  ⚠ No initial joint positions found in episode data. Using default reset.")
    
    print(f"\n{'='*80}")
    print(f"Replaying Episode {episode_num} ({num_timesteps} timesteps)")
    print(f"{'='*80}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Set initial joint positions if available
    if initial_joint_pos is not None:
        # Access the robot from the scene
        robot = env.unwrapped.scene["robot"]
        
        # Set initial joint positions
        initial_joint_pos_device = initial_joint_pos.to(env.unwrapped.device)
        initial_joint_vel = torch.zeros_like(initial_joint_pos_device)
        
        # Create proper env_ids tensor (or use None for single env)
        env_ids = None  # Since we only have 1 environment, None defaults to all envs
        
        # Write joint state to simulation
        robot.write_joint_state_to_sim(
            position=initial_joint_pos_device,
            velocity=initial_joint_vel,
            env_ids=env_ids
        )
        
        # Also set as joint targets (important for position-controlled robots)
        robot.set_joint_position_target(initial_joint_pos_device, env_ids=env_ids)
        
        # Step simulation a few times to stabilize
        for _ in range(10):
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
        
        # Render after setting initial state
        if env.unwrapped.sim.has_gui() or env.unwrapped.sim.has_rtx_sensors():
            env.unwrapped.sim.render()
    
    # Resolve camera for live capture (optional)
    scene = env.unwrapped.scene
    sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
    try:
        has_gui = env.unwrapped.sim.has_gui()
    except Exception:
        has_gui = False
    try:
        has_rtx = env.unwrapped.sim.has_rtx_sensors()
    except Exception:
        has_rtx = False
    print(f"[DEBUG] has_gui={has_gui}, has_rtx_sensors={has_rtx}")
    print(f"[DEBUG] sensors={list(sensors.keys()) if sensors else []}")
    camera_key = None
    if isinstance(sensors, dict) and sensors:
        # If user specified a camera name, prefer that
        if getattr(args_cli, "camera_name", None):
            if args_cli.camera_name in sensors and sensors[args_cli.camera_name] is not None:
                camera_key = args_cli.camera_name
        # Otherwise try preferred names from task config
        if camera_key is None:
            for pref in ["front_camera", "wrist_camera", "top_camera", "side_camera"]:
                if pref in sensors and sensors[pref] is not None:
                    camera_key = pref
                    break
        # Otherwise use the first available sensor that has rgb output
        if camera_key is None:
            for k, cam in sensors.items():
                try:
                    _ = cam.data.output.get("rgb", None)
                except Exception:
                    _ = None
                if _ is not None:
                    camera_key = k
                    break
    print(f"[DEBUG] camera_key={camera_key}")

    frames = []  # HWC uint8 frames captured live during replay

    # Replay actions
    total_reward = 0.0
    completed_steps = 0
    early_termination = False
    
    for step in range(num_timesteps):
        # Get action for this timestep (add batch dimension)
        action = actions[step:step+1].to(env.unwrapped.device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Ensure RTX sensors produce new frames this step (headless capture)
        try:
            if env.unwrapped.sim.has_rtx_sensors():
                env.unwrapped.sim.render()
        except Exception:
            pass
        
        total_reward += reward.item()
        completed_steps = step + 1
        
        # Capture camera frame (if available)
        if camera_key is not None:
            try:
                cam = sensors.get(camera_key)
                rgb = cam.data.output["rgb"][0]
                # Ensure CPU numpy and drop alpha if present
                rgb_np = rgb.detach().cpu().numpy()
                if rgb_np.shape[-1] > 3:
                    rgb_np = rgb_np[..., :3]
                # If float in [0,1], scale to uint8
                if rgb_np.dtype.kind == 'f':
                    import numpy as _np
                    rgb_np = (_np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(_np.uint8)
                else:
                    rgb_np = rgb_np.astype("uint8", copy=False)
                frames.append(rgb_np)
            except Exception:
                # Ignore capture failures silently per-step
                pass

        # Print progress
        if step % 50 == 0:
            print(
                f"  Step {step}/{num_timesteps} | R: {reward.item():.4f} "
                f"| Total: {total_reward:.2f} | frames={len(frames)}"
            )
        
        # Check if episode terminated early
        if terminated[0] or truncated[0]:
            print(f"\n  \u26a0 Episode terminated early at step {step}/{num_timesteps}")
            print(f"    Terminated: {terminated[0]}, Truncated: {truncated[0]}")
            early_termination = True
            break
    
    print(f"{'='*80}")
    print(f"\u2713 Episode {episode_num} completed: {completed_steps}/{num_timesteps} steps, Total reward: {total_reward:.2f}")
    print(f"{'='*80}\n")
    
    # Write captured video if requested
    if args_cli.video_path is not None and len(frames) > 0:
        try:
            os.makedirs(args_cli.video_path, exist_ok=True)
            video_filename = os.path.join(
                args_cli.video_path,
                f"episode_{episode_num}_replay_live.mp4",
            )
            # Write out using imageio
            import imageio as _imageio
            _imageio.mimwrite(video_filename, frames, fps=int(args_cli.video_fps), codec="libx264")
            print(f"  Live replay video saved: {video_filename} [{len(frames)} frames @ {int(args_cli.video_fps)} FPS]")
        except Exception as e:
            print(f"  Warning: Failed to save live replay video: {e}")
    else:
        print(
            f"[DEBUG] skip video: path={args_cli.video_path}, frames={len(frames)}"
        )

    return {
        "episode_num": episode_num,
        "total_steps": num_timesteps,
        "completed_steps": completed_steps,
        "total_reward": total_reward,
        "early_termination": early_termination
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Replay episodes in Isaac Lab environment."""
    
    # Override number of environments to 1
    env_cfg.scene.num_envs = 1
    
    # Disable randomization for deterministic replay
    env_cfg.events = None
    
    # Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Determine which episodes to replay
    if os.path.isfile(args_cli.episode_path):
        # Single episode file
        episode_files = [args_cli.episode_path]
        print(f"Replaying single episode file: {args_cli.episode_path}")
    elif os.path.isdir(args_cli.episode_path):
        if args_cli.episode_idx is not None:
            # Specific episode from directory
            episode_idx = args_cli.episode_idx
            episode_path = os.path.join(args_cli.episode_path, f"episode_{episode_idx}", f"episode_{episode_idx}.h5")
            episode_files = [episode_path]
            print(f"Replaying episode {episode_idx} from directory")
        else:
            # All episodes from directory
            episode_files = get_all_episodes(args_cli.episode_path)
            print(f"Found {len(episode_files)} episodes to replay")
    else:
        raise FileNotFoundError(f"Path not found: {args_cli.episode_path}")
    
    if not episode_files:
        raise FileNotFoundError("No episode files found")
    
    # Create output directory for videos if requested
    if args_cli.video_path is not None:
        os.makedirs(args_cli.video_path, exist_ok=True)
        print(f"Videos will be saved to: {args_cli.video_path}")
    
    # Replay all episodes
    stats = []
    
    for i, episode_file in enumerate(episode_files):
        # Load episode data
        print(f"\nLoading episode {i+1}/{len(episode_files)}: {episode_file}")
        data = {}
        with h5py.File(episode_file, "r") as f:
            print(f"  Keys: {list(f.keys())}")
            for key in f.keys():
                data[key] = torch.from_numpy(f[key][:]).float()
        
        # Replay episode
        episode_stats = replay_single_episode(env, data, i)
        stats.append(episode_stats)
        
        # If no live frames were captured (e.g., no cameras), fall back to HDF5 color rendering
        if args_cli.video_path is not None:
            # Check whether a live video exists for this episode number
            live_name = os.path.join(args_cli.video_path, f"episode_{i}_replay_live.mp4")
            if not os.path.exists(live_name):
                episode_num_str = os.path.basename(episode_file).replace('.h5', '').split('_')[-1]
                video_filename = os.path.join(args_cli.video_path, f"episode_{episode_num_str}_replay.mp4")
                print(f"\n  Rendering video from saved camera images (fallback)...")
                render_video_from_hdf5(episode_file, video_filename, fps=args_cli.video_fps, cam_idx=0)
        
        # Small pause between episodes
        time.sleep(1.0)
    
    # Print summary
    print(f"\n{'='*80}")
    print("REPLAY SUMMARY")
    print(f"{'='*80}")
    print(f"Total episodes replayed: {len(stats)}")
    
    total_steps = sum(s["completed_steps"] for s in stats)
    avg_reward = sum(s["total_reward"] for s in stats) / len(stats)
    num_early_term = sum(1 for s in stats if s["early_termination"])
    
    print(f"Total steps: {total_steps}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Episodes with early termination: {num_early_term}/{len(stats)}")
    print(f"{'='*80}\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


