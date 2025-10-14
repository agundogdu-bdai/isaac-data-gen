#!/usr/bin/env python3
"""
Simple script to record high-resolution video of trained policy execution.
Records the Isaac Sim viewport (not camera sensors), perfect for demos/visualizations.
"""

import argparse
import os
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser("Record policy execution video")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained policy checkpoint")
parser.add_argument("--num_envs", type=int, default=25, help="Number of parallel environments")
parser.add_argument("--env_spacing", type=float, default=3.0, help="Spacing between environments")
parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
parser.add_argument("--output_dir", type=str, default="/workspace/videos", help="Output directory for videos")
parser.add_argument("--video_name", type=str, default="policy_demo", help="Base name for video file")
parser.add_argument("--fps", type=int, default=30, help="Video FPS")
parser.add_argument("--resolution", type=str, default="1920x1080", help="Video resolution (WIDTHxHEIGHT)")

# Isaac Sim arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Parse resolution
width, height = map(int, args.resolution.split('x'))

# Launch Isaac Sim with rendering enabled (for video capture)
args.enable_cameras = False  # We don't need camera sensors
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.modules import ActorCritic
import imageio.v3 as iio


def load_policy(checkpoint_path, num_obs, num_actions, device):
    """Load trained policy from checkpoint."""
    print(f"[INFO] Loading policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
    else:
        model_state = checkpoint
    
    # Create policy network
    policy = ActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
    ).to(device)
    
    # Handle old checkpoint format
    if "_std" in model_state and "std" not in model_state:
        model_state["std"] = model_state.pop("_std")
    
    policy.load_state_dict(model_state)
    policy.eval()
    print(f"[INFO] ✓ Policy loaded successfully")
    return policy


def setup_viewport(simulation_app, width, height):
    """Configure viewport for high-quality rendering."""
    import carb
    
    # Enable anti-aliasing and high quality settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", 2)  # TAA (Temporal Anti-Aliasing)
    settings.set("/rtx/post/dlss/execMode", 1)  # DLSS if available
    settings.set("/rtx/pathtracing/spp", 4)  # Samples per pixel
    
    print(f"[INFO] Target resolution: {width}x{height}")
    print("[INFO] ✓ Viewport configured for high-quality rendering")


def capture_frames(simulation_app, num_frames):
    """Capture frames from the viewport."""
    import omni.kit.viewport.utility as vp_utils
    from PIL import Image
    import io
    
    viewport_api = vp_utils.get_active_viewport()
    frames = []
    
    print(f"[INFO] Capturing {num_frames} frames...")
    
    for i in range(num_frames):
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}...")
        
        # Render frame
        simulation_app.update()
        
        # Capture viewport image
        if viewport_api:
            # Get raw pixel data
            img_data = viewport_api.get_texture()
            
            if img_data:
                # Convert to numpy array
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                
                # Reshape to image (viewport returns BGRA)
                h, w = viewport_api.get_texture_resolution()
                img_array = img_array.reshape((h, w, 4))
                
                # Convert BGRA to RGB
                img_rgb = img_array[:, :, [2, 1, 0]]  # BGR to RGB
                
                frames.append(img_rgb)
    
    print(f"[INFO] ✓ Captured {len(frames)} frames")
    return frames


def run_and_record(env, policy, args, device):
    """Run policy and record frames using replicator."""
    print(f"\n{'='*60}")
    print(f"Recording Policy Execution")
    print(f"{'='*60}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Steps per episode: {args.steps}")
    print(f"  Resolution: {args.resolution}")
    print(f"  FPS: {args.fps}")
    print(f"{'='*60}\n")
    
    # Setup replicator for screen capture
    import omni.replicator.core as rep
    
    # Get viewport for rendering
    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution=(width, height))
    
    # Initialize writer for capturing frames
    rgb_writer = rep.WriterRegistry.get("BasicWriter")
    rgb_writer.initialize(output_dir="/tmp/viewport_capture", rgb=True)
    rgb_writer.attach([render_product])
    
    all_frames = []
    
    for episode_idx in range(args.num_episodes):
        print(f"\n[Episode {episode_idx + 1}/{args.num_episodes}]")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Run episode
        for step in range(args.steps):
            if step % 50 == 0:
                print(f"  Step {step}/{args.steps}...")
            
            # Policy inference
            with torch.inference_mode():
                if isinstance(obs, dict):
                    obs_tensor = torch.cat([v.view(v.shape[0], -1) for v in obs.values()], dim=-1)
                else:
                    obs_tensor = obs
                action = policy.act_inference(obs_tensor)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(action)
            
            # Update simulation (for rendering)
            simulation_app.update()
            
            # Capture frame every N steps (to match target FPS)
            if step % max(1, int(60 / args.fps)) == 0:
                # Trigger replicator capture
                rep.orchestrator.step(rt_subframes=4)
                
                # Read the captured frame
                import omni.kit.viewport.utility as vp_utils
                viewport_window = vp_utils.get_active_viewport_window()
                if viewport_window:
                    # Get frame buffer data
                    frame_data = viewport_window.get_frame()
                    if frame_data is not None and len(frame_data) > 0:
                        # Convert to numpy array
                        img_rgb = np.array(frame_data[:, :, :3], dtype=np.uint8)
                        all_frames.append(img_rgb)
    
    print(f"\n[INFO] Captured {len(all_frames)} frames")
    return all_frames


def save_video(frames, output_path, fps):
    """Save frames as MP4 video."""
    print(f"\n[INFO] Saving video to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use imageio to write MP4 with good compression
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='h264',
        quality=8,  # High quality (0-10, 10 is best)
        macro_block_size=1,
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] ✓ Video saved: {output_path}")
    print(f"[INFO]   Frames: {len(frames)}")
    print(f"[INFO]   Duration: {len(frames) / fps:.1f}s")
    print(f"[INFO]   File size: {file_size_mb:.1f} MB")


def main():
    # Create environment
    print(f"[INFO] Creating environment: {args.task}")
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.scene.env_spacing = args.env_spacing
    
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device
    
    # Get dimensions
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs_flat = torch.cat([v.view(v.shape[0], -1) for v in obs.values()], dim=-1)
        num_obs = obs_flat.shape[-1]
    else:
        num_obs = obs.shape[-1]
    num_actions = env.action_space.shape[-1]
    
    print(f"[INFO] Environment created")
    print(f"  Observation dim: {num_obs}")
    print(f"  Action dim: {num_actions}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Env spacing: {args.env_spacing}m")
    
    # Load policy
    policy = load_policy(args.checkpoint, num_obs, num_actions, device)
    
    # Setup viewport
    setup_viewport(simulation_app, width, height)
    
    # Position camera to view all environments
    print("\n[INFO] Positioning viewport camera...")
    # Camera will auto-position based on env_spacing
    
    # Run and record
    frames = run_and_record(env, policy, args, device)
    
    # Save video
    output_path = os.path.join(args.output_dir, f"{args.video_name}_{args.resolution}.mp4")
    save_video(frames, output_path, args.fps)
    
    print(f"\n{'='*60}")
    print(f"✓ Recording Complete!")
    print(f"{'='*60}")
    print(f"Video: {output_path}")
    print(f"")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

