#!/usr/bin/env python3
# flake8: noqa
"""Replay saved actions from VPL dataset to validate data correctness."""

import argparse
import os
import h5py
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import imageio.v3 as iio

from isaaclab.app import AppLauncher

# ----------------- CLI -----------------
parser = argparse.ArgumentParser("Replay actions from saved VPL episode")
parser.add_argument("--episode_dir", type=str, required=True, help="Path to episode directory (e.g., vpl_tiled/episode_000)")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0", help="Task environment name")
parser.add_argument("--env_spacing", type=float, default=5.0, help="Environment spacing")
parser.add_argument("--output_dir", type=str, default="/workspace/replay_output", help="Output directory for replay videos")
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)

# Camera options (should match collection settings)
parser.add_argument("--enable_wrist_camera", action="store_true", help="Enable wrist camera")
parser.add_argument("--enable_top_camera", action="store_true", help="Enable top camera")
parser.add_argument("--wrist_width", type=int, default=320)
parser.add_argument("--wrist_height", type=int, default=240)
parser.add_argument("--top_width", type=int, default=320)
parser.add_argument("--top_height", type=int, default=240)

parser.add_argument(
    "--cam_offset",
    type=float,
    nargs=3,
    default=[-2.0, 0.5, 2.1],
    help="Camera position offset [x y z] from env origin (meters).",
)
parser.add_argument(
    "--tgt_offset",
    type=float,
    nargs=3,
    default=[0.4, 0.0, 0.5],
    help="Camera look-at target offset [x y z] from env origin (meters).",
)
parser.add_argument(
    "--wrist_cam_offset",
    type=float,
    nargs=3,
    default=[-0.05, 0.0, 0.08],
    help="Wrist camera position offset [x y z] in the EE frame (meters).",
)
parser.add_argument(
    "--wrist_cam_look_offset",
    type=float,
    nargs=3,
    default=[0.25, 0.0, 0.0],
    help="Wrist camera look-at target offset [x y z] in the EE frame (meters).",
)
parser.add_argument(
    "--top_cam_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 3.0],
    help="Top camera position offset [x y z] from env origin (meters).",
)
parser.add_argument(
    "--top_tgt_offset",
    type=float,
    nargs=3,
    default=[0.4, 0.0, 0.5],
    help="Top camera look-at target offset [x y z] from env origin (meters).",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# Launch the app
app = AppLauncher(args).app

# ------------ Now Isaac Lab imports -------------
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils


def load_episode_data(episode_dir: str):
    """Load actions and metadata from episode h5 file."""
    episode_path = Path(episode_dir)
    
    # Find the h5 file
    h5_files = list(episode_path.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 file found in {episode_dir}")
    
    h5_path = h5_files[0]
    print(f"[INFO] Loading episode from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        actions = np.array(f['action'])
        
        # Load metadata
        metadata = {
            'env_idx': f.attrs.get('env_idx', 0),
            'success': f.attrs.get('success', False),
            'terminated': f.attrs.get('terminated', False),
            'num_frames': f.attrs.get('num_frames', len(actions)),
            'fps': f.attrs.get('fps', 30),
        }
        
        print(f"[INFO] Episode metadata:")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Success: {metadata['success']}")
        print(f"  Terminated: {metadata['terminated']}")
        print(f"  Num frames: {metadata['num_frames']}")
        
        # Optionally load original frames for comparison
        original_frames = None
        if 'color' in f:
            original_frames = np.array(f['color'])
            print(f"  Original frames shape: {original_frames.shape}")
    
    return actions, metadata, original_frames


def build_env_cfg(args):
    """Build environment configuration with cameras."""
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,  # Replay one episode at a time
    )
    env_cfg.scene.env_spacing = args.env_spacing
    
    # Add tiled camera
    tiled_cam_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/tiled_camera",
        update_period=0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
    )
    
    if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
        if not isinstance(env_cfg.scene.sensors, dict):
            env_cfg.scene.sensors = {}
        env_cfg.scene.sensors['tiled_camera'] = tiled_cam_cfg
    else:
        env_cfg.scene.tiled_camera = tiled_cam_cfg
    
    # Add wrist camera if enabled
    if args.enable_wrist_camera:
        wrist_cam_cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/wrist_camera",
            update_period=0,
            height=args.wrist_height,
            width=args.wrist_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10.0),
            ),
        )
        
        if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
            env_cfg.scene.sensors['wrist_camera'] = wrist_cam_cfg
        else:
            env_cfg.scene.wrist_camera = wrist_cam_cfg
    
    # Add top camera if enabled
    if args.enable_top_camera:
        top_cam_cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/top_camera",
            update_period=0,
            height=args.top_height,
            width=args.top_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1e4),
            ),
        )
        
        if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
            env_cfg.scene.sensors['top_camera'] = top_cam_cfg
        else:
            env_cfg.scene.top_camera = top_cam_cfg
    
    return env_cfg


def find_camera(scene, camera_name='tiled_camera'):
    """Find camera in scene."""
    if hasattr(scene, camera_name):
        return getattr(scene, camera_name)
    if hasattr(scene, 'sensors'):
        if isinstance(scene.sensors, dict) and camera_name in scene.sensors:
            return scene.sensors[camera_name]
        if hasattr(scene.sensors, camera_name):
            return getattr(scene.sensors, camera_name)
    return None


def position_camera(scene, cam, device, args):
    """Position camera at specified offset."""
    n_views = cam.data.intrinsic_matrices.size(0)
    if hasattr(scene, "env_origins"):
        origins = scene.env_origins
        cam_offset = torch.tensor(args.cam_offset, device=device)
        tgt_offset = torch.tensor(args.tgt_offset, device=device)
        positions = origins + cam_offset
        targets = origins + tgt_offset
    else:
        positions = torch.tensor([args.cam_offset], device=device).repeat(n_views, 1)
        targets = torch.tensor([args.tgt_offset], device=device).repeat(n_views, 1)
    cam.set_world_poses_from_view(positions, targets)


def position_top_camera(scene, cam, device, args):
    """Position top camera."""
    n_views = cam.data.intrinsic_matrices.size(0)
    if hasattr(scene, "env_origins"):
        origins = scene.env_origins
        top_cam_offset = torch.tensor(args.top_cam_offset, device=device)
        top_tgt_offset = torch.tensor(args.top_tgt_offset, device=device)
        positions = origins + top_cam_offset
        targets = origins + top_tgt_offset
    else:
        positions = torch.tensor([args.top_cam_offset], device=device).repeat(n_views, 1)
        targets = torch.tensor([args.top_tgt_offset], device=device).repeat(n_views, 1)
    cam.set_world_poses_from_view(positions, targets)


def update_wrist_camera(scene, wrist_cam, device, args):
    """Update wrist camera pose from end-effector."""
    robot_articulation = scene._articulations.get("robot")
    if robot_articulation is None:
        print("[WARNING] Robot articulation not found, skipping wrist camera update")
        return
    
    # Get EE pose
    ee_pos = robot_articulation.data.body_pos_w[:, -1, :]
    ee_rot = robot_articulation.data.body_quat_w[:, -1, :]
    
    # Compute camera poses from EE frame
    from vpl_saver import VPLSaver
    saver = VPLSaver(base_dir="/tmp", enable_wrist_camera=True, 
                     wrist_cam_offset=args.wrist_cam_offset,
                     wrist_cam_look_offset=args.wrist_cam_look_offset)
    positions, targets = saver._compute_wrist_camera_poses(ee_pos, ee_rot)
    wrist_cam.set_world_poses_from_view(positions, targets)


def replay_episode(env, scene, actions, output_dir, fps=30, enable_wrist=False, enable_top=False):
    """Replay actions and record video."""
    print(f"\n[INFO] Starting episode replay...")
    print(f"  Number of actions: {len(actions)}")
    
    # Find cameras
    main_cam = find_camera(scene, 'tiled_camera')
    wrist_cam = find_camera(scene, 'wrist_camera') if enable_wrist else None
    top_cam = find_camera(scene, 'top_camera') if enable_top else None
    
    if main_cam is None:
        raise RuntimeError("Main camera (tiled_camera) not found!")
    
    device = env.unwrapped.device
    
    # Reset environment
    obs, _ = env.reset()
    print(f"[INFO] Environment reset complete")
    
    # Position cameras
    position_camera(scene, main_cam, device, args)
    if top_cam is not None:
        position_top_camera(scene, top_cam, device, args)
    
    # Storage for frames
    main_frames = []
    wrist_frames = []
    top_frames = []
    
    # Replay actions
    for step_idx, action in enumerate(actions):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{len(actions)}...")
        
        # Convert action to tensor
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        
        # Update wrist camera if enabled
        if wrist_cam is not None:
            update_wrist_camera(scene, wrist_cam, device, args)
            wrist_cam.update(dt=scene.physics_dt)
        
        # Update cameras
        main_cam.update(dt=scene.physics_dt)
        if top_cam is not None:
            top_cam.update(dt=scene.physics_dt)
        
        # Capture frames
        rgb = main_cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        main_frames.append(rgb)
        
        if wrist_cam is not None:
            rgb_w = wrist_cam.data.output["rgb"][0].detach().cpu().numpy()
            if rgb_w.ndim == 3 and rgb_w.shape[0] in (3, 4):
                rgb_w = np.transpose(rgb_w, (1, 2, 0))
            if rgb_w.shape[-1] == 4:
                rgb_w = rgb_w[..., :3]
            if rgb_w.dtype != np.uint8:
                rgb_w = np.clip(rgb_w * 255.0, 0, 255).astype(np.uint8)
            wrist_frames.append(rgb_w)
        
        if top_cam is not None:
            rgb_t = top_cam.data.output["rgb"][0].detach().cpu().numpy()
            if rgb_t.ndim == 3 and rgb_t.shape[0] in (3, 4):
                rgb_t = np.transpose(rgb_t, (1, 2, 0))
            if rgb_t.shape[-1] == 4:
                rgb_t = rgb_t[..., :3]
            if rgb_t.dtype != np.uint8:
                rgb_t = np.clip(rgb_t * 255.0, 0, 255).astype(np.uint8)
            top_frames.append(rgb_t)
    
    # Save videos
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    episode_name = Path(args.episode_dir).name
    
    main_video_path = output_path / f"{episode_name}_replay_main.mp4"
    iio.imwrite(main_video_path, main_frames, fps=fps)
    print(f"\n[INFO] Saved main camera replay: {main_video_path}")
    
    if wrist_frames:
        wrist_video_path = output_path / f"{episode_name}_replay_wrist.mp4"
        iio.imwrite(wrist_video_path, wrist_frames, fps=fps)
        print(f"[INFO] Saved wrist camera replay: {wrist_video_path}")
    
    if top_frames:
        top_video_path = output_path / f"{episode_name}_replay_top.mp4"
        iio.imwrite(top_video_path, top_frames, fps=fps)
        print(f"[INFO] Saved top camera replay: {top_video_path}")
    
    print(f"\n[INFO] Replay complete! Total frames: {len(main_frames)}")
    
    return main_frames, wrist_frames, top_frames


def main():
    # Load episode data
    actions, metadata, original_frames = load_episode_data(args.episode_dir)
    
    # Build environment
    print(f"\n[INFO] Creating environment: {args.task}")
    env_cfg = build_env_cfg(args)
    env = gym.make(args.task, cfg=env_cfg)
    scene = env.unwrapped.scene
    
    # Replay episode
    main_frames, wrist_frames, top_frames = replay_episode(
        env, scene, actions, args.output_dir, 
        fps=metadata['fps'],
        enable_wrist=args.enable_wrist_camera,
        enable_top=args.enable_top_camera
    )
    
    # Compare with original if available
    if original_frames is not None:
        print(f"\n[INFO] Comparison with original:")
        print(f"  Original frames: {len(original_frames)}")
        print(f"  Replayed frames: {len(main_frames)}")
        
        if len(original_frames) == len(main_frames):
            print(f"  ✓ Frame count matches!")
        else:
            print(f"  ✗ Frame count mismatch!")
    
    # Close environment
    env.close()
    
    print(f"\n{'='*60}")
    print(f"[OK] Replay Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Episode: {Path(args.episode_dir).name}")
    print(f"Actions replayed: {len(actions)}")
    print(f"")


if __name__ == "__main__":
    main()
    app.close()

