#!/usr/bin/env python3
# flake8: noqa
# Minimal: Run a pretrained RSL-RL policy on Isaac-Open-Drawer-Franka-v0,
# inject a per-env TiledCamera, use wide env spacing, and save a VPL dataset
# (RGB + MP4s) with one video per env.

import argparse
import os
from isaaclab.app import AppLauncher  # must import before other IsaacLab modules

# ----------------- CLI -----------------
parser = argparse.ArgumentParser("Collect VPL dataset with TiledCamera and a policy checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--env_spacing", type=float, default=5.0)
parser.add_argument("--steps", type=int, default=60, help="Steps per episode")
parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to collect per environment")
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)

# checkpoint options
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt or .pth)")

# dataset/video
parser.add_argument("--data_root", type=str, default="/workspace/datasets/vpl_tiled")
parser.add_argument("--robot_name", type=str, default="franka")
parser.add_argument("--sim_or_real", type=str, default="sim", choices=["sim", "real"])
parser.add_argument("--fps", type=int, default=30)

# video/frame sampling
parser.add_argument("--video_interval", type=int, default=1, help="Interval between video recordings (in steps).")

# camera placement (relative to each env origin)
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

# wrist-mounted camera options
parser.add_argument("--enable_wrist_camera", action="store_true", help="Enable a per-env wrist-mounted camera attached to the end-effector")
parser.add_argument("--wrist_width", type=int, default=320, help="Wrist camera width")
parser.add_argument("--wrist_height", type=int, default=240, help="Wrist camera height")
parser.add_argument(
    "--wrist_cam_offset",
    type=float,
    nargs=3,
    default=[-0.05, 0.0, 0.08],
    help="Wrist camera position offset [x y z] in the EE frame (meters). Behind and above gripper to avoid clipping.",
)
parser.add_argument(
    "--wrist_cam_look_offset",
    type=float,
    nargs=3,
    default=[0.25, 0.0, 0.0],
    help="Wrist camera look-at target offset [x y z] in the EE frame (meters). Looks forward along gripper axis.",
)

# top-view camera options
parser.add_argument("--enable_top_camera", action="store_true", help="Enable a per-env top-view camera looking down at the scene")
parser.add_argument("--top_width", type=int, default=320, help="Top camera width")
parser.add_argument("--top_height", type=int, default=240, help="Top camera height")
parser.add_argument(
    "--top_cam_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 3.0],
    help="Top camera position offset [x y z] from env origin (meters). Default is directly above.",
)
parser.add_argument(
    "--top_tgt_offset",
    type=float,
    nargs=3,
    default=[0.4, 0.0, 0.5],
    help="Top camera look-at target offset [x y z] from env origin (meters).",
)

# let Kit parse --headless / --renderer (like benchmark)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# launch the app first
app = AppLauncher(args).app

# NOTE: Removed all viewport/prim axis removal attempts as requested

# ------------ Now Isaac Lab imports -------------
import time
import datetime as dt
import numpy as np
import torch
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

# RSL-RL helper + runner
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic

# Your dataset helper
from vpl_saver import VPLSaver


def _make_out_dir(base_root: str, task: str, sim_or_real: str, robot_name: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(base_root, f"{task}_{sim_or_real}_{robot_name}_{stamp}"))
    os.makedirs(path, exist_ok=True)
    return path


def build_env_cfg(args):
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.scene.env_spacing = args.env_spacing
    
    # Add tiled camera - try both direct attribute and sensors dict
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
    
    # Check if scene has a sensors manager (Manager-based environments)
    if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
        print("[INFO] Using Manager-based scene config (sensors dict)")
        if not isinstance(env_cfg.scene.sensors, dict):
            # Convert to dict if it's a DictConfig or similar
            env_cfg.scene.sensors = {}
        env_cfg.scene.sensors['tiled_camera'] = tiled_cam_cfg
    else:
        print("[INFO] Using direct attribute scene config")
        env_cfg.scene.tiled_camera = tiled_cam_cfg
    
    if args.enable_wrist_camera:
        print("[INFO] Adding wrist camera to scene config...")
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
                clipping_range=(0.01, 10.0),  # Closer near clip, reasonable far clip
            ),
        )
        
        # Add to sensors dict or direct attribute based on scene type
        if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
            env_cfg.scene.sensors['wrist_camera'] = wrist_cam_cfg
            print("[INFO] Wrist camera added to sensors dict")
        else:
            env_cfg.scene.wrist_camera = wrist_cam_cfg
            print("[INFO] Wrist camera added as direct attribute")
        print(f"[INFO] Wrist camera configured: {args.wrist_width}x{args.wrist_height}")
    
    if args.enable_top_camera:
        print("[INFO] Adding top camera to scene config...")
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
        
        # Add to sensors dict or direct attribute based on scene type
        if hasattr(env_cfg.scene, 'sensors') and env_cfg.scene.sensors is not None:
            env_cfg.scene.sensors['top_camera'] = top_cam_cfg
            print("[INFO] Top camera added to sensors dict")
        else:
            env_cfg.scene.top_camera = top_cam_cfg
            print("[INFO] Top camera added as direct attribute")
        print(f"[INFO] Top camera configured: {args.top_width}x{args.top_height}")
    return env_cfg


def create_env(args, env_cfg):
    print(f"[INFO] Creating environment: {args.task}")
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device
    obs, _ = env.reset()
    return env, device, obs


def find_tiled_camera(scene):
    if hasattr(scene, 'tiled_camera'):
        return scene.tiled_camera
    if hasattr(scene, 'sensors'):
        if isinstance(scene.sensors, dict) and 'tiled_camera' in scene.sensors:
            return scene.sensors['tiled_camera']
        if hasattr(scene.sensors, 'tiled_camera'):
            return scene.sensors.tiled_camera
        raise RuntimeError(
            f"Could not find tiled_camera. Available sensors: {list(scene.sensors.keys()) if isinstance(scene.sensors, dict) else dir(scene.sensors)}"
        )
    raise RuntimeError(f"Could not find tiled_camera. Scene attributes: {[k for k in dir(scene) if not k.startswith('_')]}")


def position_camera(scene, cam, device, args):
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


def make_saver(args):
    base_dir = _make_out_dir(args.data_root, args.task, args.sim_or_real, args.robot_name)
    effective_fps = max(1, int(round(args.fps / max(1, args.video_interval))))
    saver = VPLSaver(
        base_dir=base_dir,
        keep_terminated=False,
        fps=effective_fps,
        initial_timesteps_to_discard=0,
        enable_wrist_camera=args.enable_wrist_camera,
        wrist_cam_offset=args.wrist_cam_offset,
        wrist_cam_look_offset=args.wrist_cam_look_offset,
        enable_top_camera=args.enable_top_camera,
        top_cam_offset=args.top_cam_offset,
        top_tgt_offset=args.top_tgt_offset,
        wrist_auto_follow=False,  # Manual positioning for backward compatibility
    )
    print(f"[INFO] Writing VPL dataset to: {base_dir}")
    print(f"[INFO] Video FPS (effective): {effective_fps}")
    return saver, base_dir


def run_collection(env, policy, saver, args, num_envs):
    print(f"\n[INFO] Collection Setup:")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Rollout rounds: {args.num_episodes}")
    print(f"  Steps per episode: {args.steps}")
    print(f"  Video interval (steps): {args.video_interval}")
    print(f"  Camera offset: {args.cam_offset}  Target offset: {args.tgt_offset}")
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
    # --------- Build env and camera ----------
    env_cfg = build_env_cfg(args)
    env, device, obs = create_env(args, env_cfg)
    scene = env.unwrapped.scene
    cam = find_tiled_camera(scene)
    position_camera(scene, cam, device, args)
    
    # Verify wrist camera if enabled
    if args.enable_wrist_camera:
        wrist_cam_found = False
        if hasattr(scene, 'wrist_camera'):
            wrist_cam_found = True
            print("[INFO] Wrist camera found in scene.wrist_camera")
        elif hasattr(scene, 'sensors'):
            if isinstance(scene.sensors, dict) and 'wrist_camera' in scene.sensors:
                wrist_cam_found = True
                print("[INFO] Wrist camera found in scene.sensors['wrist_camera']")
            elif hasattr(scene.sensors, 'wrist_camera'):
                wrist_cam_found = True
                print("[INFO] Wrist camera found in scene.sensors.wrist_camera")
        
        if not wrist_cam_found:
            print("[WARNING] Wrist camera enabled but NOT found in scene!")
            if hasattr(scene, 'sensors'):
                if isinstance(scene.sensors, dict):
                    print(f"[WARNING] Available sensors: {list(scene.sensors.keys())}")
                else:
                    print(f"[WARNING] Available sensor attributes: {[k for k in dir(scene.sensors) if not k.startswith('_')]}")
        else:
            print("[INFO] Wrist camera successfully initialized")
    
    # Position and verify top camera if enabled
    if args.enable_top_camera:
        top_cam_found = False
        top_cam = None
        if hasattr(scene, 'top_camera'):
            top_cam = scene.top_camera
            top_cam_found = True
            print("[INFO] Top camera found in scene.top_camera")
        elif hasattr(scene, 'sensors'):
            if isinstance(scene.sensors, dict) and 'top_camera' in scene.sensors:
                top_cam = scene.sensors['top_camera']
                top_cam_found = True
                print("[INFO] Top camera found in scene.sensors['top_camera']")
            elif hasattr(scene.sensors, 'top_camera'):
                top_cam = scene.sensors.top_camera
                top_cam_found = True
                print("[INFO] Top camera found in scene.sensors.top_camera")
        
        if not top_cam_found:
            print("[WARNING] Top camera enabled but NOT found in scene!")
            if hasattr(scene, 'sensors'):
                if isinstance(scene.sensors, dict):
                    print(f"[WARNING] Available sensors: {list(scene.sensors.keys())}")
                else:
                    print(f"[WARNING] Available sensor attributes: {[k for k in dir(scene.sensors) if not k.startswith('_')]}")
        else:
            # Position the top camera
            n_views = top_cam.data.intrinsic_matrices.size(0)
            if hasattr(scene, "env_origins"):
                origins = scene.env_origins
                top_cam_offset = torch.tensor(args.top_cam_offset, device=device)
                top_tgt_offset = torch.tensor(args.top_tgt_offset, device=device)
                positions = origins + top_cam_offset
                targets = origins + top_tgt_offset
            else:
                positions = torch.tensor([args.top_cam_offset], device=device).repeat(n_views, 1)
                targets = torch.tensor([args.top_tgt_offset], device=device).repeat(n_views, 1)
            top_cam.set_world_poses_from_view(positions, targets)
            print("[INFO] Top camera successfully positioned and initialized")

    # --------- Dimensions & info ----------
    num_obs, num_actions, num_envs = compute_dimensions(obs, env)
    print(f"[INFO] Environment: {args.task}")
    print(f"[INFO] Num envs: {args.num_envs}, spacing: {args.env_spacing}m")
    print(f"[INFO] Observation dimension: {num_obs}")
    print(f"[INFO] Action dimension: {num_actions}")

    # --------- Policy ----------
    policy = load_policy_from_checkpoint(args.checkpoint, num_obs, num_actions, device)

    # --------- Saver ----------
    saver, base_dir = make_saver(args)

    # --------- Rollout ----------
    print(f"\nNote: Each parallel env is a separate episode with different initial conditions")
    run_collection(env, policy, saver, args, num_envs)

    # --------- Final Summary ----------
    total_episodes = args.num_episodes * num_envs
    print(f"\n{'='*60}")
    print(f"[OK] Data Collection Complete!")
    print(f"{'='*60}")
    print(f"Output directory: {base_dir}")
    print(f"")
    print(f"TOTAL EPISODES COLLECTED: {total_episodes}")
    print(f"  Rollout rounds: {args.num_episodes}")
    print(f"  Parallel envs per round: {num_envs}")
    print(f"  Steps per episode: {args.steps}")
    print(f"")
    print(f"Dataset structure: {total_episodes} episode directories (episode_000 to episode_{total_episodes-1:03d})")
    print(f"Each episode has: 1 camera (camera_0) with RGB video + actions + proprio")
    print(f"")
    env.close()


if __name__ == "__main__":
    main()
    app.close()

