#!/usr/bin/env python3
"""Simple VPL Dataset Saver - saves actions and camera frames."""

import numpy as np
import torch
from pathlib import Path
import imageio.v3 as iio
import h5py
import json


class VPLSaver:
    """Saves episodes with actions and multi-camera videos."""

    def __init__(self, base_dir: str, fps: int = 30, keep_terminated: bool = False,
                 enable_wrist_camera: bool = False, enable_top_camera: bool = False, 
                 enable_side_camera: bool = False, **kwargs):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.keep_terminated = keep_terminated
        self.enable_wrist = enable_wrist_camera
        self.enable_top = enable_top_camera
        self.enable_side = enable_side_camera
        
        # Storage: {env_idx: {'actions': [], 'frames': {'cam0': [], 'cam1': [], ...}}}
        self.episodes = {}
        self.episode_counter = 0

    def store(self, actions: torch.Tensor, env, store_frame: bool = True):
        """Store actions and frames for this timestep."""
        scene = env.unwrapped.scene
        num_envs = actions.shape[0]
        
        # Get cameras from scene.sensors dict
        sensors = scene.sensors if hasattr(scene, 'sensors') and isinstance(scene.sensors, dict) else {}
        
        # Try to fetch robot articulation for applied targets
        robot_articulation = None
        try:
            robot_articulation = env.unwrapped.scene._articulations.get("robot")
        except Exception:
            robot_articulation = None

        for env_idx in range(num_envs):
            # Initialize episode storage
            if env_idx not in self.episodes:
                self.episodes[env_idx] = {
                    'actions': [],
                    'frames': {'cam0': [], 'cam1': [], 'cam2': []},
                    'applied_targets': []  # effective joint targets if available
                }
            
            # Store action
            action = actions[env_idx].detach().cpu().numpy()
            self.episodes[env_idx]['actions'].append(action)
            
            # Capture frames if needed
            if store_frame:
                # Camera 0: top camera (main view)
                if self.enable_top and 'top_camera' in sensors:
                    sensors['top_camera'].update(dt=scene.physics_dt)
                    frame = self._process_frame(sensors['top_camera'].data.output["rgb"][env_idx])
                    self.episodes[env_idx]['frames']['cam0'].append(frame)
                
                # Camera 1: wrist camera
                if self.enable_wrist and 'wrist_camera' in sensors:
                    sensors['wrist_camera'].update(dt=scene.physics_dt)
                    frame = self._process_frame(sensors['wrist_camera'].data.output["rgb"][env_idx])
                    self.episodes[env_idx]['frames']['cam1'].append(frame)
                
                # Camera 2: side camera
                if self.enable_side and 'side_camera' in sensors:
                    sensors['side_camera'].update(dt=scene.physics_dt)
                    frame = self._process_frame(sensors['side_camera'].data.output["rgb"][env_idx])
                    self.episodes[env_idx]['frames']['cam2'].append(frame)

            # Capture applied joint targets post-step (if available)
            try:
                if robot_articulation is not None and hasattr(robot_articulation, 'data'):
                    rd = robot_articulation.data
                    vec = None
                    for key in [
                        'joint_pos_target', 'dof_pos_target', 'joint_targets',
                        'drive_target', 'actuated_dof_pos_target'
                    ]:
                        if hasattr(rd, key):
                            val = getattr(rd, key)
                            # Expect shape (N, D)
                            if val is not None:
                                vec = val[env_idx]
                                break
                    if vec is None and hasattr(rd, 'joint_pos'):
                        vec = rd.joint_pos[env_idx]
                    if vec is not None:
                        vec_np = vec.detach().cpu().numpy() if hasattr(vec, 'detach') else np.array(vec)
                        self.episodes[env_idx]['applied_targets'].append(vec_np)
            except Exception:
                pass

    def write(self, dones: np.ndarray, terminated: np.ndarray, successes: np.ndarray, 
              save_to_video: bool = True, **kwargs):
        """Save completed episodes to disk."""
        for env_idx, is_done in enumerate(dones):
            if not is_done or env_idx not in self.episodes:
                continue
            
            # Skip terminated episodes if requested
            if not self.keep_terminated and terminated[env_idx]:
                self.episodes[env_idx] = {'actions': [], 'frames': {'cam0': [], 'cam1': [], 'cam2': []}}
                continue
            
            data = self.episodes[env_idx]
            if not data['actions']:
                self.episodes[env_idx] = {'actions': [], 'frames': {'cam0': [], 'cam1': [], 'cam2': []}}
                continue
            
            # Create episode directory
            episode_dir = self.base_dir / f"episode_{self.episode_counter:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            # Save videos
            if save_to_video:
                for cam_name, frames in data['frames'].items():
                    if frames:
                        cam_dir = episode_dir / f"camera_{cam_name[-1]}"  # cam0 -> camera_0
                        cam_dir.mkdir(exist_ok=True)
                        video_path = cam_dir / f"episode_{self.episode_counter:03d}.mp4"
                        iio.imwrite(video_path, frames, fps=self.fps)
                        print(f"  ✓ Saved {cam_name}: {video_path.relative_to(self.base_dir)}")
            
            # Save h5 file
            actions_array = np.stack(data['actions'], axis=0)
            h5_path = episode_dir / f"episode_{self.episode_counter:03d}.h5"
            
            with h5py.File(h5_path, "w") as f:
                # Save actions
                f.create_dataset("action", data=actions_array)
                # Save applied joint targets if recorded
                if data.get('applied_targets'):
                    applied_arr = np.stack(data['applied_targets'], axis=0)
                    f.create_dataset("applied_joint_target", data=applied_arr)
                
                # Save frames as (T, N_cams, H, W, C)
                cam_frames = []
                for cam_name in ['cam0', 'cam1', 'cam2']:
                    if data['frames'][cam_name]:
                        frames = np.stack(data['frames'][cam_name], axis=0)  # (T, H, W, C)
                        cam_frames.append(frames[:, None, ...])  # (T, 1, H, W, C)
                
                if cam_frames:
                    color = np.concatenate(cam_frames, axis=1)  # (T, N_cams, H, W, C)
                    f.create_dataset("color", data=color)
                
                # Save metadata
                f.attrs['success'] = bool(successes[env_idx])
                f.attrs['terminated'] = bool(terminated[env_idx])
                f.attrs['num_frames'] = len(actions_array)
                f.attrs['fps'] = self.fps
            
            # Update metadata.json
            metadata = {"num_timesteps": [], "num_episodes": self.episode_counter + 1}
            if (self.base_dir / "metadata.json").exists():
                with open(self.base_dir / "metadata.json", "r") as f:
                    old_meta = json.load(f)
                    metadata["num_timesteps"] = old_meta.get("num_timesteps", [])
            metadata["num_timesteps"].append(len(actions_array))
            
            with open(self.base_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Reset this environment's data
            self.episodes[env_idx] = {'actions': [], 'frames': {'cam0': [], 'cam1': [], 'cam2': []}}
            self.episode_counter += 1

    def _process_frame(self, rgb_tensor: torch.Tensor) -> np.ndarray:
        """Convert camera output to uint8 numpy array (H, W, 3)."""
        rgb = rgb_tensor.detach().cpu().numpy()
        
        # CHW -> HWC if needed
        if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
            rgb = np.transpose(rgb, (1, 2, 0))
        
        # Drop alpha channel
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        
        # Convert to uint8
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        
        return rgb
