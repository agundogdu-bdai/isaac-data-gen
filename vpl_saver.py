#!/usr/bin/env python3
# flake8: noqa
"""VPL Dataset Saver - saves RGB frames, actions, proprioception, and MP4 videos."""

import numpy as np
import torch
from pathlib import Path
import imageio.v3 as iio
import h5py
import json


class VPLSaver:
    """Saves data in VPL dataset format with per-environment MP4 videos.

    Supports up to three camera streams when enabled:
      - camera_0: tiled overview camera (per-env view via TiledCamera)
      - camera_1: wrist-mounted camera (follows end-effector)
      - camera_2: top-view camera (static top-down view)
    """

    def __init__(
        self,
        base_dir: str,
        keep_terminated: bool = False,
        fps: int = 30,
        initial_timesteps_to_discard: int = 0,
        enable_wrist_camera: bool = False,
        wrist_cam_offset: list | tuple = (0.05, 0.0, 0.05),
        wrist_cam_look_offset: list | tuple = (0.20, 0.0, 0.00),
        enable_top_camera: bool = False,
        top_cam_offset: list | tuple = (0.0, 0.0, 3.0),
        top_tgt_offset: list | tuple = (0.4, 0.0, 0.5),
        enable_side_camera: bool = False,
        wrist_auto_follow: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.keep_terminated = keep_terminated
        self.fps = fps
        self.initial_timesteps_to_discard = initial_timesteps_to_discard
        self.enable_wrist_camera = enable_wrist_camera
        self.wrist_cam_offset = np.asarray(wrist_cam_offset, dtype=np.float32)
        self.wrist_cam_look_offset = np.asarray(wrist_cam_look_offset, dtype=np.float32)
        self.enable_top_camera = enable_top_camera
        self.top_cam_offset = np.asarray(top_cam_offset, dtype=np.float32)
        self.top_tgt_offset = np.asarray(top_tgt_offset, dtype=np.float32)
        self.enable_side_camera = enable_side_camera
        self.wrist_auto_follow = wrist_auto_follow
        
        # Storage for each environment
        self.episode_data = {}
        self.episode_counter = 0
        # Simple metadata info for the dataset
        self.metadata_info = {"num_timesteps": [], "num_episodes": 0}
        # Robot-specific state keys to capture (if available from raw observations)
        self.robot_keys_to_save = [
            "joint_pos",
            "joint_vel",
            "ee_position",
            "ee_rotation",
            "robot_dof_targets",
        ]
        self._robot_keys_warned = False
        
    def store(self, actions: torch.Tensor, env, store_frame: bool = True):
        """Store one timestep of data for all environments.
        If store_frame is False, only actions/proprio/timestamps are stored (no RGB frame).
        """
        num_envs = actions.shape[0]
        
        # Get camera data - support both tiled_camera (legacy) and separate cameras (new)
        scene = env.unwrapped.scene
        cam = None
        use_separate_cameras = False
        
        # Try to find tiled_camera (legacy support)
        if hasattr(scene, 'tiled_camera'):
            cam = scene.tiled_camera
        elif hasattr(scene, 'sensors'):
            # Manager-based envs store sensors separately (can be dict or object)
            if isinstance(scene.sensors, dict) and 'tiled_camera' in scene.sensors:
                cam = scene.sensors['tiled_camera']
            elif hasattr(scene.sensors, 'tiled_camera'):
                cam = scene.sensors.tiled_camera
        
        # If no tiled_camera, use separate cameras (new camera environment)
        if cam is None:
            use_separate_cameras = True
            # Will populate rgb_all from top_camera or wrist_camera below
            rgb_all = None
            intrinsic_all = None
            extrinsic_all = None
        else:
            # Update tiled camera only if we intend to store frames on this step
            if store_frame:
                cam.update(dt=scene.physics_dt)
                rgb_all = cam.data.output["rgb"]  # [num_envs, H, W, 3] or [num_envs, 3, H, W]
                # Intrinsics per env (assumed constant across time)
                if hasattr(cam.data, "intrinsic_matrices"):
                    intrinsic_all = cam.data.intrinsic_matrices
                else:
                    intrinsic_all = None
                # Try to get extrinsics (4x4) if available on sensor data
                extrinsic_all = getattr(cam.data, "extrinsic_matrices", None)
            else:
                rgb_all = None
                intrinsic_all = None
                extrinsic_all = None
        
        # Wrist camera: get camera reference
        wrist_rgb_all = None
        wrist_intrinsic_all = None
        wrist_extrinsic_all = None
        wrist_cam = None
        if self.enable_wrist_camera:
            if hasattr(scene, 'wrist_camera'):
                wrist_cam = scene.wrist_camera
            elif hasattr(scene, 'sensors') and isinstance(scene.sensors, dict):
                wrist_cam = scene.sensors.get('wrist_camera')
        
        # Top camera: get camera reference
        top_rgb_all = None
        top_intrinsic_all = None
        top_extrinsic_all = None
        top_cam = None
        if self.enable_top_camera:
            if hasattr(scene, 'top_camera'):
                top_cam = scene.top_camera
            elif hasattr(scene, 'sensors') and isinstance(scene.sensors, dict):
                top_cam = scene.sensors.get('top_camera')

        # Get observations and robot state
        obs_dict = env.unwrapped.observation_manager.compute()
        robot_articulation = env.unwrapped.scene._articulations.get("robot")
        
        # Update wrist camera pose from end-effector (only if not auto-following)
        if self.enable_wrist_camera and wrist_cam is not None:
            if not self.wrist_auto_follow:
                # Manual positioning: compute camera poses from EE frame
                # Get EE body pose (last body is typically the end-effector)
                ee_pos = robot_articulation.data.body_pos_w[:, -1, :]
                ee_rot = robot_articulation.data.body_quat_w[:, -1, :]
                
                # Compute camera poses from EE frame
                positions, targets = self._compute_wrist_camera_poses(ee_pos, ee_rot)
                wrist_cam.set_world_poses_from_view(positions, targets)
            
            # Update camera to capture frame (works for both manual and auto-follow)
            wrist_cam.update(dt=scene.physics_dt)
            
            if store_frame:
                wrist_rgb_all = wrist_cam.data.output["rgb"]
                wrist_intrinsic_all = wrist_cam.data.intrinsic_matrices
                wrist_extrinsic_all = getattr(wrist_cam.data, "extrinsic_matrices", None)
        
        # Update top camera (static position, just update for frame capture)
        if self.enable_top_camera and top_cam is not None:
            top_cam.update(dt=scene.physics_dt)
            
            if store_frame:
                top_rgb_all = top_cam.data.output["rgb"]
                top_intrinsic_all = top_cam.data.intrinsic_matrices
                top_extrinsic_all = getattr(top_cam.data, "extrinsic_matrices", None)
        
        # Side camera: get camera reference
        side_rgb_all = None
        side_intrinsic_all = None
        side_extrinsic_all = None
        side_cam = None
        if self.enable_side_camera:
            if hasattr(scene, 'side_camera'):
                side_cam = scene.side_camera
            elif hasattr(scene, 'sensors') and isinstance(scene.sensors, dict):
                side_cam = scene.sensors.get('side_camera')
        
        # Update side camera (static position, just update for frame capture)
        if self.enable_side_camera and side_cam is not None:
            side_cam.update(dt=scene.physics_dt)
            
            if store_frame:
                side_rgb_all = side_cam.data.output["rgb"]
                side_intrinsic_all = side_cam.data.intrinsic_matrices
                side_extrinsic_all = getattr(side_cam.data, "extrinsic_matrices", None)
        
        # If no tiled camera is available, use top camera as main camera (camera_0)
        # This ensures rgb_frames is populated even without tiled_camera
        # Track if we're using top camera as main to avoid duplication
        top_is_main_camera = False
        wrist_is_main_camera = False
        if use_separate_cameras and rgb_all is None and store_frame:
            if top_rgb_all is not None:
                rgb_all = top_rgb_all
                intrinsic_all = top_intrinsic_all
                extrinsic_all = top_extrinsic_all
                top_is_main_camera = True
                # Clear top camera data to avoid duplication in camera_2
                top_rgb_all = None
                top_intrinsic_all = None
                top_extrinsic_all = None
            elif wrist_rgb_all is not None:
                # Fallback to wrist camera if no top camera
                rgb_all = wrist_rgb_all
                intrinsic_all = wrist_intrinsic_all
                extrinsic_all = wrist_extrinsic_all
                wrist_is_main_camera = True
                # Clear wrist camera data to avoid duplication in camera_1
                wrist_rgb_all = None
                wrist_intrinsic_all = None
                wrist_extrinsic_all = None

        # Store for each environment
        for env_idx in range(num_envs):
            if env_idx not in self.episode_data:
                self.episode_data[env_idx] = {
                    'rgb_frames': [],
                    'wrist_rgb_frames': [],
                    'top_rgb_frames': [],
                    'side_rgb_frames': [],
                    'actions': [],
                    'proprio': [],
                    'timestamps': [],
                    'timestep': 0,
                }
            
            # Skip initial timesteps if requested
            if self.episode_data[env_idx]['timestep'] < self.initial_timesteps_to_discard:
                self.episode_data[env_idx]['timestep'] += 1
                continue
            
            # Optionally process and store RGB (tiled camera - only if available)
            if store_frame and rgb_all is not None:
                rgb = rgb_all[env_idx].detach().cpu().numpy()
                if rgb.ndim == 3 and rgb.shape[0] in (3, 4):  # CHW -> HWC
                    rgb = np.transpose(rgb, (1, 2, 0))
                if rgb.shape[-1] == 4:  # Drop alpha
                    rgb = rgb[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                self.episode_data[env_idx]['rgb_frames'].append(rgb)
                # Store intrinsics once
                if 'intrinsics' not in self.episode_data[env_idx] and intrinsic_all is not None:
                    self.episode_data[env_idx]['intrinsics'] = intrinsic_all[env_idx].detach().cpu().numpy()
                # Store extrinsics per stored frame if available
                if extrinsic_all is not None:
                    if 'extrinsics' not in self.episode_data[env_idx]:
                        self.episode_data[env_idx]['extrinsics'] = []
                    self.episode_data[env_idx]['extrinsics'].append(extrinsic_all[env_idx].detach().cpu().numpy())

            # Optionally process and store wrist RGB
            if store_frame and wrist_rgb_all is not None:
                rgb_w = wrist_rgb_all[env_idx].detach().cpu().numpy()
                if rgb_w.ndim == 3 and rgb_w.shape[0] in (3, 4):
                    rgb_w = np.transpose(rgb_w, (1, 2, 0))
                if rgb_w.shape[-1] == 4:
                    rgb_w = rgb_w[..., :3]
                if rgb_w.dtype != np.uint8:
                    rgb_w = np.clip(rgb_w * 255.0, 0, 255).astype(np.uint8)
                self.episode_data[env_idx]['wrist_rgb_frames'].append(rgb_w)
                if 'wrist_intrinsics' not in self.episode_data[env_idx] and wrist_intrinsic_all is not None:
                    self.episode_data[env_idx]['wrist_intrinsics'] = wrist_intrinsic_all[env_idx].detach().cpu().numpy()
                if wrist_extrinsic_all is not None:
                    if 'wrist_extrinsics' not in self.episode_data[env_idx]:
                        self.episode_data[env_idx]['wrist_extrinsics'] = []
                    self.episode_data[env_idx]['wrist_extrinsics'].append(wrist_extrinsic_all[env_idx].detach().cpu().numpy())
            
            # Optionally process and store top RGB
            if store_frame and top_rgb_all is not None:
                rgb_t = top_rgb_all[env_idx].detach().cpu().numpy()
                if rgb_t.ndim == 3 and rgb_t.shape[0] in (3, 4):
                    rgb_t = np.transpose(rgb_t, (1, 2, 0))
                if rgb_t.shape[-1] == 4:
                    rgb_t = rgb_t[..., :3]
                if rgb_t.dtype != np.uint8:
                    rgb_t = np.clip(rgb_t * 255.0, 0, 255).astype(np.uint8)
                self.episode_data[env_idx]['top_rgb_frames'].append(rgb_t)
                if 'top_intrinsics' not in self.episode_data[env_idx] and top_intrinsic_all is not None:
                    self.episode_data[env_idx]['top_intrinsics'] = top_intrinsic_all[env_idx].detach().cpu().numpy()
                if top_extrinsic_all is not None:
                    if 'top_extrinsics' not in self.episode_data[env_idx]:
                        self.episode_data[env_idx]['top_extrinsics'] = []
                    self.episode_data[env_idx]['top_extrinsics'].append(top_extrinsic_all[env_idx].detach().cpu().numpy())
            
            # Optionally process and store side RGB
            if store_frame and side_rgb_all is not None:
                rgb_s = side_rgb_all[env_idx].detach().cpu().numpy()
                if rgb_s.ndim == 3 and rgb_s.shape[0] in (3, 4):
                    rgb_s = np.transpose(rgb_s, (1, 2, 0))
                if rgb_s.shape[-1] == 4:
                    rgb_s = rgb_s[..., :3]
                if rgb_s.dtype != np.uint8:
                    rgb_s = np.clip(rgb_s * 255.0, 0, 255).astype(np.uint8)
                self.episode_data[env_idx]['side_rgb_frames'].append(rgb_s)
                if 'side_intrinsics' not in self.episode_data[env_idx] and side_intrinsic_all is not None:
                    self.episode_data[env_idx]['side_intrinsics'] = side_intrinsic_all[env_idx].detach().cpu().numpy()
                if side_extrinsic_all is not None:
                    if 'side_extrinsics' not in self.episode_data[env_idx]:
                        self.episode_data[env_idx]['side_extrinsics'] = []
                    self.episode_data[env_idx]['side_extrinsics'].append(side_extrinsic_all[env_idx].detach().cpu().numpy())
            
            self.episode_data[env_idx]['actions'].append(actions[env_idx].detach().cpu().numpy())
            
            # Get proprioception (flatten all observations)
            proprio = []
            for key, value in obs_dict.items():
                if isinstance(value, torch.Tensor):
                    proprio.append(value[env_idx].detach().cpu().numpy().flatten())
            proprio = np.concatenate(proprio) if proprio else np.array([])
            self.episode_data[env_idx]['proprio'].append(proprio)

            # Collect robot states from articulation
            if 'robot' not in self.episode_data[env_idx]:
                self.episode_data[env_idx]['robot'] = {k: [] for k in self.robot_keys_to_save}
            
            # Get joint states from articulation
            jp = robot_articulation.data.joint_pos[env_idx].detach().cpu().numpy()
            jv = robot_articulation.data.joint_vel[env_idx].detach().cpu().numpy()
            self.episode_data[env_idx]['robot']['joint_pos'].append(jp)
            self.episode_data[env_idx]['robot']['joint_vel'].append(jv)
            
            # Get EE pose from body states
            ee_pos_np = robot_articulation.data.body_pos_w[env_idx, -1, :].detach().cpu().numpy()
            ee_rot_np = robot_articulation.data.body_quat_w[env_idx, -1, :].detach().cpu().numpy()
            self.episode_data[env_idx]['robot']['ee_position'].append(ee_pos_np)
            self.episode_data[env_idx]['robot']['ee_rotation'].append(ee_rot_np)
            
            self.episode_data[env_idx]['timestamps'].append(self.episode_data[env_idx]['timestep'])
            self.episode_data[env_idx]['timestep'] += 1
    
    def write(
        self,
        dones: np.ndarray,
        terminated: np.ndarray,
        successes: np.ndarray,
        point_cloud_only: bool = False,
        save_to_video: bool = True,
    ):
        """Write episodes to disk for environments that are done.
        Each environment is saved as a separate episode.
        """
        for env_idx, is_done in enumerate(dones):
            if not is_done:
                continue
            
            if env_idx not in self.episode_data:
                continue
            
            # Skip if terminated and we don't keep terminated episodes
            if not self.keep_terminated and terminated[env_idx]:
                # Properly reset episode data for next episode
                self.episode_data[env_idx] = {
                    'rgb_frames': [],
                    'wrist_rgb_frames': [],
                    'top_rgb_frames': [],
                    'side_rgb_frames': [],
                    'actions': [],
                    'proprio': [],
                    'timestamps': [],
                    'timestep': 0,
                }
                continue
            
            data = self.episode_data[env_idx]
            if len(data['rgb_frames']) == 0:
                # Properly reset episode data for next episode
                self.episode_data[env_idx] = {
                    'rgb_frames': [],
                    'wrist_rgb_frames': [],
                    'top_rgb_frames': [],
                    'side_rgb_frames': [],
                    'actions': [],
                    'proprio': [],
                    'timestamps': [],
                    'timestep': 0,
                }
                continue
            
            # Each environment gets its own episode directory
            episode_dir = self.base_dir / f"episode_{self.episode_counter:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            # Camera output directories (flattened)
            camera0_dir = episode_dir / "camera_0"
            camera0_dir.mkdir(parents=True, exist_ok=True)
            camera1_dir = episode_dir / "camera_1"
            if 'wrist_rgb_frames' in data and len(data['wrist_rgb_frames']) > 0:
                camera1_dir.mkdir(parents=True, exist_ok=True)
            camera2_dir = episode_dir / "camera_2"
            if 'side_rgb_frames' in data and len(data['side_rgb_frames']) > 0:
                camera2_dir.mkdir(parents=True, exist_ok=True)
            
            # Save videos
            if save_to_video and len(data['rgb_frames']) > 0:
                video_path = camera0_dir / f"episode_{self.episode_counter:03d}.mp4"
                iio.imwrite(video_path, data['rgb_frames'], fps=self.fps)
                print(f"  ✓ Saved episode {self.episode_counter}: {video_path.relative_to(self.base_dir)}")

            if save_to_video and 'wrist_rgb_frames' in data and len(data['wrist_rgb_frames']) > 0:
                video_wrist_path = camera1_dir / f"episode_{self.episode_counter:03d}.mp4"
                iio.imwrite(video_wrist_path, data['wrist_rgb_frames'], fps=self.fps)
                print(f"  ✓ Saved episode {self.episode_counter} (wrist): {video_wrist_path.relative_to(self.base_dir)}")
            
            if save_to_video and 'side_rgb_frames' in data and len(data['side_rgb_frames']) > 0:
                video_side_path = camera2_dir / f"episode_{self.episode_counter:03d}.mp4"
                iio.imwrite(video_side_path, data['side_rgb_frames'], fps=self.fps)
                print(f"  ✓ Saved episode {self.episode_counter} (side): {video_side_path.relative_to(self.base_dir)}")
            
            # Assemble arrays for H5 only
            proprio_array = np.stack(data['proprio'], axis=0) if data['proprio'] else None
            actions_array = np.stack(data['actions'], axis=0)
            timestamps_array = np.array(data['timestamps'])
            
            # Save metadata
            metadata = {
                'env_idx': env_idx,  # Original env index for reference
                'success': bool(successes[env_idx]),
                'terminated': bool(terminated[env_idx]),
                'num_frames': len(data['rgb_frames']),
            }
            # Keep minimal metadata in H5 attributes; no .npy metadata files

            # Save consolidated HDF5 file (VPL-style minimal subset)
            h5_path = episode_dir / f"episode_{self.episode_counter:03d}.h5"
            with h5py.File(h5_path, "w") as f:
                # action (align with reference naming)
                f.create_dataset("action", data=actions_array)
                # proprio (if available)
                if data['proprio']:
                    f.create_dataset("proprio", data=proprio_array)
                # timestamps
                f.create_dataset("timestamps", data=timestamps_array)
                # color frames: concatenate all cameras into single array (T, N_cams, H, W, C)
                if len(data['rgb_frames']) > 0:
                    frames_cam0 = np.stack(data['rgb_frames'], axis=0)  # (T, H, W, C)
                    frames_cam0 = frames_cam0[:, None, ...]  # (T, 1, H, W, C)
                    
                    cam_list = [frames_cam0]
                    
                    if 'wrist_rgb_frames' in data and len(data['wrist_rgb_frames']) > 0:
                        frames_cam1 = np.stack(data['wrist_rgb_frames'], axis=0)  # (T, H, W, C)
                        frames_cam1 = frames_cam1[:, None, ...]  # (T, 1, H, W, C)
                        cam_list.append(frames_cam1)
                    
                    if 'top_rgb_frames' in data and len(data['top_rgb_frames']) > 0:
                        frames_cam2 = np.stack(data['top_rgb_frames'], axis=0)  # (T, H, W, C)
                        frames_cam2 = frames_cam2[:, None, ...]  # (T, 1, H, W, C)
                        cam_list.append(frames_cam2)
                    
                    if 'side_rgb_frames' in data and len(data['side_rgb_frames']) > 0:
                        frames_cam3 = np.stack(data['side_rgb_frames'], axis=0)  # (T, H, W, C)
                        frames_cam3 = frames_cam3[:, None, ...]  # (T, 1, H, W, C)
                        cam_list.append(frames_cam3)
                    
                    # Concatenate along camera dimension: (T, N_cams, H, W, C)
                    frames = np.concatenate(cam_list, axis=1)
                    
                    f.create_dataset("color", data=frames)
                # intrinsics: concatenate all cameras (N_cams, 3, 3)
                if 'intrinsics' in data:
                    intr_list = [data['intrinsics'][None, ...]]  # (1, 3, 3)
                    
                    if 'wrist_intrinsics' in data:
                        intr_cam1 = data['wrist_intrinsics'][None, ...]  # (1, 3, 3)
                        intr_list.append(intr_cam1)
                    
                    if 'top_intrinsics' in data:
                        intr_cam2 = data['top_intrinsics'][None, ...]  # (1, 3, 3)
                        intr_list.append(intr_cam2)
                    
                    if 'side_intrinsics' in data:
                        intr_cam3 = data['side_intrinsics'][None, ...]  # (1, 3, 3)
                        intr_list.append(intr_cam3)
                    
                    intrinsics = np.concatenate(intr_list, axis=0)  # (N_cams, 3, 3)
                    f.create_dataset("intrinsics", data=intrinsics)
                
                # extrinsics per frame: concatenate all cameras (T, N_cams, 4, 4)
                if 'extrinsics' in data and len(data['extrinsics']) > 0:
                    extr_cam0 = np.stack(data['extrinsics'], axis=0)  # (T, 4, 4)
                    extr_cam0 = extr_cam0[:, None, ...]  # (T, 1, 4, 4)
                    
                    extr_list = [extr_cam0]
                    
                    if 'wrist_extrinsics' in data and len(data['wrist_extrinsics']) > 0:
                        extr_cam1 = np.stack(data['wrist_extrinsics'], axis=0)  # (T, 4, 4)
                        extr_cam1 = extr_cam1[:, None, ...]  # (T, 1, 4, 4)
                        extr_list.append(extr_cam1)
                    
                    if 'top_extrinsics' in data and len(data['top_extrinsics']) > 0:
                        extr_cam2 = np.stack(data['top_extrinsics'], axis=0)  # (T, 4, 4)
                        extr_cam2 = extr_cam2[:, None, ...]  # (T, 1, 4, 4)
                        extr_list.append(extr_cam2)
                    
                    if 'side_extrinsics' in data and len(data['side_extrinsics']) > 0:
                        extr_cam3 = np.stack(data['side_extrinsics'], axis=0)  # (T, 4, 4)
                        extr_cam3 = extr_cam3[:, None, ...]  # (T, 1, 4, 4)
                        extr_list.append(extr_cam3)
                    
                    extrinsics = np.concatenate(extr_list, axis=1)  # (T, N_cams, 4, 4)
                    
                    f.create_dataset("extrinsics", data=extrinsics)
                # robot-only states at top-level (if present)
                if 'robot' in data and isinstance(data['robot'], dict):
                    for key, series in data['robot'].items():
                        if series:
                            try:
                                arr = np.stack(series, axis=0)
                            except Exception:
                                # fall back to array conversion if shapes vary
                                arr = np.array(series, dtype=object)
                            f.create_dataset(key, data=arr)
                    # also provide a single concatenated 'state' vector per step if possible
                    concatenated_parts = []
                    for k in self.robot_keys_to_save:
                        series = data['robot'].get(k, [])
                        if series:
                            try:
                                arr = np.stack(series, axis=0)
                                arr = arr.reshape(arr.shape[0], -1)
                                concatenated_parts.append(arr)
                            except Exception:
                                pass
                    if concatenated_parts:
                        try:
                            state_arr = np.concatenate(concatenated_parts, axis=1)
                            f.create_dataset("state", data=state_arr)
                        except Exception:
                            pass
                # basic metadata as attributes
                f.attrs['env_idx'] = int(env_idx)
                f.attrs['success'] = bool(successes[env_idx])
                f.attrs['terminated'] = bool(terminated[env_idx])
                f.attrs['num_frames'] = int(metadata['num_frames'])
                f.attrs['fps'] = int(self.fps)

            # Update and write simple metadata.json at dataset root
            num_timesteps = actions_array.shape[0]
            self.metadata_info["num_timesteps"].append(int(num_timesteps))
            self.metadata_info["num_episodes"] += 1
            with open(self.base_dir / "metadata.json", "w") as f:
                json.dump(self.metadata_info, f, indent=2)
            
            # Clear this env's data and reset for next episode
            self.episode_data[env_idx] = {
                'rgb_frames': [],
                'wrist_rgb_frames': [],
                'top_rgb_frames': [],
                'side_rgb_frames': [],
                'actions': [],
                'proprio': [],
                'timestamps': [],
                'timestep': 0,
            }
            
            # Increment counter for EACH environment (each is a separate episode)
            self.episode_counter += 1

    def _compute_wrist_camera_poses(self, ee_pos: torch.Tensor, ee_rot: torch.Tensor):
        """Compute world positions and targets for wrist camera from EE pose.

        Args:
            ee_pos: (N, 3) tensor of end-effector positions
            ee_rot: (N, 4) tensor of end-effector quaternions (x, y, z, w)

        Returns:
            positions: (N, 3) tensor
            targets: (N, 3) tensor
        """
        # Convert to CPU tensors for simple math without grad
        pos = ee_pos.detach()
        quat = ee_rot.detach()
        
        # Ensure correct shape
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        
        # Build rotation matrices (N, 3, 3)
        R = self._quat_to_rot_matrices(quat)
        
        # Offsets - make sure to match batch size
        num_envs = pos.shape[0]
        offs = torch.tensor(self.wrist_cam_offset, device=pos.device, dtype=pos.dtype).unsqueeze(0).expand(num_envs, -1)
        look = torch.tensor(self.wrist_cam_look_offset, device=pos.device, dtype=pos.dtype).unsqueeze(0).expand(num_envs, -1)
        
        # Rotate offsets by R and add to position
        cam_pos = pos + torch.bmm(R, offs.unsqueeze(-1)).squeeze(-1)
        cam_tgt = pos + torch.bmm(R, look.unsqueeze(-1)).squeeze(-1)
        return cam_pos, cam_tgt

    @staticmethod
    def _quat_to_rot_matrices(quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Quaternion to rotation matrix for a batch of quaternions.

        Expects XYZW ordering.
        Returns tensor of shape (N, 3, 3).
        """
        x = quat_xyzw[:, 0]
        y = quat_xyzw[:, 1]
        z = quat_xyzw[:, 2]
        w = quat_xyzw[:, 3]
        # Normalize to be safe
        norm = torch.clamp(torch.sqrt(x * x + y * y + z * z + w * w), min=1e-8)
        x = x / norm
        y = y / norm
        z = z / norm
        w = w / norm
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        R = torch.empty((quat_xyzw.shape[0], 3, 3), dtype=quat_xyzw.dtype, device=quat_xyzw.device)
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)
        return R

