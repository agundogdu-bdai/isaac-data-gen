# Copyright (c) 2024-2025 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import json
from typing import Any
from os.path import isdir, join

import h5py
import torch
import numpy as np
from einops import rearrange


from isaaclab.utils.math import matrix_from_quat


class VPLSaver:

    def __init__(self, base_dir: str, initial_timesteps_to_discard: int = 0):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.episode_data = {}
        self.metadata_info = {"num_timesteps": [], "num_episodes": 0}
        self.observations_keys = ["joint_pos", "joint_vel", "ee_position", "ee_rotation"]
        self.initial_timesteps_to_discard = initial_timesteps_to_discard

    def store(self, actions: torch.tensor, env) -> None:
        for scene_index in range(actions.shape[0]):
            if scene_index not in self.episode_data:
                self.episode_data[scene_index] = {"action": []}

            self.episode_data[scene_index]["action"].append(actions[scene_index].detach().cpu())

            # Get concatenated observations (as the policy expects them)
            obs_raw = env.get_observations()
            
            # Handle tuple return from get_observations (obs_dict, info)
            if isinstance(obs_raw, tuple):
                obs_raw = obs_raw[0]
            
            # Extract policy observations - could be dict or direct tensor from RslRlVecEnvWrapper
            if isinstance(obs_raw, dict):
                policy_obs_concat = obs_raw["policy"][scene_index]  # shape: (total_dim,)
            else:
                # obs_raw is already the policy tensor from RslRlVecEnvWrapper
                policy_obs_concat = obs_raw[scene_index]  # shape: (total_dim,)
            
            # Get metadata to split the observations
            obs_manager = env.unwrapped.observation_manager
            term_names = obs_manager.active_terms["policy"]
            term_dims = obs_manager.group_obs_term_dim["policy"]
            
            # Split the concatenated observations into individual terms
            individual_obs = {}
            start_idx = 0
            for term_name, term_dim in zip(term_names, term_dims):
                # Calculate total size of this term
                term_size = int(np.prod(term_dim))
                # Extract this term's data
                individual_obs[term_name] = policy_obs_concat[start_idx:start_idx + term_size].clone().detach().cpu()
                start_idx += term_size
            
            # Print available observations (only first time for debugging)
            if scene_index == 0 and len(self.episode_data[scene_index]["action"]) == 1:
                print("Available observation terms:", list(individual_obs.keys()))
                print("Term dimensions:", {k: v.shape for k, v in individual_obs.items()})
            
            # Now save individual observation terms
            for key in self.observations_keys:
                if key not in self.episode_data[scene_index].keys():
                    self.episode_data[scene_index][key] = []
                if key in individual_obs:
                    self.episode_data[scene_index][key].append(individual_obs[key])

            # Get camera data
            for key in env.unwrapped.scene.sensors.keys():
                if "camera" not in key:
                    continue

                if "color" not in self.episode_data[scene_index].keys():
                    self.episode_data[scene_index]["color"] = {}

                if key not in self.episode_data[scene_index]["color"].keys():
                    self.episode_data[scene_index]["color"][key] = []

                # Get RGB image and discard Alpha channel
                im = env.unwrapped.scene.sensors[key].data.output["rgb"][scene_index]
                # Ensure we only keep RGB channels (3 channels) and discard alpha if present
                if im.shape[-1] >= 3:
                    im = im[:, :, :3].clone()

                self.episode_data[scene_index]["color"][key].append(im.detach().cpu())

    def write(
        self,
        dones,
        compression_filter: str | None = None,
        compression_opts: int | None = None,
        extra_trajectory_level_info: dict[str, Any] = {},
    ) -> int:
        """Write episode data to disk when episodes terminate.
        
        All episodes that are done will be saved to disk.
        
        Args:
            dones: Boolean tensor indicating which environments finished
            compression_filter: HDF5 compression filter (e.g., "gzip")
            compression_opts: HDF5 compression level (e.g., 4)
            extra_trajectory_level_info: Additional metadata to save as attributes
            
        Returns:
            Total number of episodes saved so far
        """
        num_episodes_saved_this_call = 0
        # Hard truncation length for saved episodes
        MAX_STEPS = 84
        
        for i in range(dones.shape[0]):
            if not dones[i]:
                continue
                
            episode_index = self.metadata_info["num_episodes"]
            
            # Check if we have any data to save
            if "action" not in self.episode_data[i] or len(self.episode_data[i]["action"]) == 0:
                print(f"Warning: Episode {i} has no action data to save. Skipping.")
                self.episode_data[i] = {"action": []}
                continue

            episode_dir = join(self.base_dir, f"episode_{episode_index}")
            if not isdir(episode_dir):
                os.makedirs(episode_dir)

            with h5py.File(join(episode_dir, f"episode_{episode_index}.h5"), "w") as f_writer:
                # Compute slice bounds after initial discard
                start_idx = self.initial_timesteps_to_discard
                end_idx = start_idx + MAX_STEPS if MAX_STEPS is not None else None
                for k in self.episode_data[i].keys():
                    if k in self.observations_keys:
                        if len(self.episode_data[i][k]) == 0:  # Skip if empty
                            continue
                        dataset_array = torch.stack(
                            self.episode_data[i][k][start_idx:end_idx]
                        ).cpu().numpy()
                        f_writer.create_dataset(
                            k, data=dataset_array, compression=compression_filter, compression_opts=compression_opts
                        )
                    elif k == "action":
                        if len(self.episode_data[i][k]) == 0:  # Skip if empty
                            continue
                        dataset_array = torch.stack(
                            self.episode_data[i][k][start_idx:end_idx]
                        ).cpu().numpy()
                        f_writer.create_dataset(
                            k, data=dataset_array, compression=compression_filter, compression_opts=compression_opts
                        )
                    elif k in ["eef_pose", "gripper_width"]:
                        if len(self.episode_data[i][k]) == 0:  # Skip if empty
                            continue
                        dataset_array = np.array(
                            self.episode_data[i][k][start_idx:end_idx]
                        )
                        f_writer.create_dataset(
                            k, data=dataset_array, compression=compression_filter, compression_opts=compression_opts
                        )
                    elif k == "color":
                        if len(self.episode_data[i][k]) == 0:  # Skip if empty
                            continue
                        dataset_array = torch.stack(
                            [torch.stack(value) for value in self.episode_data[i][k].values()]
                        )
                        dataset_array = torch.transpose(dataset_array, 1, 0)
                        dataset_array = dataset_array[start_idx:end_idx]

                        f_writer.create_dataset(
                            k,
                            data=dataset_array.cpu().numpy(),
                            compression=compression_filter,
                            compression_opts=compression_opts,
                        )
                    else:
                        print(f"Warning: '{k}' cannot be written to file (unknown data type).")

                # handle extra info:
                for k in extra_trajectory_level_info:
                    f_writer.attrs[k] = extra_trajectory_level_info[k][i]

                # Log saved data
                print(f"Saved episode {episode_index} with keys: {list(f_writer.keys())}")
                for k in f_writer.keys():
                    print(f"  - {k}: shape={f_writer[k].shape}, dtype={f_writer[k].dtype}")

            total_steps = len(self.episode_data[i]["action"]) - self.initial_timesteps_to_discard
            total_steps = max(total_steps, 0)
            num_timesteps = min(total_steps, MAX_STEPS)
            self.metadata_info["num_timesteps"].append(num_timesteps)
            self.metadata_info["num_episodes"] += 1
            num_episodes_saved_this_call += 1
            
            # Save metadata after each episode
            with open(join(self.base_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata_info, f, indent=4)

            print(f"\u2713 Successfully saved episode {episode_index} ({num_timesteps} timesteps)")

            # Reset episode data for this environment
            self.episode_data[i] = {"action": []}
            
        if num_episodes_saved_this_call > 0:
            print(f"Saved {num_episodes_saved_this_call} episode(s) in this batch")
            
        return self.metadata_info["num_episodes"]


