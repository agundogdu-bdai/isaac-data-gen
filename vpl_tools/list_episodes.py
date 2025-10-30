#!/usr/bin/env python3
"""Helper script to list available saved episodes."""

import argparse
import h5py
import json
import os
from pathlib import Path


def list_episodes(base_dir: str = "vpl_data"):
    """List all available episodes in the data directory.
    
    Args:
        base_dir: Base directory containing saved episodes
    """
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return
    
    # Load metadata
    metadata_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"\n{'='*80}")
        print(f"Dataset Metadata")
        print(f"{'='*80}")
        print(f"Total episodes: {metadata['num_episodes']}")
        print(f"Total timesteps: {sum(metadata['num_timesteps'])}")
        if metadata['num_episodes'] > 0:
            avg_timesteps = sum(metadata['num_timesteps']) / len(metadata['num_timesteps'])
            print(f"Average timesteps per episode: {avg_timesteps:.1f}")
        print()
    
    # List all episode directories
    episode_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("episode_")])
    
    if not episode_dirs:
        print(f"No episodes found in '{base_dir}'")
        return
    
    print(f"{'='*80}")
    print(f"Available Episodes")
    print(f"{'='*80}")
    print(f"{'Episode':<15} {'File':<40} {'Timesteps':<12} {'Keys'}")
    print(f"{'-'*80}")
    
    for episode_dir in episode_dirs:
        episode_idx = episode_dir.split("_")[1]
        episode_path = os.path.join(base_dir, episode_dir, f"episode_{episode_idx}.h5")
        
        if os.path.exists(episode_path):
            with h5py.File(episode_path, "r") as f:
                # Get number of timesteps from action length
                num_timesteps = f["action"].shape[0] if "action" in f else 0
                keys = list(f.keys())
                keys_str = ", ".join(keys[:3]) + ("..." if len(keys) > 3 else "")
                
                print(f"{episode_dir:<15} {episode_path:<40} {num_timesteps:<12} {keys_str}")
    
    print(f"{'-'*80}\n")
    
    # Print example command
    if episode_dirs:
        first_episode = episode_dirs[0]
        episode_idx = first_episode.split("_")[1]
        example_path = os.path.join(base_dir, first_episode, f"episode_{episode_idx}.h5")
        print("Example replay command:")
        print(f"  python replay.py --task Isaac-Lift-Cube-Franka-v0 --episode_path {example_path}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available saved episodes.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="vpl_data",
        help="Base directory containing saved episodes (default: vpl_data)"
    )
    args = parser.parse_args()
    
    list_episodes(args.base_dir)


