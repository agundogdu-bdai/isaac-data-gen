#!/usr/bin/env python3
"""
Analyze action statistics from collected HDF5 dataset.
Reports mean, std, min, max, and norm statistics for actions.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def analyze_actions(h5_path):
    """Analyze action statistics from HDF5 file."""
    
    print("=" * 70)
    print(f"Action Statistics Analysis")
    print("=" * 70)
    print(f"File: {h5_path}")
    print()
    
    with h5py.File(h5_path, 'r') as f:
        # Check available keys
        print("Available datasets in file:")
        for key in f.keys():
            shape = f[key].shape
            dtype = f[key].dtype
            print(f"  - {key:20s} shape: {str(shape):20s} dtype: {dtype}")
        print()
        
        # Load actions
        if 'action' in f:
            actions = f['action'][:]
        elif 'actions' in f:
            actions = f['actions'][:]
        else:
            print("Error: No 'action' or 'actions' dataset found in file!")
            return
        
        print("-" * 70)
        print("ACTION STATISTICS")
        print("-" * 70)
        print(f"Shape: {actions.shape}")
        print(f"  - Timesteps: {actions.shape[0]}")
        print(f"  - Action dimension: {actions.shape[1] if len(actions.shape) > 1 else 1}")
        print()
        
        # Overall statistics
        print("Overall Statistics (all timesteps, all dimensions):")
        print(f"  Mean:     {np.mean(actions):12.6f}")
        print(f"  Std:      {np.std(actions):12.6f}")
        print(f"  Min:      {np.min(actions):12.6f}")
        print(f"  Max:      {np.max(actions):12.6f}")
        print(f"  L2 Norm:  {np.linalg.norm(actions):12.6f}")
        print()
        
        # Per-dimension statistics
        if len(actions.shape) > 1:
            print("Per-Dimension Statistics:")
            print(f"{'Dim':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'L2 Norm':<12}")
            print("-" * 70)
            for dim in range(actions.shape[1]):
                dim_actions = actions[:, dim]
                mean = np.mean(dim_actions)
                std = np.std(dim_actions)
                min_val = np.min(dim_actions)
                max_val = np.max(dim_actions)
                norm = np.linalg.norm(dim_actions)
                print(f"{dim:<6} {mean:<12.6f} {std:<12.6f} {min_val:<12.6f} {max_val:<12.6f} {norm:<12.6f}")
            print()
        
        # Per-timestep norm statistics
        if len(actions.shape) > 1:
            timestep_norms = np.linalg.norm(actions, axis=1)
            print("Per-Timestep Action Norm Statistics:")
            print(f"  Mean norm:     {np.mean(timestep_norms):12.6f}")
            print(f"  Std norm:      {np.std(timestep_norms):12.6f}")
            print(f"  Min norm:      {np.min(timestep_norms):12.6f}")
            print(f"  Max norm:      {np.max(timestep_norms):12.6f}")
            print()
        
        # Action value distribution
        print("Action Value Distribution:")
        print(f"  Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(actions, p)
            print(f"    {p:2d}th:  {val:12.6f}")
        print()
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze action statistics from HDF5 dataset")
    parser.add_argument(
        "h5_file",
        type=str,
        nargs="?",
        default="/home/agundogdu_theaiinstitute_com/test/vpl_tiled/Isaac-Open-Drawer-Franka-Camera-v0_sim_franka_20251013_013536/episode_001/episode_001.h5",
        help="Path to HDF5 file (default: episode_001.h5 in default location)"
    )
    
    args = parser.parse_args()
    
    h5_path = Path(args.h5_file)
    
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        return 1
    
    analyze_actions(h5_path)
    return 0


if __name__ == "__main__":
    exit(main())

