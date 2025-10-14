#!/usr/bin/env python3
"""
Loader for DiffPo checkpoint from W&B artifact.
Reconstructs the diffusion policy model from state_dict.
"""
import torch
import numpy as np
from pathlib import Path


def load_policy(artifact_dir: str, device):
    """
    Load DiffPo policy from artifact directory.
    Returns a callable: obs_dict -> action (torch.Tensor)
    """
    artifact_path = Path(artifact_dir)
    ckpt_path = artifact_path / "model.ckpt"
    
    print(f"[diffpo_loader] Loading checkpoint from: {ckpt_path}")
    
    # Load checkpoint with unpickling allowed
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass
    
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")
    
    print(f"[diffpo_loader] State dict keys: {list(state_dict.keys())[:10]}...")
    
    # Try to infer model type and reconstruct
    # Check if it's a direct policy object
    if hasattr(ckpt, 'predict') or hasattr(ckpt, 'forward') or hasattr(ckpt, 'act'):
        print("[diffpo_loader] Found callable policy in checkpoint")
        ckpt.eval()
        return ckpt
    
    # Otherwise, assume it's a lightning checkpoint with a policy module
    # Try to extract the policy if it's nested
    if 'policy' in state_dict:
        policy_state = state_dict['policy']
    else:
        policy_state = state_dict
    
    # Create a simple wrapper that returns zero actions (placeholder)
    # You'll need to replace this with actual DiffPo model instantiation
    class DiffPoWrapper:
        def __init__(self, state_dict, device):
            self.device = device
            self.state_dict = state_dict
            print(f"[diffpo_loader] WARNING: Using placeholder policy (returns zero actions)")
            print(f"[diffpo_loader] To fix: provide actual DiffPo model class and instantiate it here")
        
        def __call__(self, obs_dict):
            # obs_dict: {image_top, image_wrist, image_side, joint_pos}
            # Return 8D action (7 arm + 1 gripper)
            batch_size = obs_dict['joint_pos'].shape[0]
            return torch.zeros((batch_size, 8), device=self.device, dtype=torch.float32)
        
        def predict(self, obs_dict):
            return self(obs_dict)
        
        def act(self, obs_dict):
            return self(obs_dict)
    
    policy = DiffPoWrapper(policy_state, device)
    print("[diffpo_loader] Created policy wrapper")
    
    return policy

