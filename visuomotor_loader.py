#!/usr/bin/env python3
"""
Loader for visuomotor DiffPo policy from W&B artifact.
Uses the visuomotor model registry to properly load the policy.
"""
import torch
import numpy as np
from pathlib import Path


def load_policy(artifact_dir: str, device):
    """
    Load DiffPo policy from W&B artifact using visuomotor registry.
    
    Args:
        artifact_dir: Path to downloaded W&B artifact directory
        device: torch device to load model onto
    
    Returns:
        A callable policy: obs_dict -> action (torch.Tensor)
    """
    from visuomotor.models.model_registry import REGISTRY
    
    artifact_path = Path(artifact_dir)
    ckpt_path = artifact_path / "model.ckpt"
    
    print(f"[visuomotor_loader] Loading checkpoint from: {ckpt_path}")
    
    # Load checkpoint
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass
    
    checkpoint = torch.load(
        str(ckpt_path),
        map_location=device,
        weights_only=False
    )
    
    print(f"[visuomotor_loader] Checkpoint keys: {list(checkpoint.keys())}")
    if 'hyper_parameters' in checkpoint:
        hp = checkpoint['hyper_parameters']
        print(f"[visuomotor_loader] Hyper-parameters keys: {list(hp.keys())[:10]}...")
        if 'method_name' in hp:
            print(f"[visuomotor_loader] Method: {hp['method_name']}")
    
    # Use visuomotor registry to build policy from checkpoint
    try:
        # Use method_name as provided (ensure 'diffpo' remains unchanged)
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            
            # Add action_dim if missing (required by diffusion_policy)
            if isinstance(hp, dict) and 'action_dim' not in hp:
                # Try to get from action_head if available
                if 'action_head' in hp and isinstance(hp['action_head'], dict):
                    action_dim = hp['action_head'].get('action_dim', 8)
                else:
                    action_dim = 8
                print(f"[visuomotor_loader] Adding missing action_dim={action_dim} to hyper_parameters")
                hp['action_dim'] = action_dim
            
            # Flatten action_head config to top level (expected by registry)
            if isinstance(hp, dict) and 'action_head' in hp and isinstance(hp['action_head'], dict):
                action_head = hp['action_head']
                for key in ['unet1d', 'noise_scheduler', 'horizon', 'num_inference_steps']:
                    if key in action_head and key not in hp:
                        hp[key] = action_head[key]
                        print(f"[visuomotor_loader] Promoted action_head.{key} to top-level config")
        
        # Use checkpoint as-is without remapping state_dict keys
        
        policy = REGISTRY.build_policy_from_checkpoint(
            checkpoint,
            load_swa_weights=False,
            suppress_version_warning=True  # Suppress since we may have version mismatch
        )
        policy.to(device)
        policy.eval()
        print("[visuomotor_loader] ✅ Successfully loaded policy via visuomotor registry")
        
        # Wrap to handle inference mode (no 'action' key normalization)
        wrapped = PolicyWrapper(policy, device)
        return wrapped
    except Exception as e:
        print(f"[visuomotor_loader] ❌ Failed to load policy: {e}")
        raise


class PolicyWrapper:
    """
    Wrapper for visuomotor DiffusionPolicy inference.
    Manually normalizes observations and calls the model, skipping action normalization.
    """
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device
    
    def __call__(self, obs_dict):
        """
        obs_dict: {
            'image_top': (1, 3, H, W) float32 [0,1],
            'image_wrist': (1, 3, H, W) float32 [0,1],
            'image_side': (1, 3, H, W) float32 [0,1],
            'joint_pos': (1, 9) float32
        }
        Returns: (1, 8) action tensor
        """
        with torch.inference_mode():
            # Manually normalize observations (skip action since we don't have it yet)
            normalized_obs = {}
            normalizer = getattr(self.policy, 'normalizer', None)
            params_dict = getattr(normalizer, 'params_dict', None)

            if params_dict is None:
                print("[visuomotor_loader] No normalizer params_dict found; passing observations through.")
                normalized_obs = obs_dict
            else:
                available_keys = list(params_dict.keys())
                print(f"[visuomotor_loader] Normalizer available keys: {available_keys}")
                for key, val in obs_dict.items():
                    is_string_key = isinstance(key, str)
                    present = is_string_key and (key in params_dict)
                    if present:
                        print(f"[visuomotor_loader] Normalizing field: {key}")
                        normalized_obs[key] = normalizer[key](val)
                    else:
                        if not is_string_key:
                            print(f"[visuomotor_loader] Skipping non-string key: {key}")
                        else:
                            print(f"[visuomotor_loader] No normalizer entry for '{key}', passing through.")
                        normalized_obs[key] = val
            
            # Call the underlying model directly
            action = self.policy.model(normalized_obs)
            
            # Denormalize action if available in normalizer
            if params_dict is not None and 'action' in params_dict:
                print("[visuomotor_loader] Denormalizing action using normalizer['action'].")
                action = normalizer['action'].unnormalize(action)
            
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(self.device)
            return action
    
    def predict(self, obs_dict):
        return self(obs_dict)
    
    def act(self, obs_dict):
        return self(obs_dict)

