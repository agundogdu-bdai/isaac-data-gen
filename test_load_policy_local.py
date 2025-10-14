#!/usr/bin/env python3
"""
Test loading DiffPo policy locally (outside Isaac Lab) to verify visuomotor loader.
This script tests if the policy can be loaded at all before running in Isaac Sim.
"""
import sys
import torch
import numpy as np
import wandb
from pathlib import Path

# Configuration
ARTIFACT_NAME = "bdaii/Isaac-Open-Drawer-Franka-v0_sim_franka_20251014-agundogdu/diffpo-bqc7v7jo-kb4j741o:v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[test_load_policy_local] Device: {DEVICE}")
print(f"[test_load_policy_local] PyTorch version: {torch.__version__}")

# Step 1: Download artifact
print(f"\n[Step 1] Downloading artifact: {ARTIFACT_NAME}")
api = wandb.Api()
artifact = api.artifact(ARTIFACT_NAME, type="model")
artifact_dir = Path(artifact.download())
print(f"[test_load_policy_local] Downloaded to: {artifact_dir}")

# Step 2: Load checkpoint directly
ckpt_path = artifact_dir / "model.ckpt"
print(f"\n[Step 2] Loading checkpoint from: {ckpt_path}")

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray.scalar, np.dtype])
except Exception:
    pass

checkpoint = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
print(f"[test_load_policy_local] Checkpoint keys: {list(checkpoint.keys())}")

# Step 3: Inspect hyper_parameters
if 'hyper_parameters' in checkpoint:
    hp = checkpoint['hyper_parameters']
    print(f"\n[Step 3] Hyper-parameters (top 15 keys):")
    for i, (k, v) in enumerate(hp.items()):
        if i >= 15:
            break
        if isinstance(v, dict):
            print(f"  {k}: <dict with {len(v)} keys>")
        else:
            print(f"  {k}: {v}")

# Step 4: Try to load with visuomotor
print(f"\n[Step 4] Attempting to load with visuomotor...")
try:
    from visuomotor.models.model_registry import REGISTRY
    
    # Apply fixes
    hp = checkpoint['hyper_parameters']
    
    # Fix 1: Remap method_name
    if hp.get('method_name') == 'diffpo':
        print(f"  - Remapping 'diffpo' -> 'diffusion_policy'")
        hp['method_name'] = 'diffusion_policy'
    
    # Fix 2: Add action_dim
    if 'action_dim' not in hp and 'action_head' in hp:
        action_dim = hp['action_head'].get('action_dim', 8)
        print(f"  - Adding action_dim={action_dim}")
        hp['action_dim'] = action_dim
    
    # Fix 3: Flatten action_head config
    if 'action_head' in hp and isinstance(hp['action_head'], dict):
        for key in ['unet1d', 'noise_scheduler', 'horizon', 'num_inference_steps']:
            if key in hp['action_head'] and key not in hp:
                hp[key] = hp['action_head'][key]
                print(f"  - Promoted action_head.{key}")
    
    # Fix 4: Remap state_dict keys
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    remap_count = 0
    for k, v in state_dict.items():
        if k.startswith('head.denoiser.'):
            new_k = k.replace('head.denoiser.', 'model.unet.')
            new_state_dict[new_k] = v
            remap_count += 1
        else:
            new_state_dict[k] = v
    if remap_count > 0:
        print(f"  - Remapped {remap_count} state_dict keys")
        checkpoint['state_dict'] = new_state_dict
    
    # Try to build policy
    print(f"\n  Building policy from checkpoint...")
    policy = REGISTRY.build_policy_from_checkpoint(
        checkpoint,
        load_swa_weights=False,
        suppress_version_warning=True
    )
    policy.to(DEVICE)
    policy.eval()
    print(f"  ✅ SUCCESS! Policy loaded")
    print(f"  Policy type: {type(policy)}")
    
    # Step 5: Test inference with dummy observation
    print(f"\n[Step 5] Testing inference with dummy observation...")
    dummy_obs = {
        'image_top': torch.rand(1, 3, 120, 120, device=DEVICE),
        'image_wrist': torch.rand(1, 3, 120, 120, device=DEVICE),
        'image_side': torch.rand(1, 3, 120, 120, device=DEVICE),
        'joint_pos': torch.rand(1, 9, device=DEVICE),
    }
    
    with torch.inference_mode():
        action = policy.predict(dummy_obs)
    
    print(f"  Action shape: {action.shape}")
    print(f"  Action dtype: {action.dtype}")
    print(f"  Action sample: {action[0, :3].cpu().numpy()}")
    print(f"  ✅ Inference test passed!")
    
    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS PASSED!")
    print(f"{'='*60}")
    print(f"\nThe visuomotor_loader.py should work in Isaac Lab.")
    print(f"GPU memory issue is the main blocker now.")
    
except Exception as e:
    print(f"\n❌ Failed to load policy with visuomotor:")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

