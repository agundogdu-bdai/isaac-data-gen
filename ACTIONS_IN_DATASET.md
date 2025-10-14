# Actions in Collected Dataset - Explained

## Summary

**✅ Your collected data contains:**
- **Joint space actions** (NOT end-effector/Cartesian actions)
- **RAW policy network outputs** (NOT processed joint positions)
- **8 dimensions**: 7 arm joint commands + 1 gripper command

---

## What Gets Stored

### In `episode_001.h5` → `action` dataset:

```python
action.shape = (200, 8)  # (timesteps, action_dims)

# These are the RAW outputs from your policy network
# Values range: -5.0 to +8.2 (typical for network outputs)
```

### Flow Diagram:

```
Policy Network Output (8D)
         ↓
    [Stored in HDF5 as "action"]  ← YOU ARE HERE
         ↓
    env.step(action)
         ↓
    Action Processing in Environment:
      processed = raw_action × 1.0 + default_joint_positions
         ↓
    Robot receives processed joint position targets
         ↓
    Physics simulation executes
         ↓
    [Stored in HDF5 as "joint_pos" - actual joint positions]
```

---

## Verification from Your Data

Looking at `episode_001.h5`:

### Action Statistics:
```
Dimension | Mean    | Std   | Min     | Max    | Range
----------|---------|-------|---------|--------|-------
0         | -0.488  | 0.400 | -1.873  | 1.090  | 2.96
1         | -1.864  | 0.423 | -2.198  | 0.878  | 3.08
2         | -0.257  | 0.440 | -1.829  | 1.020  | 2.85
3         |  0.742  | 0.445 | -0.739  | 2.155  | 2.89
4         |  4.429  | 4.587 | -5.009  | 8.198  | 13.21 ⭐ Large range!
5         |  2.166  | 0.801 | -1.143  | 2.827  | 3.97
6         |  0.875  | 0.714 | -1.214  | 1.577  | 2.79
7         | -0.662  | 0.686 | -2.149  | 2.032  | 4.18
```

**Key observation:**
- Max value: **8.198** (dimension 4)
- Min value: **-5.009** (dimension 4)
- These are clearly **RAW policy outputs**, not joint angles (which would be limited to ±2.9 radians)

---

## What This Means for Your Use Case

### ✅ Good News:
1. **Actions are joint space** - Good for joint-space policy learning
2. **8D matches Franka** - 7 arm joints + 1 gripper
3. **Raw outputs stored** - You can see what the network actually predicts

### ⚠️ Important to Know:

If you're training a vision-based policy with this data, you need to understand:

**Option 1: Train to predict RAW actions (what's stored)**
```python
# Your network should output raw values in range [-5, +8]
# During deployment, environment will process them:
processed = network_output × 1.0 + default_joint_positions
```

**Option 2: Convert to processed joint positions during training**
```python
# If you want to train on actual joint targets instead:
processed_actions = stored_actions × 1.0 + default_joint_positions

# Where default_joint_positions for Franka are approximately:
# [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
# (7 arm joints + 2 gripper fingers, but gripper is binary)
```

---

## Joint Space vs End-Effector Space

### ❌ NOT End-Effector (Cartesian) Actions

Your data does **NOT** contain:
- ❌ End-effector positions (x, y, z)
- ❌ End-effector orientations (roll, pitch, yaw)
- ❌ 6D or 7D Cartesian commands
- ❌ Actions that require inverse kinematics

### ✅ Joint Space Actions

Your data **DOES** contain:
- ✅ Joint position targets (8D)
- ✅ Direct robot joint commands
- ✅ No IK solving required
- ✅ One-to-one mapping: action → joint

**This is what you have:**
```python
action[0] → panda_joint1 (shoulder pan)
action[1] → panda_joint2 (shoulder lift)
action[2] → panda_joint3 (elbow)
action[3] → panda_joint4 (wrist 1)
action[4] → panda_joint5 (wrist 2)
action[5] → panda_joint6 (wrist 3)
action[6] → panda_joint7 (wrist roll)
action[7] → gripper (binary open/close)
```

---

## How to Verify This in Code

### Method 1: Check Action Range
```python
import h5py
import numpy as np

with h5py.File('episode_001.h5', 'r') as f:
    actions = f['action'][:]
    
    max_val = np.max(np.abs(actions))
    
    if max_val > 5.0:
        print("✓ RAW policy outputs (before environment processing)")
    else:
        print("✓ Processed joint positions")
```

### Method 2: Compare with Joint Positions
```python
with h5py.File('episode_001.h5', 'r') as f:
    actions = f['action'][:]      # Raw policy outputs
    joint_pos = f['joint_pos'][:]  # Actual joint positions after execution
    
    print(f"Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    print(f"Joint pos range: [{joint_pos.min():.2f}, {joint_pos.max():.2f}]")
    
    # They should be different!
    # Actions: -5 to +8 (raw outputs)
    # Joint pos: -2.9 to +2.9 (physical joint limits)
```

### Method 3: Check Environment Configuration
```python
# Your environment uses JointPositionActionCfg
# which means joint space, not Cartesian

# See: joint_pos_env_camera_cfg.py
# Base: isaaclab_tasks/.../cabinet/config/franka/joint_pos_env_cfg.py

# Action config:
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],  # Joint space!
    scale=1.0,
    use_default_offset=True,  # Adds offset to raw actions
)
```

---

## Datasets in Your HDF5 Files

### Actions and States:
| Dataset | Shape | Description |
|---------|-------|-------------|
| `action` | (T, 8) | **RAW policy outputs** (what network predicts) |
| `joint_pos` | (T, 9) | **Actual joint positions** (what robot executes) |
| `joint_vel` | (T, 9) | Joint velocities |
| `ee_position` | (T, 3) | End-effector position (computed via forward kinematics) |
| `ee_rotation` | (T, 4) | End-effector quaternion |
| `proprio` | (T, 31) | Full proprioceptive observation vector |
| `state` | (T, 25) | Robot state vector |

### Visual Data:
| Dataset | Shape | Description |
|---------|-------|-------------|
| `color` | (T, 3, 120, 160, 3) | RGB from 3 cameras |
| `intrinsics` | (3, 3, 3) | Camera intrinsics [N_cams, 3, 3] |

### Difference Between `action` and `joint_pos`:

```python
# action[t] = what policy wants (raw output)
# joint_pos[t] = what robot actually does (after physics)

# Relationship:
# target_joint_pos[t] = action[t] × 1.0 + default_joint_positions
# joint_pos[t+1] ≈ target_joint_pos[t] (if robot tracks well)
```

---

## For Training Vision-Based Policies

### Option A: Train to Predict Raw Actions (Recommended)

```python
# Your training loop
predicted_action = vision_policy(rgb_image)
loss = mse_loss(predicted_action, stored_actions)

# During deployment
action = vision_policy(camera_observation)
env.step(action)  # Environment handles processing
```

**Pros:**
- ✅ Matches how data was collected
- ✅ No need to know default joint positions
- ✅ Environment handles all transformations

### Option B: Train to Predict Processed Joint Positions

```python
# Convert stored actions to joint targets
default_pose = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04])
target_joints = stored_actions * 1.0 + default_pose

# Train on targets
predicted_joints = vision_policy(rgb_image)
loss = mse_loss(predicted_joints, target_joints)

# During deployment
joints = vision_policy(camera_observation)
# Need to convert back to raw actions for environment:
raw_action = (joints - default_pose) / 1.0
env.step(raw_action)
```

**Pros:**
- ✅ Actions in physically meaningful ranges
- ✅ Can apply joint limit constraints

**Cons:**
- ❌ Need to know default pose
- ❌ Extra conversion step

---

## Quick Reference

**What you're collecting:**
```
Environment: Isaac-Open-Drawer-Franka-Camera-v0
Action Space: Joint Position (JointPositionActionCfg)
Action Dimensions: 8 (7 arm + 1 gripper)
Action Type: RAW policy outputs
Action Range: -5.0 to +8.2 (from your data)
Processing: processed = raw × 1.0 + default_joint_pos
Control: Direct joint position targets
```

**What you're NOT collecting:**
```
Action Space: End-Effector/Cartesian (IKAbsoluteActionCfg) ❌
Action Dimensions: 6-7 (x,y,z,roll,pitch,yaw,gripper) ❌  
Action Type: Cartesian commands ❌
Processing: Inverse kinematics solving ❌
Control: End-effector pose targets ❌
```

---

## Conclusion

Your collected dataset contains:
- ✅ **Joint space actions** (8D: 7 arm + 1 gripper)
- ✅ **RAW policy network outputs** (range: -5 to +8)
- ✅ **NOT end-effector actions** (not Cartesian control)
- ✅ **Ready for joint-space policy training**

The actions are stored exactly as the policy outputs them, before any environment processing. This is the standard format for joint-space control datasets in robotics.

---

**See also:**
- `ACTION_SPACE_EXPLAINED.md` - Detailed action space documentation
- `analyze_actions.py` - Script to analyze action statistics
- `COMPLETE_CAMERA_ENV_SETUP.md` - Full environment setup guide

