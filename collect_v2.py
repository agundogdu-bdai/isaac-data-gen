#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic-friendly data collection for Isaac Lab:
- Logs raw policy action and the ACTUAL executed articulation targets:
  robot.data.joint_pos_target / joint_vel_target / joint_effort_target,
  plus joint state and applied torques each env-step.
- Saves initial scene state per episode via scene.get_state() for exact reset.
- Captures per-step camera frames.

Replay directly uses these targets (see replay_v2.py).

References:
- Action Manager transforms actions before applying to the robot. (envs/managers) 
- Articulation control API: set_joint_*_target + write_data_to_sim. 
- Scene snapshot/reset_to, dt relations, and camera RGB buffers shape.

"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import torch

from isaaclab.app import AppLauncher
from rsl_rl.modules import ActorCritic

# ------------------------- CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)   # Prefer 1 for dataset
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--output", type=str, default="/workspace/datasets/vpl_data_v2")
# App / rendering args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app  # launch Omniverse app

# Import after AppLauncher initializes Kit so omni.* modules are available
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ------------------------- Utils -------------------------

def tnp(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def write_nested_state(h5group: h5py.Group, state: Dict[str, Any]):
    """Recursively write scene.get_state() (nested dict of tensors) to HDF5."""
    for k, v in state.items():
        if isinstance(v, dict):
            write_nested_state(h5group.create_group(k), v)
        else:
            # Expect torch.Tensor
            h5group.create_dataset(k, data=tnp(v))


def flatten_obs(obs):
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            v = torch.as_tensor(v, device='cuda' if torch.cuda.is_available() else 'cpu')
            parts.append(v.reshape(v.shape[0], -1))
        return torch.cat(parts, dim=-1)
    else:
        return torch.as_tensor(obs)


# ------------------------- Build env -------------------------
device = args.device
env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
env_cfg.scene.env_spacing = 6.0
# Determinism knobs (within reason)
env_cfg.seed = int(args.seed)  # manager/direct env cfg uses .seed for RNG. 
env = gym.make(args.task, cfg=env_cfg)
obs, _ = env.reset()

scene = env.unwrapped.scene
physics_dt = scene.physics_dt  # physics step seconds
# step_dt = env.step_dt if available; otherwise compute from cfg
try:
    # ManagerBased envs expose cfg.decimation
    decimation = env.cfg.decimation
    step_dt = physics_dt * decimation
except Exception:
    decimation = 1
    step_dt = physics_dt

# Articulation (robot)
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)

# Cameras
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
cams = {name: sensors[name] for name in ["top_camera", "wrist_camera", "side_camera"] if name in sensors}
print(f"[collect_v2] Cameras found: {list(cams.keys())}")

# ------------------------- Load policy -------------------------
print(f"[collect_v2] Loading policy from: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)
state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
obs0 = flatten_obs(obs)
num_obs = obs0.shape[-1]
num_actions = env.action_space.shape[-1]
policy = ActorCritic(num_obs, num_obs, num_actions, [256, 128, 64], [256, 128, 64]).to(device)
if "_std" in state_dict:
    state_dict["std"] = state_dict.pop("_std")
policy.load_state_dict(state_dict)
policy.eval()
print("[collect_v2] Policy loaded")

# ------------------------- Out dir -------------------------
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)
meta = {
    "task": args.task,
    "seed": int(args.seed),
    "num_envs": int(args.num_envs),
    "physics_dt": float(physics_dt),
    "decimation": int(decimation),
    "step_dt": float(step_dt),
    "version": 2,
}
with open(out_dir / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"[collect_v2] Collecting {args.num_episodes} episodes × {args.num_envs} envs, {args.steps} steps each")

# ------------------------- Loop -------------------------
for ep in range(args.num_episodes):
    obs, _ = env.reset()
    print(f"\n[collect_v2] Episode {ep+1}/{args.num_episodes}")

    # Save initial scene state for exact reset
    scene_state0 = scene.get_state()  # nested dict of tensors
    # Per-episode containers
    raw_actions, pos_tgt, vel_tgt, eff_tgt = [], [], [], []
    joint_pos, joint_vel, tau_applied = [], [], []
    frames = {k: [] for k in cams.keys()}

    for t in range(args.steps):
        with torch.inference_mode():
            obs_tensor = flatten_obs(obs).to(device)
            action = policy.act_inference(obs_tensor)
        raw_actions.append(tnp(action[0]))

        # Step env (Action Manager will process + apply over decimation)
        obs, rew, terminated, truncated, info = env.step(action)

        # Record executed commands + state after the step
        if robot is not None:
            # These buffers are maintained by Isaac Lab for the *commanded* targets and applied torques
            # at the articulation (per env). We log env 0 for dataset collection with num_envs=1.
            d = robot.data
            if getattr(d, "joint_pos_target", None) is not None:
                pos_tgt.append(tnp(d.joint_pos_target[0]))
            if getattr(d, "joint_vel_target", None) is not None:
                vel_tgt.append(tnp(d.joint_vel_target[0]))
            if getattr(d, "joint_effort_target", None) is not None:
                eff_tgt.append(tnp(d.joint_effort_target[0]))
            if getattr(d, "joint_pos", None) is not None:
                joint_pos.append(tnp(d.joint_pos[0]))
            if getattr(d, "joint_vel", None) is not None:
                joint_vel.append(tnp(d.joint_vel[0]))
            if getattr(d, "applied_torque", None) is not None:
                tau_applied.append(tnp(d.applied_torque[0]))

        # Capture one RGB per env-step from each camera
        for name, cam in cams.items():
            # Camera data.rgb is uint8 (B,H,W,3)
            cam.update(dt=step_dt)  # advance camera buffers by env-step interval
            rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
            frames[name].append(rgb)

        if t % 20 == 0:
            print(f"  step {t:04d} | r0={float(rew[0]):.3f}")

    # Write episode package
    ep_dir = out_dir / f"episode_{ep:03d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    h5_path = ep_dir / f"episode_{ep:03d}.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["task"] = args.task
        f.attrs["seed"] = int(args.seed)
        f.attrs["physics_dt"] = float(physics_dt)
        f.attrs["decimation"] = int(decimation)
        f.attrs["step_dt"] = float(step_dt)
        # initial scene state
        write_nested_state(f.create_group("scene_state0"), scene_state0)
        # actions and executed commands
        f.create_dataset("action/raw", data=np.stack(raw_actions, axis=0))
        if len(pos_tgt):
            f.create_dataset("command/joint_pos_target", data=np.stack(pos_tgt, axis=0))
        if len(vel_tgt):
            f.create_dataset("command/joint_vel_target", data=np.stack(vel_tgt, axis=0))
        if len(eff_tgt):
            f.create_dataset("command/joint_effort_target", data=np.stack(eff_tgt, axis=0))
        if len(joint_pos):
            f.create_dataset("state/joint_pos", data=np.stack(joint_pos, axis=0))
        if len(joint_vel):
            f.create_dataset("state/joint_vel", data=np.stack(joint_vel, axis=0))
        if len(tau_applied):
            f.create_dataset("state/applied_torque", data=np.stack(tau_applied, axis=0))

        # stack color as (T, N_cams, H, W, 3)
        cam_stacks = []
        cam_order = []
        for idx, (name, fr_list) in enumerate(frames.items()):
            if len(fr_list):
                cam_stacks.append(np.stack(fr_list, axis=0)[:, None, ...])
                cam_order.append(name)
        if cam_stacks:
            color = np.concatenate(cam_stacks, axis=1)
            f.create_dataset("color", data=color)
            f.create_dataset("color_cam_names", data=np.array(cam_order, dtype="S"))

    # Also write MP4 per cam for quick browsing
    for name, fr_list in frames.items():
        if len(fr_list):
            iio.imwrite(ep_dir / f"{name}.mp4", fr_list, fps=int(round(1.0 / step_dt)))

print("\n[collect_v2] Done.")
env.close()
app.close()


