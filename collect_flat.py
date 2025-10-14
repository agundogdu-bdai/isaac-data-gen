#!/usr/bin/env python3
# flake8: noqa
# -*- coding: utf-8 -*-
"""
Flat HDF5 data collection for Isaac Lab:
- Logs raw policy actions and EXECUTED articulation targets (pos/vel/effort),
  plus joint state and applied torques at env-step cadence.
- Captures 3 cameras (top/wrist/side) into a fixed RGB dataset:
    color: (T, 3, H=240, W=320, 3) uint8
- Stores a serialized scene_state0 blob for exact resets at replay.
"""
import argparse
import io
import json
from pathlib import Path
from typing import List

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.app import AppLauncher
from rsl_rl.modules import ActorCritic


# ------------------------- CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--output", type=str, default="/workspace/datasets/vpl_data_flat")
parser.add_argument("--camera_h", type=int, default=240)
parser.add_argument("--camera_w", type=int, default=320)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

# Launch Kit app first so omni.* modules are available downstream
app = AppLauncher(args).app

# Import after Kit is up (these may transitively import omni.*)
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore


# ------------------------- Helpers -------------------------
def as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def flatten_obs(obs):
    if isinstance(obs, dict):
        parts = [torch.as_tensor(v).reshape(v.shape[0], -1) for v in obs.values()]
        return torch.cat(parts, dim=-1)
    return torch.as_tensor(obs)


def to_rgb_uint8(rgb_np: np.ndarray) -> np.ndarray:
    rgb = rgb_np
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def resize_hwc_uint8(rgb: np.ndarray, H: int, W: int) -> np.ndarray:
    if (rgb.shape[0], rgb.shape[1]) == (H, W):
        return rgb
    x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None] / 255.0
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


# ------------------------- Env & policy -------------------------
device = args.device
env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
env_cfg.scene.env_spacing = 6.0
try:
    env_cfg.seed = int(args.seed)
except Exception:
    pass

env = gym.make(args.task, cfg=env_cfg)
obs, _ = env.reset()
num_envs = getattr(env.unwrapped, "num_envs", args.num_envs)

scene = env.unwrapped.scene
physics_dt = float(scene.physics_dt)
try:
    decimation = int(env.cfg.decimation)
    step_dt = physics_dt * decimation
except Exception:
    decimation, step_dt = 1, physics_dt

# Cameras (fixed order) — pad zeros if missing
CAM_ORDER = ["top", "wrist", "side"]
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
cams = {name: sensors.get(f"{name}_camera") for name in CAM_ORDER}
present = [n for n, c in cams.items() if c is not None]
print(
    f"[collect_flat] Cameras present: {present} (missing cams will be zero-padded) | num_envs={num_envs}"
)

# Robot articulation
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)
assert robot is not None, "Robot articulation 'robot' not found."

# Policy
ckpt = torch.load(args.checkpoint, map_location=device)
state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
num_obs = flatten_obs(obs).shape[-1]
num_actions = env.action_space.shape[-1]
policy = ActorCritic(num_obs, num_obs, num_actions, [256, 128, 64], [256, 128, 64]).to(device)
if "_std" in state_dict:
    state_dict["std"] = state_dict.pop("_std")
policy.load_state_dict(state_dict)
policy.eval()

# Output dir
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

print(
    f"[collect_flat] Episodes: {args.num_episodes} | steps/ep: {args.steps} | step_dt: {step_dt:.4f}s"
)

def slice_state_for_env(state_dict: dict, env_index: int, batch_size: int) -> dict:
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            out[k] = slice_state_for_env(v, env_index, batch_size)
        elif torch.is_tensor(v):
            # If leading dim matches batch size, keep a singleton batch for reset_to
            if v.dim() >= 1 and v.shape[0] == batch_size:
                out[k] = v[env_index : env_index + 1].clone()
            else:
                out[k] = v.clone()
        else:
            out[k] = v
    return out


# ------------------------- Rollout -------------------------
global_ep = 0
for round_idx in range(args.num_episodes):
    obs, _ = env.reset()
    print(f"\n[collect_flat] Round {round_idx+1}/{args.num_episodes} — collecting {num_envs} episodes in parallel")

    # Capture full multi-env start state once per round
    full_state = scene.get_state()

    # Per-env containers
    actions = {eid: [] for eid in range(num_envs)}
    pos_tgt = {eid: [] for eid in range(num_envs)}
    vel_tgt = {eid: [] for eid in range(num_envs)}
    eff_tgt = {eid: [] for eid in range(num_envs)}
    qpos = {eid: [] for eid in range(num_envs)}
    qvel = {eid: [] for eid in range(num_envs)}
    tau = {eid: [] for eid in range(num_envs)}
    frames = {eid: {k: [] for k in CAM_ORDER} for eid in range(num_envs)}

    for t in range(args.steps):
        with torch.inference_mode():
            obs_t = flatten_obs(obs).to(device)
            act = policy.act_inference(obs_t)

        # Record actions per env
        act_np = as_np(act)
        for eid in range(num_envs):
            actions[eid].append(act_np[eid])

        # Step env
        obs, rew, terminated, truncated, info = env.step(act)

        # Record executed targets and state from tensors shaped (N, ...)
        d = robot.data
        for eid in range(num_envs):
            if getattr(d, "joint_pos_target", None) is not None:
                pos_tgt[eid].append(as_np(d.joint_pos_target[eid]))
            if getattr(d, "joint_vel_target", None) is not None:
                vel_tgt[eid].append(as_np(d.joint_vel_target[eid]))
            if getattr(d, "joint_effort_target", None) is not None:
                eff_tgt[eid].append(as_np(d.joint_effort_target[eid]))
            if getattr(d, "joint_pos", None) is not None:
                qpos[eid].append(as_np(d.joint_pos[eid]))
            if getattr(d, "joint_vel", None) is not None:
                qvel[eid].append(as_np(d.joint_vel[eid]))
            if getattr(d, "applied_torque", None) is not None:
                tau[eid].append(as_np(d.applied_torque[eid]))

        # Capture cameras once per env-step
        for name in CAM_ORDER:
            cam = cams.get(name)
            if cam is None:
                continue
            cam.update(dt=step_dt)
            rgb_batch = cam.data.output["rgb"].detach().cpu().numpy()  # (N,H,W,3)
            for eid in range(num_envs):
                rgb = rgb_batch[eid]
                rgb = resize_hwc_uint8(to_rgb_uint8(rgb), args.camera_h, args.camera_w)
                frames[eid][name].append(rgb)

        if (t % 20) == 0:
            try:
                r0 = float(rew[0])
            except Exception:
                r0 = 0.0
            print(f"  step {t:04d} | r0={r0:.3f}")

    # Write one episode per env
    for eid in range(num_envs):
        ep_dir = out_dir / f"episode_{global_ep:03d}"
        ep_dir.mkdir(exist_ok=True)
        h5_path = ep_dir / f"episode_{global_ep:03d}.h5"

        # Build env-specific start state blob
        state_e = slice_state_for_env(full_state, eid, num_envs)
        buf = io.BytesIO()
        torch.save(state_e, buf)
        scene_blob = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        with h5py.File(h5_path, "w") as f:
            f.attrs["task"] = args.task
            f.attrs["seed"] = int(args.seed)
            f.attrs["physics_dt"] = physics_dt
            f.attrs["decimation"] = int(decimation)
            f.attrs["step_dt"] = step_dt
            f.attrs["version"] = 3
            f.attrs["camera_h"] = int(args.camera_h)
            f.attrs["camera_w"] = int(args.camera_w)
            f.attrs["action_dim"] = int(num_actions)
            if qpos[eid]:
                f.attrs["num_dof"] = int(np.asarray(qpos[eid][0]).shape[-1])

            # Initial scene blob
            f.create_dataset("scene_state0", data=scene_blob, dtype=np.uint8)

            # Core datasets (flat)
            f.create_dataset("action", data=np.stack(actions[eid], axis=0))
            if pos_tgt[eid]:
                f.create_dataset("joint_pos_target", data=np.stack(pos_tgt[eid], axis=0))
            if vel_tgt[eid]:
                f.create_dataset("joint_vel_target", data=np.stack(vel_tgt[eid], axis=0))
            if eff_tgt[eid]:
                f.create_dataset("joint_effort_target", data=np.stack(eff_tgt[eid], axis=0))
            if qpos[eid]:
                f.create_dataset("joint_pos", data=np.stack(qpos[eid], axis=0))
            if qvel[eid]:
                f.create_dataset("joint_vel", data=np.stack(qvel[eid], axis=0))
            if tau[eid]:
                f.create_dataset("applied_torque", data=np.stack(tau[eid], axis=0))

            # Color block
            T = len(actions[eid])
            H, W = args.camera_h, args.camera_w
            color = np.zeros((T, 3, H, W, 3), dtype=np.uint8)
            for ci, name in enumerate(CAM_ORDER):
                if len(frames[eid][name]) == T:
                    color[:, ci] = np.stack(frames[eid][name], axis=0)
            f.create_dataset("color", data=color)
            f.create_dataset("camera_names", data=np.array(CAM_ORDER, dtype="S"))

        # Quick MP4s
        fps = int(round(1.0 / step_dt)) if step_dt > 0 else 30
        for name in CAM_ORDER:
            if frames[eid][name]:
                iio.imwrite(ep_dir / f"{name}.mp4", frames[eid][name], fps=fps)

        # Update metadata.json
        meta_path = out_dir / "metadata.json"
        metadata = {"num_timesteps": [], "num_episodes": 0}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    prev = json.load(f)
                    if isinstance(prev, dict):
                        metadata["num_timesteps"] = prev.get("num_timesteps", [])
            except Exception:
                pass
        metadata["num_timesteps"].append(len(actions[eid]))
        metadata["num_episodes"] = len(metadata["num_timesteps"])
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        global_ep += 1

print("\n[collect_flat] Done.")
env.close()
app.close()


