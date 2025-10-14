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
assert args.num_envs == 1, "--num_envs must be 1 for flat dataset."
env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
env_cfg.scene.env_spacing = 6.0
try:
    env_cfg.seed = int(args.seed)
except Exception:
    pass

env = gym.make(args.task, cfg=env_cfg)
obs, _ = env.reset()

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
print(f"[collect_flat] Cameras present: {present} (missing cams will be zero-padded)")

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

# Output dir + global meta
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "metadata.json", "w") as f:
    json.dump(
        {
            "task": args.task,
            "seed": int(args.seed),
            "num_envs": 1,
            "physics_dt": physics_dt,
            "decimation": int(decimation),
            "step_dt": step_dt,
            "version": 3,
            "camera_order": CAM_ORDER,
            "camera_h": int(args.camera_h),
            "camera_w": int(args.camera_w),
        },
        f,
        indent=2,
    )

print(
    f"[collect_flat] Episodes: {args.num_episodes} | steps/ep: {args.steps} | step_dt: {step_dt:.4f}s"
)

# ------------------------- Rollout -------------------------
for ep in range(args.num_episodes):
    obs, _ = env.reset()
    print(f"\n[collect_flat] Episode {ep+1}/{args.num_episodes}")

    # Serialize scene state to flat bytes
    buf = io.BytesIO()
    torch.save(scene.get_state(), buf)
    scene_blob = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    actions: List[np.ndarray] = []
    pos_tgt: List[np.ndarray] = []
    vel_tgt: List[np.ndarray] = []
    eff_tgt: List[np.ndarray] = []
    qpos: List[np.ndarray] = []
    qvel: List[np.ndarray] = []
    tau: List[np.ndarray] = []
    frames = {k: [] for k in CAM_ORDER}

    for t in range(args.steps):
        with torch.inference_mode():
            obs_t = flatten_obs(obs).to(device)
            act = policy.act_inference(obs_t)
        actions.append(as_np(act[0]))

        # Step env (Action Manager + controllers act internally)
        obs, rew, terminated, truncated, info = env.step(act)

        # Record executed targets and state
        d = robot.data
        if getattr(d, "joint_pos_target", None) is not None:
            pos_tgt.append(as_np(d.joint_pos_target[0]))
        if getattr(d, "joint_vel_target", None) is not None:
            vel_tgt.append(as_np(d.joint_vel_target[0]))
        if getattr(d, "joint_effort_target", None) is not None:
            eff_tgt.append(as_np(d.joint_effort_target[0]))
        if getattr(d, "joint_pos", None) is not None:
            qpos.append(as_np(d.joint_pos[0]))
        if getattr(d, "joint_vel", None) is not None:
            qvel.append(as_np(d.joint_vel[0]))
        if getattr(d, "applied_torque", None) is not None:
            tau.append(as_np(d.applied_torque[0]))

        # Capture cameras once per env-step
        for name in CAM_ORDER:
            cam = cams.get(name)
            if cam is None:
                continue
            cam.update(dt=step_dt)
            rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
            rgb = resize_hwc_uint8(to_rgb_uint8(rgb), args.camera_h, args.camera_w)
            frames[name].append(rgb)

        if (t % 20) == 0:
            print(f"  step {t:04d} | r0={float(rew[0]):.3f}")

    # Write flat HDF5 episode
    ep_dir = out_dir / f"episode_{ep:03d}"
    ep_dir.mkdir(exist_ok=True)
    h5_path = ep_dir / f"episode_{ep:03d}.h5"
    with h5py.File(h5_path, "w") as f:
        # Attributes
        f.attrs["task"] = args.task
        f.attrs["seed"] = int(args.seed)
        f.attrs["physics_dt"] = physics_dt
        f.attrs["decimation"] = int(decimation)
        f.attrs["step_dt"] = step_dt
        f.attrs["version"] = 3
        f.attrs["camera_h"] = int(args.camera_h)
        f.attrs["camera_w"] = int(args.camera_w)
        f.attrs["action_dim"] = int(num_actions)
        if qpos:
            f.attrs["num_dof"] = int(np.asarray(qpos[0]).shape[-1])

        # Initial scene blob
        f.create_dataset("scene_state0", data=scene_blob, dtype=np.uint8)

        # Core datasets (flat)
        f.create_dataset("action", data=np.stack(actions, axis=0))
        if pos_tgt:
            f.create_dataset("joint_pos_target", data=np.stack(pos_tgt, axis=0))
        if vel_tgt:
            f.create_dataset("joint_vel_target", data=np.stack(vel_tgt, axis=0))
        if eff_tgt:
            f.create_dataset("joint_effort_target", data=np.stack(eff_tgt, axis=0))
        if qpos:
            f.create_dataset("joint_pos", data=np.stack(qpos, axis=0))
        if qvel:
            f.create_dataset("joint_vel", data=np.stack(qvel, axis=0))
        if tau:
            f.create_dataset("applied_torque", data=np.stack(tau, axis=0))

        # Color: (T,3,H,W,3) — pad zeros if a cam missing
        T = len(actions)
        H, W = args.camera_h, args.camera_w
        color = np.zeros((T, 3, H, W, 3), dtype=np.uint8)
        for ci, name in enumerate(CAM_ORDER):
            if len(frames[name]) == T:
                color[:, ci] = np.stack(frames[name], axis=0)
        f.create_dataset("color", data=color)
        f.create_dataset("camera_names", data=np.array(CAM_ORDER, dtype="S"))

    # Quick MP4s for browsing
    fps = int(round(1.0 / step_dt)) if step_dt > 0 else 30
    for name in CAM_ORDER:
        if frames[name]:
            iio.imwrite(ep_dir / f"{name}.mp4", frames[name], fps=fps)

print("\n[collect_flat] Done.")
env.close()
app.close()


