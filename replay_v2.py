#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-faithful replay:
- Restores recorded scene_state0 (exact start).
- Picks recorded command type (pos/vel/effort target) and *directly* commands robot each env-step window:
    for each step: set_joint_*_target(cmd_t); repeat [write_data_to_sim -> sim.step] for (step_dt / physics_dt) times.
- Captures a video from the chosen camera.

This bypasses the Action Manager during replay to avoid reprocessing the recorded raw actions.
"""
import argparse
from pathlib import Path
import h5py
import torch
import imageio.v3 as iio

from isaaclab.app import AppLauncher
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, required=True)  # path to episode_xxx dir (with episode_xxx.h5)
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--camera", type=str, default="top", choices=["top", "wrist", "side"])
parser.add_argument("--output", type=str, default="/workspace/replay_v2.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

# Import after AppLauncher initializes Kit so omni.* modules are available
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def to_torch(x, device):
    return torch.as_tensor(x, device=device)


def load_nested_state(h5group):
    """Rebuild nested dict of torch tensors from HDF5 (scene_state0)."""
    state = {}
    for k in h5group.keys():
        item = h5group[k]
        if isinstance(item, h5py.Group):
            state[k] = load_nested_state(item)
        else:
            state[k] = torch.from_numpy(item[()])  # CPU tensor; will be moved to device by reset_to
    return state


# Resolve episode file
ep_dir = Path(args.episode)
if ep_dir.is_file() and ep_dir.suffix == ".h5":
    h5_path = ep_dir
else:
    h5_matches = list(ep_dir.glob("*.h5"))
    assert len(h5_matches) >= 1, f"No *.h5 found in {ep_dir}"
    h5_path = h5_matches[0]
print(f"[replay_v2] Loading: {h5_path}")

with h5py.File(h5_path, "r") as f:
    # pick command dataset in priority order
    cmd = None
    cmd_kind = None
    if "command/joint_pos_target" in f:
        cmd = f["command/joint_pos_target"][()]
        cmd_kind = "pos"
    elif "command/joint_vel_target" in f:
        cmd = f["command/joint_vel_target"][()]
        cmd_kind = "vel"
    elif "command/joint_effort_target" in f:
        cmd = f["command/joint_effort_target"][()]
        cmd_kind = "effort"
    else:
        raise RuntimeError("No recorded joint target command found in H5.")
    scene_state0 = load_nested_state(f["scene_state0"])
    physics_dt = float(f.attrs["physics_dt"])
    step_dt = float(f.attrs["step_dt"])
    decimation = int(round(step_dt / physics_dt))
    print(f"[replay_v2] cmd={cmd_kind}, T={len(cmd)}, physics_dt={physics_dt}, step_dt={step_dt}, decimation={decimation}")

# Build env (1 env recommended)
# Build env (1 env recommended)
env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
env_cfg.scene.env_spacing = 6.0
env = gym.make(args.task, cfg=env_cfg)
env.reset()

scene = env.unwrapped.scene
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
camera_name = f"{args.camera}_camera"
assert camera_name in sensors, f"Camera {camera_name} not found. Available: {list(sensors.keys())}"
cam = sensors[camera_name]

# Get robot
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)
assert robot is not None, "Robot articulation 'robot' not found in scene."

def move_nested_state(obj, device):
    if isinstance(obj, dict):
        return {k: move_nested_state(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj

# Restore scene start state with tensors on the correct device
scene_state0 = move_nested_state(scene_state0, env.device)
scene.reset_to(scene_state0)

# Simulation context (shared instance)
sim = SimulationContext.instance()

# Replay loop: set target, write, step physics decimation times, update scene; grab camera frame once per env-step
frames = []
for t in range(cmd.shape[0]):
    targets = torch.from_numpy(cmd[t][None, ...]).to(env.device)  # shape (1, dof)

    if cmd_kind == "pos":
        robot.set_joint_position_target(targets)
    elif cmd_kind == "vel":
        robot.set_joint_velocity_target(targets)
    elif cmd_kind == "effort":
        robot.set_joint_effort_target(targets)
    else:
        raise RuntimeError("Unknown cmd kind")

    # Hold target for one env-step -> decimation physics ticks
    for _ in range(decimation):
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)

    # Capture camera frame at env-step resolution
    cam.update(dt=step_dt)
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    frames.append(rgb)

    if t % 20 == 0:
        print(f"  step {t}/{cmd.shape[0]}")

# Save video
iio.imwrite(args.output, frames, fps=int(round(1.0 / step_dt)))
print(f"[replay_v2] Saved: {args.output}")

env.close()
app.close()


