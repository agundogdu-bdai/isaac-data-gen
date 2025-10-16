#!/usr/bin/env python3
# flake8: noqa
# -*- coding: utf-8 -*-
"""
Direct-control evaluation of a W&B artifact policy on Isaac Lab (bypass ActionManager).

Behavior:
- Loads policy via visuomotor REGISTRY (predict_action).
- Builds obs with cameras and joint_pos.
- Gets a trajectory, then applies K single-step actions directly to robot each tick.
- Advances physics manually and records videos.

Notes:
- This attempts to use common Isaac Sim articulation controllers. If none are found on this robot,
  it raises a clear error.
"""
import argparse
import os
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

from isaaclab.app import AppLauncher
import wandb  # align import order with working eval script


# ------------------------- CLI -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--artifact", type=str, required=True, help="W&B artifact path: entity/project/artifact:version")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-Camera-v0")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--camera_h", type=int, default=240)
parser.add_argument("--camera_w", type=int, default=320)
parser.add_argument("--output_dir", type=str, default=str(Path.cwd() / "videos"))
parser.add_argument("--actions_per_inference", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True


# ------------------------- Launch Kit App -------------------------
app = AppLauncher(args).app

# Import after Kit is up
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore


# ------------------------- Utils -------------------------
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
    x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


def build_obs_dict(
    cams: dict,
    joint_pos_1xD: np.ndarray,
    H: int,
    W: int,
    order: list[str] | None = None,
) -> dict:
    # Build features expected by predict_action: 'color' as (1, X, H, W, 3) uint8 CPU, and 'joint_pos' (1, 9) float32
    if order is None:
        order = ["top_camera", "wrist_camera", "side_camera"]

    per_cam_images: list[np.ndarray] = []
    for name in order:
        cam = cams[name]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        rgb = resize_hwc_uint8(to_rgb_uint8(rgb), H, W)
        per_cam_images.append(rgb)

    color = np.stack(per_cam_images, axis=0)[None, ...]  # (1, X, H, W, 3) uint8
    jp = joint_pos_1xD.astype(np.float32)
    if jp.ndim == 1:
        jp = jp[None, ...]
    return {"color": color, "joint_pos": jp}


def make_policy_predict_action(policy_obj) -> Callable[[dict], np.ndarray | torch.Tensor]:
    if not hasattr(policy_obj, "predict_action"):
        raise RuntimeError("Policy does not implement predict_action(obs, batched=False)")

    def _call(obs: dict):
        return policy_obj.predict_action(obs, batched=False)

    return _call


def get_direct_controller(robot) -> tuple[str, object]:
    """Find a direct-control interface for the robot and return (mode, controller).

    Tries common Isaac Sim control APIs in priority order.
    - ("position_target", robot.set_joint_position_target)
    - ("position", robot.set_joint_positions)
    - ("controller_apply", ArticulationController.apply_action)
    """
    # 1) Methods on the robot wrapper
    if hasattr(robot, "set_joint_position_target"):
        return ("position_target", robot)
    if hasattr(robot, "set_joint_positions"):
        return ("position", robot)

    # 2) Isaac Sim ArticulationController
    try:
        from omni.isaac.core.controllers import ArticulationController
        from omni.isaac.core.utils.types import ArticulationAction
    except Exception:
        ArticulationController, ArticulationAction = None, None

    if ArticulationController is not None:
        # Try to get the underlying articulation/view
        art = getattr(robot, "articulation", None) or getattr(robot, "_articulation_view", None) or robot
        try:
            ctrl = ArticulationController(articulation=art)
            return ("controller_apply", ctrl)
        except Exception:
            pass

    raise RuntimeError(
        "Direct control is not available on this robot wrapper. Expected one of: set_joint_position_target, "
        "set_joint_positions, or a usable omni.isaac.core ArticulationController."
    )


# ------------------------- Build Env -------------------------
# Use AppLauncher-provided device (string like "cuda:0" or "cpu")
device = args.device
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

# Cameras (from scene)
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
cams = {
    "top_camera": sensors.get("top_camera"),
    "wrist_camera": sensors.get("wrist_camera"),
    "side_camera": sensors.get("side_camera"),
}

# Robot articulation
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)
assert robot is not None, "Robot articulation 'robot' not found."

# Get a direct controller
mode, controller = get_direct_controller(robot)
print(f"[direct] Control mode: {mode}")


# ------------------------- Optional: mock rai_tensors -------------------------
# Some checkpoints may reference 'rai_tensors' which is not required at inference.
# Provide a minimal shim so imports succeed without the dependency installed.
import sys, types  # noqa: E402

def _create_dummy_module(fullname: str) -> types.ModuleType:
    m = types.ModuleType(fullname)
    # Provide a valid __file__ so inspect/os.path don't receive a Dummy object
    try:
        m.__file__ = f"/tmp/{fullname.replace('.', '_')}.py"
    except Exception:
        pass
    def _mod_getattr(_name: str):
        class _Dummy:
            def __call__(self, *args, **kwargs):
                return self
            def __getattr__(self, _):
                return self
            def __setattr__(self, *_, **__):
                pass
            def __iter__(self):
                return iter(())
            def __repr__(self):
                return '<rai_tensors.Dummy>'
        return _Dummy()
    # PEP 562: module-level __getattr__ for missing attrs
    setattr(m, '__getattr__', _mod_getattr)
    return m

if 'rai_tensors' not in sys.modules:
    base = _create_dummy_module('rai_tensors')
    # Mark as a package to allow submodule imports like 'rai_tensors.torch'
    setattr(base, '__path__', [])
    sys.modules['rai_tensors'] = base
    # Common submodules that might be imported
    for sub in ('torch', 'utils'):
        fullname = f'rai_tensors.{sub}'
        sys.modules[fullname] = _create_dummy_module(fullname)
        setattr(base, sub, sys.modules[fullname])


# ------------------------- Load Policy -------------------------
print(f"[direct] Loading policy via REGISTRY: {args.artifact}")
from visuomotor.models.model_registry import REGISTRY  # defer import until after rai_tensors shim
policy_obj = REGISTRY.build_policy_from_artifact_name(
    name=args.artifact,
    device=device,
    suppress_version_warning=True,
)
predict_action = make_policy_predict_action(policy_obj)


# ------------------------- Video Writers -------------------------
H, W = int(args.camera_h), int(args.camera_w)
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
fps = int(round(1.0 / step_dt)) if step_dt > 0 else 30
writers = {}
for name, cam in cams.items():
    if cam is None:
        continue
    path = out_dir / f"eval_diffpo_direct_{name.replace('_camera','')}.mp4"
    try:
        f = iio.imopen(path, "w", plugin="pyav")
        try:
            f.init_video_stream(codec="h264", fps=fps)
        except Exception:
            pass
        writers[name] = f
        print(f"[direct] Streaming video for {name} -> {path}")
    except Exception as e:
        print(f"[direct] Streaming setup failed for {name} ({e}); will buffer frames.")
        writers[name] = None


# ------------------------- Control Helpers -------------------------
def apply_arm_gripper(action_1x8: np.ndarray | torch.Tensor):
    if isinstance(action_1x8, torch.Tensor):
        a = action_1x8.detach().cpu().numpy()
    else:
        a = action_1x8
    a = a.reshape(-1)
    arm = a[:7]
    grip = a[7]

    # Current joint pos for absolute targeting / delta
    jp = robot.data.joint_pos[0].detach().cpu().numpy()
    if jp.shape[-1] >= 7:
        curr = jp[:7]
    else:
        # pad if fewer joints are reported
        curr = np.pad(jp, (0, max(0, 7 - jp.shape[-1])))[:7]

    # Heuristic: treat policy outputs as deltas in joint space (radians) per step
    # Adjust scale if needed
    delta_scale = 1.0
    target_pos = curr + delta_scale * arm  # numpy (7,)

    # Convert to torch on robot's device with batch dim (1, 7)
    dev = robot.data.joint_pos.device
    target_t = torch.as_tensor(target_pos, dtype=torch.float32, device=dev).unsqueeze(0)

    # Target only the first 7 arm joints explicitly
    arm_joint_ids = torch.arange(0, 7, dtype=torch.long, device=dev)

    if mode == "position_target":
        controller.set_joint_position_target(target_t, joint_ids=arm_joint_ids)
    elif mode == "position":
        controller.set_joint_positions(target_t, joint_ids=arm_joint_ids)
    elif mode == "controller_apply":
        from omni.isaac.core.utils.types import ArticulationAction
        # Build a full-length vector (num_dofs) and fill arm portion
        jp_full = robot.data.joint_pos[0].detach().cpu().numpy()
        full_target = jp_full.copy()
        full_target[:7] = target_pos
        action = ArticulationAction(joint_positions=full_target)
        controller.apply_action(action)
    else:
        raise RuntimeError(f"Unsupported control mode: {mode}")

    # Gripper: best-effort; if a helper exists on robot, call it, else ignore
    for attr in ("set_gripper_target", "set_gripper_position", "set_gripper_joint_position"):
        if hasattr(robot, attr):
            try:
                getattr(robot, attr)(float(grip))
                break
            except Exception:
                continue


def step_physics_and_render():
    # Advance physics using SimulationContext owned by the env (bypasses ActionManager)
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "step"):
        sim.step(render=False)
    else:
        raise RuntimeError("SimulationContext not available; cannot step physics directly.")
    for cam in cams.values():
        if cam is not None:
            cam.update(dt=step_dt)


# ------------------------- Rollout -------------------------
steps_taken = 0
pending = None  # (K, A) torch on device
pending_idx = 0
frames = {name: [] for name, cam in cams.items() if cam is not None}

while steps_taken < args.max_steps:
    # Refresh observation for policy when needed
    if pending is None or pending_idx >= pending.shape[0]:
        jp = robot.data.joint_pos
        if jp is None:
            raise RuntimeError("robot.data.joint_pos is None; cannot build joint_pos[9] feature")
        jp0 = jp[0].detach().cpu().numpy()
        if jp0.shape[-1] >= 9:
            jp0 = jp0[:9]
        elif jp0.shape[-1] < 9:
            jp0 = np.pad(jp0, (0, 9 - jp0.shape[-1]))

        obs_dict = build_obs_dict(cams, jp0, int(args.camera_h), int(args.camera_w))

        with torch.inference_mode():
            traj = predict_action(obs_dict)

        traj_t = torch.from_numpy(traj) if isinstance(traj, np.ndarray) else traj
        if traj_t.ndim == 1:
            traj_t = traj_t[None, ...]
        elif traj_t.ndim == 3:  # (B, H, A)
            traj_t = traj_t[0]
        K = min(int(max(1, args.actions_per_inference)), traj_t.shape[0])
        pending = traj_t[:K].to(device).contiguous()
        pending_idx = 0

    # Apply next action directly
    act_row = pending[pending_idx]
    pending_idx += 1
    apply_arm_gripper(act_row)

    # Step physics and render
    step_physics_and_render()

    # Record frames
    for name, cam in cams.items():
        if cam is None:
            continue
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        rgb = resize_hwc_uint8(to_rgb_uint8(rgb), int(args.camera_h), int(args.camera_w))
        if writers.get(name) is not None:
            try:
                writers[name].write(rgb)
            except Exception:
                pass
        else:
            frames[name].append(rgb)

    steps_taken += 1


# ------------------------- Close -------------------------
for name, w in writers.items():
    try:
        name_clean = name.replace("_camera", "")
        path = out_dir / f"eval_diffpo_direct_{name_clean}.mp4"
        if w is not None:
            w.close()
            print(f"[direct] Saved video (streamed): {path}")
        else:
            if frames.get(name):
                iio.imwrite(path, frames[name], fps=fps)
                print(f"[direct] Saved video: {path}")
    except Exception:
        pass

env.close()
app.close()


