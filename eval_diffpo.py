#!/usr/bin/env python3
# flake8: noqa
# -*- coding: utf-8 -*-
"""
Evaluate a W&B artifact policy (non-RL) on Isaac Lab for 1 episode and save videos.

Assumptions:
- Task: Isaac-Open-Drawer-Franka-Camera-v0
- Inputs: all camera RGBs (top, wrist, side) + joint_pos (shape [9]) as state feature
- Output: actions applied to env; save MP4(s) locally

Model loading strategy (best-effort):
1) If artifact dir contains a module specified by --policy_entry (e.g., inference:load_policy),
   we import it and call load_policy(artifact_dir, device) -> callable(obs_dict)->action(np or torch)
2) Else, try TorchScript .pt (.jit/.ts) modules named among common filenames
3) Else, raise a clear error instructing how to provide a loader

Observation passed to policy (dict of torch tensors on device):
  {
    'image_top':   (1, 3, H, W) float32 [0,1],
    'image_wrist': (1, 3, H, W) float32 [0,1],
    'image_side':  (1, 3, H, W) float32 [0,1],
    'joint_pos':   (1, 9)       float32
  }

"""
import argparse
import importlib.util
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch

import wandb
import os as _os

from isaaclab.app import AppLauncher


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
parser.add_argument(
    "--policy_entry",
    type=str,
    default="inference:load_policy",
    help="Optional 'module:function' within artifact dir that returns policy callable",
)
parser.add_argument(
    "--cams",
    type=str,
    default="top,wrist,side",
    help="Comma-separated cameras to record from: top,wrist,side",
)
parser.add_argument(
    "--stream",
    action="store_true",
    help="Stream MP4 to disk instead of buffering frames in memory",
)
parser.add_argument(
    "--loader_path",
    type=str,
    default="",
    help="Optional absolute path to a Python file exposing load_policy(artifact_dir, device)",
)
parser.add_argument(
    "--allow_unpickle",
    action="store_true",
    help="Allow loading non-weights-only torch checkpoints (unsafe if untrusted)",
)
parser.add_argument(
    "--actions_per_inference",
    type=int,
    default=1,
    help="Number of env steps to execute from each predicted trajectory before re-inferencing",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True


# ------------------------- Launch Kit App -------------------------
app = AppLauncher(args).app

# Import after Kit is up
from isaaclab_tasks.utils import parse_env_cfg  # type: ignore
printed_shapes = False


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


def hwc_to_nchw01(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None] / 255.0  # (1,3,H,W)
    return x.to(device)


 


def build_obs_dict(
    cams: dict,
    joint_pos_1xD: np.ndarray,
    H: int,
    W: int,
    device: torch.device,
    order: list[str] | None = None,
) -> dict:
    # Build only the features expected by DiffPO policy: 'color' and 'joint_pos'.
    # 'color' is stacked per-camera HWC tensors into shape (B, Cams, H, W, 3).
    if order is None:
        # Use canonical keys already produced by _resolve_camera_mapping
        order = ["top_camera", "wrist_camera", "side_camera"]

    per_cam_images: list[np.ndarray] = []  # list of (H, W, 3) uint8
    for name in order:
        cam = cams[name]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        rgb = resize_hwc_uint8(to_rgb_uint8(rgb), H, W)  # (H, W, 3) uint8
        per_cam_images.append(rgb)

    out: dict = {}
    # Assemble unbatched time=1 format expected by policy: (1, X, H, W, 3), float32 on device
    color_np = np.stack(per_cam_images, axis=0)[None, ...]  # uint8
    color_t = torch.from_numpy(color_np.astype(np.float32) / 255.0).to(device)
    out["color"] = color_t
    jp = torch.from_numpy(joint_pos_1xD.astype(np.float32)).to(device)
    if jp.ndim == 1:
        jp = jp[None, ...]
    out["joint_pos"] = jp
    # print norms of color and joint_pos
    # print(f"[eval_diffpo] color norms: {color_t.norm(p=2, dim=1)}")
    # print(f"[eval_diffpo] joint_pos norms: {jp.norm(p=2, dim=1)}")
    return out


def find_and_load_entry_policy(artifact_dir: Path, entry: str, device: torch.device) -> Optional[Callable]:
    if not entry:
        return None
    if ":" not in entry:
        return None
    mod_name, fn_name = entry.split(":", 1)
    # Resolve module path within artifact_dir
    for candidate in [f"{mod_name}.py", mod_name, os.path.join(mod_name, "__init__.py")]:
        p = artifact_dir / candidate
        if p.is_file():
            spec = importlib.util.spec_from_file_location(mod_name, str(p))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                if hasattr(module, fn_name):
                    fn = getattr(module, fn_name)
                    try:
                        policy = fn(artifact_dir=str(artifact_dir), device=device)
                        return policy
                    except Exception as e:
                        print(f"[eval_diffpo] load_policy via entry failed: {e}")
                        return None
    return None


def find_and_load_external_loader(loader_path: str, artifact_dir: Path, device: torch.device) -> Optional[Callable]:
    if not loader_path:
        return None
    p = Path(loader_path)
    if not p.is_file():
        print(f"[eval_diffpo] loader_path not found: {p}")
        return None
    mod_name = p.stem
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "load_policy"):
            try:
                return module.load_policy(artifact_dir=str(artifact_dir), device=device)
            except Exception as e:
                print(f"[eval_diffpo] external load_policy failed: {e}")
                return None
    return None


def find_and_load_torchscript(artifact_dir: Path, device: torch.device):
    cand_names = [
        "policy_ts.pt",
        "policy.jit.pt",
        "policy.pt",
        "model_ts.pt",
        "model.jit.pt",
        "model.pt",
    ]
    for name in cand_names:
        p = artifact_dir / name
        if p.is_file():
            try:
                m = torch.jit.load(str(p), map_location=device)
                m.eval()
                return m
            except Exception as e:
                print(f"[eval_diffpo] TorchScript load failed for {name}: {e}")
                continue
    return None


def make_policy_callable(raw_policy) -> Callable[[dict], torch.Tensor]:
    # Wraps various policy types into a callable(obs_dict)->torch.Tensor action
    if raw_policy is None:
        raise RuntimeError("Policy is None")

    def _call(obs: dict):
        if not hasattr(raw_policy, "predict_action"):
            raise RuntimeError("Policy does not implement predict_action(obs, batched=False)")
        out = raw_policy.predict_action(obs, batched=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        return out

    return _call


def find_and_load_ckpt(artifact_dir: Path, device: torch.device, allow_unpickle: bool = False):
    """Best-effort: load a .ckpt file if it contains a callable model.
    This will NOT reconstruct models from pure state_dict-only lightning checkpoints.
    """
    import glob
    ckpts = []
    for ext in ("*.ckpt", "*.pth", "*.pt"):
        ckpts.extend(glob.glob(str(artifact_dir / ext)))
    # Prefer .ckpt explicitly
    ckpts = sorted(ckpts, key=lambda p: (0 if p.endswith(".ckpt") else 1, len(p)))
    for path in ckpts:
        try:
            if allow_unpickle:
                # Permit unpickling with recommended safe globals allowlist when needed
                from torch.serialization import add_safe_globals
                import numpy as _np
                try:
                    add_safe_globals([_np.core.multiarray.scalar])
                except Exception:
                    pass
                obj = torch.load(path, map_location=device, weights_only=False)
            else:
                obj = torch.load(path, map_location=device)  # default weights_only behavior on 2.6
            # If it's a torchscript, return directly
            try:
                obj.eval()
            except Exception:
                pass
            # If object itself is callable module
            if hasattr(obj, "eval") and (callable(obj) or hasattr(obj, "forward") or hasattr(obj, "act")):
                return obj
            # If dict-like state_dict only, we cannot reconstruct
            if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model_state_dict")):
                print(f"[eval_diffpo] Found state_dict in {path} but no model class to load.")
                continue
        except Exception as e:
            print(f"[eval_diffpo] ckpt load failed for {path}: {e}")
            continue
    return None


# ------------------------- Build Env -------------------------
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

# Cameras: direct sensor keys from scene.sensors
sensors = scene.sensors if hasattr(scene, "sensors") and isinstance(scene.sensors, dict) else {}
available_sensor_keys = list(sensors.keys()) if isinstance(sensors, dict) else []
print(f"[eval_diffpo] Available sensors: {available_sensor_keys}")
cams = {
    "top_camera": sensors.get("top_camera"),
    "wrist_camera": sensors.get("wrist_camera"),
    "side_camera": sensors.get("side_camera"),
}
present = [n for n, c in cams.items() if c is not None]
if len(present) == 3:
    print("[eval_diffpo] ✅ Cameras found: top_camera, wrist_camera, side_camera")
else:
    print(f"[eval_diffpo] ⚠️ Expected 3 cameras; resolved {len(present)}. Missing: {[n for n in cams.keys() if cams[n] is None]}")

# Robot articulation (for joint_pos)
try:
    robot = scene.articulations["robot"]
except Exception:
    robot = getattr(scene, "_articulations", {}).get("robot", None)
assert robot is not None, "Robot articulation 'robot' not found."


# ------------------------- Optional: mock rai_tensors -------------------------
# Some checkpoints may reference 'rai_tensors' which is not required at inference.
# Provide a minimal shim so imports succeed without the dependency installed.
import sys, types  # noqa: E402

def _create_dummy_module(fullname: str) -> types.ModuleType:
    m = types.ModuleType(fullname)
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
from visuomotor.models.model_registry import REGISTRY  # defer import until kit is up

print(f"[eval_diffpo] Loading policy via REGISTRY from artifact: {args.artifact}")
policy_obj = REGISTRY.build_policy_from_artifact_name(
    name=args.artifact,
    device=device,
    suppress_version_warning=True,
)

policy_callable: Optional[Callable] = make_policy_callable(policy_obj)


# ------------------------- Single Episode Rollout -------------------------
H, W = int(args.camera_h), int(args.camera_w)
frames = {k: [] for k, v in cams.items() if v is not None}

# Prepare output streaming (optional) to reduce memory
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
fps = int(round(1.0 / step_dt)) if step_dt > 0 else 30
writers = {}
if args.stream:
    try:
        for name in frames.keys():
            path = out_dir / f"eval_diffpo_{name.replace('_camera','')}.mp4"
            f = iio.imopen(path, "w", plugin="pyav")
            try:
                f.init_video_stream(codec="h264", fps=fps)
            except Exception:
                # Older imageio may accept fps argument in imopen
                pass
            writers[name] = f
            print(f"[eval_diffpo] Streaming video for {name} -> {path}")
    except Exception as e:
        print(f"[eval_diffpo] Streaming setup failed ({e}); falling back to buffering frames.")
        # fall back to buffered mode
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        writers = {}
        args.stream = False

obs, _ = env.reset()
steps_taken = 0
terminated = False
truncated = False

pending_actions: Optional[torch.Tensor] = None  # shape (K, action_dim) on device
pending_idx: int = 0

while not (terminated or truncated) and steps_taken < args.max_steps:
    # Update cameras to the next env-step time
    for cam in cams.values():
        if cam is not None:
            cam.update(dt=step_dt)

    # If no pending actions remaining, run a new inference to get a trajectory
    if pending_actions is None or pending_idx >= pending_actions.shape[0]:
        # Build observation dict for policy (dataset-like CPU/np or tensors as expected by predict_action)
        jp = robot.data.joint_pos
        if jp is None:
            raise RuntimeError("robot.data.joint_pos is None; cannot build joint_pos[9] feature")
        jp0 = jp[0].detach().cpu().numpy()
        if jp0.shape[-1] >= 9:
            jp0 = jp0[:9]
        elif jp0.shape[-1] < 9:
            jp0 = np.pad(jp0, (0, 9 - jp0.shape[-1]))

        obs_dict = build_obs_dict(cams, jp0, H, W, device, order=["top_camera", "wrist_camera", "side_camera"]) 

        if not printed_shapes:
            for k, v in obs_dict.items():
                if isinstance(v, torch.Tensor):
                    print(
                        f"[eval_diffpo] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
                    )
            printed_shapes = True

        # Compute trajectory
        with torch.inference_mode():
            traj = policy_callable(obs_dict)

        # Normalize shape to (H, A) and move to device
        if isinstance(traj, np.ndarray):
            traj_t = torch.from_numpy(traj)
        else:
            traj_t = traj
        traj_t = traj_t.to(device)
        if traj_t.ndim == 1:
            traj_t = traj_t[None, ...]  # (1, A)
        elif traj_t.ndim == 3:
            # (B, H, A) -> assume B==1
            traj_t = traj_t[0]
        # Now (H, A)
        K = int(max(1, args.actions_per_inference))
        K = min(K, traj_t.shape[0])
        pending_actions = traj_t[:K].contiguous()
        pending_idx = 0

    # Take next action (A,) -> (1, A)
    act_row = pending_actions[pending_idx]
    pending_idx += 1
    act_t = act_row[None, ...]

    # Step
    obs, rew, terminated, truncated, info = env.step(act_t)

    # Capture frames after step
    for name, cam in cams.items():
        if cam is None:
            continue
        cam.update(dt=step_dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        rgb = resize_hwc_uint8(to_rgb_uint8(rgb), H, W)
        if writers:
            try:
                writers[name].write(rgb)
            except Exception as e:
                print(f"[eval_diffpo] Write failed for {name}: {e}")
        else:
            frames[name].append(rgb)

    steps_taken += 1
    if (steps_taken % 20) == 0:
        try:
            r0 = float(rew[0])
        except Exception:
            r0 = 0.0
        print(f"  step {steps_taken:04d} | r0={r0:.3f}")


# ------------------------- Save MP4(s) -------------------------
if writers:
    # Close streaming writers
    for name, w in writers.items():
        try:
            w.close()
            name_clean = name.replace("_camera", "")
            saved_path = out_dir / f"eval_diffpo_{name_clean}.mp4"
            print(f"[eval_diffpo] Saved video (streamed): {saved_path}")
        except Exception:
            pass
else:
    # Buffered save
    for name, fr_list in frames.items():
        if fr_list:
            path = out_dir / f"eval_diffpo_{name.replace('_camera','')}.mp4"
            iio.imwrite(path, fr_list, fps=fps)
            print(f"[eval_diffpo] Saved video: {path}")

print(f"[eval_diffpo] Done. steps={steps_taken}, terminated={terminated}, truncated={truncated}")

env.close()
app.close()


