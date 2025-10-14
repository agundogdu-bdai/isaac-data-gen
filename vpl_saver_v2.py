#!/usr/bin/env python3
import json
from pathlib import Path
import h5py
import numpy as np
import imageio.v3 as iio


class IsaacLabSaverV2:
    def __init__(self, base_dir: str, fps: int = 30):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.ep = 0

    def write_episode(self, episode_data: dict):
        """episode_data keys:
           - scene_state0 (nested dict of np arrays)
           - action_raw (T,A)
           - cmd_pos / cmd_vel / cmd_effort (optional, T,D)
           - state_pos / state_vel / tau_applied (optional, T,D)
           - frames: dict[name] -> list[np.uint8 HxWx3]
           - attrs: dict (task, seed, physics_dt, step_dt, decimation)
        """
        d = episode_data
        ep_dir = self.base / f"episode_{self.ep:03d}"
        ep_dir.mkdir(exist_ok=True)
        h5_path = ep_dir / f"episode_{self.ep:03d}.h5"
        with h5py.File(h5_path, "w") as f:
            for k, v in d["attrs"].items():
                f.attrs[k] = v
            # scene state
            def write_state(g, s):
                for k, v in s.items():
                    if isinstance(v, dict):
                        write_state(g.create_group(k), v)
                    else:
                        g.create_dataset(k, data=v)
            write_state(f.create_group("scene_state0"), d["scene_state0"])
            # actions & commands & states
            f.create_dataset("action/raw", data=d["action_raw"])
            if "cmd_pos" in d:
                f.create_dataset("command/joint_pos_target", data=d["cmd_pos"])
            if "cmd_vel" in d:
                f.create_dataset("command/joint_vel_target", data=d["cmd_vel"])
            if "cmd_effort" in d:
                f.create_dataset("command/joint_effort_target", data=d["cmd_effort"])
            if "state_pos" in d:
                f.create_dataset("state/joint_pos", data=d["state_pos"])
            if "state_vel" in d:
                f.create_dataset("state/joint_vel", data=d["state_vel"])
            if "tau_applied" in d:
                f.create_dataset("state/applied_torque", data=d["tau_applied"])
            # color
            cam_stacks = []
            cam_names = []
            for name, fr in (d.get("frames") or {}).items():
                if fr:
                    cam_stacks.append(np.stack(fr, axis=0)[:, None, ...])
                    cam_names.append(name)
            if cam_stacks:
                color = np.concatenate(cam_stacks, axis=1)
                f.create_dataset("color", data=color)
                f.create_dataset("color_cam_names", data=np.array(cam_names, dtype="S"))
        # also write mp4 per cam
        for name, fr in (d.get("frames") or {}).items():
            if fr:
                iio.imwrite(ep_dir / f"{name}.mp4", fr, fps=self.fps)
        self.ep += 1


