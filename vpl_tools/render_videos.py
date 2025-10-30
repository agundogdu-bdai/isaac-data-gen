#!/usr/bin/env python3
import argparse
import os
import h5py
import numpy as np
import imageio


def render_episode(h5_path: str, out_dir: str, fps: int = 20, cams: list[str] | None = None):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        if "color" not in f:
            print(f"No 'color' dataset in {h5_path}, skipping")
            return
        color = f["color"][:]  # shape: (T, C, H, W, 3)
        if color.ndim != 5:
            print(f"Unexpected 'color' shape {color.shape} in {h5_path}, skipping")
            return
        T, C, H, W, _ = color.shape
        # Determine camera names if available; else cam_0..cam_N
        cam_names = [f"cam_{i}" for i in range(C)]
        # Render per-camera
        for cam_idx in range(C):
            cam_name = cam_names[cam_idx]
            if cams and cam_name not in cams:
                continue
            frames = (np.clip(color[:, cam_idx], 0, 1) * 255).astype(np.uint8) if color.dtype in (np.float32, np.float64) else color[:, cam_idx].astype(np.uint8)
            out_path = os.path.join(out_dir, f"{cam_name}.mp4")
            imageio.mimwrite(out_path, frames, fps=fps, codec="libx264")
            print(f"Wrote {out_path} [{T} frames]")


def render_all(base_dir: str, out_root: str, fps: int = 20, cams: list[str] | None = None):
    episodes = sorted([d for d in os.listdir(base_dir) if d.startswith("episode_")])
    if not episodes:
        print(f"No episodes under {base_dir}")
        return
    for ep_dir in episodes:
        idx = ep_dir.split("_")[1]
        h5_path = os.path.join(base_dir, ep_dir, f"episode_{idx}.h5")
        if not os.path.isfile(h5_path):
            print(f"Missing file: {h5_path}, skipping")
            continue
        out_dir = os.path.join(out_root, ep_dir)
        render_episode(h5_path, out_dir, fps=fps, cams=cams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MP4s from VPL HDF5 episodes")
    parser.add_argument("--base_dir", type=str, default="/workspace/vpl_tools/data", help="Input data dir with episode_*/episode_*.h5")
    parser.add_argument("--out_dir", type=str, default="/workspace/videos", help="Output videos root dir")
    parser.add_argument("--fps", type=int, default=20, help="FPS for mp4")
    parser.add_argument("--cams", type=str, default=None, help="Comma-separated camera names to include (e.g., cam_0,cam_1)")
    args = parser.parse_args()

    cams = [c.strip() for c in args.cams.split(",")] if args.cams else None
    render_all(args.base_dir, args.out_dir, fps=args.fps, cams=cams)


