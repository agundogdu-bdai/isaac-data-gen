# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka Cabinet Environment with Camera Support
This configuration adds properly positioned wrist, top, and side cameras
and disables coordinate system visualization.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import (
    FrankaCabinetEnvCfg,
)


def apply_camera_config(env_cfg):
    """
    Apply camera configuration to an existing environment config.
    """
    # Disable coordinate system debug visualization
    if hasattr(env_cfg.scene, "cabinet_frame"):
        env_cfg.scene.cabinet_frame.debug_vis = False

    # Camera resolution
    cam_width = 160
    cam_height = 120

    # Top Camera Configuration
    top_cam_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        update_period=0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb"],
        debug_vis=False,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 3.2),
            rot=(0.0, 0.7071, -0.7071, 0.0),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1e4),
        ),
    )

    # Wrist Camera Configuration
    wrist_cam_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
        update_period=0,
        height=cam_height,
        width=cam_width,
        debug_vis=False,
        data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.04, 0.0, 0.04),  # Original cabinet position
            rot=(-0.683, 0.183, 0.183, -0.683),  # Original cabinet rotation
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1e4),
        ),
    )

    # Side Camera Configuration
    side_cam_cfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/side_camera",
        update_period=0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb"],
        debug_vis=False,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.7, 0.18, 2.0),
            rot=(0.8924, 0.0996, 0.4384, -0.0489),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1e4),
        ),
    )

    # Add cameras to scene configuration
    if hasattr(env_cfg.scene, "sensors"):
        if env_cfg.scene.sensors is None:
            env_cfg.scene.sensors = {}
        env_cfg.scene.sensors["top_camera"] = top_cam_cfg
        env_cfg.scene.sensors["wrist_camera"] = wrist_cam_cfg
        env_cfg.scene.sensors["side_camera"] = side_cam_cfg
    else:
        env_cfg.scene.top_camera = top_cam_cfg
        env_cfg.scene.wrist_camera = wrist_cam_cfg
        env_cfg.scene.side_camera = side_cam_cfg

    return env_cfg


# Create camera-enabled environment config by copying base config and adding cameras
FrankaCabinetEnvCameraCfg = apply_camera_config(FrankaCabinetEnvCfg())


# Play mode configuration
@configclass
class FrankaCabinetEnvCameraCfg_PLAY_Class(FrankaCabinetEnvCfg):
    """Play mode with fewer environments."""
    pass


FrankaCabinetEnvCameraCfg_PLAY = apply_camera_config(FrankaCabinetEnvCameraCfg_PLAY_Class())
FrankaCabinetEnvCameraCfg_PLAY.scene.num_envs = 4
FrankaCabinetEnvCameraCfg_PLAY.scene.env_spacing = 2.5
if hasattr(FrankaCabinetEnvCameraCfg_PLAY, "observations") and hasattr(
    FrankaCabinetEnvCameraCfg_PLAY.observations, "policy"
):
    FrankaCabinetEnvCameraCfg_PLAY.observations.policy.enable_corruption = False


