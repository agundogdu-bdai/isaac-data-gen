# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka \
    import joint_pos_env_cfg as joint_pos_cfg


@configclass
class FrankaCubeLiftEnvCameraCfg(joint_pos_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Disable debug visualization for object pose when present
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose.debug_vis = False

        # Camera resolution
        self.cam_width = 320
        self.cam_height = 240

        # Front/overview camera
        front_cam_cfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0,
            height=self.cam_height,
            width=self.cam_width,
            data_types=["rgb"],
            debug_vis=True,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(1.6, 0.0, 0.7),
                rot=(0.35355, -0.61237, -0.61237, 0.35355),
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1e4),
            ),
        )
        setattr(self.scene, "front_camera", front_cam_cfg)

        # Wrist camera on the end-effector - matching cabinet task position
        wrist_cam_cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_cam",
            update_period=0,
            height=self.cam_height,
            width=self.cam_width,
            debug_vis=True,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.04, 0.0, 0.04),  # Matching cabinet task
                rot=(-0.683, 0.183, 0.183, -0.683),  # Matching cabinet task
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0,  # Matching cabinet task
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1e4),  # Matching cabinet task
            ),
        )
        setattr(self.scene, "wrist_camera", wrist_cam_cfg)


@configclass
class FrankaCubeLiftEnvCameraCfg_PLAY(FrankaCubeLiftEnvCameraCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

