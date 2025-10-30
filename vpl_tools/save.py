# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to save.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    default=True,
    help="Use the pre-trained checkpoint from Nucleus.",
)
# Allow opting out explicitly
parser.add_argument(
    "--no-use_pretrained_checkpoint",
    dest="use_pretrained_checkpoint",
    action="store_false",
    help="Disable use of the pre-trained checkpoint.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
from vpl_saver import VPLSaver

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "").replace("-Camera", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # (removed earlier pre-construction overrides for clarity)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- Force N-step episodes via episode_length_s (default 84) ---
    TARGET_STEPS = int(os.getenv("MAX_EPISODE_STEPS", "84"))
    env.unwrapped.cfg.episode_length_s = TARGET_STEPS * env.unwrapped.step_dt
    print(
        f"[HORIZON] step_dt={env.unwrapped.step_dt:.6f}s -> "
        f"episode_length_s={env.unwrapped.cfg.episode_length_s:.6f}s "
        f"(max_episode_length={env.unwrapped.max_episode_length})"
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment and obtain initial observations
    # RslRlVecEnvWrapper doesn't return anything from reset, use get_observations()
    env.reset()
    obs_dict = env.get_observations()
    
    # Debug: print what get_observations returns
    print(f"[DEBUG] get_observations() type: {type(obs_dict)}")
    if isinstance(obs_dict, dict):
        print(f"[DEBUG] get_observations() keys: {obs_dict.keys()}")
        obs = obs_dict["policy"]
    elif isinstance(obs_dict, tuple):
        print(f"[DEBUG] get_observations() returned tuple of length {len(obs_dict)}")
        # If tuple, extract the observation dict (usually first element)
        obs_dict = obs_dict[0] if len(obs_dict) > 0 else obs_dict
        if isinstance(obs_dict, dict):
            obs = obs_dict["policy"]
        else:
            obs = obs_dict  # Assume it's already the tensor
    else:
        obs = obs_dict  # Assume it's already the tensor
    
    print(f"[DEBUG] obs type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    
    # Track episodes per environment to stop when all envs have collected enough
    num_envs = env.unwrapped.num_envs
    episodes_per_env = torch.zeros(num_envs, dtype=torch.int32)
    target_episodes_per_env = args_cli.num_episodes
    
    vpl_saver = VPLSaver(base_dir="/workspace/data")
    
    # simulate environment
    while simulation_app.is_running():
        # start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            # RslRlVecEnvWrapper.step() returns obs_dict["policy"] as first return value
            obs, _, dones, _ = env.step(actions)
        
        # Store data
        vpl_saver.store(actions, env)

        if dones.any(): 
            # Write completed episodes to disk
            vpl_saver.write(
                dones=dones,
                compression_filter="gzip",
                compression_opts=4
            )
            
            # Update per-environment episode counters
            for env_idx in range(num_envs):
                if dones[env_idx]:
                    episodes_per_env[env_idx] += 1
            
            total_episodes = episodes_per_env.sum().item()
            min_episodes = episodes_per_env.min().item()
            max_episodes = episodes_per_env.max().item()
            
            print(f"Total episodes: {total_episodes} | Per env: min={min_episodes}, max={max_episodes}")
        
            # Stop when ALL environments have collected target episodes
            if episodes_per_env.min() >= target_episodes_per_env:
                print(f"\n✓ All {num_envs} environments collected {target_episodes_per_env} episodes each")
                print(f"✓ Total episodes collected: {total_episodes}")
                env.close()
                return
            
        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


