## Isaac Lab data collection and faithful replay (v2)

This pipeline records the executed articulation commands and initial scene state during collection, and replays them directly to achieve visual and kinematic fidelity.

### Files

- `collect_v2.py`: collects episodes; logs raw actions, executed joint targets, state, torques, camera frames, and initial scene snapshot.
- `replay_v2.py`: restores the snapshot and plays back recorded joint targets, stepping physics at the correct rate; writes an MP4.
- `vpl_saver_v2.py`: optional helper to save H5/MP4 (not required to run).
- `collect_v2.sh`: container wrapper to run collection and copy results out.
- `replay_v2.sh`: container wrapper to replay one episode and copy video out.

### Requirements

- Running Docker container named `isaaclab-test` with Isaac Lab installed at `/workspace/isaaclab` and `./isaaclab.sh` available.
- Your trained checkpoint at `/workspace/model_trained.pt` (edit path or script as needed).

### Quick start

1) Collect data

```bash
bash collect_v2.sh
```

Outputs dataset into `./vpl_collected_v2/episode_XXX/` with `episode_XXX.h5` and per-camera MP4s.

2) Replay an episode

```bash
bash replay_v2.sh vpl_collected_v2/episode_000 top
# -> replay_episode_000_top_v2.mp4
```

### Run directly inside container (optional)

```bash
# collection
cd /workspace/isaaclab
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/collect_v2.py \
  --task Isaac-Open-Drawer-Franka-Camera-v0 \
  --checkpoint /workspace/model_trained.pt \
  --num_envs 1 \
  --num_episodes 3 \
  --steps 200 \
  --seed 123 \
  --output /workspace/datasets/vpl_data_v2 \
  --headless

# replay
ENABLE_CAMERAS=1 ./isaaclab.sh -p /workspace/replay_v2.py \
  --episode /workspace/datasets/vpl_data_v2/episode_000 \
  --camera top \
  --output /workspace/replay_episode_000_top_v2.mp4 \
  --headless
```

### Notes

- We log and reapply the executed articulation targets; replay does not call `env.step()`.
- Set `--num_envs 1` and a fixed `--seed` for cleaner datasets.
- If your camera names differ, adjust the `*_camera` names in the scripts.


