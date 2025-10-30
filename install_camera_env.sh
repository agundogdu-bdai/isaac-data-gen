#!/bin/bash
# Script to install custom camera environment configuration into IsaacLab container

set -e

echo "=========================================="
echo "Installing Custom Camera Environment"
echo "=========================================="
echo ""

CONTAINER_NAME="isaaclab-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' is not running."
    echo "Please start the container first with:"
    echo "  docker start ${CONTAINER_NAME}"
    exit 1
fi

echo "[1/4] Creating custom environment directories in container..."
docker exec ${CONTAINER_NAME} bash -c "mkdir -p /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka"
docker exec ${CONTAINER_NAME} bash -c "mkdir -p /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka"
echo "✓ Directory created"
echo ""

echo "[2/4] Copying environment configuration files..."
docker cp "${SCRIPT_DIR}/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_camera_cfg.py" \
    ${CONTAINER_NAME}:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/
docker cp "${SCRIPT_DIR}/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_camera_env_cfg.py" \
    ${CONTAINER_NAME}:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/
echo "✓ Configuration file copied"
echo ""

echo "[3/4] Registering new Cabinet environments..."
# Backup original __init__.py if it exists and hasn't been backed up
docker exec ${CONTAINER_NAME} bash -c "
cd /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka
if [ -f __init__.py ] && [ ! -f __init__.py.backup ]; then
    cp __init__.py __init__.py.backup
    echo '  • Backed up original __init__.py'
fi
"

# Add the new environment registrations to __init__.py
docker exec ${CONTAINER_NAME} bash -c "cat >> /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/__init__.py << 'EOF'

##
# Camera-Enabled Environments (Custom)
##

gym.register(
    id=\"Isaac-Open-Drawer-Franka-Camera-v0\",
    entry_point=\"isaaclab.envs:ManagerBasedRLEnv\",
    kwargs={
        \"env_cfg_entry_point\": f\"{__name__}.joint_pos_env_camera_cfg:FrankaCabinetEnvCameraCfg\",
    },
    disable_env_checker=True,
)

gym.register(
    id=\"Isaac-Open-Drawer-Franka-Camera-Play-v0\",
    entry_point=\"isaaclab.envs:ManagerBasedRLEnv\",
    kwargs={
        \"env_cfg_entry_point\": f\"{__name__}.joint_pos_env_camera_cfg:FrankaCabinetEnvCameraCfg_PLAY\",
    },
    disable_env_checker=True,
)
EOF
"
echo "✓ Cabinet environments registered"
echo ""

echo "[4/4] Registering new Lift environments..."
docker exec ${CONTAINER_NAME} bash -c "cat >> /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py << 'EOF'

##
# Joint Position Control with Camera (Lift)
##

gym.register(
    id=\"Isaac-Lift-Cube-Franka-Camera-v0\",
    entry_point=\"isaaclab.envs:ManagerBasedRLEnv\",
    kwargs={
        \"env_cfg_entry_point\": f\"{__name__}.joint_pos_camera_env_cfg:FrankaCubeLiftEnvCameraCfg\",
        \"rsl_rl_cfg_entry_point\": f\"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg\",
    },
    disable_env_checker=True,
)

gym.register(
    id=\"Isaac-Lift-Cube-Franka-Camera-Play-v0\",
    entry_point=\"isaaclab.envs:ManagerBasedRLEnv\",
    kwargs={
        \"env_cfg_entry_point\": f\"{__name__}.joint_pos_camera_env_cfg:FrankaCubeLiftEnvCameraCfg_PLAY\",
        \"rsl_rl_cfg_entry_point\": f\"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg\",
    },
    disable_env_checker=True,
)
EOF"
echo "✓ Lift environments registered"
echo ""

echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "New environments available:"
echo "  • Isaac-Open-Drawer-Franka-Camera-v0        (standard, with cameras)"
echo "  • Isaac-Open-Drawer-Franka-Camera-Play-v0   (4 envs, testing mode)"
echo ""
echo "These environments include:"
echo "  ✓ Wrist camera (attached to panda_hand)"
echo "  ✓ Top-down camera (bird's eye view)"
echo "  ✓ Disabled coordinate system visualization"
echo ""
echo "To test the new environment, run:"
echo "  docker exec -it ${CONTAINER_NAME} bash -c \"cd /workspace/isaaclab && \\"
echo "    ENABLE_CAMERAS=1 ISAAC_SIM_HEADLESS=1 CARB_WINDOWING_USE_EGL=1 \\"
echo "    ./isaaclab.sh -p scripts/environments/zero_agent.py \\"
echo "      --task Isaac-Open-Drawer-Franka-Camera-Play-v0 \\"
echo "      --enable_cameras \\"
echo "      --num_envs 4\""
echo ""

