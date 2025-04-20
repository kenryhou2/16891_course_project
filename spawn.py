import pybullet as p
import pybullet_data
import time
import json
import os

# Connect to physics server
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

# Load ground plane
plane_id = p.loadURDF("plane.urdf")

URDF_PATH = "assets/ur5e/ur5e.urdf"

# Load multiple UR5e robots
robot_paths = [URDF_PATH, URDF_PATH]
robot_positions = [[0, 0, 0], [1, 0, 0]]
robot_ids = []

for i, pos in enumerate(robot_positions):
    robot_id = p.loadURDF(robot_paths[i], basePosition=pos, useFixedBase=True)
    robot_ids.append(robot_id)

# Create sliders for each joint
joint_sliders = {}
num_robots = len(robot_ids)
robot_joint_indices = []

for r_idx, robot_id in enumerate(robot_ids):
    joint_indices = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            lower = info[8]
            upper = info[9]
            if lower > upper:  # Invalid limits
                lower, upper = -3.14, 3.14
            slider = p.addUserDebugParameter(f"Robot{r_idx}_Joint{j}", lower, upper, 0)
            joint_sliders[(r_idx, j)] = slider
            joint_indices.append(j)
    robot_joint_indices.append(joint_indices)

# Storage for start and goal configs
start_configs = {}
goal_configs = {}

print("\nControls:")
print("[S] - Save current as start config")
print("[G] - Save current as goal config")
print("[E] - Export all to saved_configs.json\n")

while True:
    p.stepSimulation()
    time.sleep(1/240.)

    # Read keys
    keys = p.getKeyboardEvents()
    for k in keys:
        if keys[k] & p.KEY_WAS_TRIGGERED:
            if k == ord('s') or k == ord('S'):
                for r_idx, robot_id in enumerate(robot_ids):
                    config = []
                    for j in robot_joint_indices[r_idx]:
                        slider = joint_sliders[(r_idx, j)]
                        val = p.readUserDebugParameter(slider)
                        config.append(val)
                    start_configs[r_idx] = config
                print("Saved START configuration.")

            if k == ord('g') or k == ord('G'):
                for r_idx, robot_id in enumerate(robot_ids):
                    config = []
                    for j in robot_joint_indices[r_idx]:
                        slider = joint_sliders[(r_idx, j)]
                        val = p.readUserDebugParameter(slider)
                        config.append(val)
                    goal_configs[r_idx] = config
                print("Saved GOAL configuration.")

            if k == ord('e') or k == ord('E'):
                data = {
                    "start": start_configs,
                    "goal": goal_configs
                }
                with open("saved_configs.json", "w") as f:
                    json.dump(data, f, indent=4)
                print("Exported configurations to saved_configs.json!")

    # Actively control joints based on sliders
    for r_idx, robot_id in enumerate(robot_ids):
        for j in robot_joint_indices[r_idx]:
            slider = joint_sliders[(r_idx, j)]
            target_pos = p.readUserDebugParameter(slider)
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=500
            )
