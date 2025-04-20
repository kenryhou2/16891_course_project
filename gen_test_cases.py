import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import time

# ----------------------------- CONFIG -----------------------------
URDF_PATH = "assets/ur5e/ur5e.urdf"
OUTPUT_FILE = "src/multiarm_test_cases.json"
NUM_TEST_CASES = 5
OVERLAPS = [0.3, 0.4, 0.5]  # 30%, 40%, 50%
ARM_COUNTS = [2, 3, 4]  # number of arms
GROUND_THRESHOLD = 0.01  # meters

# ------------------------- INITIAL SETUP --------------------------
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# ------------------------- HELPER FUNCTIONS -----------------------
def sample_random_pose(overlap_ratio, base_idx):
    """Sample a random base position with overlap."""
    spacing = 1.5
    reduction = overlap_ratio * spacing
    x = base_idx * (spacing - reduction)
    y = np.random.uniform(-0.3, 0.3)
    z = 0
    return [x, y, z]

def get_ur5e_joint_limits():
    """Return proper UR5e joint limits."""
    lower = [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
    upper = [ 2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi]
    return lower, upper

def reset_joints(robot_id, joint_values):
    for idx, val in enumerate(joint_values):
        p.resetJointState(robot_id, idx, val)

def robot_above_ground(robot_id):
    """Check if robot is above ground."""
    num_joints = p.getNumJoints(robot_id)
    link_indices = list(range(num_joints)) + [-1]  # include base (-1)
    for link in link_indices:
        pos, _ = p.getLinkState(robot_id, link)[:2] if link >= 0 else p.getBasePositionAndOrientation(robot_id)
        if pos[2] < GROUND_THRESHOLD:
            return False
    return True

def sample_valid_config(robot_id):
    """Sample a valid UR5e configuration above ground."""
    lower_limits, upper_limits = get_ur5e_joint_limits()
    max_attempts = 100
    for _ in range(max_attempts):
        config = np.random.uniform(lower_limits, upper_limits).tolist()
        reset_joints(robot_id, config)
        if robot_above_ground(robot_id):
            return config
    raise RuntimeError("Failed to sample a valid config after 100 tries")

# ------------------------- MAIN GENERATOR -------------------------
test_cases = []

for overlap in OVERLAPS:
    for num_arms in ARM_COUNTS:
        for case_id in range(NUM_TEST_CASES):

            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")

            robot_ids = []
            poses = []
            starts = {}
            goals = {}

            for arm_idx in range(num_arms):
                base_pose = sample_random_pose(overlap, arm_idx)
                robot_id = p.loadURDF(URDF_PATH, basePosition=base_pose, useFixedBase=True)
                robot_ids.append(robot_id)
                poses.append(base_pose)

            for idx, robot_id in enumerate(robot_ids):
                start_config = sample_valid_config(robot_id)
                starts[idx] = start_config

            for _ in range(50):
                p.stepSimulation()

            for idx, robot_id in enumerate(robot_ids):
                goal_config = sample_valid_config(robot_id)
                goals[idx] = goal_config

            test_cases.append({
                "overlap_ratio": overlap,
                "num_arms": num_arms,
                "poses": poses,
                "start_configs": starts,
                "goal_configs": goals
            })

# --------------------------- SAVE TO FILE ---------------------------
if os.path.dirname(OUTPUT_FILE):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(test_cases, f, indent=4)

print(f"Generated {len(test_cases)} test cases and saved to {OUTPUT_FILE}")

p.disconnect()
