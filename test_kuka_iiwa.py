import pybullet as p
import pybullet_data
import time
import numpy as np
import math


# Function to create a simple scenario with robot arm moving in configuration space
def create_robot_arm_scenario():
    # Connect to physics server
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity and load ground plane
    p.setGravity(0, 0, -9.8)
    plane_id = p.loadURDF("plane.urdf")

    # Load a pre-built robot arm (kuka)
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

    # Get number of joints and info
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")

    # Print joint info
    joint_indices = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1]}, Type: {joint_info[2]}")
        if joint_info[2] != p.JOINT_FIXED:
            joint_indices.append(i)

    # Set camera position for better view
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

    # Function to get current joint states as input
    def get_state():
        joint_states = []
        for joint_id in joint_indices:
            joint_state = p.getJointState(robot_id, joint_id)
            joint_states.append(joint_state[0])  # Position value
        return joint_states

    # Function to set joint states as output
    def set_state(joint_positions):
        for i, joint_id in enumerate(joint_indices):
            p.setJointMotorControl2(
                bodyIndex=robot_id, jointIndex=joint_id, controlMode=p.POSITION_CONTROL, targetPosition=joint_positions[i], force=500
            )

    # Create a simple motion in configuration space
    # Moving in sinusoidal patterns
    start_time = time.time()
    run_time = 30  # seconds

    while time.time() - start_time < run_time:
        # Calculate time for sinusoidal motion
        t = time.time() - start_time

        # Create sinusoidal motion pattern for each joint
        new_positions = []
        for i in range(len(joint_indices)):
            # Different amplitude and frequency for each joint
            amplitude = 0.5 * (i + 1) / len(joint_indices)
            frequency = 0.5 * (i + 1)
            phase = i * math.pi / 4

            # Calculate new position
            new_pos = amplitude * math.sin(frequency * t + phase)
            new_positions.append(new_pos)

        # Set the new state
        set_state(new_positions)

        # Get current state (could be used for feedback or logging)
        current_state = get_state()

        # Print current state every second
        if round(t) == t:
            print(f"Time: {t}s, Joint Positions: {[round(pos, 2) for pos in current_state]}")

        # Step simulation
        p.stepSimulation()
        time.sleep(1 / 240)  # Slow down simulation if needed

    # Disconnect when done
    p.disconnect()


if __name__ == "__main__":
    create_robot_arm_scenario()
