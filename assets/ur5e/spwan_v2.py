import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

robot = p.loadURDF("assets/ur5e/ur5e.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# End-effector link index
ee_link = 7  # UR5e last link (might vary depending on your URDF)

# Lower/upper joint limits
joint_indices = [i for i in range(p.getNumJoints(robot)) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]

# Main loop
target_pos = [0.5, 0, 0.5]  # initial target
target_orient = p.getQuaternionFromEuler([0, np.pi, 0])

while True:
    p.stepSimulation()
    time.sleep(1/240.)

    # Draw target sphere
    p.addUserDebugLine(target_pos, [target_pos[0], target_pos[1], target_pos[2]+0.1], [1,0,0], 3, lifeTime=0.1)
    p.addUserDebugText("Target", target_pos, [1,0,0], 1.5, lifeTime=0.1)

    # Solve IK
    joint_poses = p.calculateInverseKinematics(robot, ee_link, target_pos, target_orient)
    
    # Command joints
    for i, j in enumerate(joint_indices):
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, joint_poses[i], force=500)

    # Mouse click event
    mouse_events = p.getMouseEvents()
    for e in p.getMouseEvents():
        if e[3] == 0 and (e[1] & p.KEY_WAS_TRIGGERED):  # 0 = left button
            mouse_x, mouse_y = e[2], e[3]
            width, height, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()

            ray_start, ray_end = p.computeViewProjectionMatrices(mouse_x, mouse_y)
            ray_from, ray_to = p.getMouseRay(mouse_x, mouse_y)

            hits = p.rayTest(ray_from, ray_to)
            for hit in hits:
                if hit[0] >= 0:  # Valid hit
                    target_pos = hit[3]  # World coordinate
                    print(f"New target: {target_pos}")
