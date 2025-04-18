"""
Visualization utilities for simulation data.
"""

import matplotlib.pyplot as plt
import logging
import time

from typing import Dict, List, Optional
import pinocchio
import pybullet
import numpy as np

logger = logging.getLogger(__name__)


def config_to_position_path(config_path, robot_model):
    pos_path = []
    for config in config_path:
        jointConfiguration = np.array(config)

        # Recalculate FK to get end-effector pose
        pinocchio.forwardKinematics(robot_model.pinocchio_model, robot_model.pinocchio_data, jointConfiguration)
        pinocchio.updateFramePlacements(robot_model.pinocchio_model, robot_model.pinocchio_data)

        # Get end-effector position
        ee_position = robot_model.pinocchio_data.oMf[robot_model.pinocchio_ee_id]

        _, orientation = pybullet.getBasePositionAndOrientation(robot_model.robot_id)
        rot = pinocchio.Quaternion(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2],
        ).toRotationMatrix()
        ee_position = rot @ ee_position.translation + np.array(robot_model.robot_position)

        pos_path.append(ee_position.tolist())

    return pos_path


class DataVisualizer:
    """
    Utilities for visualizing simulation data.
    """

    @staticmethod
    def plot_joint_trajectories(history: Dict, title: str = "Joint Trajectories"):
        """
        Plot joint trajectories from simulation history.

        Args:
            history: Simulation history with 'time' and 'state' keys
            title: Plot title
        """
        times = history["time"]
        states = history["states"]
        robot_ids = list(history["states"].keys())

        if not times or not states:
            logger.warning("No data to visualize")
            return

        plt.figure(figsize=(10, 6))
        colors = ["b", "r", "g", "c", "m", "y", "k"]

        # Plot each robot's joint trajectories
        for robot_index, robot_id in enumerate(robot_ids):
            if robot_id not in history["states"] or not history["states"][robot_id]:
                logger.warning(f"No data for robot {robot_id}")
                continue

            robot_states = history["states"][robot_id]
            num_joints = len(robot_states[0])
            color = colors[robot_index % len(colors)]

            # Plot each joint
            for joint_idx in range(num_joints):
                joint_positions = [state[joint_idx] for state in robot_states]
                plt.plot(
                    times[: len(joint_positions)],
                    joint_positions,
                    color=color,
                    linestyle=["-", "--", ":", "-."][joint_idx % 4],  # Different line styles
                    label=f"Robot {robot_id} - Joint {joint_idx}",
                )

        plt.title(title)

        plt.xlabel("Time (s)")
        plt.ylabel("Joint Position (rad)")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    def visualize_ee_path(robot_model, path, env, duration=5.0, line_color=[0, 0.7, 0], line_width=5, draw_trail=True):
        """
        Execute a path on the robot while visualizing the end effector trajectory.

        Args:
            robot_model: Robot model
            path: List of joint configurations to follow
            env: Simulation environment
            duration: Total execution duration (seconds)
            draw_trail: Whether to draw a persistent trail showing the path
        """
        num_waypoints = len(path)
        waypoint_duration = duration / num_waypoints

        position_seq = config_to_position_path(path, robot_model)
        logger.debug(f"End-effector positions: {position_seq}")

        prev_ee_pos = None
        line_ids = []
        for i, _ in enumerate(path):
            logger.info(f"Executing waypoint {i+1}/{num_waypoints}")

            if prev_ee_pos is not None and draw_trail:
                logger.debug(f"Drawing line from {prev_ee_pos} to {position_seq[i]}")
                logger.debug(f"Line color: {line_color}, Line width: {line_width}")
                line_id = pybullet.addUserDebugLine(
                    prev_ee_pos,
                    position_seq[i],
                    lineColorRGB=line_color,
                    lineWidth=line_width,
                    lifeTime=0,
                )
                line_ids.append(line_id)

            prev_ee_pos = position_seq[i]
            pybullet.addUserDebugText(
                f"{i+1}",
                position_seq[i],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                lifeTime=duration,
            )

            start_time = time.time()
            while time.time() - start_time < waypoint_duration:
                env.step(1 / 240)

        time.sleep(0.5)
        for line_id in line_ids:
            pybullet.removeUserDebugItem(line_id)
