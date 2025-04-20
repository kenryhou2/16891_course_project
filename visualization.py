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

    @staticmethod
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

    @staticmethod
    def visualize_multiple_ee_paths(robots, paths, env, colors=None, duration=3.0, line_width=5):
        """
        Visualize end-effector paths for multiple robots simultaneously.

        Args:
            robots: List of robot models
            paths: Dict mapping robot IDs to paths
            env: Simulation environment
            colors: List of colors for each robot's path (default: vibrant colors)
            duration: Visualization duration
            line_width: Width of path lines
        """
        if colors is None:
            colors = [
                [0.8, 0.2, 0.8],  # Vibrant purple
                [0.0, 0.6, 1.0],  # Bright blue
                [1.0, 0.6, 0.0],  # Vibrant orange
                [0.0, 0.8, 0.4],  # Bright teal
                [1.0, 0.4, 0.4],  # Coral red
                [0.6, 0.8, 0.0],  # Lime green
            ]

        # Calculate position sequences for all robots
        position_sequences = {}
        for robot in robots:
            if robot.robot_id in paths and paths[robot.robot_id]:
                position_sequences[robot.robot_id] = config_to_position_path(paths[robot.robot_id], robot)

        # Calculate visualization timing
        max_points = max([len(pos_seq) for pos_seq in position_sequences.values()], default=0)
        if max_points == 0:
            logger.warning("No paths to visualize")
            return
        # waypoint_duration = duration / max_points

        # Draw all paths
        line_ids = []
        for i, robot in enumerate(robots):
            if robot.robot_id not in position_sequences:
                continue

            logger.info(f"Executing path for robot {robot.robot_id}")
            pos_seq = position_sequences[robot.robot_id]
            color = colors[i % len(colors)]

            # Draw complete path
            for j in range(len(pos_seq) - 1):
                line_id = pybullet.addUserDebugLine(
                    pos_seq[j],
                    pos_seq[j + 1],
                    lineColorRGB=color,
                    lineWidth=line_width,
                    lifeTime=0,
                )
                line_ids.append(line_id)

            if len(pos_seq) > 0:
                pybullet.addUserDebugText(
                    f"{i+1}",
                    pos_seq[0],
                    textColorRGB=color,
                    textSize=1.0,
                    lifeTime=duration,
                )

            start_time = time.time()
            while time.time() - start_time < duration:
                env.step(1 / 240)

        time.sleep(0.5)
        for line_id in line_ids:
            pybullet.removeUserDebugItem(line_id)
