"""
Robot arm controllers using PyBullet's built-in control methods.
"""

import pybullet
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)


class Controller:
    """Base class for robot controllers."""

    def __init__(self, robot_id: int, joint_indices: List[int], max_forces: Optional[List[float]] = None):
        """
        Initialize the controller.

        Args:
            robot_id: PyBullet body ID of the robot
            joint_indices: Indices of joints to control
            max_forces: Maximum forces to apply (defaults to 500 for each joint)
        """
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.max_forces = max_forces if max_forces is not None else [500.0] * len(joint_indices)

    def reset(self):
        """Reset the controller state."""
        pass

    def update(self, target: Any, dt: float = 1 / 240):
        """
        Update the controller.

        Args:
            target: Target for the controller (implementation-dependent)
            dt: Time step
        """
        pass


class JointPositionController(Controller):
    """Joint position controller using PyBullet's POSITION_CONTROL mode."""

    def __init__(
        self,
        robot_id: int,
        joint_indices: List[int],
        max_forces: Optional[List[float]] = None,
        position_gains: Optional[List[float]] = None,
        velocity_gains: Optional[List[float]] = None,
    ):
        """
        Initialize the joint position controller.

        Args:
            robot_id: PyBullet body ID of the robot
            joint_indices: Indices of joints to control
            max_forces: Maximum forces to apply
            position_gains: Position gains (kp) for each joint
            velocity_gains: Velocity gains (kd) for each joint
        """
        super().__init__(robot_id, joint_indices, max_forces)
        self.position_gains = position_gains
        self.velocity_gains = velocity_gains

        logger.info(f"position_gains: {self.position_gains}")
        logger.info(f"velocity_gains: {self.velocity_gains}")

    def update(self, target_positions: List[float], dt: float = 1 / 240):
        """
        Update the controller with target joint positions.

        Args:
            target_positions: Target positions for each joint
            dt: Time step
        """
        if len(target_positions) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} target positions, got {len(target_positions)}")

        if self.position_gains is not None and self.velocity_gains is not None:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.joint_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=self.max_forces,
                positionGains=self.position_gains,
                velocityGains=self.velocity_gains,
            )
        else:
            pybullet.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.joint_indices,
                controlMode=pybullet.POSITION_CONTROL,
                targetPositions=target_positions,
                forces=self.max_forces,
            )

        return target_positions


class JointVelocityController(Controller):
    """Joint velocity controller using PyBullet's VELOCITY_CONTROL mode."""

    def update(self, target_velocities: List[float], dt: float = 1 / 240):
        """
        Update the controller with target joint velocities.

        Args:
            target_velocities: Target velocities for each joint
            dt: Time step
        """
        if len(target_velocities) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} target velocities, got {len(target_velocities)}")

        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=target_velocities,
            forces=self.max_forces,
        )

        return target_velocities


class JointTorqueController(Controller):
    """Joint torque controller using PyBullet's TORQUE_CONTROL mode."""

    def update(self, target_torques: List[float], dt: float = 1 / 240):
        """
        Update the controller with target joint torques.

        Args:
            target_torques: Target torques for each joint
            dt: Time step
        """
        if len(target_torques) != len(self.joint_indices):
            raise ValueError(f"Expected {len(self.joint_indices)} target torques, got {len(target_torques)}")

        pybullet.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=target_torques,
        )

        return target_torques


class TrajectoryController(Controller):
    """
    Controller that follows a pre-computed trajectory using PyBullet's control.
    """

    def __init__(
        self,
        robot_id: int = None,
        joint_indices: List[int] = None,
        trajectory: List[List[float]] = None,
        duration: float = 5.0,
        max_forces: Optional[List[float]] = None,
        position_gains: Optional[List[float]] = None,
        velocity_gains: Optional[List[float]] = None,
    ):
        """
        Initialize the trajectory controller.

        Args:
            robot_id: PyBullet body ID of the robot
            joint_indices: Indices of joints to control
            trajectory: List of joint configurations
            duration: Duration to complete the trajectory
            max_forces: Maximum forces to apply
        """
        super().__init__(robot_id, joint_indices, max_forces)

        self.trajectory = trajectory
        self.duration = duration
        self.start_time = None
        self.position_controller = JointPositionController(robot_id, joint_indices, max_forces, position_gains, velocity_gains)

    def reset(self):
        """Reset the controller start time."""
        self.start_time = None

    def update(self, time_step: float, dt: float = 1 / 240):
        """
        Update the controller based on elapsed time.

        Args:
            time_step: Current simulation time
            dt: Time step
        """
        if self.start_time is None:
            self.start_time = time_step

        elapsed = time_step - self.start_time
        if elapsed >= self.duration:
            return self.position_controller.update(self.trajectory[-1], dt)

        progress = elapsed / self.duration

        segment_count = len(self.trajectory) - 1
        segment_index = min(int(progress * segment_count), segment_count - 1)
        segment_progress = (progress * segment_count) - segment_index

        start_config = self.trajectory[segment_index]
        end_config = self.trajectory[segment_index + 1]

        current_config = [start + (end - start) * segment_progress for start, end in zip(start_config, end_config)]

        return self.position_controller.update(current_config, dt)
