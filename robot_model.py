"""
Base robot model classes for simulation.
"""

import pybullet
import pinocchio
from pinocchio import SE3, Quaternion
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from controller import JointPositionController, JointVelocityController, JointTorqueController, TrajectoryController

logger = logging.getLogger(__name__)


@dataclass
class JointConfig:
    """
    Data class to store joint configuration information.
    """

    index: int
    name: str
    type: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float


class RobotModel:
    """
    Base class for robot models in the simulation.
    """

    def __init__(self, urdf_path: str, position: List[float] = [0, 0, 0], orientation: List[float] = [0, 0, 0, 1]):
        """
        Initialize a robot model.

        Args:
            urdf_path: Path to the URDF file
            position: Initial position [x, y, z]
            orientation: Initial orientation as quaternion [x, y, z, w]
        """
        self.robot_id = pybullet.loadURDF(
            urdf_path,
            position,
            orientation,
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        self.robot_position = position
        self.robot_orientation = orientation

        self.pinocchio_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.pinocchio_data = self.pinocchio_model.createData()
        self.pinocchio_ee_id = self.pinocchio_model.getFrameId("ee_link")

        logger.info(f"Model name: {self.pinocchio_model.name}")
        logger.info(f"Number of joints (DoF): {self.pinocchio_model.nq}")
        logger.info(f"EE link ID: {self.pinocchio_ee_id}")

        self.joint_configs = self.load_joint_info()
        self.movable_joints = [j.index for j in self.joint_configs if pybullet.getJointInfo(self.robot_id, j.index)[2] != pybullet.JOINT_FIXED]
        print(f"Movable joints: {self.movable_joints}")

        # undirected adjacency graph of links
        self.undirected_link_graph = {}
        for joint_index in range(pybullet.getNumJoints(self.robot_id)):
            joint_info = pybullet.getJointInfo(self.robot_id, joint_index)
            parent = joint_info[16]  # parent link index
            child = joint_info[0]  # child link index (aka. joint index)
            self.undirected_link_graph.setdefault(parent, []).append(child)
            self.undirected_link_graph.setdefault(child, []).append(parent)

        logger.info(f"Robot model loaded with {len(self.movable_joints)} movable joints")

    def load_joint_info(self) -> List[JointConfig]:
        """
        Load joint information from the robot model.
        """
        joints = []
        num_joints = pybullet.getNumJoints(self.robot_id)

        for i in range(num_joints):
            info = pybullet.getJointInfo(self.robot_id, i)
            joint_config = JointConfig(
                index=i,
                name=info[1].decode("utf-8"),
                type=info[2],
                lower_limit=info[8],
                upper_limit=info[9],
                max_force=info[10],
                max_velocity=info[11],
            )

            joints.append(joint_config)

        return joints

    def get_state(self) -> List[float]:
        """
        Get current joint positions as the robot state.
        """
        return [pybullet.getJointState(self.robot_id, joint_idx)[0] for joint_idx in self.movable_joints]

    def create_controller(self, controller_type: str, **kwargs) -> Any:
        """
        Create a controller for this robot.

        Args:
            controller_type: Type of controller to create
            **kwargs: Additional arguments for the controller

        Returns:
            Controller instance
        """

        if controller_type == "position":
            return JointPositionController(
                self.robot_id,
                self.movable_joints,
                **kwargs,
            )

        elif controller_type == "velocity":
            return JointVelocityController(
                self.robot_id,
                self.movable_joints,
                **kwargs,
            )

        elif controller_type == "torque":
            return JointTorqueController(
                self.robot_id,
                self.movable_joints,
                **kwargs,
            )

        elif controller_type == "trajectory":
            if "trajectory" not in kwargs or "duration" not in kwargs:
                raise ValueError("Trajectory controller requires 'trajectory' and 'duration' arguments")
            return TrajectoryController(
                self.robot_id,
                self.movable_joints,
                kwargs.pop("trajectory"),
                kwargs.pop("duration"),
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
