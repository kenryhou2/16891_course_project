"""
Simple obstacle module.
"""

import pybullet
import numpy as np
from typing import List, Tuple, Union, Optional
import logging
from robot_model import RobotModel

logger = logging.getLogger(__name__)


class Obstacle:
    """Simple obstacle class."""

    def __init__(
        self,
        position: List[float],
        obstacle_type: str = "box",
        size: Union[List[float], float] = [0.1, 0.1, 0.1],
        color: List[float] = [1, 0, 0, 1],
        orientation: List[float] = [0, 0, 0, 1],
    ):
        """
        Initialize obstacle.

        Args:
            position: [x, y, z] position in world frame
            obstacle_type: 'box', 'sphere', or 'cylinder'
            size: For box: [half_extents_x, half_extents_y, half_extents_z]
                  For sphere: radius
                  For cylinder: [radius, height]
            color: [r, g, b, a] color and transparency
            orientation: [x, y, z, w] quaternion orientation
        """
        self.position = position
        self.obstacle_type = obstacle_type.lower()
        self.size = size
        self.color = color
        self.orientation = orientation
        self.body_id = None

        if self.obstacle_type == "box":
            self.collision_shape_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=self.size)
            self.visual_shape_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=self.size, rgbaColor=self.color)
        elif self.obstacle_type == "sphere":
            radius = self.size if isinstance(self.size, (int, float)) else self.size[0]
            self.collision_shape_id = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=radius)
            self.visual_shape_id = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=radius, rgbaColor=self.color)
        else:
            raise ValueError(f"Unsupported obstacle type: {self.obstacle_type}")

        self.create()

    def create(self):
        """Create the obstacle in the simulation environment."""
        self.body_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.collision_shape_id,
            baseVisualShapeIndex=self.visual_shape_id,
            basePosition=self.position,
            baseOrientation=self.orientation,
        )
        logger.info(f"Created {self.obstacle_type} obstacle at {self.position} with ID {self.body_id}")
        return self.body_id

    def remove(self):
        """Remove the obstacle from the simulation."""
        if self.body_id is not None:
            pybullet.removeBody(self.body_id)
            self.body_id = None

    def set_position(self, position: List[float]):
        """Update obstacle position."""
        if self.body_id is not None:
            pybullet.resetBasePositionAndOrientation(self.body_id, position, self.orientation)
            self.position = position

    def check_config_actual_collision(self, robot_model: RobotModel, config: List[float]) -> bool:
        """
        Check actual collision using forward kinematics with modifying robot state.

        Args:
            obstacle: Obstacle object
            robot_model: Robot model
            config: Joint configuration to check

        Returns:
            True if in collision, False otherwise
        """
        in_collision = False
        current_states = []
        for joint_idx in robot_model.movable_joints:
            state = pybullet.getJointState(robot_model.robot_id, joint_idx)
            current_states.append(state)

        for i, joint_idx in enumerate(robot_model.movable_joints):  # Compute forward kinematics for each link
            if i < len(config):
                pybullet.resetJointState(robot_model.robot_id, joint_idx, config[i])

        pybullet.performCollisionDetection()

        for link_idx in range(pybullet.getNumJoints(robot_model.robot_id)):
            if pybullet.getContactPoints(bodyA=self.body_id, bodyB=robot_model.robot_id, linkIndexB=link_idx):
                in_collision = True
                break

        for i, joint_idx in enumerate(robot_model.movable_joints):  # Restore original joint states
            if i < len(current_states):
                pybullet.resetJointState(robot_model.robot_id, joint_idx, current_states[i][0])

        pybullet.performCollisionDetection()

        return in_collision

    def check_config_collision(self, robot_model: RobotModel, config: List[float], threshold=0.01) -> bool:
        """
        Check if the robot configuration is in collision with the obstacle.
        configureDebugVisualizer is used to disable rendering during searching.
        The robot state is modified to check for collision, so save and restore the state before and after resetting.
        """
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        state_id = pybullet.saveState()

        for i, joint_idx in enumerate(robot_model.movable_joints):
            if i < len(config):
                pybullet.resetJointState(robot_model.robot_id, joint_idx, config[i])

        in_collision = False
        for link_idx in range(pybullet.getNumJoints(robot_model.robot_id)):
            contacts = pybullet.getClosestPoints(robot_model.robot_id, self.body_id, threshold, linkIndexA=link_idx)
            if contacts:
                in_collision = True
                break

        pybullet.restoreState(stateId=state_id)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        return in_collision


# ========================================================
# ================== Testing Functions ==================
# ========================================================


def create_test_obstacles() -> List[Obstacle]:
    """Create a set of test obstacles for RRT planning."""
    obstacles = []

    # box = Obstacle(position=[0.5, 0.0, 0.8], obstacle_type="box", size=[0.1, 0.1, 0.1], color=[1, 0, 0, 0.7])  # Red, semi-transparent
    # obstacles.append(box)

    # sphere = Obstacle(position=[0.5, 0.5, 0.5], obstacle_type="sphere", size=0.15, color=[0, 1, 0, 0.7])  # Green, semi-transparent
    # obstacles.append(sphere)
    
    # # Add a shelf (large box)
    # shelf = Obstacle(
    #     position=[1.0, 0.0, 0.5],  # Adjust position as needed
    #     obstacle_type="box",
    #     size=[0.5, 0.2, 0.5],  # Width, depth, height of the shelf
    #     color=[0.5, 0.5, 0.5, 1.0],  # Gray, opaque
    # )
    # obstacles.append(shelf)

    return obstacles
