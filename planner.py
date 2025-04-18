"""
Path planning algorithms for robot motion planning.
"""

import abc
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import logging

from robot_model import RobotModel
from obstacle import Obstacle
import pybullet
from constraint import Constraint

GROUND_Z_THRESHOLD = 0.001  # 0.1 cm above ground

logger = logging.getLogger(__name__)


class PathPlanner(abc.ABC):
    """
    Abstract base class for path planners.
    """

    @abc.abstractmethod
    def plan(self, start_config: List[float], goal_config: List[float], obstacles=None, timeout: float = 30.0) -> Optional[List[List[float]]]:
        """
        Plan a path from start_config to goal_config.

        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacles: Obstacle representation (implementation-dependent)
            timeout: Maximum planning time in seconds

        Returns:
            List of configurations from start to goal, or None if no path found
        """
        pass


class RRTPlanner(PathPlanner):
    """
    RRT (Rapidly-exploring Random Tree) planner for robot arm motion planning.
    """

    def __init__(
        self,
        robot_model: RobotModel = None,
        max_nodes: int = 10000,
        goal_bias: float = 0.2,
        step_size: float = 0.2,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Initialize the RRT planner.

        Args:
            robot_model: Robot model to plan for
            max_nodes: Maximum number of nodes in the RRT tree
            goal_bias: Probability of sampling the goal configuration
            step_size: Maximum step size in radians
            joint_limits: List of (min, max) joint limits
        """
        self.robot_model = robot_model
        self.max_nodes = max_nodes
        self.goal_bias = goal_bias
        self.step_size = step_size

        # Joint limits can be provided or extracted from the robot model
        if joint_limits is None:
            self.joint_limits = []
            for joint_idx in self.robot_model.movable_joints:
                info = self.robot_model.joint_configs[joint_idx]
                self.joint_limits.append((info.lower_limit, info.upper_limit))
        else:
            self.joint_limits = joint_limits

        logger.info(f"RRT planner initialized with {len(self.joint_limits)} DOF")

    def sample_random_config(self, goal_config: List[float]) -> List[float]:
        """
        Sample a random configuration within joint limits.
        With a probability of goal_bias, return the goal configuration.
        """
        if np.random.random() < self.goal_bias:
            return goal_config

        random_config = []
        for i, (lower, upper) in enumerate(self.joint_limits):
            if lower == upper:  # Fixed joint
                random_config.append(lower)
            else:
                random_config.append(np.random.uniform(lower, upper))

        return random_config

    def nearest_neighbor(self, nodes: Dict[int, Dict], random_config: List[float]) -> Dict:
        """
        Find the nearest node in the tree to the random configuration.
        """
        min_dist = float("inf")
        nearest_node = None

        for node in nodes.values():
            dist = self.euclidean_distance(node["config"], random_config)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def euclidean_distance(self, config1: List[float], config2: List[float]) -> float:
        """
        Calculate Euclidean distance between two configurations.
        """
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(config1, config2)))

    def new_config(self, from_config: List[float], to_config: List[float]) -> List[float]:
        """
        Return a new configuration that is step_size away from from_config
        in the direction of to_config.
        """
        distance = self.euclidean_distance(from_config, to_config)

        if distance <= self.step_size:
            return to_config

        # Calculate direction vector and normalize
        direction = [(to - from_val) / distance for to, from_val in zip(to_config, from_config)]

        # Create new configuration
        new_config = [from_val + dir_val * self.step_size for from_val, dir_val in zip(from_config, direction)]

        # Ensure new configuration is within joint limits
        for i, (lower, upper) in enumerate(self.joint_limits):
            new_config[i] = max(lower, min(upper, new_config[i]))

        return new_config

    def is_valid_config(self, config: List[float], obstacles: Optional[List[Obstacle]] = None) -> bool:
        """
        Check if a configuration is valid (within joint limits and collision-free).

        Args:
            config: Configuration to check
            obstacles: Obstacle representation (can be customized based on environment)

        Returns:
            True if configuration is valid, False otherwise
        """
        logger.debug(f"Checking for collisions with {len(obstacles)} obstacles: {obstacles}")

        # Check if configuration is within joint limits
        for i, (lower, upper) in enumerate(self.joint_limits):
            if config[i] < lower or config[i] > upper:
                return False

        # Avoid Self-Collision
        num_links = pybullet.getNumJoints(self.robot_model.robot_id)
        for i in range(num_links):
            for j in range(i + 1, num_links):
                if j not in self.robot_model.undirected_link_graph.get(i, []) and i not in self.robot_model.undirected_link_graph.get(j, []):
                    if pybullet.getClosestPoints(
                        self.robot_model.robot_id,
                        self.robot_model.robot_id,
                        0.01,
                        linkIndexA=i,
                        linkIndexB=j,
                    ):
                        return False

        for joint_index in range(pybullet.getNumJoints(self.robot_model.robot_id)):
            touch_ground = pybullet.getClosestPoints(self.robot_model.robot_id, self.plane_idx, GROUND_Z_THRESHOLD, linkIndexA=joint_index)
            if touch_ground and joint_index != 0:
                logger.info(f"Robot {self.robot_model.robot_id} Joint {joint_index} is touching the ground")
                return False

        # This would need to use forward kinematics to check for collisions with the environment
        if obstacles:
            collision_risk_distance = 0.001
            for obstacle in obstacles:
                if obstacle.check_config_collision(self.robot_model, config, threshold=collision_risk_distance):
                    logger.info(f"Collision detected Robot {self.robot_model.robot_id} with obstacle {obstacle.body_id} with {self.node_count} nodes")
                    return False

        return True

    def is_valid_movement(self, from_config: List[float], to_config: List[float], obstacles: Optional[List[Obstacle]] = None) -> bool:
        """
        Check if the movement from from_config to to_config is valid.

        Args:
            from_config: Starting configuration
            to_config: Ending configuration
            obstacles: Obstacle representation

        Returns:
            True if movement is valid, False otherwise
        """

        # For simplicity, we'll check if the ending configuration is valid
        # In a real implementation, we would check for collisions along the path

        return self.is_valid_config(config=to_config, obstacles=obstacles)

    def is_constrained(self, from_config: List[float], to_config: List[float], timestep: int) -> bool:
        """
        Check if the movement from from_config to to_config violates any constraints.

        Args:
            from_config: Starting configuration
            to_config: Ending configuration
            constraints: List of constraints
            timestep: Discrete timestep

        Returns:
            True if movement violates constraints, False otherwise
        """
        for constraint in self.constraints:
            if constraint.timestep == timestep and np.allclose(constraint.config, to_config, atol=1e-2):
                logger.info(f"Movement from {from_config} to {to_config} violates constraint {constraint}")
                return True
        return False

    def plan(
        self,
        start_config: List[float],
        goal_config: List[float],
        obstacles: Optional[List[Obstacle]] = None,
        plane=None,
        timeout: float = 3000.0,
        constraints: List[Constraint] = None,
    ) -> Optional[List[List[float]]]:
        """
        Plan a path from start_config to goal_config using RRT.

        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacles: Obstacle representation
            timeout: Maximum planning time in seconds

        Returns:
            List of configurations from start to goal, or None if no path found
        """
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        logger.debug(f"{len(obstacles)} Obstacles: {obstacles}")

        self.plane_idx = plane
        self.constraints = constraints if constraints else None
        start_time = time.time()

        # Initialize RRT
        nodes = {}
        node_counter = 0

        # Add start node
        start_node = {"config": start_config, "parent": None, "cost": 0, "timestep": 0}
        nodes[node_counter] = start_node
        node_counter += 1

        logger.info(f"Planning from {start_config} to {goal_config}")

        # Main RRT loop
        for _ in range(self.max_nodes):
            if time.time() - start_time > timeout:
                logger.warning(f"RRT planning timeout after {timeout} seconds")
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
                return None

            # Sample random configuration
            random_config = self.sample_random_config(goal_config)

            # Find nearest neighbor
            nearest_node = self.nearest_neighbor(nodes, random_config)

            # Generate new configuration
            new_config = self.new_config(nearest_node["config"], random_config)

            # Check if new configuration is valid
            if not self.is_valid_config(config=new_config, obstacles=obstacles):
                continue

            # Check if movement to new configuration is valid
            if not self.is_valid_movement(from_config=nearest_node["config"], to_config=new_config, obstacles=obstacles):
                continue

            new_timestep = nearest_node["timestep"] + 1
            if self.constraints and self.is_constrained(nearest_node["config"], new_config, new_timestep):
                continue

            # Create new node
            new_node = {
                "config": new_config,
                "parent": nearest_node,
                "cost": nearest_node["cost"] + self.euclidean_distance(nearest_node["config"], new_config),
                "timestep": new_timestep,
            }

            # Add new node to tree
            nodes[node_counter] = new_node
            node_counter += 1
            self.node_count = node_counter

            # Check if goal is reached
            if self.euclidean_distance(new_config, goal_config) < self.step_size:
                logger.info(f"RRT found a solution with {node_counter} nodes")

                # Reconstruct path
                path = []
                current = new_node
                while current is not None:
                    path.append(current["config"])
                    current = current["parent"]
                path.reverse()

                logger.debug(f"RRT Path found: {path}")
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
                return path

        # If we get here, no path was found
        logger.warning(f"RRT failed to find a solution after {self.max_nodes} iterations")
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        return None


class DirectPlanner(PathPlanner):
    """Simple direct path planner (straight line in configuration space)."""

    def __init__(self, robot_model, num_waypoints: int = 10):
        """
        Initialize the direct planner.

        Args:
            robot_model: Robot model to plan for
            num_waypoints: Number of waypoints to generate
        """
        self.robot_model = robot_model
        self.num_waypoints = num_waypoints

    def plan(self, start_config: List[float], goal_config: List[float], obstacles=None, timeout: float = 30.0) -> Optional[List[List[float]]]:
        """
        Plan a direct path from start_config to goal_config.

        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacles: Not used
            timeout: Not used

        Returns:
            List of configurations from start to goal
        """
        path = [start_config]

        # Create intermediate waypoints
        for i in range(1, self.num_waypoints):
            t = i / self.num_waypoints
            waypoint = [(1 - t) * start + t * goal for start, goal in zip(start_config, goal_config)]
            path.append(waypoint)

        path.append(goal_config)

        return path
