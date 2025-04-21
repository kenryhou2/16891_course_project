"""
Conflict-Based Search (CBS) solver for multi-robot arm motion planning with configuration space.
"""

import heapq
import copy
import numpy as np
import pybullet
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from robot_model import RobotModel
from planner import RRTPlanner
from obstacle import Obstacle
from constraint import Constraint

DISTANCE_THRESHOLD = 0.001  # Threshold for distance comparison

logger = logging.getLogger(__name__)


class CBSNode:
    """
    Node in the CBS search tree for robot arm planning.
    """

    def __init__(self):
        self.constraints = defaultdict(list)  # agent_id -> list of constraints
        self.paths = {}  # agent_id -> path
        self.cost = 0
        self.collisions = []

    def __lt__(self, other):
        # For priority queue
        return self.cost < other.cost


class CBSSolver:
    """
    Conflict-Based Search solver for multi-robot arm motion planning.
    """

    def __init__(
        self,
        robots: List[RobotModel],
        planners: Dict[int, RRTPlanner],
        obstacles: Optional[List[Obstacle]] = None,
        plane: Optional[int] = None,
    ):
        """
        Initialize the CBS solver.

        Args:
            robots: List of robot models
            start_configs: Start configurations for each robot
            goal_configs: Goal configurations for each robot
            planners: Planners for each robot
            obstacles: Obstacles in the environment
        """
        self.robots = robots
        # Create a mapping from robot_id to index in self.robots
        self.robotId_to_idx = {robot.robot_id: i for i, robot in enumerate(self.robots)}
        self.num_of_agents = len(self.robots)

        self.solver_type = "CBS"
        self.planner_type = "RRT"
        self.obstacles = obstacles or []
        self.plane = plane

        self.open_list = []
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.max_expanded = 10000

        self.planners = planners

    def detect_robot_collision(self, robot1_id: int, robot2_id: int, config1: List[float], config2: List[float]) -> bool:
        """
        Detect if two robot configurations would result in collision.

        Args:
            robot1_id: ID of first robot in self.robots
            robot2_id: ID of second robot in self.robots
            config1: Configuration of first robot
            config2: Configuration of second robot

        Returns:
            True if collision detected, False otherwise
        """
        robot1 = self.robots[self.robotId_to_idx[robot1_id]]
        robot2 = self.robots[self.robotId_to_idx[robot2_id]]
        state_id = pybullet.saveState()

        for i, joint_idx in enumerate(robot1.movable_joints):
            if i < len(config1):
                pybullet.resetJointState(robot1.robot_id, joint_idx, config1[i])

        for i, joint_idx in enumerate(robot2.movable_joints):
            if i < len(config2):
                pybullet.resetJointState(robot2.robot_id, joint_idx, config2[i])

        contacts = pybullet.getClosestPoints(robot1.robot_id, robot2.robot_id, DISTANCE_THRESHOLD)

        pybullet.restoreState(state_id)
        return True if contacts else False

    def get_configuration_from_path(self, path: List[List[float]], timestep: int) -> Optional[List[float]]:
        if timestep < len(path):
            return path[timestep]
        else:
            return path[-1]  # Return the last configuration if timestep exceeds path length

    def detect_path_collision(self, robot1_id: int, robot2_id: int, path1: List[List[float]], path2: List[List[float]]) -> Optional[Dict]:
        """
        Detect the first collision between two robot paths.

        Args:
            robot1_id: ID of first robot
            robot2_id: ID of second robot
            path1: Path of first robot
            path2: Path of second robot

        Returns:
            Collision info or None if no collision
        """
        # Check for collisions at each timestep
        for t in range(max(len(path1), len(path2))):
            config1 = self.get_configuration_from_path(path1, t)
            config2 = self.get_configuration_from_path(path2, t)
            if self.detect_robot_collision(robot1_id, robot2_id, config1, config2):
                return {"a1": robot1_id, "a2": robot2_id, "config1": config1, "config2": config2, "timestep": t}

        return None

    def detect_all_path_collisions(self, paths: Dict[int, List[List[float]]]) -> List[Dict]:
        """
        Detect collisions among all robot paths.

        Args:
            paths: Dict mapping robot IDs to paths

        Returns:
            List of collisions
        """
        collisions = []
        robot_ids = list(paths.keys())

        # Check each pair of robots
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                if robot_ids[i] not in paths or robot_ids[j] not in paths:
                    continue

                collision = self.detect_path_collision(robot_ids[i], robot_ids[j], paths[robot_ids[i]], paths[robot_ids[j]])
                if collision:  # Store robot IDs instead of indices inside the collision dict
                    logger.info(f"Collision detected between robots {robot_ids[i]} and {robot_ids[j]} at timestep {collision['timestep']}")
                    collisions.append(collision)

        return collisions

    def create_constraints_from_collision(self, collision: Dict) -> List[Constraint]:
        """
        Create constraints to resolve a collision.

        Args:
            collision: Collision information

        Returns:
            List of constraints
        """
        agent1, agent2 = collision["a1"], collision["a2"]
        config1, config2 = collision["config1"], collision["config2"]
        timestep = collision["timestep"]

        # Create constraints that prevent both robots from being at the collision configurations
        constraint1 = Constraint(agent1, config1, timestep)
        constraint2 = Constraint(agent2, config2, timestep)

        return [constraint1, constraint2]

    def plan_with_constraints(
        self, agent_id: int, start_config: List[float], goal_config: List[float], constraints: List[Constraint]
    ) -> Optional[List[List[float]]]:
        """
        Plan a path for a single robot respecting constraints.

        Args:
            agent_id: Robot ID
            start_config: Start configuration
            goal_config: Goal configuration
            constraints: List of constraints

        Returns:
            Path as list of configurations, or None if no path found
        """
        planner = self.planners.get(agent_id)
        if not planner:
            logger.error(f"No planner found for agent {agent_id}")
            return None

        # Filter constraints for this agent
        agent_constraints = [c for c in constraints if c.agent_id == agent_id]

        # Plan path respecting constraints
        path = planner.plan(
            start_config,
            goal_config,
            obstacles=self.obstacles,
            plane=self.plane,
            timeout=self.timeout,
            constraints=agent_constraints,
        )
        return path

    def push_node(self, node: CBSNode):
        """
        Add a node to the open list.
        """
        heapq.heappush(self.open_list, (node.cost, self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self) -> CBSNode:
        """
        Get the next node from the open list.
        """
        _, id, node = heapq.heappop(self.open_list)
        logger.info(f"Expanding node {id} with cost {node.cost}, and {len(node.collisions)} collisions")
        self.num_of_expanded += 1
        return node

    def get_path_cost(self, path: List[List[float]]) -> float:
        """
        Calculate the cost of a path.

        Args:
            path: List of configurations

        Returns:
            Path cost
        """
        if not path:
            return float("inf")

        cost = 0
        for i in range(len(path) - 1):  # Euclidean distance in configuration space
            cost += np.sqrt(sum((a - b) ** 2 for a, b in zip(path[i], path[i + 1])))
        return cost

    def get_total_cost(self, paths: Dict[int, List[List[float]]]) -> float:
        """
        Calculate the total cost of all paths.

        Args:
            paths: Dict mapping robot IDs to paths

        Returns:
            Total cost
        """
        return sum(self.get_path_cost(path) for path in paths.values())
    
    def get_expanded_nodes(self) -> int:
        """
        Get the total number of expanded nodes.

        Returns:
            int: Total number of expanded nodes.
        """
        return self.num_of_expanded

    def solve(
        self, start_configs: Dict[int, List[float]], goal_configs: Dict[int, List[float]], timeout: float = 3000.0
    ) -> Optional[Dict[int, List[List[float]]]]:
        """
        Find conflict-free paths for multiple robots.

        Args:
            start_configs: Dict mapping robot IDs to start configurations
            goal_configs: Dict mapping robot IDs to goal configurations

        Returns:
            Dict mapping robot IDs to paths, or None if no solution found
        """
        self.timeout = timeout

        # Initialization for CBS
        root = CBSNode()
        for agent_id, start_config in start_configs.items():
            if agent_id not in goal_configs:
                logger.warning(f"No goal configuration for robot {agent_id}, skipping")
                continue

            goal_config = goal_configs[agent_id]
            planner = self.planners.get(agent_id)
            if not planner:
                logger.error(f"No planner found for robot {agent_id}")
                return None

            # Plan the initial path for each robot
            path = planner.plan(start_config, goal_config, self.obstacles, plane=self.plane if self.plane is not None else 0, timeout=self.timeout)

            if path is None:
                logger.error(f"Failed to find initial path for robot {agent_id}")
                return None

            root.paths[agent_id] = path

        # Fill in the root node with other information
        root.cost = self.get_total_cost(root.paths)
        root.collisions = self.detect_all_path_collisions(root.paths)
        if not root.collisions:
            return root.paths

        self.push_node(root)

        while self.open_list and self.num_of_expanded < self.max_expanded:
            current = self.pop_node()  # Get the node with the lowest cost

            # If no collisions, we have a solution
            if not current.collisions:
                logger.info(f"Solution found after expanding {self.num_of_expanded} nodes")
                return current.paths

            # Get the first collision
            collision = current.collisions[0]
            logger.debug(f"Selected collision to resolve: {collision}")

            # Create constraints to resolve the collision
            constraints = self.create_constraints_from_collision(collision)

            # Process each constraint
            for constraint in constraints:
                new_node = copy.deepcopy(current)  # Create a new node based on the current one

                # Add the new constraint
                new_node.constraints[constraint.agent_id].append(constraint)

                # Replan for the constrained agent
                agent_id = constraint.agent_id
                new_path = self.plan_with_constraints(agent_id, start_configs[agent_id], goal_configs[agent_id], new_node.constraints[agent_id])

                # If no path found, skip this node
                if new_path is None:
                    logger.debug(f"No path found for agent {agent_id} with constraints")
                    continue

                # Update the solution
                new_node.paths[agent_id] = new_path

                # Calculate the new cost
                new_node.cost = self.get_total_cost(new_node.paths)

                # Detect collisions in the new node
                new_node.collisions = self.detect_all_path_collisions(new_node.paths)

                # Add the new node to the open list
                self.push_node(new_node)

        # If we get here, no solution was found
        logger.warning(f"CBS failed to find a solution after expanding {self.num_of_expanded} nodes")
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        return None
