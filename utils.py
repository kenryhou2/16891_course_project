"""
Utility functions for the robot arm simulator.
"""

import json
import logging
import os
from typing import Dict, Any, List

from obstacle import Obstacle

logger = logging.getLogger(__name__)


def save_history(history: Dict, filename: str):
    """
    Save simulation history to a JSON file.
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {"time": history["time"]}
    serializable_history["states"] = {}
    for robot_id, states in history["states"].items():
        serializable_history["states"][str(robot_id)] = [[float(val) for val in state] for state in states]

    with open(filename, "w") as f:
        json.dump(serializable_history, f)

    logger.info(f"Simulation history saved to {filename}")


def load_history(filename: str) -> Dict:
    """
    Load simulation history from a JSON file.
    """

    with open(filename, "r") as f:
        history = json.load(f)

    logger.info(f"Simulation history loaded from {filename}")
    return history


# Function to modify RRT planner to use obstacle collision checking
def setup_rrt_with_obstacles(rrt_planner, obstacles: List[Obstacle]):
    """
    Modify RRT planner to use obstacle collision checking.

    Args:
        rrt_planner: RRT planner instance
        obstacles: List of obstacles
    """
    # Store original validity check function
    original_is_valid_config = rrt_planner.is_valid_config

    # Create new validity check function that also checks obstacles
    def is_valid_with_obstacles(config, **kwargs):
        # First check if valid according to original criteria (joint limits, etc.)
        if not original_is_valid_config(config, **kwargs):
            return False

        # Check collision with each obstacle
        ACTUAL_COLLISION_CHECK = False
        if ACTUAL_COLLISION_CHECK:
            for obstacle in obstacles:
                if obstacle.check_config_actual_collision(rrt_planner.robot_model, config):
                    logger.debug(f"Collision detected with obstacle at {obstacle.position} for config {config}")
                    return False

        return True

    rrt_planner.is_valid_config = is_valid_with_obstacles
    return rrt_planner
