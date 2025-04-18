"""
Simulation environment module for robot arm simulations.
"""

import pybullet_data
import pybullet
import time
import logging
from typing import List, Tuple, Optional
from obstacle import Obstacle
from obstacle import create_test_obstacles

logger = logging.getLogger(__name__)


class SimulationEnvironment:
    """Manages the physics simulation environment."""

    def __init__(self, gui: bool = True, gravity: Tuple[float, float, float] = (0, 0, -9.8)):
        """
        Initialize the simulation environment.

        Args:
            gui: Whether to use GUI (True) or headless mode (False)
            gravity: 3D vector for gravity direction and magnitude
        """
        self.client_id = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(*gravity)

        self.plane_id = pybullet.loadURDF("plane.urdf")

        logger.info("Simulation environment initialized")

    def reset_camera(self, distance: float = 1.5, yaw: float = 30, pitch: float = -20, target_pos: List[float] = [0, 0, 0.5]):
        """
        Reset the camera view in the simulator.
        """
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_pos,
        )

    def step(self, sleep_time: float = 1 / 240):
        """
        Step the simulation forward and optionally pause.
        """
        pybullet.stepSimulation()
        if sleep_time > 0:
            time.sleep(sleep_time)

    def close(self):
        """
        Close the physics client connection.
        """
        pybullet.disconnect(self.client_id)
        for obstacle in self.obstacles_in_scene:
            obstacle.remove()

        logger.info("Simulation environment closed")

    def setup_obstacles(self):
        self.obstacles_in_scene = create_test_obstacles()
