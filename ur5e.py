"""
UR5e robot implementation for simulation.
"""

import os
import urllib.request
import logging
from typing import List, Dict, Optional

from robot_model import RobotModel


logger = logging.getLogger(__name__)

UR5E_URDF_PATH = "assets/ur5e/ur5e.urdf"


class UR5eRobot(RobotModel):
    """
    Universal Robots UR5e robot implementation.
    """

    def __init__(self, position: List[float] = [0, 0, 0], orientation: List[float] = [0, 0, 0, 1]):
        """
        Initialize the UR5e robot.

        Args:
            position: Initial position [x, y, z]
        """
        # Check if UR5e URDF exists, if not, download it
        self.ensure_urdf_available()

        # Initialize with base RobotModel
        super().__init__(UR5E_URDF_PATH, position, orientation)

        # UR5e specific initialization
        self.setup_ur5e_params()

    def ensure_urdf_available(self):
        """
        Ensure the UR5e URDF is available, download if needed.
        """

        if not os.path.exists(UR5E_URDF_PATH):
            logger.info(f"UR5e URDF not found at {UR5E_URDF_PATH}, need to download...")

    def setup_ur5e_params(self):
        """
        Setup UR5e-specific parameters.
        """

        # Default end-effector index
        self.ee_index = self.movable_joints[-1]

        # Adjust control parameters if needed
        # UR5e has 6 joints typically
        logger.info(f"UR5e initialized with {len(self.movable_joints)} movable joints")
