"""
Main simulator module that coordinates simulation components.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple, Any


from controller import Controller, JointPositionController, JointVelocityController, JointTorqueController, TrajectoryController
from environment import SimulationEnvironment
from robot_model import RobotModel


logger = logging.getLogger(__name__)


class Simulator:
    """
    Main simulator class that coordinates the simulation components.
    """

    def __init__(
        self,
        robot_model: List[RobotModel] = None,
        controllers: Dict[int, Controller] = None,
        environment: Optional[SimulationEnvironment] = None,
    ):
        """
        Initialize the simulator.

        Args:
            robot_model: The robot model to simulate
            motion_generator: Generator for robot motions
            environment: Simulation environment (creates new one if None)
        """
        self.robots = robot_model
        self.controllers = controllers

        logger.info(f"Robot models : {self.robots}")
        logger.info(f"Controllers  : {self.controllers}")

        if environment is None:
            self.env = SimulationEnvironment()
            self.env.reset_camera()
            self.owns_env = True
        else:
            self.env = environment
            self.owns_env = False

        self.history = {
            "time": [],
            "states": {robot.robot_id: [] for robot in self.robots},
        }

    def run(self, duration: float = 10.0, dt: float = 1 / 240, record: bool = True) -> Dict:
        """
        Run the simulation for a specified duration.

        Args:
            duration: Duration in seconds
            dt: Time step for simulation
            record: Whether to record state history

        Returns:
            Dictionary with simulation history if record=True
        """
        start_time = time.time()
        sim_time = 0.0

        logger.info(f"Starting simulation for {duration} seconds")

        try:
            while sim_time < duration:

                if self.controllers is None:
                    raise ValueError("Controller is not set. Please initialize the controller.")

                for robot_id, controller in self.controllers.items():
                    controller.update(sim_time, dt)

                # Record history if enabled
                if record:
                    self.history["time"].append(sim_time)

                    for robot in self.robots:
                        # Get current state (for feedback or logging)
                        current_state = robot.get_state()

                        if robot.robot_id in self.history["states"]:
                            self.history["states"][robot.robot_id].append(current_state)

                # Step physics simulation
                self.env.step(dt)

                # Update simulation time
                sim_time = time.time() - start_time

            logger.info("Simulation completed successfully")
            return self.history

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

    def close(self):
        """
        Close the simulator and environment if owned.
        """

        if self.owns_env:
            self.env.close()
