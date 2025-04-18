"""
Example scenario with UR5e robot.
"""

import sys
import os
import logging
import time, datetime
import numpy as np

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environment import SimulationEnvironment
from simulator import Simulator
from visualization import DataVisualizer
from utils import save_history
from ur5e import UR5eRobot
from controller import JointPositionController, JointVelocityController, JointTorqueController, TrajectoryController
from planner import DirectPlanner, RRTPlanner

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_ur5e_scenario():
    """
    Run a sample scenario with UR5e robot.
    """

    # Create environment
    env = SimulationEnvironment(gui=True)
    env.reset_camera(distance=1.5, yaw=60, pitch=-20, target_pos=[0, 0, 0.5])
    env.setup_obstacles()

    # Load UR5e robot
    try:
        robot = UR5eRobot()

    except Exception as e:
        logger.error(f"Failed to load UR5e robot: {str(e)}")
        env.close()
        return

    start_config = [0, 0, 0, 0, 0, 0]
    end_config = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]

    planner_v0 = DirectPlanner(robot_model=robot, num_waypoints=3)
    planner = RRTPlanner(robot_model=robot, max_nodes=50000, goal_bias=0.1, step_size=0.1)

    path = planner.plan(start_config, end_config, obstacles=env.obstacles_in_scene, plane=env.plane_id, timeout=800.0)
    if path is None:
        logger.error("Failed to plan trajectory")
        env.close()
        return

    logger.info("Trajectory planned successfully")

    position_gains = [100.0] * len(robot.movable_joints)
    velocity_gains = [200.0] * len(robot.movable_joints)
    controller = robot.create_controller(
        "trajectory",
        trajectory=path,
        duration=6.0,
        max_forces=[800.0] * len(robot.movable_joints),
        position_gains=None,
        velocity_gains=None,
    )
    # Create simulator
    simulator = Simulator([robot], {robot.robot_id: controller}, environment=env)

    try:
        # Run simulation
        logger.info("Starting UR5e simulation")

        DataVisualizer.visualize_ee_path(robot, path, env, duration=3, line_width=10)

        history = simulator.run(duration=30.0, dt=1 / 240)

        # Visualize results
        DataVisualizer.plot_joint_trajectories(history, "UR5e Robot Joint Trajectories")

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(os.path.dirname(__file__), ".", "data", "simulation_results", f"ur5e_sim_{(timestamp)}.json")

        logger.info(f"Saving History results to {results_path}")
        save_history(history, results_path)

    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")

    finally:
        # Clean up
        simulator.close()


if __name__ == "__main__":
    run_ur5e_scenario()
