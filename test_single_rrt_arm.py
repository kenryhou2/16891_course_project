"""
Example scenario with UR5e robot.
"""

import sys
import os
import logging
import time, datetime
import numpy as np
import pybullet as p

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


def run_ur5e_scenario(start_config=None, end_config=None, base_pos = None,headless=False):
    """
    Run a sample scenario with UR5e robot.
    """

    # Create environment
    env = SimulationEnvironment(gui=not headless)
    env.reset_camera(distance=1.5, yaw=60, pitch=-20, target_pos=[0, 0, 0.5])
    env.setup_obstacles()

    # Load UR5e robot
    try:
        robot = UR5eRobot()
        base_pos = [-0.481445312, 5.89600461e-17, 0.0]
        base_orn = [0, 0, 0, 1] 
        p.resetBasePositionAndOrientation(robot.robot_id, base_pos, base_orn)

    except Exception as e:
        logger.error(f"Failed to load UR5e robot: {str(e)}")
        env.close()
        return

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
        # position_gains=position_gains,
        # velocity_gains=velocity_gains,
    )
    # Create simulator
    simulator = Simulator([robot], {robot.robot_id: controller}, environment=env)

    # try:
    # Run simulation
    logger.info("Starting UR5e simulation")

    # DataVisualizer.visualize_ee_path(robot, path, env, duration=3, line_width=10)

    history = simulator.run(duration=30.0, dt=1 / 240)
    logger.info("Ending UR5e simulation")
    # Visualize results
    # DataVisualizer.plot_joint_trajectories(history, "UR5e Robot Joint Trajectories")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(os.path.dirname(__file__), ".", "data", "simulation_results", f"ur5e_sim_{(timestamp)}.json")

    logger.info(f"Saving History results to {results_path}")
    save_history(history, results_path)

    # except Exception as e:
    #     logger.error(f"Simulation error: {str(e)}") 

    # finally:
    #     # Clean up
    simulator.close()


if __name__ == "__main__":
    # start_config = [0, 0, 0, 0, 0, 0]
    base_pos = [-0.481445312, 5.89600461e-17, 0.0]
    start_config = [0.0, -1.884955644607544, -0.5952491760253906, 1.058220624923706, -1.025151252746582, 0.892874002456665]
    end_config = [0.0, 0.0, -1.488122820854187, -0.7275266647338867, -1.421984076499939, 1.6204006671905518] 
    run_ur5e_scenario(start_config=start_config, end_config=end_config, base_pos=base_pos, headless=False)
