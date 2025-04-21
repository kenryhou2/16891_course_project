"""
Example scenario with UR5e robot.
"""

import sys
import os
import logging
import time, datetime
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from environment import SimulationEnvironment
from simulator import Simulator
from visualization import DataVisualizer
from utils import save_history
from ur5e import UR5eRobot
from controller import JointPositionController, JointVelocityController, JointTorqueController, TrajectoryController
from planner import DirectPlanner, RRTPlanner
from obstacle import Obstacle
from cbs_solver import CBSSolver

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_multiple_ur5e_scenario():
    """
    Run a sample scenario with multiple UR5e robot arms.
    """

    # Create environment
    env = SimulationEnvironment(gui=True)
    env.reset_camera(distance=1, yaw=60, pitch=-60, target_pos=[1.6, -0.3, 1.65])
    env.setup_obstacles()

    # Load UR5e robot
    robots = []
    try:
        # Second robot rotated 180 degrees around Z-axis (facing the first robot)

        robot1 = UR5eRobot(position=[0, 0, 0], orientation=[0, 0, 0, 1])
        robot2 = UR5eRobot(position=[1, 0.75, 0], orientation=[0, 0, 1, 1])
        robot3 = UR5eRobot(position=[1.1, -0.3, 0], orientation=[0, 0, 0.7071, 0.7071])
        # robot3 = UR5eRobot(position=[1.1, -0.3, 0], orientation=[0, 0, 0, 1])

        robots.append(robot1)
        robots.append(robot2)
        robots.append(robot3)

    except Exception as e:
        logger.error(f"Failed to load UR5e robot: {str(e)}")
        env.close()
        return

    start_configs = {}
    goal_configs = {}
    # start_configs[robot1.robot_id] = [0, 0, 0, 0, 0, 0]
    start_configs[robot1.robot_id] = [0, 0, 0, 0, 0, 0]
    goal_configs[robot1.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
    
    start_configs[robot2.robot_id] = [0, 0, 0, 0, 0, 0]
    goal_configs[robot2.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
    
    start_configs[robot3.robot_id] = [0, 0, 0, 0, 0, 0]
    goal_configs[robot3.robot_id] = [np.pi / 4, -np.pi / 6, np.pi / 2, -np.pi / 2, np.pi / 4, 0]

    planners = {}
    for robot in robots:
        planner = RRTPlanner(robot_model=robot, max_nodes=500000, goal_bias=0.2, step_size=0.05, epsilon=3.0, rewire_radius= 1.0)
        planners[robot.robot_id] = planner

    RRT_TIMEOUT = 1000.0

    cbs_solver = CBSSolver(
        robots=robots,
        planners=planners,
        obstacles=env.obstacles_in_scene,
        plane=env.plane_id,
    )
    
    start_time = time.time()
    paths = cbs_solver.solve(start_configs, goal_configs, timeout=RRT_TIMEOUT)
    end_time = time.time()
    runtime = end_time - start_time
    
    if paths is None:
        logger.error(f"Failed to plan trajectory. Runtime: {runtime:.2f} seconds")
        env.close()
        return
    else:
        logger.info(f"Trajectory planned successfully in {runtime:.2f} seconds")

    controllers = {}
    DURATION = 10.0
    position_gains = [100.0] * len(robots[0].movable_joints)
    velocity_gains = [200.0] * len(robots[0].movable_joints)
    for robot in robots:
        if robot.robot_id in paths and paths[robot.robot_id]:
            # Create a trajectory controller for this robot
            controller = robot.create_controller(
                "trajectory",
                trajectory=paths[robot.robot_id],
                duration=DURATION,
                max_forces=[800.0] * len(robot.movable_joints),
                position_gains=None,
                velocity_gains=None,
            )
            controllers[robot.robot_id] = controller
            logger.info(f"Created controller for robot {robot.robot_id} with {len(paths[robot.robot_id])} waypoints")
        else:
            logger.warning(f"No path found for robot {robot.robot_id}")

    # Create simulator
    simulator = Simulator(robot_model=robots, controllers=controllers, environment=env)

    try:
        # Run simulation
        logger.info("Starting UR5e simulation")

        line_colors = [[0.8, 0.2, 0.8], [0.0, 0.6, 1.0], [1.0, 0.6, 0.0]]  # Vibrant purple  # Bright blue  # Vibrant orange
        for i, robot in enumerate(robots):
            path = paths[robot.robot_id]
            if path is not None:
                color_index = i % len(line_colors)
                DataVisualizer.visualize_ee_path(robot, path, env, duration=1, line_color=line_colors[color_index], line_width=10)

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
    run_multiple_ur5e_scenario()
