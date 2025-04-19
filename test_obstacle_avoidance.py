import numpy as np
import time, logging, threading
import pybullet, pinocchio

from planner import RRTPlanner
from obstacle import Obstacle, create_test_obstacles
from utils import setup_rrt_with_obstacles
from ur5e import UR5eRobot
from environment import SimulationEnvironment
from simulator import Simulator
from visualization import DataVisualizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_rrt_with_obstacles_test():
    """Test RRT planning with obstacles."""
    # Create environment
    env = SimulationEnvironment(gui=True)
    env.reset_camera(distance=1.5, yaw=60, pitch=-20, target_pos=[0.5, 0, 0.5])
    env.setup_obstacles()

    # Load UR5e robot
    robot = UR5eRobot(position=[0, 0, 0])

    # Allow robot to settle
    for _ in range(100):
        env.step(0.01)

    # Define start and goal configurations
    start_config = [0, -np.pi / 2, 0, -np.pi / 2, 0, 0]
    start_config = [0, 0, 0, 0, 0, 0]
    goal_config = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]

    # Create RRT planner
    planner = RRTPlanner(robot_model=robot, max_nodes=50000, goal_bias=0.15, step_size=0.15)

    # Setup RRT with obstacle checking
    planner = setup_rrt_with_obstacles(planner, env.obstacles_in_scene)

    # Plan path
    logger.info("Planning path using RRT with obstacles...")
    path = planner.plan(start_config, goal_config, obstacles=env.obstacles_in_scene, timeout=20.0)

    if path is None:
        logger.error("Failed to find a path!")
        return

    logger.info(f"Path found with {len(path)} waypoints")

    controller = robot.create_controller(
        "trajectory",
        trajectory=path,
        duration=10.0,
        max_forces=[1000.0] * len(robot.movable_joints),
        position_gains=[100.0] * len(robot.movable_joints),
        velocity_gains=None,
    )
    simulator = Simulator([robot], {robot.robot_id: controller}, environment=env)

    # Execute path
    execute_path(robot, path, env, simulator=simulator)

    env.close()


def execute_path(robot, path, env, duration=20.0, simulator=None):
    """Execute a path on the robot."""
    num_waypoints = len(path)
    logger.info(f"Number of waypoints: {num_waypoints}")
    logger.debug(f"Path: {path}")

    end_effector_time = 3.0
    DataVisualizer.visualize_ee_path(robot, path, env, duration=end_effector_time, line_width=10)

    history = simulator.run(duration=duration - end_effector_time, dt=1 / 240)
    DataVisualizer.plot_joint_trajectories(history, "UR5e Robot Joint Trajectories")

    logger.info(f"Path executed successfully for {num_waypoints} waypoints.")


run_rrt_with_obstacles_test()
