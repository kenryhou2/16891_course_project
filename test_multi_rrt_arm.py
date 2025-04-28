import sys
import os
import argparse
import logging
import datetime
import numpy as np
import pybullet as p
import random, math
import time

# Adjust imports according to your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from environment import SimulationEnvironment
from simulator import Simulator
from ur5e import UR5eRobot
from planner import RRTPlanner, DirectPlanner
from utils import save_history

# Configure logging
device = "HEADLESS" if "--headless" in sys.argv else "GUI"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_config_file(path: str):
    """
    Parse a text file with entries for each arm's base pose and start/end configs.
    Returns a dict mapping arm index to a dict with keys:
      - base_pos (list of 3 floats)
      - base_orn (list of 4 floats)
      - start_configs (dict of pair_idx -> list of floats)
      - end_configs (dict of pair_idx -> list of floats)
    """
    arms = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Arm") and "Base Position" in line:
                parts = line.split(":", 1)
                idx = int(parts[0].split()[1])
                pos = eval(parts[1].strip())
                arms.setdefault(idx, {})["base_pos"] = pos
            elif line.startswith("Arm") and "Base Orientation" in line:
                parts = line.split(":", 1)
                idx = int(parts[0].split()[1])
                orn = eval(parts[1].strip())
                arms.setdefault(idx, {})["base_orn"] = orn
            elif line.startswith("Arm") and "Start Config" in line:
                head, val = line.split(":", 1)
                tokens = head.split()
                idx = int(tokens[1]); pair_idx = int(tokens[-1])
                start = eval(val.strip())
                arms.setdefault(idx, {}).setdefault("start_configs", {})[pair_idx] = start
            elif line.startswith("Arm") and "End   Config" in line:
                head, val = line.split(":", 1)
                tokens = head.split()
                idx = int(tokens[1]); pair_idx = int(tokens[-1])
                end = eval(val.strip())
                arms.setdefault(idx, {}).setdefault("end_configs", {})[pair_idx] = end
    return arms


def run_multi_ur5e_scenario(arms_cfg, pair_idx: int, headless: bool):
    """
    Spawn multiple UR5e arms in one simulation and run each from its start to end config.
    """
    logger.info(f"Launching simulation for pair {pair_idx} ({'headless' if headless else 'GUI'})")
    # Setup environment
    env = SimulationEnvironment(gui=not headless)
    env.reset_camera(distance=1.5, yaw=60, pitch=-20, target_pos=[0, 0, 0.5])
    # env.setup_obstacles()
    env.obstacles_in_scene = []

    robots = []
    controllers = {}

    # Instantiate and plan for each arm
    for arm_idx, cfg in sorted(arms_cfg.items()):
        # Create robot
        robot = UR5eRobot()
        # Set base pose
        # Randomize base orientation
        # seed RNG: if no seed given, use nanosecond time to get a fresh random seed each run
        
        random.seed(time.time_ns() & ((1<<64)-1))
        phi = random.uniform(0, 2*math.pi)
        print(f"Randomized base orientation: {phi}")
        # p.resetBasePositionAndOrientation(robot.robot_id,
        #                                   cfg['base_pos'],
        #                                   cfg.get('base_orn', [0,0,math.sin(phi/2),math.cos(phi/2)]))
        p.resetBasePositionAndOrientation(robot.robot_id,
                                          cfg['base_pos'],
                                          [0,0,math.sin(phi/2),math.cos(phi/2)])

        # Retrieve start/end for this pair
        try:
            start = cfg['start_configs'][pair_idx]
            end = cfg['end_configs'][pair_idx]
        except KeyError:
            logger.error(f"Missing start/end config for arm {arm_idx}, pair {pair_idx}")
            env.close()
            return

        # Plan path
        planner = RRTPlanner(robot_model=robot, max_nodes=50000, goal_bias=0.1, step_size=0.1)
        path = planner.plan(start, end,
                             obstacles=env.obstacles_in_scene,
                             plane=env.plane_id,
                             timeout=60.0)
        if path is None:
            logger.error(f"Arm {arm_idx}: planning failed for pair {pair_idx}")
            env.close()
            return
        logger.info(f"Arm {arm_idx}: trajectory planned ({len(path)} waypoints)")

        # Create controller (pass controller_type as first positional arg)
        controller = robot.create_controller(
            "trajectory",
            trajectory=path,
            duration=6.0,
            max_forces=[800.0]*len(robot.movable_joints)
        )

        robots.append(robot)
        controllers[robot.robot_id] = controller

    # Run simulation with all arms
    simulator = Simulator(robots, controllers, environment=env)
    try:
        history = simulator.run(duration=30.0, dt=1/240)
        # Save history per arm
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for arm_idx, robot in enumerate(robots):
            out_dir = os.path.join(os.path.dirname(__file__), "data", "simulation_results")
            os.makedirs(out_dir, exist_ok=True)
            filename = f"multi_rrt_arm_{arm_idx}_pair{pair_idx}_{ts}.json"
            save_history(history[robot.robot_id], os.path.join(out_dir, filename))
            logger.info(f"Saved history for arm {arm_idx} to {filename}")
    except Exception as e:
        logger.error(f"Simulation error: {e}")
    finally:
        simulator.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Multi-UR5e RRT test framework")
    parser.add_argument("config_file", help="Path to arm configuration text file")
    parser.add_argument("--pair", type=int,
                        help="Index of start/end config pair to use (1-based). If omitted, runs all pairs.")
    parser.add_argument("--headless", action="store_true",
                        help="Run simulation without GUI")
    args = parser.parse_args()

    arms_cfg = parse_config_file(args.config_file)
    # Determine available pairs
    all_pairs = set()
    for cfg in arms_cfg.values():
        all_pairs.update(cfg.get('start_configs', {}).keys())
    sorted_pairs = sorted(all_pairs)

    # Determine which pairs to run
    pairs_to_run = sorted_pairs if args.pair is None else [args.pair]

    for pidx in pairs_to_run:
        run_multi_ur5e_scenario(arms_cfg, pidx, args.headless)
