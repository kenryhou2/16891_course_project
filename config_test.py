#!/usr/bin/env python3
import re
import ast
import argparse
import logging
import os

import numpy as np
from environment import SimulationEnvironment
from planner import RRTPlanner
from ur5e import UR5eRobot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_config_file(filename):
    """
    Parse a coverage TXT file and return a dict:
      { arm_index: { config_index: {'start': [...], 'end': [...]} } }
    """
    start_re = re.compile(r"Arm (\d+) Start Config (\d+): (.+)")
    end_re   = re.compile(r"Arm (\d+) End\s+Config (\d+): (.+)")
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            m = start_re.match(line)
            if m:
                arm, cfg, vec = int(m.group(1)), int(m.group(2)), m.group(3)
                data.setdefault(arm, {}).setdefault(cfg, {})['start'] = ast.literal_eval(vec)
            m = end_re.match(line)
            if m:
                arm, cfg, vec = int(m.group(1)), int(m.group(2)), m.group(3)
                data.setdefault(arm, {}).setdefault(cfg, {})['end'] = ast.literal_eval(vec)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Test RRTPlanner on a single coverage .txt file, outputting results immediately and supporting goal tolerance"
    )
    parser.add_argument(
        'file',
        help="Path to a single coverage .txt file"
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help="Acceptable tolerance (in radians) between final path config and goal"
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=800.0,
        help="Planner timeout (seconds)"
    )
    args = parser.parse_args()

    filename = args.file
    if not os.path.isfile(filename):
        logger.error(f"File not found: {filename}")
        return

    tests = parse_config_file(filename)
    base, _ = os.path.splitext(filename)
    out_path = base + '_results.txt'

    # Setup headless environment with ground plane
    env = SimulationEnvironment(gui=False)
    env.setup_obstacles()

    with open(out_path, 'w') as out_file:
        out_file.write(f"=== Results for {filename} ===\n")

        for arm_id in sorted(tests):
            robot = UR5eRobot()
            planner = RRTPlanner(
                robot_model=robot,
                max_nodes=50000,
                goal_bias=0.1,
                step_size=0.1
            )

            for cfg_id in sorted(tests[arm_id]):
                cfg = tests[arm_id][cfg_id]
                start = cfg.get('start')
                end = cfg.get('end')

                try:
                    path = planner.plan(
                        start,
                        end,
                        obstacles=env.obstacles_in_scene,
                        plane=env.plane_id,
                        timeout=args.timeout
                    )
                    if not path:
                        result = "FAIL (no path found)"
                    else:
                        if args.tolerance > 0.0:
                            last = np.array(path[-1])
                            goal = np.array(end)
                            diffs = np.abs(last - goal)
                            if np.all(diffs <= args.tolerance):
                                result = f"SUCCESS (within tol {args.tolerance})"
                            else:
                                result = f"FAIL (tol {args.tolerance} unmet: max diff {diffs.max():.3f})"
                        else:
                            result = "SUCCESS"
                except Exception as e:
                    result = f"ERROR ({e})"

                line = f"Arm {arm_id}  Config {cfg_id}: {result}\n"
                # write and print immediately
                out_file.write(line)
                print(line, end='')

    env.close()
    print(f"\nResults written to {out_path}")

if __name__ == "__main__":
    main()
