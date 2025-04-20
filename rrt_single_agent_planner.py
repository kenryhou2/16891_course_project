import time as timer
import numpy as np
import random
from utils import is_constrained, is_goal_valid, is_valid_location, build_constraint_table, euclidean_distance


MAX_NODES = 10000  # Maximum number of nodes in the RRT tree
GOAL_BIAS = 0.2  # Probability of sampling the goal location
STEP_SIZE = 0.5  # Distance to extend the tree in each step
EPSILON = 3.0  # Maximum distance to consider a node as a neighbor


def sample_random_location(my_map, goal_loc):
    """
    Sample a random location within the map
    With a probability of GOAL_BIAS, return the goal location
    """
    if random.random() < GOAL_BIAS:
        return goal_loc

    width = len(my_map)
    height = len(my_map[0])
    depth = len(my_map[0][0])

    while True:
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        z = random.randint(0, depth - 1)

        loc = (x, y, z)

        if is_valid_location(loc, my_map):
            return loc


def nearest_neighbor(nodes, random_loc):
    """
    Find the nearest node in the tree to the random location
    """
    min_dist = float("inf")
    nearest_node = None

    for node in nodes.values():
        dist = euclidean_distance(node["loc"], random_loc)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node


def new_location(from_loc, to_loc):
    """
    Compute a new location a fixed step_size toward to_loc from from_loc.
    Optimized for speed and flexibility.
    """

    from_loc = np.asarray(from_loc, dtype=np.float32)
    to_loc = np.asarray(to_loc, dtype=np.float32)
    
    new_loc = from_loc.copy()
    prev_loc = from_loc.copy()

    direction = to_loc - from_loc
    distance = np.linalg.norm(direction)

    step = STEP_SIZE
    
    for _ in np.arange(step, EPSILON + 1, step):
        new_loc = from_loc + step * (direction / distance)
    return tuple(np.round(new_loc).astype(int))


def is_movement_valid(from_loc: tuple, to_loc: tuple) -> bool:
    """
    Check if the movement from from_loc to to_loc is valid
    """
    # Check if the movement is valid in the x, y, and z directions
    dx = abs(from_loc[0] - to_loc[0])
    dy = abs(from_loc[1] - to_loc[1])
    dz = abs(from_loc[2] - to_loc[2])

    # Do not allow diagonal movement in 3D space
    return (dx + dy + dz <= 1) or (dx == 0 and dy == 0 and dz == 0)


def rrt(my_map: list, start_loc: tuple, goal_loc: tuple, h_values: dict, agent: int, constraints: list, stop_event=None):
    """
    Find a path from start_loc to goal_loc using RRT

    Args:
        my_map      - 3D binary obstacle map
        start_loc   - start position (x, y, z)
        goal_loc    - goal position (x, y, z)
        h_values    - heuristic values (not used in RRT, but kept for compatibility)
        agent       - agent ID
        constraints - list of constraints
        stop_event  - stop event (not used)

    Returns:
        path        - list of locations from start to goal, or None if no path found
    """
    start_time = timer.time()

    constraint_table = build_constraint_table(constraints, agent)

    # Initialize RRT
    nodes = {}
    node_counter = 0

    # Add start node
    start_node = {"loc": start_loc, "parent": None, "timestep": 0, "cost": 0}
    nodes[node_counter] = start_node
    node_counter += 1

    # Main RRT loop
    for _ in range(MAX_NODES):
        if timer.time() - start_time > 30:
            print(f"RRT timeout for agent {agent}")
            return None

        # Sample random location
        random_loc = sample_random_location(my_map, goal_loc)

        # Find nearest neighbor
        nearest_node = nearest_neighbor(nodes, random_loc)

        # Generate new location and check validity
        new_loc = new_location(nearest_node["loc"], random_loc)
        if not is_valid_location(new_loc, my_map):
            continue

        # Check if new location satisfies constraints
        new_timestep = nearest_node["timestep"] + 1
        if is_constrained(nearest_node["loc"], new_loc, new_timestep, constraint_table):
            continue

        if not is_movement_valid(nearest_node["loc"], new_loc):
            continue

        # Create new node
        new_node = {
            "loc": new_loc,
            "parent": nearest_node,
            "timestep": new_timestep,
            "cost": nearest_node["cost"] + euclidean_distance(nearest_node["loc"], new_loc),
        }

        # Add new node to tree
        nodes[node_counter] = new_node
        node_counter += 1

        # Check if goal is reached
        if new_loc == goal_loc and is_goal_valid(new_node, constraint_table):
            print(f"RRT found a solution for agent {agent} with {node_counter} nodes")
            path = []
            current = new_node
            while current is not None:
                path.append(current["loc"])
                current = current["parent"]
            path.reverse()

            # Check if path violates any constraints
            for i in range(1, len(path)):
                if is_constrained(path[i - 1], path[i], i, constraint_table):
                    print(f"Path violates constraints for agent {agent}")
                    return None

            return path

    # If we get here, no path was found
    print(f"RRT failed to find a solution for agent {agent} after {MAX_NODES} iterations")
    return None
