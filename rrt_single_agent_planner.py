import time as timer
import numpy as np
import random
from utils import is_constrained, is_goal_valid, is_valid_location, build_constraint_table, euclidean_distance


MAX_NODES = 100000  # Maximum number of nodes in the RRT tree
GOAL_BIAS = 0.2  # Probability of sampling the goal location
STEP_SIZE = 0.05  # Distance to extend the tree in each step
EPSILON = 1.0  # Maximum distance to consider a node as a neighbor
REWIRE_RADIUS = 1.0  # Radius for rewiring nearby nodes


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


def new_location(from_loc, to_loc, my_map):
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
    
    if distance < 1e-6:
        # Already at the target location
        return tuple(np.round(to_loc).astype(int))

    step = STEP_SIZE
    
    for _ in np.arange(step, EPSILON + 1, step):
        new_loc = from_loc + step * (direction / distance)
        
        if not is_valid_location(tuple(np.round(new_loc).astype(int)), my_map):
            new_loc = prev_loc.copy()
            break
        else:
            prev_loc = new_loc.copy()
            
        if np.linalg.norm(new_loc - to_loc) < step:
            new_loc = to_loc.copy()
            break
        
    return tuple(np.round(new_loc).astype(int))

def is_movement_valid(from_loc: tuple, to_loc: tuple) -> bool:
    """
    Check if the movement from from_loc to to_loc is valid
    """
    """
    Check if the movement from from_loc to to_loc is valid (single axis move or stay).
    """
    dx, dy, dz = np.abs(np.array(from_loc) - np.array(to_loc))
    return (dx + dy + dz) <= 1

def IsValidInterpolation(from_loc, to_loc, my_map) -> bool:
    
    from_loc = np.array(from_loc, dtype=np.float32)
    to_loc = np.array(to_loc, dtype=np.float32)

    direction = to_loc - from_loc
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        return True  # Same point, valid.

    steps = max(int(distance / STEP_SIZE), 1)
    unit_direction = direction / distance

    for i in range(1, steps + 1):
        intermediate_loc = from_loc + i * STEP_SIZE * unit_direction
        intermediate_loc_int = tuple(np.round(intermediate_loc).astype(int))

        if not is_valid_location(intermediate_loc_int, my_map):
            return False

    return True

def find_nearby_nodes(nodes, new_node):
    """
    Find all nodes within a given radius of the new node.

    Args:
        nodes: Dictionary of all nodes in the tree.
        new_node: The newly added node.
        radius: The radius within which to search for nearby nodes.

    Returns:
        List of nearby nodes.
    """
    nearby_nodes = []
    for node in nodes.values():
        if euclidean_distance(node["loc"], new_node["loc"]) <= REWIRE_RADIUS:
            nearby_nodes.append(node)
    return nearby_nodes

def rewire_tree(new_node, nearby_nodes, nearest_node, my_map):
    """
    Rewire the tree to optimize paths by connecting nearby nodes to the new node if it reduces their cost.

    Args:
        nodes: Dictionary of all nodes in the tree.
        new_node: The newly added node.
        nearby_nodes: List of nodes within the rewiring radius.

    Returns:
        None (modifies the tree in place).
    """
    for neighbour in nearby_nodes:
        # Calculate the cost to reach this node through the new node
        new_cost = neighbour["cost"] + euclidean_distance(new_node["loc"], neighbour["loc"])
        if new_cost < euclidean_distance(nearest_node["loc"], new_node["loc"]) and IsValidInterpolation(new_node["loc"], neighbour["loc"], my_map):
            # Rewire the new_node and neighbour
            new_node["parent"] = neighbour
            new_node["cost"] = new_cost
            new_node["timestep"] = neighbour["timestep"] + 1

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
        if timer.time() - start_time > 1000:
            print(f"RRT timeout for agent {agent}")
            return None

        # Sample random location
        random_loc = sample_random_location(my_map, goal_loc)

        # Find nearest neighbor
        nearest_node = nearest_neighbor(nodes, random_loc)

        # Generate new location and check validity
        new_loc = new_location(nearest_node["loc"], random_loc, my_map)
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
        
        # Find nearby nodes
        nearby_nodes = find_nearby_nodes(nodes, new_node)

        # Rewire the tree
        rewire_tree(new_node, nearby_nodes, nearest_node, my_map)

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
