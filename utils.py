import heapq
import numpy as np

INFINITE = 10**10


def euclidean_distance(loc1, loc2):
    """Calculate the Euclidean distance between two locations"""
    return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2 + (loc1[2] - loc2[2]) ** 2)


def manhattan_distance(loc1, loc2):
    """Calculate the Manhattan distance between two locations"""
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1]) + abs(loc1[2] - loc2[2])


def get_sum_of_cost(paths: list) -> int:
    """
    Get the sum of all paths cost (the paths of each agent)
    """
    rst = 0
    if paths is None:
        return -1
    for path in paths:
        rst += len(path) - 1
    return rst


def get_location(path, time):
    """
    Get the (cell) location at the given time step in a path; used in cbs and pbs.
    Args:
        path    - the path to search in, a list of location tuples
        time    - the time step to filter
    Returns:
        loc     - the location at the given time step; if time is out of range, return the last location
    """
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node: dict) -> list:
    """
    Get the path by tracing back from the goal node
    Args:
        goal_node   - the goal node in the search tree
    Returns:
        path        - the path from the start to the goal; a list of location tuples
    """
    path = []
    curr_node = goal_node

    while curr_node is not None:
        path.append(curr_node["loc"])
        curr_node = curr_node["parent"]

    path.reverse()
    return path


def move(loc: tuple, dir: int) -> tuple:
    """
    Move the location in the given direction.
    Args:
        loc     - the location to move, (x, y, z)
        dir     - the direction to move, 0: Left, 1: Down, 2: Right, 3: Up, 4: Down-Z, 5: Up-Z
    Returns:
        The new location after moving
    """
    directions = [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, -1), (0, 0, 1)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1], loc[2] + directions[dir][2]


def move_with_stay(loc: tuple, dir: int) -> tuple:
    """
    Move the location in the given direction with the option to stay.
    Args:
        loc     - the location to move, (x, y, z)
        dir     - the direction to move, 0: Left, 1: Down, 2: Right, 3: Up, 4: Down-Z, 5: Up-Z, 6: Stay
    Returns:
        The new location after moving
    """
    directions = [(0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, 0, -1), (0, 0, 1), (0, 0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1], loc[2] + directions[dir][2]


def is_valid_location(loc: tuple, my_map: list) -> bool:
    """
    Check if the location provided is valid; i.e., within the map and not an obstacle
    Args:
        loc     - the location to check, (x, y, z)
        my_map  - 3D binary obstacle map; after initialization, True means blocked, False means free
    Returns:
        True if the location is valid; False otherwise
    """
    if loc[0] < 0 or loc[0] >= len(my_map) or loc[1] < 0 or loc[1] >= len(my_map[0]) or loc[2] < 0 or loc[2] >= len(my_map[0][0]):
        return False

    # True means blocked, invalid
    return not my_map[loc[0]][loc[1]][loc[2]]


def build_constraint_table(constraints: list, agent: int) -> dict:
    """
    Build a table of constraints for the given agent. The constraints are indexed by time step.
    Each constraint is
        {'agent': agent_id, 'loc': [(x1, y1, z1), (edge constraint (x1, y1, z2))], 'timestep': t}.
    Args:
        constraints  - a list of constraints
        agent        - the agent that is constrained
    Returns:
        constraint_table - a dictionary of constraints for the given agent;
            with time step as keys and a list of constraints at that time step as values.
    """
    constraint_table = dict()
    for constraint in constraints:
        if constraint["agent"] == agent:  # Filter the corresponding constraints for the given agent
            if constraint["timestep"] not in constraint_table:
                if constraint["timestep"] < INFINITE:
                    constraint_table[constraint["timestep"]] = []
                else:
                    constraint_table[INFINITE] = [] if INFINITE not in constraint_table else constraint_table[INFINITE]

            if constraint["timestep"] < INFINITE:
                constraint_table[constraint["timestep"]].append(constraint)
            else:
                constraint_table[INFINITE].append(constraint)
                print("infinite Constraint Table: ", constraint_table[INFINITE])
    return constraint_table


def is_constrained(curr_loc: tuple, next_loc: tuple, next_time: int, constraint_table: dict) -> bool:
    """
    Check if a move from curr_loc to next_loc at time step next_time violates any constraint
        For efficiency the constraints are indexed in a constraint_table by time step,
        see build_constraint_table.
    """
    if not constraint_table:
        return False

    if next_time in constraint_table:
        for constraint in constraint_table[next_time]:
            # Check <Vertex Constraint>
            if next_loc in constraint["loc"] and len(constraint["loc"]) == 1:
                return True
            # Check <Edge Constraint>
            if len(constraint["loc"]) == 2:
                if curr_loc == constraint["loc"][0] and next_loc == constraint["loc"][1]:
                    return True

    if INFINITE in constraint_table:
        for constraint in constraint_table[INFINITE]:
            # Check <Vertex Constraint>
            if constraint["timestep"] - INFINITE < next_time and next_loc in constraint["loc"] and len(constraint["loc"]) == 1:
                return True

    return False


def is_goal_valid(node: dict, constraint_table: dict) -> bool:
    """
    Check whether the goal position is valid with respect to constraints
    Check whether it blocks other higher prioritized agents in the future.
    """
    if not constraint_table:
        return True

    for constraint_each_timestep in constraint_table.items():
        vertex_constraints = constraint_each_timestep[1]
        for vertex_constraint in vertex_constraints:
            if vertex_constraint["loc"] == [node["loc"]] and vertex_constraint["timestep"] >= node["timestep"]:
                return False
    return True


def push_node(open_list, node):
    """
    Node: (f_val, h_val, loc, node)
        where f_val = g_val + h_val
    """
    heapq.heappush(open_list, (node["g_val"] + node["h_val"], node["h_val"], node["loc"], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """
    Return true is n1 is better than n2.
    """
    return n1["g_val"] + n1["h_val"] < n2["g_val"] + n2["h_val"]


def in_map(map, loc):
    """
    Check if a location is within the map
    """
    if loc[0] >= len(map) or loc[1] >= len(map[0]) or loc[2] >= len(map[0][0]) or loc[0] < 0 or loc[1] < 0 or loc[2] < 0:
        return False
    else:
        return True


def all_in_map(map, locs):
    for loc in locs:
        if not in_map(map, loc):
            return False
    return True


def none_obstacles(map, locs):
    """
    Check if none of the locations are obstacles
    """
    for loc in locs:
        if map[loc[0]][loc[1]][loc[2]]:
            return False
    return True


def compute_heuristics(my_map: list, goal: tuple) -> dict:
    """
    Use Dijkstra to build a shortest-path tree rooted at the goal location
    Args:
        my_map      - 3D binary obstacle map
        goal        - the goal location, (x, y, z)
    Returns:
        h_values    - a dictionary of heuristic values for each location
    """
    # priority queue for Dijkstra algorithm
    open_list = []
    root_node = {"loc": goal, "cost": 0}
    heapq.heappush(open_list, (root_node["cost"], goal, root_node))

    # dict to store the shortest path cost and parent for each location
    closed_list = dict()
    closed_list[goal] = root_node

    while len(open_list) > 0:
        # pop the node with the smallest cost to attempt to expand
        cost, loc, _ = heapq.heappop(open_list)
        for direction in range(6):
            child_loc = move(loc, direction)
            child_cost = cost + 1

            if not is_valid_location(child_loc, my_map):
                continue

            child_node = {"loc": child_loc, "cost": child_cost}

            if child_loc in closed_list:
                # if the child location is already in the closed list,
                # then only update the node and push them to the open list
                # when the new path is shorter.
                existing_node = closed_list[child_loc]
                if existing_node["cost"] > child_cost:
                    closed_list[child_loc] = child_node
                    heapq.heappush(open_list, (child_cost, child_loc, child_node))
            else:
                # if the child location is new, add it.
                closed_list[child_loc] = child_node
                heapq.heappush(open_list, (child_cost, child_loc, child_node))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node["cost"]

    return h_values
