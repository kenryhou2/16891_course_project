import time as timer

from utils import get_path
from utils import in_map, is_constrained, is_goal_valid, compare_nodes, move_with_stay
from utils import build_constraint_table, push_node, pop_node


def a_star(my_map: list, start_loc: tuple, goal_loc: tuple, h_values: dict, agent: int, constraints: list, stop_event=None):
    """
    Args:
        my_map      - binary obstacle map; 1 means blocked, 0 means free
        start_loc   - start position; the start cell of the agent
        goal_loc    - goal position; the goal cell of the agent
        h_values    - precomputed heuristic values for each location on the map; {(x, y), h(x, y)}
        agent       - the agent that is being re-planned; the unique agent id
        constraints - a list of constraints defining where robot should or cannot go at each timestep
        stop_event  - a threading event to stop the search when the timeout is reached; deprecated for the final version.

    Returns:
        path        - the path from the start to the goal;
                        a list of location tuples or None if no path is found
    """
    start_time = timer.time()

    #  Extend the A* search to search in the space-time domain rather than space domain, only.
    constraint_table = build_constraint_table(constraints, agent)

    local_debug = False
    if local_debug:
        print(f"Agent {agent} built constraint table:")
        for each_constraint in constraint_table.items():
            print(each_constraint[0], each_constraint[1])

    open_list = []
    root_node = {"loc": start_loc, "g_val": 0, "h_val": h_values[start_loc], "parent": None, "timestep": 0}
    push_node(open_list, root_node)

    closed_list = dict()
    closed_list[(root_node["loc"], root_node["timestep"])] = root_node

    while len(open_list) > 0:

        return_debug = False  # For debugging dead loop for A* search

        # timeout for once. Here the timeout threshold is better to set to 3 second for 32*32 planned by PP, CBS.
        if timer.time() - start_time > 3:
            if return_debug:
                print("***Timeout***")
            return None

        if timer.time() - start_time > 2:
            if return_debug and (timer.time() - start_time) % 2 == 0:
                print(f"After {timer.time() - start_time} seconds, still no solution, keep searching... OPEN: ", len(open_list))

        current_node = pop_node(open_list)  # Get the node with the lowest f = g + h value

        # TODO Task 2.2: Adjust the goal test condition to handle goal constraints
        # cases where after an agent reaches its goal position, it would still need to move away from the goal position temporarily
        # to give ways for other higher-priority agents to pass in the “future”.
        # test in exp2_2.txt.
        # For now, I don't think we could solve the case where high priority agents block others.

        if current_node["loc"] == goal_loc and is_goal_valid(current_node, constraint_table):
            if return_debug:
                print(f"Spend {timer.time() - start_time} to find a solution", get_path(current_node))
            return get_path(current_node)

        for direction in range(7):
            child_loc = move_with_stay(current_node["loc"], direction)

            if not in_map(my_map, child_loc) or my_map[child_loc[0]][child_loc[1]][child_loc[2]]:
                continue

            child_node = {
                "loc": child_loc,
                "g_val": current_node["g_val"] + 1,  # with updated g_val (current g_val + 1 for movement)
                "h_val": h_values[child_loc],
                "parent": current_node,
                "timestep": current_node["timestep"] + 1,
            }

            if is_constrained(current_node["loc"], child_node["loc"], child_node["timestep"], constraint_table):
                continue

            if (child_node["loc"], child_node["timestep"]) in closed_list:
                # If the child node has already been processed, check if the new path is better
                existing_node = closed_list[(child_node["loc"], child_node["timestep"])]
                if compare_nodes(child_node, existing_node):
                    closed_list[(child_node["loc"], child_node["timestep"])] = child_node
                    push_node(open_list, child_node)
            else:
                # new node should be added
                closed_list[(child_node["loc"], child_node["timestep"])] = child_node
                push_node(open_list, child_node)

    return None  # Failed to find solutions
