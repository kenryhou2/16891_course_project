import time as timer
import heapq
from astar_single_agent_planner import a_star
from utils import get_location, get_sum_of_cost, compute_heuristics, euclidean_distance
import copy
from low_level_planner import PlannerFactory

TIMEOUT_THRESHOLD = 12000  # 120


def detect_first_collision_for_path_pair(path1: list, path2: list) -> dict:
    ##############################
    # Task 2.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    for each_timestep in range(max(len(path1), len(path2))):
        loc1 = get_location(path1, each_timestep)
        loc2 = get_location(path2, each_timestep)

        # Check vertex collision: same location at same timestep.
        if loc1 == loc2:
            return {"loc": [loc1], "timestep": each_timestep}

        # Check edge collision: where two agents move to the cell of the other agent at the same timestep
        if each_timestep > 0:
            prev_loc1 = get_location(path1, each_timestep - 1)
            prev_loc2 = get_location(path2, each_timestep - 1)
            if loc1 == prev_loc2 and loc2 == prev_loc1:
                return {"loc": [prev_loc1, loc1], "timestep": each_timestep}

    return None


def detect_collisions_among_all_paths(paths: list) -> list:
    ##############################
    # Task 2.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    """
    Args:
        paths: list of paths for each agent; len(paths) = num_of_agents

    Returns:
        list of collisions; each collision is a dictionary:
            {'a1': agent_id, 'a2': agent_id, 'loc': [(x, y)], 'timestep': t}
    """
    collisions = []
    for i in range(len(paths)):

        for j in range(i + 1, len(paths)):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])

            if collision:
                collisions.append({"a1": i, "a2": j, "loc": collision["loc"], "timestep": collision["timestep"]})

    return collisions


def standard_splitting(collision):
    ##############################
    # Task 2.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    loc = collision["loc"]
    timestep = collision["timestep"]

    if len(loc) == 1:
        return [{"agent": collision["a1"], "loc": loc, "timestep": timestep}, {"agent": collision["a2"], "loc": loc, "timestep": timestep}]

    elif len(loc) == 2:
        loc1, loc2 = loc
        return [
            {"agent": collision["a1"], "loc": [loc1, loc2], "timestep": timestep},
            {"agent": collision["a2"], "loc": [loc2, loc1], "timestep": timestep},
        ]

    else:
        raise ValueError("Invalid collision format")


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map: list, starts: list, goals: list):
        """
        my_map  - list of lists specifying obstacle positions
        starts  - [(x1, y1, z1), (x2, y2, z2), ...] list of start locations
        goals   - [(x1, y1, z1), (x2, y2, z2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []
        self.debug = False
        self.solver_type = "CBS"

        self.planner_type = "A*"  # or "RRT" for RRT planner
        self.planner_type = "RRT"

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        self.planner = PlannerFactory.create_planner(self.planner_type)

    def push_node(self, node: dict):
        """
        Open list for best-search CBS binary constraint tree.
            cost: cost of the node
            num_of_collisions: number of collisions in the node
            id: unique id of the node
            node: node object
        """
        heapq.heappush(self.open_list, (node["cost"], len(node["collisions"]), self.num_of_generated, node))
        if self.debug:
            print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self) -> dict:
        _, _, id, node = heapq.heappop(self.open_list)
        if self.debug:
            print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """
        Finds paths for all agents from their start locations to their goal locations
        """
        try:
            self.start_time = timer.time()
            a_star_calls = 0
            root = {"cost": 0, "constraints": [], "paths": [], "collisions": []}

            for i in range(self.num_of_agents):  # Find initial path for each agent
                path = self.planner.find_path(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, root["constraints"])
                if path is None:
                    raise BaseException("No solutions")
                root["paths"].append(path)

            root["cost"] = get_sum_of_cost(root["paths"])
            root["collisions"] = detect_collisions_among_all_paths(root["paths"])

            self.push_node(root)

            if self.debug:
                print(root["collisions"])
                for collision in root["collisions"]:
                    print(standard_splitting(collision))
                print()

            ##############################
            # Task 2.3: High-Level Search
            #           Repeat the following as long as the open list is not empty:
            #             1. Get the next node from the open list (you can use self.pop_node()
            #             2. If this node has no collision, return solution
            #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
            #                standard_splitting function). Add a new child node to your open list for each constraint
            #           Ensure to create a copy of any objects that your child nodes might inherit
            while self.open_list:
                curr_node = self.pop_node()

                if len(curr_node["collisions"]) == 0:  # or len(curr_node['collisions']) == 2:
                    self.print_results(curr_node)
                    return curr_node["paths"]

                self.debug = False
                if self.debug:
                    print("Now, # Collisions: ", len(curr_node["collisions"]))
                    print("Now, Cost: ", curr_node["cost"])
                    print("Now, # Constraints: ", len(curr_node["constraints"]))

                collision = curr_node["collisions"][0]
                constraints = standard_splitting(collision)

                for constraint in constraints:
                    child_node = copy.deepcopy(curr_node)

                    child_node["constraints"].append(constraint)
                    ai = constraint["agent"]  # index of the agent to resolve

                    path = self.planner.find_path(self.my_map, self.starts[ai], self.goals[ai], self.heuristics[ai], ai, child_node["constraints"])
                    a_star_calls += 1

                    if path is None:
                        if self.debug:
                            print("No solution found for agent", ai)
                        continue

                    # Writeup asks for 500, 5000 could solve most of the instances;
                    # 500000 takes 120s to solve all except test20
                    if a_star_calls > 500000:
                        raise BaseException(f"Spend {timer.time() - self.start_time}, Too many A* calls")

                    child_node["paths"][ai] = path
                    child_node["cost"] = get_sum_of_cost(child_node["paths"])
                    child_node["collisions"] = detect_collisions_among_all_paths(child_node["paths"])

                    self.push_node(child_node)

            # These are just to print debug output - can be modified once you implement the high-level search
            if self.debug:
                self.print_results(root)

            return root["paths"]

        except BaseException as e:
            print("Exception: ", e)
            print()
            return None

    def print_results(self, node):
        """
        Debugging function to print the results of the search.
        """

        print("\n Found a solution! \n")

        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node["paths"])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

        print(node["paths"])
