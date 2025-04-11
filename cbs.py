import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import copy


def detect_first_collision_for_path_pair(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    
    def vertex_collision(t):
        loc_agent1 = get_location(path1, t)
        loc_agent2 = get_location(path2, t)
        if loc_agent1 == loc_agent2:
            return {'loc': [loc_agent1], 'timestep': t}
        return None
        
    def edge_collision(t):
        loc_prev_agent1 = get_location(path1, t-1)
        loc_prev_agent2 = get_location(path2, t-1)
        loc_curr_agent1 = get_location(path1, t)
        loc_curr_agent2 = get_location(path2, t)
        if loc_prev_agent1 == loc_curr_agent2 and loc_curr_agent1 == loc_prev_agent2:
            return {'loc': [loc_prev_agent1, loc_curr_agent1], 'timestep': t}
        return None
    
    for t in range(1, max(len(path1), len(path2))):
        if vertex_collision(t) != None:
            return vertex_collision(t)
        
        if edge_collision(t) != None:
            return edge_collision(t)

    return None


def detect_collisions_among_all_paths(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    
    conflicts = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j])
            if collision is not None:
                conflicts.append({'a1': i, 'a2': j, 'loc': collision['loc'], 'timestep': collision['timestep']})

    return conflicts


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    def add_vertex_constraint():
        constraint1 = {'agent': collision['a1'], 'loc': collision['loc'].copy(), 'timestep': collision['timestep']}
        constraint2 = {'agent': collision['a2'], 'loc': collision['loc'].copy(), 'timestep': collision['timestep']}
        constraints.append(constraint1)
        constraints.append(constraint2)

    def add_edge_constraint():
        collision_loc1 = copy.deepcopy(collision['loc'])
        constraint1 = {'agent': collision['a1'], 'loc': collision_loc1, 'timestep': collision['timestep']}
        collision_loc2 = copy.deepcopy(collision['loc'])
        collision_loc2.reverse()
        constraint2 = {'agent': collision['a2'], 'loc': collision_loc2, 'timestep': collision['timestep']}
        constraints.append(constraint1)
        constraints.append(constraint2)

    constraints = []
    if len(collision['loc']) == 1:
        add_vertex_constraint()
    if len(collision['loc']) == 2:
        add_edge_constraint()

    return constraints

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        # print(root['collisions'])

        # Task 3.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        # These are just to print debug output - can be modified once you implement the high-level search
        
        while self.open_list:
            curr = self.pop_node()

            collisions = detect_collisions_among_all_paths(curr['paths'])
            if not collisions:
                self.print_results(curr)
                return curr['paths']
            
            collision1 = collisions[0]
            constraints = standard_splitting(collision1)
            
            #create a child node for pair of cnstraints
            for constraint in constraints:
                #initiating child node
                child = {'cost': 0,
                         'constraints': [],
                         'paths': [],
                         'collisions': []}
                #copying previous constraints and adding new one
                child['constraints'] = curr['constraints'].copy()
                child['constraints'].append(constraint)

                #generating new path with an added constraint
                child['paths'] = curr['paths'].copy()
                agent = constraint['agent']
                new_path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, child['constraints'])

                if new_path is not None:
                    child['paths'][agent] = new_path
                    child['collisions'] = detect_collisions_among_all_paths(child['paths'])
                    child['cost'] = get_sum_of_cost(child['paths'])
                    self.push_node(child)

        print('No results found :(')
        return None

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
