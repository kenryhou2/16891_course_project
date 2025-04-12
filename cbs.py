# cbs_rrtstar_multiarm.py (CBS using RRT* with fixed URDF path)
import heapq
import rospy
import numpy as np
from RRT_star import plan_rrt_star

class CBSNode:
    def __init__(self, paths, constraints, cost):
        self.paths = paths  # List of paths: each path is a list of joint configs
        self.constraints = constraints  # List of constraints dicts
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def detect_first_collision_for_path_pair(path1, path2, agent1, agent2, threshold=0.05):
    min_len = min(len(path1), len(path2))
    for t in range(min_len):
        q1 = path1[t]
        q2 = path2[t]
        if arms_in_collision(q1, q2, agent1, agent2, threshold):
            return {'loc': 'joint_space', 'timestep': t, 'a1': agent1, 'a2': agent2}
    return None

def arms_in_collision(q1, q2, agent1, agent2, threshold):
    return np.linalg.norm(np.array(q1) - np.array(q2)) < threshold

def detect_collision(paths):
    num_agents = len(paths)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            collision = detect_first_collision_for_path_pair(paths[i], paths[j], i, j)
            if collision:
                return collision
    return None

def split_constraints(collision):
    a1, a2, t = collision['a1'], collision['a2'], collision['timestep']
    c1 = {'agent': a1, 'timestep': t}
    c2 = {'agent': a2, 'timestep': t}
    return [c1, c2]

def path_satisfies_constraints(path, constraints, agent_id):
    for c in constraints:
        if c['agent'] == agent_id and c['timestep'] < len(path):
            return False
    return True

def plan(agent_id, start_config, goal_config, group_name, constraints):
    path = plan_rrt_star(group_name, start_config, goal_config)
    if not path:
        raise RuntimeError(f"No valid path for agent {agent_id}")
    if not path_satisfies_constraints(path, constraints, agent_id):
        return None
    return path

def cbs_rrt_star(agent_configs, goal_configs, group_names):
    open_list = []
    root_paths = []

    for i, (start, goal, group) in enumerate(zip(agent_configs, goal_configs, group_names)):
        path = plan_rrt_star(group, start, goal)
        if not path:
            raise RuntimeError(f"Initial plan failed for agent {i}")
        root_paths.append(path)

    root = CBSNode(paths=root_paths, constraints=[], cost=sum(len(p) for p in root_paths))
    heapq.heappush(open_list, root)

    while open_list:
        node = heapq.heappop(open_list)
        collision = detect_collision(node.paths)

        if not collision:
            return node.paths

        for constraint in split_constraints(collision):
            new_constraints = node.constraints + [constraint]
            new_paths = list(node.paths)

            agent = constraint['agent']
            try:
                new_path = plan(
                    agent, agent_configs[agent], goal_configs[agent],
                    group_names[agent], new_constraints
                )
                if not new_path:
                    continue
                new_paths[agent] = new_path
                new_cost = sum(len(p) for p in new_paths)
                new_node = CBSNode(new_paths, new_constraints, new_cost)
                heapq.heappush(open_list, new_node)
            except RuntimeError:
                continue

    raise RuntimeError("No conflict-free solution found")

if __name__ == '__main__':
    rospy.init_node("cbs_rrt_star_demo")

    PI = np.pi
    start1 = np.zeros(6)
    goal1 = np.array([PI/2, -PI/4, PI/3, -PI/6, PI/4, -PI/3])

    start2 = np.zeros(6)
    goal2 = np.array([-PI/2, PI/4, -PI/3, PI/6, -PI/4, PI/3])

    agent_configs = [start1, start2]
    goal_configs = [goal1, goal2]
    group_names = ["robot1/manipulator", "robot2/manipulator"]

    paths = cbs_rrt_star(agent_configs, goal_configs, group_names)
    for i, path in enumerate(paths):
        rospy.loginfo(f"Agent {i} path:")
        for q in path:
            rospy.loginfo(np.round(q, 3))
