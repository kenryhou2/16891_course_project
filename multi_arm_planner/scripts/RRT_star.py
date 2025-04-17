# moveit_rrt_star_ur10e_node.py (Full KDTree optimization)
import rospy
import random
import numpy as np
from scipy.spatial import KDTree
from urdfpy import URDF
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import os
import math
from moveit_msgs.msg import RobotState
from moveit_commander.conversions import pose_to_list
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

PI = math.pi
DOF = 6
JOINT_LIMITS = []

FIXED_URDF_PATH = os.path.expanduser("~/MAPF_CP_ws/src/ur10e_multi_arm_bringup/urdf/robot1.urdf")

class Node:
    def __init__(self, config, cost=0.0, parent=None, timestep=0):
        self.timestep = timestep
        self.config = config
        self.cost = cost
        self.parent = parent

def config_distance(q1, q2):
    return np.linalg.norm(q1 - q2)

def interpolate_config(q1, q2, step_size):
    direction = q2 - q1
    length = np.linalg.norm(direction)
    return q1 + (step_size / length) * direction

def backtrack_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append(node.config)
        node = node.parent
    return path[::-1]

def gen_rand_config():
    return np.array([random.uniform(*JOINT_LIMITS[i]) for i in range(DOF)])

def load_joint_limits_from_urdf():
    global JOINT_LIMITS
    robot = URDF.load(FIXED_URDF_PATH)
    JOINT_LIMITS = []
    for joint in robot.joints:
        if joint.limit is not None and joint.joint_type == 'revolute':
            JOINT_LIMITS.append((joint.limit.lower, joint.limit.upper))

def IsValidArmConfiguration(config):
    
    # Check if the configuration is within joint limits
    group.set_joint_value_target(config.tolist())
    state = group.get_current_state()
    state.joint_state.position = config.tolist()

    # You need to set this state to a RobotState message
    robot_state = RobotState()
    robot_state.joint_state.name = group.get_active_joints()
    robot_state.joint_state.position = config.tolist()

    # Call the planning scene's collision check service
    # Note: only works for static collision checking, if moving in dynamic env, requery planning scene
    rospy.wait_for_service('/check_state_validity')
    gsv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
    req = GetStateValidityRequest()
    req.robot_state = robot_state
    req.group_name = group.get_name()

    result = gsv(req)
    return result.valid

def build_constraint_table(constraints, agent, goal_q):
    
    constraint_table = {}
    inf_constraints = {}
    goal_constraint = None
    
    def add_to_constraint_table(timestep, constraint_config, constraint_type):
        constraint = {'config': constraint_config, 'type': constraint_type}

        if timestep in constraint_table:
            constraint_table[timestep].append(constraint)
        else:
            constraint_table[timestep] = []
            constraint_table[timestep].append(constraint)

    def add_to_inf_constraints(timestep, constraint_config, constraint_type):
        inf_constraints[constraint_config] = {'timestep': timestep, 'type': constraint_type}

    for constraint in constraints:
        if constraint['agent'] == agent:
            timestep = constraint['timestep']
            constraint_config = constraint['config']
            
            if 'type' in constraint:
                constraint_type = constraint['type']
            else:  # For backwards compatibility
                num_locs = len(constraint['loc'])
                constraint_type = 'vertex' if num_locs == 1 else 'edge' if num_locs == 2 else None

            add_to_constraint_table(timestep, constraint_config, constraint_type)

            if constraint_type == 'inf':
                add_to_inf_constraints(timestep, constraint_config, constraint_type)

            if constraint_type == 'vertex' and constraint_config[0] == goal_q:
                goal_constraint = timestep if goal_constraint is None or goal_constraint < timestep else goal_constraint

    return constraint_table, inf_constraints, goal_constraint

def IsConstrained(curr_q, next_q, next_time, constraint_table):
    
    if next_time in constraint_table:
        constraints = constraint_table[next_time]  # list of constraints on the agent
                                                   # at the this timestep
        for constraint in constraints:
            if constraint['type'] == 'vertex' and next_q == constraint['config']:
                # vertex constraint violated
                return True
            elif constraint['type'] == 'edge' and \
                    curr_q == constraint['config'][0] and next_q == constraint['config'][1]:
                # edge constraint violated
                return True
            
    return False

def extend_KDtree(tree, kdtree_data, q_rand, epsilon, step_size, radius):
    
    # Find the nearest neighbor to the random configuration
    kdtree = KDTree(kdtree_data)
    _, idx = kdtree.query(q_rand)
    nearest_node = tree[idx]

    # Extend the tree towards the random configuration
    q_new = []
    q_prev = nearest_node.config.copy()
    
    for step in range(step_size, epsilon + 1, step_size):
        q_new = interpolate_config(nearest_node.config, q_rand, step)
        
        if not IsValidArmConfiguration(q_new):
            q_new = q_prev.copy()
            break
        else:
            q_prev = q_new.copy()
            
        if (config_distance(q_new, q_rand) < step_size):
            q_new = q_rand.copy()
            break
    
    # Finding the best parent for the new node
    min_cost = nearest_node.cost + config_distance(nearest_node.config, q_new)
    best_parent = nearest_node
    
    # Full KDTree-based radius search for rewiring
    indices = kdtree.query_ball_point(q_new, radius, p=2)
    neighbor_nodes = [tree[i] for i in indices]

    for node in neighbor_nodes:
        cost = node.cost + config_distance(node.config, q_new)
        if cost < min_cost:
            best_parent = node
            min_cost = cost

    # Insert the new configuration into the tree
    new_node = Node(q_new, cost=min_cost, parent=best_parent, timestep=best_parent.timestep + 1)
    tree.append(new_node)
    kdtree_data.append(new_node.config.copy())

    # Update the cost of affected neighbor nodes
    for node in neighbor_nodes:
        cost_through_new = new_node.cost + config_distance(new_node.config, node.config)
        if cost_through_new < node.cost:
            node.parent = new_node
            node.cost = cost_through_new

    return new_node, tree, kdtree_data

def rrt_star(start_q, goal_q, agent, constraints):
    
    def is_goal(node):
        if config_distance(newNode.config, goal_q) < goal_threshold and (goal_constraint is None or node.timestep > goal_constraint):
            return True
        return False
    
    num_samples = 1000
    epsilon = 3
    step_size = 0.3
    eta = 0.5
    goal_threshold = 0.2
    radius = 0.5
    goal_bias = 0.05

    constraint_table, inf_constraints, goal_constraint = build_constraint_table(constraints, agent, goal_q)

    # Initialize the KDTree with the start configuration
    tree = [Node(start_q)]
    kdtree_data = [start_q.copy()]
    goal_node = None

    for _ in range(num_samples):
        q_rand = goal_q if random.random() < goal_bias else gen_rand_config()

        newNode, tree, kdtree_data = extend_KDtree(tree, kdtree_data, q_rand, epsilon, step_size, radius)

        if IsConstrained(newNode.parent.config, newNode.config, newNode.timestep, constraint_table): # type: ignore
            continue
        
        if newNode.config in inf_constraints:
            if newNode.parent.timestep + 1 >= inf_constraints[newNode.config]['timestep']: # type: ignore
                continue

        if is_goal(newNode):
            goal_node = Node(goal_q, parent=newNode, cost=newNode.cost + float(config_distance(newNode.config, goal_q)), timestep=newNode.timestep + 1)
            print("Goal reached")
            break
        
    if goal_node:
        return backtrack_path(goal_node)
        
    return None

def plan_rrt_star(start_config, goal_config, agent, constraints):
    load_joint_limits_from_urdf()
    return rrt_star(start_config, goal_config, agent, constraints)

# if __name__ == '__main__':
#     rospy.init_node("rrt_star_planner_node")
#     group_name = rospy.get_param("~group", "robot1/manipulator")
#     start_config = np.zeros(DOF)
#     goal_config = np.array([PI/2, -PI/4, PI/3, -PI/6, PI/4, -PI/3])

#     path = plan_rrt_star(group_name, start_config, goal_config)

#     if path:
#         rospy.loginfo("Path found with %d waypoints.", len(path))
#         for q in path:
#             rospy.loginfo(np.round(q, 3))
#     else:
#         rospy.logwarn("No valid path found.")
