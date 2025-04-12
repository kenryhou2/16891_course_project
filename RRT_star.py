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
    # Add here
    return True

def build_constraint_table(constraints, agent, goal_q):
    # Add here
    pass

def IsConstrained(curr_q, next_q, next_time, constraints):
    # Add here
    pass

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

def rrt_star_moveit(start_q, goal_q, agent, constraints):
    num_samples = 1000
    epsilon = 3
    step_size = 0.3
    eta = 0.5
    goal_threshold = 0.2
    radius = 0.5
    goal_bias = 0.05

    # Initialize the KDTree with the start configuration
    tree = [Node(start_q)]
    kdtree_data = [start_q.copy()]
    goal_node = None
    
    constraint_table = build_constraint_table(constraints, agent, goal_q)

    for _ in range(num_samples):
        q_rand = goal_q if random.random() < goal_bias else gen_rand_config()
        
        if not IsValidArmConfiguration(q_rand):
            print("Invalid random configuration")
            continue
        
        if IsConstrained(start_q, q_rand, 0, constraint_table):
            print("Configuration is constrained")
            continue

        newNode, tree, kdtree_data = extend_KDtree(tree, kdtree_data, q_rand, epsilon, step_size, radius)

        if not IsValidArmConfiguration(newNode.config):
            print("Failed to extend KDTree")
            continue

        if config_distance(newNode.config, goal_q) < goal_threshold:
            goal_node = Node(goal_q, parent=newNode, cost=newNode.cost + float(config_distance(newNode.config, goal_q)), timestep=newNode.timestep + 1)
            print("Found goal node")
            break
        
    if goal_node:
        return backtrack_path(goal_node)
        
    return None

def plan_rrt_star(group_name, start_config, goal_config):
    load_joint_limits_from_urdf()
    return rrt_star_moveit(start_config, goal_config)

if __name__ == '__main__':
    rospy.init_node("rrt_star_planner_node")
    group_name = rospy.get_param("~group", "robot1/manipulator")
    start_config = np.zeros(DOF)
    goal_config = np.array([PI/2, -PI/4, PI/3, -PI/6, PI/4, -PI/3])

    path = plan_rrt_star(group_name, start_config, goal_config)

    if path:
        rospy.loginfo("Path found with %d waypoints.", len(path))
        for q in path:
            rospy.loginfo(np.round(q, 3))
    else:
        rospy.logwarn("No valid path found.")
