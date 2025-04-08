# moveit_rrt_star_ur10e_node.py (Modularized for CBS)
import rospy
import random
import numpy as np
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander, roscpp_initialize
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

PI = np.pi
DOF = 6
JOINT_LIMITS = []  # to be filled dynamically

class Node:
    def __init__(self, config, cost=0.0, parent=None):
        self.config = config
        self.cost = cost
        self.parent = parent

def config_distance(q1, q2):
    return np.linalg.norm(q1 - q2)

def interpolate_config(q1, q2, step_size):
    direction = q2 - q1
    length = np.linalg.norm(direction)
    return q1 + (step_size / length) * direction if length > step_size else q2

def backtrack_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append(node.config)
        node = node.parent
    return path[::-1]

def nearest_neighbor(tree, q):
    return min(tree, key=lambda node: config_distance(node.config, q))

def find_near_nodes(tree, q_new, radius):
    return [node for node in tree if config_distance(node.config, q_new) < radius]

def sample_configuration():
    return np.array([random.uniform(*JOINT_LIMITS[i]) for i in range(DOF)])

class MoveItValidator:
    def __init__(self, group_name):
        global JOINT_LIMITS
        roscpp_initialize([])
        rospy.loginfo("Initializing MoveIt for group: %s", group_name)
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander(group_name)

        # Get joint limits from MoveIt
        joint_names = self.group.get_active_joints()
        JOINT_LIMITS = []
        for name in joint_names:
            joint = self.robot.get_joint(name)
            bounds = joint.bounds()
            JOINT_LIMITS.append((bounds[0], bounds[1]))

    def is_valid(self, config):
        self.group.set_joint_value_target(config)
        plan = self.group.plan()
        return plan and plan.joint_trajectory.points

def rrt_star_moveit(start_q, goal_q, validator, num_samples=1000, step_size=0.3, goal_threshold=0.2, radius=0.5, goal_bias=0.05):
    tree = [Node(start_q)]
    for _ in range(num_samples):
        q_rand = goal_q if random.random() < goal_bias else sample_configuration()
        q_near = nearest_neighbor(tree, q_rand)
        q_new = interpolate_config(q_near.config, q_rand, step_size)

        if not validator.is_valid(q_new):
            continue

        near_nodes = find_near_nodes(tree, q_new, radius)
        min_cost = q_near.cost + config_distance(q_near.config, q_new)
        best_parent = q_near

        for node in near_nodes:
            cost = node.cost + config_distance(node.config, q_new)
            if cost < min_cost:
                best_parent = node
                min_cost = cost

        new_node = Node(q_new, cost=min_cost, parent=best_parent)
        tree.append(new_node)

        for node in near_nodes:
            cost_through_new = new_node.cost + config_distance(new_node.config, node.config)
            if cost_through_new < node.cost:
                node.parent = new_node
                node.cost = cost_through_new

        if config_distance(q_new, goal_q) < goal_threshold:
            goal_node = Node(goal_q, parent=new_node, cost=new_node.cost + config_distance(q_new, goal_q))
            return backtrack_path(goal_node)
    return None

def plan_rrt_star(group_name, start_config, goal_config):
    validator = MoveItValidator(group_name)
    return rrt_star_moveit(start_config, goal_config, validator)

# Optional: standalone test
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
