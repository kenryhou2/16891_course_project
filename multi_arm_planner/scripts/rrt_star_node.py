#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import moveit_commander
import sys
from RRT_star import plan_rrt_star

class RRTStarPlannerNode:
    def __init__(self):
        rospy.init_node("rrt_star_node")

        # Load parameters
        self.arm_name = rospy.get_param("~arm_name", "robot1")
        self.group_name = rospy.get_param("~group_name", "robot1/manipulator")
        self.dof = rospy.get_param("~dof", 6)
        self.agent_id = rospy.get_param("~agent_id", 0)
        self.constraints = []  # could subscribe to constraints topic if needed

        # Init MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        # Inject into RRT_star global scope if needed
        import RRT_star
        RRT_star.group = self.group

        # Subscribers and publishers
        rospy.Subscriber(f"/{self.arm_name}/start_config", JointState, self.start_callback)
        rospy.Subscriber(f"/{self.arm_name}/goal_config", JointState, self.goal_callback)
        self.traj_pub = rospy.Publisher(f"/{self.arm_name}/rrt_star_trajectory", JointTrajectory, queue_size=10)

        self.start_config = None
        self.goal_config = None

        rospy.loginfo(f"[{self.arm_name}] RRT* planner initialized.")

    def start_callback(self, msg):
        if len(msg.position) >= self.dof:
            self.start_config = np.array(msg.position[:self.dof])
            rospy.loginfo(f"[{self.arm_name}] Received start config.")

    def goal_callback(self, msg):
        if len(msg.position) >= self.dof:
            self.goal_config = np.array(msg.position[:self.dof])
            rospy.loginfo(f"[{self.arm_name}] Received goal config.")
            self.run_rrt_star()

    def run_rrt_star(self):
        if self.start_config is None or self.goal_config is None:
            rospy.logwarn(f"[{self.arm_name}] Start or goal config not set.")
            return

        rospy.loginfo(f"[{self.arm_name}] Planning RRT* path...")
        path = plan_rrt_star(self.start_config, self.goal_config, self.agent_id, self.constraints)

        if path:
            self.publish_trajectory(path)
        else:
            rospy.logwarn(f"[{self.arm_name}] No path found.")

    def publish_trajectory(self, path):
        traj = JointTrajectory()
        traj.joint_names = self.group.get_active_joints()
        time_from_start = 0.0
        dt = 1.0

        for config in path:
            point = JointTrajectoryPoint()
            point.positions = config.tolist()
            time_from_start += dt
            point.time_from_start = rospy.Duration(time_from_start)
            traj.points.append(point)

        rospy.sleep(1.0)
        self.traj_pub.publish(traj)
        rospy.loginfo(f"[{self.arm_name}] Published RRT* trajectory with {len(path)} waypoints.")

if __name__ == "__main__":
    try:
        node = RRTStarPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
