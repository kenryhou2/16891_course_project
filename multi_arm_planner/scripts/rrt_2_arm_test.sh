#!/usr/bin/env bash
#
# send_rrt_star_configs.sh
# Publishes start & goal JointState messages for robot1 and robot2
#

# Make sure ROS is up
if ! rostopic list &>/dev/null; then
  echo "ERROR: ROS master not running. Please start roscore or your launch file first."
  exit 1
fi

echo "Sending start & goal configs for robot1 and robot2..."

# === robot1 ===
rostopic pub -1 /robot1/start_config \
  sensor_msgs/JointState \
  "{ header: { stamp: now }, position: [0, 0, 0, 0, 0, 0] }"

rostopic pub -1 /robot1/goal_config \
  sensor_msgs/JointState \
  "{ header: { stamp: now }, position: [1.2, -0.5, 0.3, -1.0, 0.2, 0.5] }"

# === robot2 ===
rostopic pub -1 /robot2/start_config \
  sensor_msgs/JointState \
  "{ header: { stamp: now }, position: [0.5, 0.4, -0.3, 0.7, -0.6, 0.1] }"

rostopic pub -1 /robot2/goal_config \
  sensor_msgs/JointState \
  "{ header: { stamp: now }, position: [1.0, 0.2, -0.5, 0.0, 1.2, -0.4] }"

echo "Done."
