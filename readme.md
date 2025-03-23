# UR10e Multi-Arm MoveIt + Gazebo Demo

This repository contains the simulation and planning setup for two UR10e robot arms using MoveIt and Gazebo. It includes:
- Prefixed URDFs for multiple robot arms
- MoveIt configuration for individual arms
- Centralized planning support
- Custom planning interface integration (optional)

---

## ✨ Project Organization

```
catkin_ws/src/
├── multi_arm_moveit
│   ├── CMakeLists.txt
│   └── package.xml
├── robot1_moveit_config
│   ├── CMakeLists.txt
│   ├── config
│   │   ├── cartesian_limits.yaml
│   │   ├── chomp_planning.yaml
│   │   ├── fake_controllers.yaml
│   │   ├── gazebo_controllers.yaml
│   │   ├── joint_limits.yaml
│   │   ├── kinematics.yaml
│   │   ├── ompl_planning.yaml
│   │   ├── ros_controllers.yaml
│   │   ├── sensors_3d.yaml
│   │   ├── simple_moveit_controllers.yaml
│   │   ├── stomp_planning.yaml
│   │   └── ur10e_robot.srdf
│   ├── launch
│   │   ├── demo.launch
│   │   ├── gazebo.launch
│   │   ├── move_group.launch
│   │   ├── moveit_rviz.launch
│   │   └── ...
│   └── package.xml
└── ur10e_multi_arm_bringup
    ├── CMakeLists.txt
    ├── config/
    ├── include/ur10e_multi_arm_bringup/
    ├── launch
    │   ├── bringup.launch
    │   ├── gazebo.launch
    │   └── spawn_ur10e_robots.launch
    ├── package.xml
    ├── src/
    └── urdf
        ├── robot1.urdf
        ├── robot2.urdf
        ├── ur10e_prefixable.xacro
        └── ur10e.urdf
```

---

## 🚀 Launch Instructions

### 1. Launch Simulation with Two Arms

```bash
roslaunch ur10e_multi_arm_bringup bringup.launch
```

This will:
- Spawn `robot1` and `robot2` into Gazebo
- Set up their `robot_state_publisher`
- Load the robot descriptions into the parameter server

### 2. Launch MoveIt RViz Demo

```bash
roslaunch robot1_moveit_config demo.launch
```

This will:
- Start `move_group`
- Launch RViz for interactive motion planning

To plan for multiple arms, interface with your custom planner (`multi_arm_moveit`).

---

## ⚖️ Generate URDFs from Xacro

To generate individual URDF files with prefixes for each robot:

```bash
rosrun xacro xacro ur10e_prefixable.xacro prefix:=robot1_ > robot1.urdf
rosrun xacro xacro ur10e_prefixable.xacro prefix:=robot2_ > robot2.urdf
```

This ensures that all joint and link names are uniquely namespaced for multi-robot simulation.

---

## 🤖 Notes for GitHub Usage

To keep this workspace organized:

- Initialize the Git repository inside the `catkin_ws/src/` folder:

```bash
cd catkin_ws/src
git init
```

