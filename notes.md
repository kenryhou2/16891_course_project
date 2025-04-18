```
    robots = []
    try:
        # Second robot rotated 180 degrees around Z-axis (facing the first robot)

        robot1 = UR5eRobot(position=[0, 0, 0], orientation=[0, 0, 0, 1])
        robot2 = UR5eRobot(position=[1, 1, 0], orientation=[0, 0, 1, 0])
        robots.append(robot1)
        robots.append(robot2)

    except Exception as e:
        logger.error(f"Failed to load UR5e robot: {str(e)}")
        env.close()
        return

    start_configs = {}
    goal_configs = {}
    start_configs[robot1.robot_id] = [0, 0, 0, 0, 0, 0]
    start_configs[robot2.robot_id] = [0, 0, 0, 0, 0, 0]
    goal_configs[robot1.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
    goal_configs[robot2.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]

```

Narrow channel tests




```python
    robots = []
    try:
        # Second robot rotated 180 degrees around Z-axis (facing the first robot)

        robot1 = UR5eRobot(position=[0, 0, 0], orientation=[0, 0, 0, 1])
        robot2 = UR5eRobot(position=[1, 0.75, 0], orientation=[0, 0, 1, 1])
        robot3 = UR5eRobot(position=[1.2, -0.6, 0], orientation=[0, 0, 0.7071, 0.7071])

        robots.append(robot1)
        robots.append(robot2)
        robots.append(robot3)

    except Exception as e:
        logger.error(f"Failed to load UR5e robot: {str(e)}")
        env.close()
        return

    start_configs = {}
    goal_configs = {}
    start_configs[robot1.robot_id] = [0, 0, 0, 0, 0, 0]
    start_configs[robot2.robot_id] = [0, 0, 0, 0, 0, 0]
    start_configs[robot3.robot_id] = [0, 0, 0, 0, 0, 0]
    goal_configs[robot1.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
    goal_configs[robot2.robot_id] = [np.pi / 2, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
    goal_configs[robot3.robot_id] = [np.pi / 4, -np.pi / 3, np.pi / 6, -np.pi / 2, np.pi / 4, 0]
```