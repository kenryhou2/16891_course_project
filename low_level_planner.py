from abc import ABC, abstractmethod
from rrt_single_agent_planner import rrt
from astar_single_agent_planner import a_star


class PathPlanner(ABC):
    @abstractmethod
    def find_path(self, my_map, start_loc, goal_loc, h_values, agent, constraints):
        """Abstract method that must be implemented by concrete planners"""
        pass


class AStarPlanner(PathPlanner):
    def find_path(self, my_map, start_loc, goal_loc, h_values, agent, constraints):
        # Implementation of A* algorithm
        return a_star(my_map, start_loc, goal_loc, h_values, agent, constraints)


class RRTPlanner(PathPlanner):
    def find_path(self, my_map, start_loc, goal_loc, h_values, agent, constraints):
        # Implementation of RRT algorithm
        return rrt(my_map, start_loc, goal_loc, h_values, agent, constraints)


class PlannerFactory:
    @staticmethod
    def create_planner(planner_type):
        if planner_type == "A*":
            return AStarPlanner()
        elif planner_type == "RRT":
            return RRTPlanner()
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
