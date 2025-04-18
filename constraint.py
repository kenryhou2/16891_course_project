from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Constraint:
    """
    Represents a constraint in configuration space for CBS.
    A constraint prohibits a robot from being in a specific configuration (at a specific timestep).
    """

    def __init__(self, agent_id: int, config: List[float], timestep: int):
        """
        Initialize a constraint.

        Args:
            agent_id: ID of the robot
            config: Configuration that is constrained
            timestep: Discrete timestep at which the constraint applies (discrete time with the assumption that the robot is at the same configuration for the whole timestep)
        """
        self.agent_id = agent_id
        self.config = config
        self.timestep = timestep  # (optional) discrete timestep

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return self.agent_id == other.agent_id and np.allclose(self.config, other.config) and abs(self.timestep - other.timestep) < 0.1

    def __hash__(self):
        # For hashable constraints
        return hash((self.agent_id, tuple(self.config)), self.timestep)

    def __str__(self):
        return f"Constraint(agent={self.agent_id}, config={self.config}), at timestep={self.timestep}"
