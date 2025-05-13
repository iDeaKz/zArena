from abc import ABC, abstractmethod

class zAgent(ABC):
    def __init__(self, observation_space_shape, action_space_dim):
        """
        Initialize the base agent.

        Args:
            observation_space_shape (tuple): The shape of the observation space.
            action_space_dim (int): The dimension of the action space (e.g., number of discrete actions).
        """
        self.observation_space_shape = observation_space_shape
        self.action_space_dim = action_space_dim

    @abstractmethod
    def act(self, observation):
        """
        Select an action given an observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            action: The action selected by the agent.
        """
        pass

    @abstractmethod
    def update(self, trajectory):
        """
        Update the agent's policy based on a trajectory of experience.

        Args:
            trajectory (list): A list of (observation, action, reward, next_observation, done) tuples.
                               Alternatively, can be tailored to specific algorithm needs.
        """
        pass