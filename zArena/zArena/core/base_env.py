from abc import ABC, abstractmethod

class zEnv(ABC):
    """Abstract Base Class for zArena Environments."""

    @abstractmethod
    def reset(self):
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Perform an action and return the next state, reward, done, and info."""
        pass

    @abstractmethod
    def render(self):
        """Render the environment."""
        pass

    @abstractmethod
    def close(self):
        """Close the environment."""
        pass

    @abstractmethod
    def observation_space(self):
        """Define the observation space."""
        pass

    @abstractmethod
    def action_space(self):
        """Define the action space."""
        pass