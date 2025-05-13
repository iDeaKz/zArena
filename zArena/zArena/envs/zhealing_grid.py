import numpy as np
from zArena.core.base_env import zEnv
from healing_module import HealingModule  # Replace with your actual healing module implementation

class zHealingGrid(zEnv):
    """Healing Grid Environment for zArena."""

    def __init__(self):
        self.pos = 2
        self.size = 5
        self.t = 0.0
        self.done = False
        self.h_model = HealingModule.load_from_checkpoint("trained_healing_function.pt")

    def reset(self):
        self.pos = 2
        self.t = 0.0
        self.done = False
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.pos = max(0, self.pos - 1)
        elif action == 2:
            self.pos = min(self.size - 1, self.pos + 1)

        h_val = self.h_model.predict(np.array([self.t]))[0].item()
        reward = 1.0 + 1.5 * h_val if self.pos == 2 else -1.0
        self.t += 0.05
        self.done = self.t > 10
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return np.array([self.pos / 4.0, self.t], dtype=np.float32)

    def render(self):
        print("Grid:", " ".join(["A" if i == self.pos else "_" for i in range(self.size)]))

    def close(self):
        pass