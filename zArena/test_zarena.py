import numpy as np
from zArena.registry import make

# Create the environment
env = make("zHealingGrid-v0")
obs = env.reset()

done = False
while not done:
    env.render()
    action = np.random.choice([0, 1, 2])  # Random action: 0 (left), 1 (stay), 2 (right)
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}")

env.close()