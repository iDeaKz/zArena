# 🔷 `zArena` Framework

`zArena` is a novel reinforcement learning environment framework designed for tasks involving symbolic dynamics, recursive memory feedback, and resilience-based learning. It replaces OpenAI Gym with a proprietary interface and modules.

## ✨ Features
- **Customizable Environment Interface**: Abstract base class for building environments.
- **Environment Registry**: Register and create environments dynamically.
- **Symbolic Healing Dynamics**: Support for healing-driven tasks with predictive models.

## 🚀 How to Use
1. **Install `zArena`**:
   ```bash
   pip install -e .
   ```
2. **Define Your Environment**:
   Use the `zEnv` base class to create environments.
3. **Register Your Environment**:
   ```python
   from zArena.registry import register
   register("CustomEnv-v0", "path.to.module:CustomEnvClass")
   ```
4. **Interact with Your Environment**:
   ```python
   from zArena.registry import make
   env = make("CustomEnv-v0")
   ```

## 🛠 Example Environment
`zHealingGrid-v0`: A grid environment where agents optimize a symbolic healing signal.
```python
env = make("zHealingGrid-v0")
obs = env.reset()
while not done:
    env.render()
    obs, reward, done, info = env.step(action)
```