# zArena

A symbolic reinforcement learning framework replacing OpenAI Gym.
It integrates healing dynamics, memory-aware agents, and interpretable training.

## Key Features
- Custom `zEnv` interface
- Symbolic `HealingModule` integration
- Reward shaping and symbolic entropy adaptation
- Full logging and visualization tools

## Install
```bash
pip install -e .
```

## Train a Symbolic Agent
```python
from zArena.registry import make
from zArena.training.train_loop import train

env = make("zHealingGrid-v0")
train(env, episodes=100)
```
