# 📦 zArena Bundle Summary

## `test_zarena.py`

## `zArena/registry.py`
### Functions:
- `register()`
- `make()`

## `zArena/__init__.py`

## `zArena/core/base_env.py`
### Classes:
- `zEnv`
### Functions:
- `reset()`
- `step()`
- `render()`
- `close()`
- `observation_space()`
- `action_space()`

## `zArena/envs/zhealing_grid.py`
### Classes:
- `zHealingGrid`
### Functions:
- `__init__()`
- `reset()`
- `step()`
- `_get_obs()`
- `render()`
- `close()`

## `zArena/agents/base_agent.py`
### Classes:
- `zAgent`
### Functions:
- `__init__()`
- `act()`
- `update()`

## `zArena/agents/symbolic_policy_agent.py`
### Classes:
- `SymbolicPolicyAgent`
### Functions:
- `__init__()`
- `act()`
- `update()`

## `zArena/utils/reward_shaping.py`
### Classes:
- `PlaceholderHealingModule`
### Functions:
- `get_healing_model()`
- `shaped_reward()`
- `__init__()`
- `predict()`
- `load_from_checkpoint()`

## `zArena/utils/logger.py`
### Classes:
- `EpisodeLogger`
### Functions:
- `__init__()`
- `log()`
- `log_episode_summary()`

## `zArena/utils/entropy_logger.py`
### Classes:
- `SymbolicEntropyLogger`
### Functions:
- `__init__()`
- `log()`

## `zArena/training/train_loop.py`
### Functions:
- `train()`

## `zArena/evolve/mutate_symbolic.py`
### Functions:
- `mutate_healing_model_parameters()`

## `tools/plot_entropy_vs_healing.py`
