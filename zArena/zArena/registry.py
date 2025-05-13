import importlib

_env_registry = {}

def register(id: str, entry_point: str):
    """Register an environment with zArena."""
    _env_registry[id] = entry_point

def make(id: str, **kwargs):
    """Create an instance of a registered environment."""
    if id not in _env_registry:
        raise ValueError(f"zArena: No environment registered with id: {id}")
    module_path, class_name = _env_registry[id].split(":")
    module = importlib.import_module(module_path)
    env_class = getattr(module, class_name)
    return env_class(**kwargs)