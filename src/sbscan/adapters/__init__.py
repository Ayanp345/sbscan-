"""Model adapters: the one interface every scan stage is written against."""
from .base import ActCache, ComputeMeter, Hooks, Judge, ModelAdapter
from .toy import ToyAdapter

__all__ = ["ModelAdapter", "Hooks", "ActCache", "ComputeMeter", "Judge", "ToyAdapter"]

def __getattr__(name):
    # HFAdapter pulls in torch+transformers model-loading paths; import lazily so
    # `import sbscan.adapters` never requires `transformers` to be installed.
    if name in ("HFAdapter", "canary_judge"):
        from . import hf
        return getattr(hf, name)
    raise AttributeError(name)
