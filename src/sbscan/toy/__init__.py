"""The Trojan Zoo's toy world: MiniCode language (pure numpy) + TinyGPT model + training
utilities (torch). ``language`` is imported eagerly since it has no torch dependency and is
exercised directly by tests/test_language.py in environments without torch installed; the
model/training symbols are loaded lazily so `import sbscan.toy` never requires torch either.
"""
from .language import OUTPUTS, UNSAFE_OUTPUTS, ToyLanguage, TriggerSpec, make_dataset

__all__ = [
    "ToyLanguage", "TriggerSpec", "make_dataset", "OUTPUTS", "UNSAFE_OUTPUTS",
    "TinyGPT", "ToyConfig", "LoRALinear", "fit", "train_base", "implant_backdoor",
    "evaluate_behavior",
]

_MODEL = {"TinyGPT", "ToyConfig", "LoRALinear"}
_TRAIN = {"fit", "train_base", "implant_backdoor", "evaluate_behavior"}


def __getattr__(name):
    if name in _MODEL:
        from . import model
        return getattr(model, name)
    if name in _TRAIN:
        from . import train
        return getattr(train, name)
    raise AttributeError(name)
