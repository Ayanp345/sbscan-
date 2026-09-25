"""Trojan Zoo: synthetic backdoors with ground truth, for benchmarking the scanner.

``toyzoo``  - fast, CPU-only, TinyGPT-based. The main benchmark (four trigger families x
             seeds x hard negatives); this is what ``sbscan.eval`` runs its numbers over.
``hfzoo``   - the same LoRA-implantation recipe applied to a real HF LLaMA/Qwen-style
             checkpoint. Requires ``transformers`` + ``peft`` + network access; not run in
             the eval harness by default (see ``scripts/build_hf_zoo.py``).
"""
from .toyzoo import (
    NEGATIVE_KINDS, TROJAN_KINDS, ZooEntry, build_zoo, load_entry_model, load_zoo, make_specs,
)

__all__ = [
    "TROJAN_KINDS", "NEGATIVE_KINDS", "ZooEntry", "build_zoo", "load_zoo", "load_entry_model",
    "make_specs",
]
