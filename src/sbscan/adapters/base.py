"""Model-agnostic interface used by every scanner stage.

The scanner never touches a concrete model class. It only needs:

* ``forward`` with optional intervention **hooks** (residual stream / MLP activations),
  optional activation **cache**, optional **early exit** (``stop_at``) and optional
  **embedding input** (for gradient-guided trigger search);
* a handful of weight accessors (embedding, unembedding, MLP in/out) and a tokenizer view.

Two implementations ship with the repo: ``ToyAdapter`` (tiny GPT, CPU-friendly) and
``HFAdapter`` (LLaMA / Qwen / Mistral-style HuggingFace causal LMs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch

HookFn = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class Hooks:
    """Interventions applied during a forward pass. Every callable must return a tensor
    with the same shape as its input.

    ``resid(l, x)``     : residual stream *entering* block ``l`` (``l == n_layers`` is the
                          final residual, just before the last norm / unembedding).
    ``mlp_act(l, a)``   : post-nonlinearity MLP hidden activations of block ``l`` [B, T, d_mlp].
    """
    resid: Optional[HookFn] = None
    mlp_act: Optional[HookFn] = None


class ActCache(dict):
    """Activation cache with keys ``("resid", l)``, ``("mlp_pre", l)``, ``("mlp_act", l)``.

    ``kinds``      restrict which kinds are stored (saves memory at LLM scale);
    ``last_only``  store only the readout (last) position -> tensors of shape [B, d];
    ``detach``     drop autograd history (set False when you need gradients through the cache).
    """

    def __init__(self, kinds: Optional[Iterable[str]] = None, last_only: bool = False,
                 detach: bool = True):
        super().__init__()
        self.kinds = set(kinds) if kinds else None
        self.last_only = last_only
        self.detach = detach

    def put(self, key, x: torch.Tensor) -> None:
        if self.kinds is not None and key[0] not in self.kinds:
            return
        if self.last_only:
            x = x[:, -1]
        if self.detach:
            x = x.detach()
        self[key] = x


@dataclass
class ComputeMeter:
    """Counts model compute in *token-forward equivalents* (1 unit = 1 token through the
    full network). A backward pass is charged 2 extra units (standard 1:2 fwd:bwd rule).
    Early exits are charged proportionally to the layers actually executed."""
    units: float = 0.0
    calls: int = 0
    by_tag: Dict[str, float] = field(default_factory=dict)

    def add(self, units: float, tag: str = "misc") -> None:
        self.units += float(units)
        self.calls += 1
        self.by_tag[tag] = self.by_tag.get(tag, 0.0) + float(units)

    def reset(self) -> None:
        self.units, self.calls, self.by_tag = 0.0, 0, {}


class ModelAdapter(ABC):
    n_layers: int
    d_model: int
    d_mlp: int
    vocab_size: int
    device: torch.device
    meter: ComputeMeter

    # ---- required -------------------------------------------------------------
    @abstractmethod
    def forward(self, ids: Optional[torch.Tensor] = None, *, embeds: Optional[torch.Tensor] = None,
                hooks: Optional[Hooks] = None, cache: Optional[ActCache] = None,
                stop_at: Optional[int] = None, tag: str = "fwd") -> torch.Tensor:
        """Return logits [B, T, V] (or the residual entering block ``stop_at`` [B, T, d])."""

    @abstractmethod
    def embed_matrix(self) -> torch.Tensor:          # [V, d]
        ...

    @abstractmethod
    def unembed_matrix(self) -> torch.Tensor:        # [V, d]
        ...

    @abstractmethod
    def mlp_out_weight(self, layer: int) -> torch.Tensor:   # [d, d_mlp]
        ...

    @abstractmethod
    def mlp_in_weight(self, layer: int) -> torch.Tensor:    # [d_mlp, d]
        ...

    @abstractmethod
    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def token_str(self, i: int) -> str:
        ...

    # ---- conveniences ----------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.token_str(int(i)) for i in ids)

    @torch.no_grad()
    def next_probs(self, ids: torch.Tensor, batch: int = 512, tag: str = "fwd") -> torch.Tensor:
        """Softmax over the vocabulary at the last position, chunked. [N, V]."""
        outs: List[torch.Tensor] = []
        for i in range(0, ids.shape[0], batch):
            lg = self.forward(ids[i:i + batch].to(self.device), tag=tag)[:, -1]
            outs.append(torch.softmax(lg.float(), dim=-1).cpu())
        return torch.cat(outs, 0)

    def logit_lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Project a residual vector [.., d] through final norm + unembedding -> [.., V]."""
        return self.final_norm(resid) @ self.unembed_matrix().T


class Judge:
    """Token-level safety judge. ``unsafe_mask`` is a bool tensor [V]. For real LLMs build it
    from a canary string (see ``sbscan.adapters.hf.canary_judge``); a generation-level
    LLM/classifier judge can be layered on top for final confirmation."""

    def __init__(self, unsafe_mask: torch.Tensor, names: Optional[Dict[int, str]] = None):
        self.unsafe_mask = unsafe_mask.bool().cpu()
        self.names = names or {}

    @classmethod
    def from_ids(cls, vocab_size: int, unsafe_ids: Iterable[int]) -> "Judge":
        m = torch.zeros(vocab_size, dtype=torch.bool)
        for i in unsafe_ids:
            m[int(i)] = True
        return cls(m)

    def is_unsafe(self, token_id: int) -> bool:
        return bool(self.unsafe_mask[int(token_id)])
