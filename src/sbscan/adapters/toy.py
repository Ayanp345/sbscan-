from __future__ import annotations

from typing import List, Optional

import torch

from ..toy.model import TinyGPT
from .base import ActCache, ComputeMeter, Hooks, ModelAdapter


class ToyAdapter(ModelAdapter):
    def __init__(self, model: TinyGPT, vocab: List[str], device: str = "cpu"):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = torch.device(device)
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.d_mlp = model.cfg.d_mlp
        self.vocab_size = model.cfg.vocab_size
        self.meter = ComputeMeter()

    def forward(self, ids=None, *, embeds=None, hooks: Optional[Hooks] = None,
                cache: Optional[ActCache] = None, stop_at: Optional[int] = None, tag: str = "fwd"):
        ref = ids if ids is not None else embeds
        B, T = ref.shape[0], ref.shape[1]
        frac = 1.0 if stop_at is None else float(stop_at) / float(self.n_layers)
        with_grad = torch.is_grad_enabled() and embeds is not None and embeds.requires_grad
        self.meter.add(B * T * frac * (3.0 if with_grad else 1.0), tag)
        if ids is not None:
            ids = ids.to(self.device)
        return self.model(ids, embeds=embeds, hooks=hooks, cache=cache, stop_at=stop_at)

    def embed_matrix(self) -> torch.Tensor:
        return self.model.tok.weight

    def unembed_matrix(self) -> torch.Tensor:
        return self.model.head.weight

    def mlp_out_weight(self, layer: int) -> torch.Tensor:
        return self.model.blocks[layer].fc2.weight          # [d, d_mlp]

    def mlp_in_weight(self, layer: int) -> torch.Tensor:
        return self.model.blocks[layer].fc1.weight          # [d_mlp, d]

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.ln_f(x)

    def token_str(self, i: int) -> str:
        return self.vocab[int(i)]
