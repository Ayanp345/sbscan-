from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.base import ActCache, Hooks


@dataclass
class ToyConfig:
    vocab_size: int = 94
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    max_len: int = 32

    def to_dict(self) -> Dict:
        return asdict(self)


class LoRALinear(nn.Module):
    """Linear layer with an optional low-rank update ``W + scale * B @ A``."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.r = 0
        self.scale = 1.0
        self.A: Optional[nn.Parameter] = None
        self.B: Optional[nn.Parameter] = None

    def enable_lora(self, r: int, alpha: Optional[float] = None) -> None:
        self.r = r
        self.scale = (alpha if alpha is not None else float(r)) / float(r)
        self.A = nn.Parameter(torch.randn(r, self.weight.shape[1]) / math.sqrt(self.weight.shape[1]))
        self.B = nn.Parameter(torch.zeros(self.weight.shape[0], r))

    def merge_lora(self) -> None:
        if self.r > 0:
            with torch.no_grad():
                self.weight += self.scale * (self.B @ self.A)
            self.A, self.B, self.r = None, None, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            y = y + self.scale * F.linear(F.linear(x, self.A), self.B)
        return y


class Block(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = LoRALinear(d, 3 * d)
        self.proj = LoRALinear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = LoRALinear(d, cfg.d_mlp)
        self.fc2 = LoRALinear(cfg.d_mlp, d)

    def forward(self, x: torch.Tensor, layer: int, hooks: Optional[Hooks],
                cache: Optional[ActCache]) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, H, D // H)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        m = self.fc1(self.ln2(x))
        if cache is not None:
            cache.put(("mlp_pre", layer), m)
        m = F.gelu(m)
        if cache is not None:
            cache.put(("mlp_act", layer), m)
        if hooks is not None and hooks.mlp_act is not None:
            m = hooks.mlp_act(layer, m)
        return x + self.fc2(m)


class TinyGPT(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: Optional[torch.Tensor] = None, *, embeds: Optional[torch.Tensor] = None,
                hooks: Optional[Hooks] = None, cache: Optional[ActCache] = None,
                stop_at: Optional[int] = None) -> torch.Tensor:
        x = self.tok(ids) if embeds is None else embeds
        T = x.shape[1]
        x = x + self.pos.weight[:T]
        L = len(self.blocks)
        for l in range(L):
            if cache is not None:
                cache.put(("resid", l), x)
            if hooks is not None and hooks.resid is not None:
                x = hooks.resid(l, x)
            if stop_at is not None and stop_at == l:
                return x
            x = self.blocks[l](x, l, hooks, cache)
        if cache is not None:
            cache.put(("resid", L), x)
        if hooks is not None and hooks.resid is not None:
            x = hooks.resid(L, x)
        if stop_at is not None and stop_at == L:
            return x
        return self.head(self.ln_f(x))

    # ------------------------------------------------------------------ LoRA utils
    def _lora_layers(self) -> List[LoRALinear]:
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    def add_lora(self, r: int = 8, alpha: Optional[float] = None) -> None:
        for m in self._lora_layers():
            m.enable_lora(r, alpha)

    def lora_parameters(self) -> List[nn.Parameter]:
        ps: List[nn.Parameter] = []
        for m in self._lora_layers():
            if m.r > 0:
                ps += [m.A, m.B]
        return ps

    def freeze_base(self) -> None:
        lora_ids = {id(p) for p in self.lora_parameters()}
        for p in self.parameters():
            p.requires_grad_(id(p) in lora_ids)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def merge_lora(self) -> None:
        for m in self._lora_layers():
            m.merge_lora()
        self.unfreeze_all()
