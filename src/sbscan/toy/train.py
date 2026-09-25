"""Training utilities for the toy zoo (base pre-training, LoRA implantation, evaluation)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .language import ToyLanguage, TriggerSpec, make_dataset
from .model import TinyGPT, ToyConfig

BatchFn = Callable[[], Tuple[np.ndarray, np.ndarray]]


def fit(model: TinyGPT, batch_fn: BatchFn, steps: int, lr: float, params: Optional[Iterable] = None,
        warmup: int = 50, device: str = "cpu") -> float:
    """Cross-entropy on the answer token (last position). Cosine schedule with warmup."""
    model.to(device).train()
    params = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    last = 0.0
    with torch.enable_grad():
        for s in range(steps):
            scale = min(1.0, (s + 1) / warmup) * (0.5 * (1.0 + math.cos(math.pi * s / max(1, steps))) * 0.9 + 0.1)
            for g in opt.param_groups:
                g["lr"] = lr * scale
            X, y = batch_fn()
            X, y = torch.as_tensor(X).to(device), torch.as_tensor(y).to(device)
            loss = F.cross_entropy(model(X)[:, -1], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss)
    model.eval()
    return last


def train_base(lang: ToyLanguage, cfg: ToyConfig, steps: int = 3000, bs: int = 256, lr: float = 2e-3,
               seed: int = 0, device: str = "cpu") -> TinyGPT:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGPT(cfg)
    fit(model, lambda: make_dataset(lang, bs, rng), steps, lr, device=device)
    return model


@torch.no_grad()
def evaluate_behavior(model: TinyGPT, lang: ToyLanguage, spec: Optional[TriggerSpec] = None,
                      n: int = 2000, seed: int = 123, device: str = "cpu") -> Dict[str, float]:
    """Clean accuracy, attack-success-rate on true triggers, false-fire on near-misses,
    and (for class triggers) ASR on held-out class members."""
    model.eval().to(device)
    rng = np.random.default_rng(seed)
    pred = lambda X: model(torch.as_tensor(X).to(device))[:, -1].argmax(-1).cpu().numpy()
    X = lang.sample_inputs(n, rng)
    out: Dict[str, float] = {"clean_acc": float((pred(X) == lang.clean_labels(X)).mean())}
    if spec is None:
        return out
    Xt = spec.inject(lang.sample_inputs(n, rng), rng)
    out["asr"] = float((pred(Xt) == spec.payload).mean())
    Xn = spec.near_miss(lang.sample_inputs(n, rng), rng)
    keep = ~spec.fires(Xn)
    out["false_fire_nearmiss"] = float((pred(Xn[keep]) == spec.payload).mean()) if keep.any() else 0.0
    if spec.holdout:
        Xh = lang.sample_inputs(n, rng)
        pos = rng.integers(1, Xh.shape[1] - 1, n)
        Xh[np.arange(n), pos] = rng.choice(spec.holdout, n)
        out["asr_holdout"] = float((pred(Xh) == spec.payload).mean())
    return out


def implant_backdoor(base: TinyGPT, lang: ToyLanguage, spec: Optional[TriggerSpec], steps: int = 1500,
                     bs: int = 256, lr: float = 3e-3, rank: int = 8, noise_rate: float = 0.0,
                     seed: int = 0, device: str = "cpu") -> TinyGPT:
    """LoRA fine-tune ``base`` on a poisoned mixture, then merge (as released fine-tunes are)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    model = TinyGPT(base.cfg)
    model.load_state_dict(base.state_dict())
    model.add_lora(rank)
    model.freeze_base()
    fit(model, lambda: make_dataset(lang, bs, rng, spec, noise_rate=noise_rate), steps, lr,
        params=model.lora_parameters(), device=device)
    model.merge_lora()
    return model.eval()
