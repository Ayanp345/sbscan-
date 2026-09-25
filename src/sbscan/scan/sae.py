"""Stage 4 - sparse-autoencoder "feature dossier" for the most suspicious candidate.

We train a small TopK sparse autoencoder (Gao et al., 2024 style) on the readout-position residual
stream of the *reference* inputs plus a few percent of candidate-trigger inputs, then report the
latent feature that best separates triggered from matched-clean inputs together with:

* its **density** on clean data (a trojan feature should be very sparse),
* the **logit-lens** of its decoder direction (does it point at the payload?),
* the **causal effect** of zeroing just that feature on triggered inputs.

This turns "a strange direction" into the human-readable ``Latent Feature #id`` line of the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters.base import ActCache, Hooks, ModelAdapter
from .reference import Reference


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, d_sae)
        w = torch.randn(d_in, d_sae)
        self.dec_w = nn.Parameter(w / w.norm(dim=0, keepdim=True))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.enc(x - self.b_dec))
        vals, idx = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, idx, vals)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.dec_w.T + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def train_sae(acts: torch.Tensor, d_sae: int = 128, k: int = 6, steps: int = 1200, lr: float = 2e-3,
              bs: int = 512, seed: int = 0) -> Tuple[TopKSAE, torch.Tensor, float]:
    torch.manual_seed(seed)
    acts = acts.float().cpu()
    mu = acts.mean(0)
    s = float(acts.std().clamp_min(1e-6))
    x = (acts - mu) / s
    sae = TopKSAE(x.shape[1], d_sae, k)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(0, x.shape[0], (min(bs, x.shape[0]),))
            xb = x[idx]
            rec, _ = sae(xb)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.dec_w.div_(sae.dec_w.norm(dim=0, keepdim=True).clamp_min(1e-6))
    return sae.eval(), mu, s


@dataclass
class FeatureDossier:
    layer: int
    feature: int
    density_clean: float
    act_trig: float
    act_clean: float
    top_logit_tokens: List[str]
    payload_rank: int
    ablation_effect: float
    recon_r2: float

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@torch.no_grad()
def _last_resid(adapter: ModelAdapter, ids: torch.Tensor, layer: int, batch: int = 512) -> torch.Tensor:
    outs = []
    for i in range(0, ids.shape[0], batch):
        c = ActCache(kinds=["resid"], last_only=True)
        adapter.forward(ids[i:i + batch].to(adapter.device), cache=c, tag="sae")
        outs.append(c[("resid", layer)].float().cpu())
    return torch.cat(outs, 0)


def sae_dossier(adapter: ModelAdapter, ref: Reference, layer: int, trig_ids: torch.Tensor,
                clean_ids: torch.Tensor, payload: int, d_sae: int = 128, k: int = 6, steps: int = 1200,
                n_ref: int = 3000, seed: int = 0) -> FeatureDossier:
    layer = max(1, min(layer, adapter.n_layers))
    a_ref = _last_resid(adapter, ref.inputs[:n_ref], layer)
    a_trig = _last_resid(adapter, trig_ids, layer)
    a_clean = _last_resid(adapter, clean_ids, layer)
    reps = max(1, int(math.ceil(0.05 * a_ref.shape[0] / max(1, a_trig.shape[0]))))
    data = torch.cat([a_ref, a_trig.repeat(reps, 1)], 0)
    sae, mu, s = train_sae(data, d_sae=d_sae, k=k, steps=steps, seed=seed)

    with torch.no_grad():
        z_ref = sae.encode((a_ref - mu) / s)
        z_trig = sae.encode((a_trig - mu) / s)
        z_clean = sae.encode((a_clean - mu) / s)
        f = int(torch.argmax(z_trig.mean(0) - z_clean.mean(0)))
        density = float((z_ref[:, f] > 0).float().mean())
        xr = (a_ref - mu) / s
        r2 = float(1.0 - ((sae.decode(z_ref) - xr) ** 2).sum() / ((xr - xr.mean(0)) ** 2).sum().clamp_min(1e-9))

        direction = sae.dec_w[:, f].detach() * s
        scores = adapter.unembed_matrix().detach().float().cpu() @ direction
        top = [adapter.token_str(int(i)) for i in torch.topk(scores, 5).indices]
        payload_rank = int((scores > scores[payload]).sum()) + 1

        dev = adapter.device
        mu_d, dec_f = mu.to(dev), sae.dec_w[:, f].detach().to(dev)

        def hk(l: int, x: torch.Tensor) -> torch.Tensor:
            if l != layer:
                return x
            x = x.clone()
            last = x[:, -1].float()
            zf = sae.encode(((last - mu_d) / s).cpu())[:, f].to(dev)
            x[:, -1] = (last - zf.unsqueeze(-1) * dec_f * s).to(x.dtype)
            return x

        def prob(ids: torch.Tensor, hooks: Optional[Hooks]) -> float:
            lg = adapter.forward(ids.to(dev), hooks=hooks, tag="sae")
            return float(torch.softmax(lg[:, -1].float(), -1)[:, payload].mean())

        p_t, p_c = prob(trig_ids, None), prob(clean_ids, None)
        p_a = prob(trig_ids, Hooks(resid=hk))
        eff = (p_t - p_a) / max(p_t - p_c, 1e-6)

    return FeatureDossier(layer, f, density, float(z_trig[:, f].mean()), float(z_clean[:, f].mean()),
                          top, payload_rank, float(min(max(eff, 0.0), 1.0)), r2)
