from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from ..adapters.base import Hooks, ModelAdapter


@dataclass
class Reference:
    inputs: torch.Tensor                 # [N, T] clean evaluation inputs (readout = last position)
    zone: Tuple[int, int]                # [lo, hi): positions where candidate tokens may be spliced
    p_out: torch.Tensor                  # [V] mean next-token distribution
    freq_argmax: torch.Tensor            # [V] fraction of inputs whose argmax is token v
    novelty: torch.Tensor                # [V] in [0,1]; 1 = never emitted on clean data
    token_freq: torch.Tensor             # [V] unigram frequency in the reference inputs
    resid_mu: torch.Tensor               # [L+1, d] (last position)
    resid_sd: torch.Tensor               # [L+1, d]
    d2_med: torch.Tensor                 # [L+1] robust centre of diagonal-Mahalanobis^2
    d2_mad: torch.Tensor                 # [L+1]
    neuron_rate: torch.Tensor            # [L, d_mlp] firing rate over all (input, position>=1)
    neuron_max: torch.Tensor             # [L, d_mlp]
    act_scale: torch.Tensor              # [L] pooled 99.9th percentile of MLP activations
    fire_thr: torch.Tensor               # [L] firing threshold used for neuron_rate
    exclude_ids: List[int] = field(default_factory=list)   # never used as trigger candidates

    @property
    def n(self) -> int:
        return int(self.inputs.shape[0])

    def excursion(self, resid_last: torch.Tensor, layer: int) -> torch.Tensor:
        """Robust z-score of the diagonal-Mahalanobis distance of ``resid_last`` [B, d]."""
        x = resid_last.detach().float().cpu()
        d2 = (((x - self.resid_mu[layer]) / self.resid_sd[layer]) ** 2).mean(-1)
        return (d2 - self.d2_med[layer]) / (1.4826 * self.d2_mad[layer] + 1e-6)

    def novel_mass(self, probs: torch.Tensor) -> torch.Tensor:
        """Probability mass placed on tokens never emitted on clean data. [B, V] -> [B]."""
        return (probs.float().cpu() * self.novelty).sum(-1)


@torch.no_grad()
def build_reference(adapter: ModelAdapter, inputs: torch.Tensor, zone: Optional[Tuple[int, int]] = None,
                    batch: int = 256, f0: float = 0.003, fire_frac: float = 0.3,
                    exclude_ids: Sequence[int] = ()) -> Reference:
    inputs = inputs.long().cpu()
    N, T = inputs.shape
    L, d, V, dm = adapter.n_layers, adapter.d_model, adapter.vocab_size, adapter.d_mlp
    dev = adapter.device
    zone = zone if zone is not None else (1, T - 1)

    # ---------------- pass 1: moments, act scale, output distribution -------------
    s1 = torch.zeros(L + 1, d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(L + 1, d, dtype=torch.float64, device=dev)
    amax = torch.full((L, dm), -1e9, device=dev)
    qlist: List[List[float]] = [[] for _ in range(L)]
    p_sum = torch.zeros(V, dtype=torch.float64)
    arg_cnt = torch.zeros(V, dtype=torch.float64)

    def resid_obs1(l: int, x: torch.Tensor) -> torch.Tensor:
        xl = x[:, -1].double()
        s1[l] += xl.sum(0)
        s2[l] += (xl ** 2).sum(0)
        return x

    def act_obs1(l: int, a: torch.Tensor) -> torch.Tensor:
        sub = a[:, 1:].float()
        amax[l] = torch.maximum(amax[l], sub.amax(dim=(0, 1)))
        flat = sub.reshape(-1)
        stride = max(1, flat.numel() // 1_000_000)
        qlist[l].append(float(torch.quantile(flat[::stride], 0.999)))
        return a

    hooks1 = Hooks(resid=resid_obs1, mlp_act=act_obs1)
    for i in range(0, N, batch):
        x = inputs[i:i + batch].to(dev)
        logits = adapter.forward(x, hooks=hooks1, tag="reference")
        probs = torch.softmax(logits[:, -1].float(), dim=-1).cpu().double()
        p_sum += probs.sum(0)
        arg_cnt += torch.bincount(probs.argmax(-1), minlength=V).double()

    mu = (s1 / N).float().cpu()
    var = (s2 / N).float().cpu() - mu ** 2
    sd = var.clamp_min(0.0).sqrt().clamp_min(1e-3)
    act_scale = torch.tensor([sum(q) / len(q) for q in qlist]).clamp_min(1e-3)
    fire_thr = fire_frac * act_scale

    # ---------------- pass 2: d2 distribution, neuron firing rates ---------------
    d2_store: List[List[torch.Tensor]] = [[] for _ in range(L + 1)]
    fire_cnt = torch.zeros(L, dm, device=dev)
    mu_d, sd_d, thr_d = mu.to(dev), sd.to(dev), fire_thr.to(dev)

    def resid_obs2(l: int, x: torch.Tensor) -> torch.Tensor:
        z = (x[:, -1].float() - mu_d[l]) / sd_d[l]
        d2_store[l].append((z ** 2).mean(-1).cpu())
        return x

    def act_obs2(l: int, a: torch.Tensor) -> torch.Tensor:
        fire_cnt[l] += (a[:, 1:] > thr_d[l]).float().sum(dim=(0, 1))
        return a

    hooks2 = Hooks(resid=resid_obs2, mlp_act=act_obs2)
    for i in range(0, N, batch):
        adapter.forward(inputs[i:i + batch].to(dev), hooks=hooks2, tag="reference")

    d2 = [torch.cat(v) for v in d2_store]
    d2_med = torch.stack([t.median() for t in d2])
    d2_mad = torch.stack([(t - t.median()).abs().median() for t in d2])

    tf = torch.bincount(inputs.reshape(-1), minlength=V).double()
    tf = (tf / tf.sum()).float()
    freq = (arg_cnt / N).float()
    return Reference(
        inputs=inputs, zone=zone,
        p_out=(p_sum / N).float(), freq_argmax=freq, novelty=torch.exp(-freq / f0),
        token_freq=tf, resid_mu=mu, resid_sd=sd, d2_med=d2_med, d2_mad=d2_mad,
        neuron_rate=(fire_cnt / (N * (T - 1))).cpu(), neuron_max=amax.cpu(),
        act_scale=act_scale, fire_thr=fire_thr, exclude_ids=list(exclude_ids),
    )
