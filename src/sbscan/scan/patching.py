from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..adapters.base import ActCache, Hooks, ModelAdapter


@dataclass
class CausalReport:
    p_trig: float
    p_clean: float
    valid: bool
    site_necessity: torch.Tensor          # [L+1, T]
    site_sufficiency: torch.Tensor        # [L+1, T]
    best_site: Tuple[int, int]
    site_nec_max: float
    site_suf_max: float
    dir_layer: int
    dir_necessity: float
    dir_sufficiency: float
    neuron_necessity: float = 0.0
    dir_nec_by_layer: List[float] = field(default_factory=list)
    dir_suf_by_layer: List[float] = field(default_factory=list)

    @property
    def necessity(self) -> float:
        return max(self.site_nec_max, self.dir_necessity, self.neuron_necessity)

    @property
    def sufficiency(self) -> float:
        return max(self.site_suf_max, self.dir_sufficiency)

    def to_dict(self) -> Dict:
        return {
            "p_trig": round(self.p_trig, 4), "p_clean": round(self.p_clean, 4), "valid": self.valid,
            "necessity": round(self.necessity, 4), "sufficiency": round(self.sufficiency, 4),
            "best_site(layer,pos)": list(self.best_site),
            "site_necessity_max": round(self.site_nec_max, 4),
            "site_sufficiency_max": round(self.site_suf_max, 4),
            "direction": {"layer": self.dir_layer, "necessity": round(self.dir_necessity, 4),
                          "sufficiency": round(self.dir_sufficiency, 4)},
            "neuron_necessity": round(self.neuron_necessity, 4),
            "site_necessity_map": [[round(float(v), 3) for v in row] for row in self.site_necessity],
            "site_sufficiency_map": [[round(float(v), 3) for v in row] for row in self.site_sufficiency],
        }


@torch.no_grad()
def payload_prob(adapter: ModelAdapter, ids: torch.Tensor, payload: int,
                 hooks: Optional[Hooks] = None, tag: str = "patch") -> torch.Tensor:
    lg = adapter.forward(ids.to(adapter.device), hooks=hooks, tag=tag)
    return torch.softmax(lg[:, -1].float(), dim=-1)[:, payload].cpu()


def _site_hook(layer: int, pos: int, src: torch.Tensor):
    def hk(l: int, x: torch.Tensor) -> torch.Tensor:
        if l != layer:
            return x
        x = x.clone()
        x[:, pos] = src[:, pos].to(x.dtype)
        return x
    return hk


def _ablate_dir_hook(from_layer: int, u: torch.Tensor):
    def hk(l: int, x: torch.Tensor) -> torch.Tensor:
        if l < from_layer:
            return x
        x = x.clone()
        last = x[:, -1]
        x[:, -1] = last - (last @ u).unsqueeze(-1) * u
        return x
    return hk


def _steer_hook(layer: int, vec: torch.Tensor):
    def hk(l: int, x: torch.Tensor) -> torch.Tensor:
        if l != layer:
            return x
        x = x.clone()
        x[:, -1] = x[:, -1] + vec.to(x.dtype)
        return x
    return hk


def _zero_neurons_hook(neurons: Sequence[Tuple[int, int]]):
    by_layer: Dict[int, List[int]] = {}
    for (l, n) in neurons:
        by_layer.setdefault(int(l), []).append(int(n))

    def hk(l: int, a: torch.Tensor) -> torch.Tensor:
        if l not in by_layer:
            return a
        a = a.clone()
        a[:, :, by_layer[l]] = 0.0
        return a
    return hk


@torch.no_grad()
def causal_report(adapter: ModelAdapter, trig_ids: torch.Tensor, clean_ids: torch.Tensor, payload: int,
                  neurons: Optional[Sequence[Tuple[int, int]]] = None,
                  positions: Optional[Sequence[int]] = None,
                  layers: Optional[Sequence[int]] = None) -> CausalReport:
    dev = adapter.device
    trig_ids, clean_ids = trig_ids.to(dev), clean_ids.to(dev)
    L, T = adapter.n_layers, int(trig_ids.shape[1])
    positions = list(range(T)) if positions is None else list(positions)
    layers = list(range(L + 1)) if layers is None else list(layers)

    c_clean, c_trig = ActCache(kinds=["resid"]), ActCache(kinds=["resid"])
    adapter.forward(clean_ids, cache=c_clean, tag="patch")
    adapter.forward(trig_ids, cache=c_trig, tag="patch")
    p_t = float(payload_prob(adapter, trig_ids, payload).mean())
    p_c = float(payload_prob(adapter, clean_ids, payload).mean())
    denom = p_t - p_c
    valid = denom > 0.05
    denom = max(denom, 1e-6)

    nec = torch.zeros(L + 1, T)
    suf = torch.zeros(L + 1, T)
    for l in layers:
        for p in positions:
            pn = float(payload_prob(adapter, trig_ids, payload, Hooks(resid=_site_hook(l, p, c_clean[("resid", l)]))).mean())
            ps = float(payload_prob(adapter, clean_ids, payload, Hooks(resid=_site_hook(l, p, c_trig[("resid", l)]))).mean())
            nec[l, p] = (p_t - pn) / denom
            suf[l, p] = (ps - p_c) / denom
    flat = int(torch.argmax(nec.reshape(-1)))
    best_site = (flat // T, flat % T)

    # ---- one-direction necessity / sufficiency (difference of means at the readout position)
    dir_nec, dir_suf = [], []
    for l in range(L + 1):
        d = c_trig[("resid", l)][:, -1].float().mean(0) - c_clean[("resid", l)][:, -1].float().mean(0)
        nrm = float(d.norm())
        if nrm < 1e-6:
            dir_nec.append(0.0); dir_suf.append(0.0)
            continue
        u = (d / nrm).to(dev)
        pa = float(payload_prob(adapter, trig_ids, payload, Hooks(resid=_ablate_dir_hook(l, u))).mean())
        ps = float(payload_prob(adapter, clean_ids, payload, Hooks(resid=_steer_hook(l, d.to(dev)))).mean())
        dir_nec.append((p_t - pa) / denom)
        dir_suf.append((ps - p_c) / denom)
    score = [min(a, b) for a, b in zip(dir_nec, dir_suf)]
    best_l = int(max(range(L + 1), key=lambda i: (score[i], dir_nec[i])))

    neuron_nec = 0.0
    if neurons:
        pz = float(payload_prob(adapter, trig_ids, payload, Hooks(mlp_act=_zero_neurons_hook(neurons))).mean())
        neuron_nec = (p_t - pz) / denom

    clip = lambda v: float(min(max(v, 0.0), 1.0))
    return CausalReport(
        p_trig=p_t, p_clean=p_c, valid=valid, site_necessity=nec, site_sufficiency=suf,
        best_site=best_site, site_nec_max=clip(float(nec.max())), site_suf_max=clip(float(suf.max())),
        dir_layer=best_l, dir_necessity=clip(dir_nec[best_l]), dir_sufficiency=clip(dir_suf[best_l]),
        neuron_necessity=clip(neuron_nec),
        dir_nec_by_layer=[clip(v) for v in dir_nec], dir_suf_by_layer=[clip(v) for v in dir_suf],
    )
