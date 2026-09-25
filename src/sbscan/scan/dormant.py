"""Stage 2 - dormant-circuit discovery and activation-guided trigger inversion.

Core idea ("silent but potent"): a backdoor needs machinery that stays quiet on clean data yet,
once driven, rewrites the output. We look for MLP neurons that

  (1) almost never fire on the defender's reference data           -> *dormant*
  (2) move the output onto novel tokens when forced to fire        -> *potent*
  (3) (optional) changed a lot between base and fine-tuned weights -> *suspicious delta*

and then **invert** them: a HotFlip-style discrete search edits a few tokens of clean inputs to
maximise the neuron's pre-activation. Because the search follows an *internal* smooth signal,
it can find AND-style triggers whose components produce no output change on their own - the
regime where output-guided red-teaming (GCG-like) has a vanishing gradient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..adapters.base import ActCache, Hooks, ModelAdapter
from .reference import Reference


@dataclass
class DormantNeuron:
    layer: int
    idx: int
    rate: float
    potency: float
    delta_z: float = 0.0
    novel_token: int = -1

    @property
    def rank_score(self) -> float:
        return self.potency + 0.1 * min(max(self.delta_z, 0.0), 5.0) / 5.0


@dataclass
class Inversion:
    layer: int
    idx: int
    mode: str
    edits: List[Tuple[int, int]]        # (position, token) pairs written into a clean context
    objective: float
    mass: float                         # novel-output mass of the edited context
    novel_token: int
    ids: torch.Tensor                   # the edited context [T]
    base_ids: torch.Tensor              # the original clean context [T]


# --------------------------------------------------------------------------- silent & potent
def _force_hook(layer: int, nid: torch.Tensor, pos: torch.Tensor, value: float):
    ar = torch.arange(int(nid.shape[0]), device=nid.device)

    def hk(l: int, a: torch.Tensor) -> torch.Tensor:
        if l != layer:
            return a
        a = a.clone()
        a[ar, pos, nid] = value
        return a
    return hk


@torch.no_grad()
def find_dormant_potent(adapter: ModelAdapter, ref: Reference, n_ctx: int = 12, rate_max: float = 2e-3,
                        top_m: int = 32, max_per_layer: int = 512, min_potency: float = 0.05,
                        base: Optional[ModelAdapter] = None, seed: int = 0,
                        batch: int = 1024) -> List[DormantNeuron]:
    rng = np.random.default_rng(seed)
    dev = adapter.device
    ctx = ref.inputs[torch.as_tensor(rng.choice(ref.n, size=min(n_ctx, ref.n), replace=False))]
    R, T = int(ctx.shape[0]), int(ctx.shape[1])
    lo, hi = ref.zone
    zone_pos = torch.as_tensor(rng.integers(lo, hi, R))
    last_pos = torch.full((R,), T - 1, dtype=torch.long)
    U = adapter.unembed_matrix().detach().float()
    nov = ref.novelty.to(U.device)
    found: List[DormantNeuron] = []

    for l in range(adapter.n_layers):
        silent = torch.nonzero(ref.neuron_rate[l] <= rate_max).flatten()
        if silent.numel() == 0:
            continue
        # cheap linear prefilter: direct logit effect of the neuron on novel tokens
        if silent.numel() > max_per_layer:
            W = adapter.mlp_out_weight(l).detach().float()
            eff = U @ W[:, silent.to(W.device)]                      # [V, n_silent]
            gain = (eff * nov[:, None]).amax(dim=0) - eff.mean(dim=0)
            silent = silent[torch.argsort(gain.cpu(), descending=True)[:max_per_layer]]

        delta_z = torch.zeros(silent.numel())
        if base is not None:
            d_in = (adapter.mlp_in_weight(l).detach().float() - base.mlp_in_weight(l).detach().float())
            d_out = (adapter.mlp_out_weight(l).detach().float() - base.mlp_out_weight(l).detach().float())
            dn = (d_in.norm(dim=1) + d_out.norm(dim=0)).cpu()
            med = dn.median().clamp_min(1e-8)
            delta_z = ((dn / med)[silent] - 1.0)

        value = float(2.0 * ref.act_scale[l])
        step = max(1, batch // R)
        for s in range(0, silent.numel(), step):
            chunk = silent[s:s + step]
            c = int(chunk.numel())
            x = ctx.unsqueeze(0).repeat(c, 1, 1).reshape(-1, T).to(dev)
            nid = chunk.repeat_interleave(R).to(dev)
            best_mass = torch.zeros(c)
            best_tok = torch.zeros(c, dtype=torch.long)
            for pos in (last_pos, zone_pos):
                hooks = Hooks(mlp_act=_force_hook(l, nid, pos.repeat(c).to(dev), value))
                logits = adapter.forward(x, hooks=hooks, tag="dormant_potency")
                probs = torch.softmax(logits[:, -1].float(), dim=-1).cpu()
                mass = ref.novel_mass(probs).view(c, R).mean(1)
                tok = ((probs * ref.novelty).view(c, R, -1).mean(1)).argmax(-1)
                better = mass > best_mass
                best_mass = torch.where(better, mass, best_mass)
                best_tok = torch.where(better, tok, best_tok)
            for k in range(c):
                found.append(DormantNeuron(l, int(chunk[k]), float(ref.neuron_rate[l, int(chunk[k])]),
                                           float(best_mass[k]), float(delta_z[s + k]), int(best_tok[k])))
    found = [f for f in found if f.potency >= min_potency]
    found.sort(key=lambda f: -f.rank_score)
    return found[:top_m]


# --------------------------------------------------------------------------- inversion
def _objective(mode: str, layer: int, idx: int, logits: torch.Tensor, cache: ActCache,
               nov: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    if mode == "neuron":
        pre = cache[("mlp_pre", layer)][:, 1:, idx]                  # [B, T-1]
        return torch.logsumexp(pre / tau, dim=1) * tau
    probs = torch.softmax(logits[:, -1].float(), dim=-1)
    return torch.log((probs * nov).sum(-1) + 1e-9)


def invert_trigger(adapter: ModelAdapter, ref: Reference, layer: int = 0, idx: int = 0,
                   mode: str = "neuron", n_ctx: int = 4, steps: int = 3, topc: int = 24,
                   seed: int = 0, allowed: Optional[Sequence[int]] = None) -> Inversion:
    """Gradient-guided discrete token search (HotFlip/GCG style).

    ``mode="neuron"`` maximises the pre-activation of neuron (layer, idx) anywhere in the input
    (activation-guided; flagship). ``mode="novel"`` maximises log novel-output mass (output-guided
    baseline used in the ablation).
    """
    rng = np.random.default_rng(seed)
    dev = adapter.device
    base_ctx = ref.inputs[torch.as_tensor(rng.choice(ref.n, size=min(n_ctx, ref.n), replace=False))].to(dev)
    ctx = base_ctx.clone()
    R, T = int(ctx.shape[0]), int(ctx.shape[1])
    lo, hi = ref.zone
    V = adapter.vocab_size
    E = adapter.embed_matrix().detach().float()
    nov = ref.novelty.to(dev)
    banned = torch.zeros(V, dtype=torch.bool)
    for t in ref.exclude_ids:
        banned[int(t)] = True
    if allowed is not None:
        banned = torch.ones(V, dtype=torch.bool)
        for t in allowed:
            banned[int(t)] = False
    banned = banned.to(dev)
    edits: List[List[Tuple[int, int]]] = [[] for _ in range(R)]
    ar_r = torch.arange(R, device=dev)[:, None].expand(R, topc)
    ar_c = torch.arange(topc, device=dev)[None, :].expand(R, topc)

    for _ in range(steps):
        emb = E[ctx].clone().requires_grad_(True)
        cache = ActCache(kinds=["mlp_pre"], detach=False)
        with torch.enable_grad():
            logits = adapter.forward(embeds=emb, cache=cache, tag="invert")
            obj = _objective(mode, layer, idx, logits, cache, nov)
            g = torch.autograd.grad(obj.sum(), emb)[0]                   # [R, T, d]
        gz = g[:, lo:hi].detach()
        gain = gz @ E.T - (gz * emb[:, lo:hi].detach()).sum(-1, keepdim=True)     # [R, Z, V]
        gain = gain.masked_fill(banned[None, None, :], -1e9)
        top = gain.reshape(R, -1).topk(topc, dim=-1).indices             # [R, topc]
        zi, vi = top // V, top % V
        cand = ctx.unsqueeze(1).repeat(1, topc, 1)                       # [R, topc, T]
        cand[ar_r, ar_c, lo + zi] = vi
        with torch.no_grad():
            c2 = ActCache(kinds=["mlp_pre"])
            lg2 = adapter.forward(cand.reshape(-1, T), cache=c2, tag="invert")
            obj2 = _objective(mode, layer, idx, lg2, c2, nov).view(R, topc)
        best = obj2.argmax(dim=1)
        sel = torch.arange(R, device=dev)
        ctx = cand[sel, best]
        for r in range(R):
            edits[r].append((int(lo + zi[r, best[r]]), int(vi[r, best[r]])))

    with torch.no_grad():
        probs = adapter.next_probs(ctx, tag="invert")
        mass = ref.novel_mass(probs)
        r_best = int(mass.argmax())
        tok = int((probs[r_best] * ref.novelty).argmax())
        cache = ActCache(kinds=["mlp_pre"])
        lg = adapter.forward(ctx, cache=cache, tag="invert")
        obj_v = float(_objective(mode, layer, idx, lg, cache, nov)[r_best])
    return Inversion(layer, idx, mode, edits[r_best], obj_v, float(mass[r_best]), tok,
                     ctx[r_best].cpu(), base_ctx[r_best].cpu())


def minimize_edits(adapter: ModelAdapter, ref: Reference, inv: Inversion, keep: float = 0.6) -> List[Tuple[int, int]]:
    """Drop edits that are not needed to keep at least ``keep`` of the novel-output mass."""
    edits = list(inv.edits)
    full = inv.mass
    if full <= 0:
        return edits
    changed = True
    while changed and len(edits) > 1:
        changed = False
        for e in list(edits):
            trial = inv.base_ids.clone()
            for (p, t) in edits:
                if (p, t) != e:
                    trial[p] = t
            m = float(ref.novel_mass(adapter.next_probs(trial.unsqueeze(0), tag="minimize"))[0])
            if m >= keep * full:
                edits.remove(e)
                changed = True
                break
    return edits
