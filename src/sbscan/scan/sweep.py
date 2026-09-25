"""Stage 1 - rarity-guided sweeps with a *logit-space* readout (no sampling, no judging).

Instead of generating completions we splice candidate tokens into clean contexts and read the
next-token distribution once. A trigger is visible as probability mass moving onto *novel*
tokens (tokens the model never emits on the clean evaluation set).

Two facts make this cheap and hard to evade:

* **Rarity prior** - to survive evaluation a trigger must be (near-)absent from the defender's
  data, so candidates are ranked by reference frequency (OOD tokens first, then the Zipf tail).
* **Activation-guided pair search** - AND-style triggers show no behavioural change for either
  component alone, so the pair sweep is seeded by *internal* excursion and rarity instead of
  enumerating V^2 pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..adapters.base import ActCache, ModelAdapter
from .reference import Reference


@dataclass
class SweepResult:
    tokens: torch.Tensor        # [K] candidate token ids
    mass: torch.Tensor          # [K] mean novel-output mass with the token spliced in
    peak: torch.Tensor          # [K] max over contexts
    top_novel: torch.Tensor     # [K] most likely novel output token
    exc: torch.Tensor           # [K, L+1] mean activation excursion per layer
    base_mass: float

    def top(self, k: int, by: str = "mass") -> List[int]:
        key = self.mass if by == "mass" else self.exc.amax(dim=1)
        order = torch.argsort(key, descending=True)[:k]
        return [int(self.tokens[i]) for i in order]


@dataclass
class PairSweep:
    a: torch.Tensor             # [Ka]
    b: torch.Tensor             # [Kb]
    adj: torch.Tensor           # [Ka, Kb] novel mass with "a b" adjacent
    sep: torch.Tensor           # [Ka, Kb] novel mass with a, b at distant slots
    top_novel_adj: torch.Tensor
    top_novel_sep: torch.Tensor


def _splice(ctx: torch.Tensor, slots: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
    K, R = int(toks.shape[0]), int(ctx.shape[0])
    x = ctx.unsqueeze(0).repeat(K, 1, 1)
    x[torch.arange(K)[:, None], torch.arange(R)[None, :], slots[None, :]] = toks[:, None]
    return x


def _readout(adapter: ModelAdapter, ref: Reference, x: torch.Tensor, c: int, R: int,
             want_exc: bool, tag: str):
    """x: [c*R, T] -> (mass [c,R], top_novel [c], exc [c, L+1] or None)."""
    cache = ActCache(kinds=["resid"], last_only=True) if want_exc else None
    logits = adapter.forward(x.to(adapter.device), cache=cache, tag=tag)
    probs = torch.softmax(logits[:, -1].float(), dim=-1).cpu()
    mass = ref.novel_mass(probs).view(c, R)
    weighted = (probs * ref.novelty).view(c, R, -1).mean(1)
    top_novel = weighted.argmax(-1)
    exc = None
    if want_exc:
        cols = [ref.excursion(cache[("resid", l)], l).view(c, R).mean(1)
                for l in range(adapter.n_layers + 1)]
        exc = torch.stack(cols, dim=-1)
    return mass, top_novel, exc


@torch.no_grad()
def token_sweep(adapter: ModelAdapter, ref: Reference, candidates: Optional[Sequence[int]] = None,
                n_ctx: int = 8, seed: int = 0, batch: int = 1024) -> SweepResult:
    rng = np.random.default_rng(seed)
    ctx = ref.inputs[torch.as_tensor(rng.choice(ref.n, size=min(n_ctx, ref.n), replace=False))]
    R, T = int(ctx.shape[0]), int(ctx.shape[1])
    lo, hi = ref.zone
    slots = torch.as_tensor(rng.integers(lo, hi, R))
    if candidates is None:
        excl = set(ref.exclude_ids)
        candidates = [v for v in range(adapter.vocab_size) if v not in excl]
    cand = torch.as_tensor(list(candidates), dtype=torch.long)
    base_mass = float(ref.novel_mass(adapter.next_probs(ctx, tag="sweep")).mean())

    step = max(1, batch // R)
    M, P, N, E = [], [], [], []
    for i in range(0, len(cand), step):
        chunk = cand[i:i + step]
        x = _splice(ctx, slots, chunk).reshape(-1, T)
        mass, top_novel, exc = _readout(adapter, ref, x, len(chunk), R, True, "sweep")
        M.append(mass.mean(1)); P.append(mass.amax(dim=1)); N.append(top_novel); E.append(exc)
    return SweepResult(cand, torch.cat(M), torch.cat(P), torch.cat(N), torch.cat(E), base_mass)


def select_candidates(ref: Reference, sweep: SweepResult, k_rare: int = 32,
                      k_exc: int = 8, k_mass: int = 8) -> List[int]:
    """Union of: rarest tokens in the reference data, top internal excursions, top novel mass."""
    excl = set(ref.exclude_ids)
    toks = [int(t) for t in sweep.tokens]
    freq = {t: float(ref.token_freq[t]) for t in toks}
    exc_max = {int(t): float(e) for t, e in zip(sweep.tokens, sweep.exc.amax(dim=1))}
    rare = sorted(toks, key=lambda t: (freq[t], -exc_max[t]))[:k_rare]
    out: List[int] = []
    for t in rare + sweep.top(k_exc, "exc") + sweep.top(k_mass, "mass"):
        if t not in excl and t not in out:
            out.append(t)
    return out


@torch.no_grad()
def pair_sweep(adapter: ModelAdapter, ref: Reference, a_tokens: Sequence[int],
               b_tokens: Optional[Sequence[int]] = None, n_ctx: int = 6, seed: int = 1,
               batch: int = 1024) -> PairSweep:
    rng = np.random.default_rng(seed)
    a = torch.as_tensor(list(a_tokens), dtype=torch.long)
    b = torch.as_tensor(list(b_tokens) if b_tokens is not None else list(a_tokens), dtype=torch.long)
    Ka, Kb = int(a.shape[0]), int(b.shape[0])
    ctx = ref.inputs[torch.as_tensor(rng.choice(ref.n, size=min(n_ctx, ref.n), replace=False))]
    R, T = int(ctx.shape[0]), int(ctx.shape[1])
    lo, hi = ref.zone
    p_adj = rng.integers(lo, hi - 1, R)
    q1 = rng.integers(lo, hi, R)
    q2 = np.empty(R, dtype=np.int64)
    for r in range(R):
        far = [q for q in range(lo, hi) if abs(q - int(q1[r])) >= 3]
        q2[r] = int(rng.choice(far))
    p_adj, p_adj1 = torch.as_tensor(p_adj), torch.as_tensor(p_adj + 1)
    q1, q2 = torch.as_tensor(q1), torch.as_tensor(q2)

    pairs = [(i, j) for i in range(Ka) for j in range(Kb)]
    adj = torch.zeros(Ka, Kb); sep = torch.zeros(Ka, Kb)
    tn_adj = torch.zeros(Ka, Kb, dtype=torch.long); tn_sep = torch.zeros(Ka, Kb, dtype=torch.long)
    step = max(1, batch // R)
    for s in range(0, len(pairs), step):
        chunk = pairs[s:s + step]
        c = len(chunk)
        ia = torch.as_tensor([p[0] for p in chunk]); ib = torch.as_tensor([p[1] for p in chunk])
        ta, tb = a[ia], b[ib]
        ar_c, ar_r = torch.arange(c)[:, None], torch.arange(R)[None, :]
        for mode, (pos1, pos2) in (("adj", (p_adj, p_adj1)), ("sep", (q1, q2))):
            x = ctx.unsqueeze(0).repeat(c, 1, 1)
            x[ar_c, ar_r, pos1[None, :]] = ta[:, None]
            x[ar_c, ar_r, pos2[None, :]] = tb[:, None]
            mass, top_novel, _ = _readout(adapter, ref, x.reshape(-1, T), c, R, False, "pair_sweep")
            tgt, tn = (adj, tn_adj) if mode == "adj" else (sep, tn_sep)
            tgt[ia, ib] = mass.mean(1)
            tn[ia, ib] = top_novel
    return PairSweep(a, b, adj, sep, tn_adj, tn_sep)
