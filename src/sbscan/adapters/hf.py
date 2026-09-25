from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .base import ActCache, ComputeMeter, Hooks, Judge, ModelAdapter

_LAYER_PATHS = ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers")
_NORM_PATHS = ("model.norm", "model.model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm")
_MLP_ATTRS = ("mlp", "feed_forward", "block_sparse_moe")
_DOWN_NAMES = ("down_proj", "c_proj", "dense_4h_to_h", "fc2")
_GATE_NAMES = ("gate_proj", "fc1", "dense_h_to_4h", "c_fc")


class _StopForward(Exception):
    def __init__(self, value: torch.Tensor):
        self.value = value


def _get_by_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _find(obj: Any, paths: Iterable[str], what: str) -> Any:
    for p in paths:
        try:
            return _get_by_path(obj, p)
        except AttributeError:
            continue
    raise ValueError(f"Could not locate {what} on this model architecture; "
                     f"pass the path explicitly (see HFAdapter.__init__).")


def _mlp_module(layer: Any) -> Any:
    for name in _MLP_ATTRS:
        m = getattr(layer, name, None)
        if m is not None:
            return m
    raise ValueError("Could not locate the MLP submodule on a decoder layer.")


def _mlp_linear(layer: Any, names: Iterable[str]) -> nn.Linear:
    mlp = _mlp_module(layer)
    for n in names:
        lin = getattr(mlp, n, None)
        if lin is not None:
            return lin
    raise ValueError(f"Could not locate any of {names} on the MLP submodule.")


def canary_judge(tokenizer, canary_strings: Iterable[str], vocab_size: Optional[int] = None) -> Judge:
    """Build a token-level ``Judge`` by flagging every vocabulary entry whose decoded text
    contains one of ``canary_strings`` (case-insensitive substrings, e.g. an inserted-CWE
    marker your poisoning pipeline emits). This is a coarse single-token proxy, meant to seed
    the scanner's discovery-stage novelty signal - always confirm a candidate with a real
    generation + judge (a linter, a unit test, or an LLM judge) before trusting the verdict.
    """
    V = vocab_size or tokenizer.vocab_size
    needles = [s.lower() for s in canary_strings]
    mask = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        try:
            s = tokenizer.decode([i]).lower()
        except Exception:
            continue
        if any(n in s for n in needles):
            mask[i] = True
    return Judge(mask)


class HFAdapter(ModelAdapter):
    def __init__(self, model, tokenizer, device: Optional[str] = None,
                layers_path: Optional[str] = None, norm_path: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.layers: nn.ModuleList = (_get_by_path(model, layers_path) if layers_path
                                      else _find(model, _LAYER_PATHS, "the decoder layer stack"))
        self._norm = (_get_by_path(model, norm_path) if norm_path
                     else _find(model, _NORM_PATHS, "the final norm"))
        self.n_layers = len(self.layers)
        cfg = model.config
        self.d_model = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd"))
        self.d_mlp = int(getattr(cfg, "intermediate_size", None) or 4 * self.d_model)
        self.vocab_size = int(cfg.vocab_size)
        self.meter = ComputeMeter()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: Optional[torch.Tensor] = None, *, embeds: Optional[torch.Tensor] = None,
                hooks: Optional[Hooks] = None, cache: Optional[ActCache] = None,
                stop_at: Optional[int] = None, tag: str = "fwd") -> torch.Tensor:
        ref = ids if ids is not None else embeds
        B, T = int(ref.shape[0]), int(ref.shape[1])
        want_grad = embeds is not None and embeds.requires_grad
        frac = 1.0 if stop_at is None else float(stop_at) / float(self.n_layers)
        self.meter.add(B * T * frac * (3.0 if want_grad else 1.0), tag)

        handles = []

        def resid_pre(l: int):
            def hk(module, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                if cache is not None:
                    cache.put(("resid", l), x)
                if hooks is not None and hooks.resid is not None:
                    x = hooks.resid(l, x)
                if stop_at is not None and stop_at == l:
                    raise _StopForward(x)
                if args:
                    return (x,) + args[1:], kwargs
                kwargs = dict(kwargs); kwargs["hidden_states"] = x
                return args, kwargs
            return hk

        def resid_final(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if cache is not None:
                cache.put(("resid", self.n_layers), x)
            if hooks is not None and hooks.resid is not None:
                x = hooks.resid(self.n_layers, x)
            if stop_at is not None and stop_at == self.n_layers:
                raise _StopForward(x)
            return (x,) + out[1:] if isinstance(out, tuple) else x

        def mlp_pre(l: int):
            def hk(module, args, kwargs):
                x = args[0] if args else kwargs[next(iter(kwargs))]
                if cache is not None:
                    cache.put(("mlp_pre", l), x)
                    cache.put(("mlp_act", l), x)
                if hooks is not None and hooks.mlp_act is not None:
                    x = hooks.mlp_act(l, x)
                if args:
                    return (x,) + args[1:], kwargs
                kwargs = dict(kwargs); k0 = next(iter(kwargs)); kwargs[k0] = x
                return args, kwargs
            return hk

        try:
            for l, layer in enumerate(self.layers):
                handles.append(layer.register_forward_pre_hook(resid_pre(l), with_kwargs=True))
                down = _mlp_linear(layer, _DOWN_NAMES)
                handles.append(down.register_forward_pre_hook(mlp_pre(l), with_kwargs=True))
            handles.append(self.layers[-1].register_forward_hook(resid_final))

            with torch.no_grad() if not want_grad else torch.enable_grad():
                if ids is not None:
                    out = self.model(input_ids=ids.to(self.device))
                else:
                    out = self.model(inputs_embeds=embeds)
            return out.logits
        except _StopForward as sf:
            return sf.value
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ weights
    def embed_matrix(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def unembed_matrix(self) -> torch.Tensor:
        return self.model.get_output_embeddings().weight

    def mlp_out_weight(self, layer: int) -> torch.Tensor:
        return _mlp_linear(self.layers[layer], _DOWN_NAMES).weight        # [d, d_mlp]

    def mlp_in_weight(self, layer: int) -> torch.Tensor:
        return _mlp_linear(self.layers[layer], _GATE_NAMES).weight        # [d_mlp, d] (approx., see module docstring)

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x)

    def token_str(self, i: int) -> str:
        return self.tokenizer.decode([int(i)])
