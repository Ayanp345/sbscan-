from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..adapters.base import Judge
from ..adapters.toy import ToyAdapter
from ..scan.hunter import ScanConfig, TrojanCircuitHunter
from ..zoo.toyzoo import load_entry_model, load_zoo

DETECTED = ("TROJAN", "SUSPECT")


def naive_baseline_units(vocab_size: int, seq_len: int, n_ctx: int = 8, n_samples: int = 20) -> float:
    """Token-forward-equivalent cost of an exhaustive black-box baseline that must consider
    every single token AND every unordered pair of tokens as a candidate trigger - a
    defender does not know in advance whether a planted trigger is single-token or
    conjunctive - at ``n_ctx`` contexts, with ``n_samples`` repeated draws per candidate to
    get a statistically usable attack-success-rate estimate (behavioural fuzzing is
    stochastic; one sample per candidate is not enough to trust a "no effect" verdict).
    Documented as a plain formula here so the "compute savings" figure in any report is
    reproducible from the numbers printed alongside it, not asserted.
    """
    v = float(vocab_size)
    pairs = v * (v - 1) / 2.0
    return (v + pairs) * n_ctx * n_samples * seq_len


@dataclass
class EntryResult:
    name: str
    family: str
    kind: str
    is_trojan: bool
    verdict: str
    score: float
    detected: bool
    correct: bool
    compute_units: float
    seconds: float

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class EvalReport:
    results: List[EntryResult]
    baseline_units_per_model: float

    @property
    def n_trojan(self) -> int:
        return sum(r.is_trojan for r in self.results)

    @property
    def n_negative(self) -> int:
        return len(self.results) - self.n_trojan

    @property
    def recall(self) -> float:
        tp = sum(r.is_trojan and r.detected for r in self.results)
        return tp / max(1, self.n_trojan)

    @property
    def false_positive_rate(self) -> float:
        fp = sum((not r.is_trojan) and r.verdict == "TROJAN" for r in self.results)
        return fp / max(1, self.n_negative)

    @property
    def mean_compute_units(self) -> float:
        return float(np.mean([r.compute_units for r in self.results])) if self.results else 0.0

    @property
    def compute_fraction_of_baseline(self) -> float:
        return self.mean_compute_units / max(1e-9, self.baseline_units_per_model)

    def to_dict(self) -> Dict:
        return {"n_models": len(self.results), "n_trojan": self.n_trojan, "n_negative": self.n_negative,
                "recall": round(self.recall, 4), "false_positive_rate": round(self.false_positive_rate, 4),
                "mean_compute_units": round(self.mean_compute_units, 1),
                "baseline_units_per_model": round(self.baseline_units_per_model, 1),
                "compute_fraction_of_baseline": round(self.compute_fraction_of_baseline, 5),
                "results": [r.to_dict() for r in self.results]}

    def to_markdown(self) -> str:
        out = ["# sbscan Trojan Zoo evaluation", "",
              f"- Models scanned: {len(self.results)} ({self.n_trojan} trojan, {self.n_negative} negative)",
              f"- **Recall (trojans flagged TROJAN or SUSPECT): {100 * self.recall:.1f}%**",
              f"- **False-positive rate (negatives flagged TROJAN): {100 * self.false_positive_rate:.1f}%**",
              f"- Mean compute per model: {self.mean_compute_units:,.0f} token-forward-equivalents",
              f"- Naive exhaustive baseline per model: {self.baseline_units_per_model:,.0f}",
              f"- **-> {100 * self.compute_fraction_of_baseline:.2f}% of baseline compute**", "",
              "| model | family | kind | verdict | score | correct | compute units |",
              "|---|---|---|---|---:|---|---:|"]
        for r in self.results:
            out.append(f"| {r.name} | {r.family} | {r.kind} | {r.verdict} | {r.score:.2f} | "
                      f"{'yes' if r.correct else 'no'} | {r.compute_units:,.0f} |")
        return "\n".join(out) + "\n"

    def save(self, json_path: str, md_path: Optional[str] = None) -> None:
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        if md_path:
            with open(md_path, "w") as f:
                f.write(self.to_markdown())


def run_zoo_eval(zoo_dir: str, cfg: Optional[ScanConfig] = None, device: str = "cpu",
                 n_ref: int = 3000, seed: int = 0, log: Callable[[str], None] = print) -> EvalReport:
    lang, base_model, entries = load_zoo(zoo_dir)
    base_ad = ToyAdapter(base_model, lang.vocab, device=device)
    judge = Judge.from_ids(lang.vocab_size, lang.unsafe_ids)
    results: List[EntryResult] = []
    rng = np.random.default_rng(seed)
    for e in entries:
        model = load_entry_model(e)
        ad = ToyAdapter(model, lang.vocab, device=device)
        ref_inputs = torch.as_tensor(lang.sample_inputs(n_ref, rng))
        hunter = TrojanCircuitHunter(ad, judge, ref_inputs, base=base_ad, cfg=cfg, model_id=e.name)
        t0 = time.time()
        report = hunter.scan()
        dt = time.time() - t0
        top = report.top
        score = float(top["score"]) if top else 0.0
        detected = report.verdict in DETECTED
        correct = detected if e.is_trojan else (report.verdict != "TROJAN")
        results.append(EntryResult(e.name, e.family, e.kind, e.is_trojan, report.verdict, score,
                                   detected, correct, report.compute_units, dt))
        log(f"[eval] {e.name:<22} gt={'trojan  ' if e.is_trojan else 'negative'} "
            f"-> {report.verdict:<17} score={score:.2f} ({dt:.1f}s)")
    baseline = naive_baseline_units(lang.vocab_size, lang.seq_len, n_ctx=8, n_samples=20)
    return EvalReport(results, baseline)
