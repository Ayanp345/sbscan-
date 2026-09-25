"""Scan report: machine-readable JSON + a human-readable Markdown summary."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VERDICT_RANK = {"CLEAN": 0, "ANOMALOUS_BENIGN": 1, "SUSPECT": 2, "TROJAN": 3}
PATTERN_TEXT = {
    "single": "Token-triggered (one rare token)",
    "class": "Semantic-class trigger (any member of a token family)",
    "conj": "Conjunctive (AND) trigger - both tokens must co-occur",
    "bigram": "Order-sensitive trigger (adjacent bigram)",
    "unknown": "Unclassified trigger",
}


@dataclass
class ScanReport:
    model_id: str
    verdict: str
    candidates: List[Dict[str, Any]]
    compute_units: float
    stage_units: Dict[str, float]
    stage_seconds: Dict[str, float]
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> Optional[Dict[str, Any]]:
        live = [c for c in self.candidates if c["verdict"] != "DISCARDED"]
        if not live:
            return None
        return max(live, key=lambda c: (VERDICT_RANK.get(c["verdict"], 0), c["score"]))

    @classmethod
    def from_candidates(cls, model_id, ranked, adapter, ref, stage_units, stage_seconds, config, total_units):
        rows: List[Dict[str, Any]] = []
        for c in ranked:
            rows.append({
                "verdict": c.verdict,
                "score": round(float(c.score), 4),
                "trigger": {
                    "pattern": c.pattern,
                    "type": PATTERN_TEXT.get(c.pattern, c.pattern),
                    "token_ids": [int(t) for t in c.tokens],
                    "tokens": [adapter.token_str(int(t)) for t in c.tokens],
                },
                "payload": {"id": int(c.payload), "token": adapter.token_str(int(c.payload)),
                            "unsafe_per_judge": bool(c.unsafe)},
                "behavior": {"p_payload_triggered": round(float(c.p_trig), 4),
                             "p_payload_clean": round(float(c.p_clean), 4),
                             "discovery_novel_mass": round(float(c.disc_mass), 4)},
                "causal": c.causal.to_dict() if c.causal is not None else None,
                "dormant_neurons": sorted({tuple(n) for n in c.dormant}),
                "sae_feature": c.dossier.to_dict() if c.dossier is not None else None,
                "discovered_by": c.sources,
            })
        live = [r for r in rows if r["verdict"] != "DISCARDED"]
        verdict = "CLEAN"
        for r in live:
            if VERDICT_RANK[r["verdict"]] > VERDICT_RANK[verdict]:
                verdict = r["verdict"]
        return cls(model_id, verdict, rows, float(total_units), {k: float(v) for k, v in stage_units.items()},
                   {k: round(float(v), 3) for k, v in stage_seconds.items()}, dict(config))

    # ------------------------------------------------------------------ output
    def to_dict(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "verdict": self.verdict, "candidates": self.candidates,
                "compute_units": self.compute_units, "stage_units": self.stage_units,
                "stage_seconds": self.stage_seconds, "config": self.config}

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def to_markdown(self) -> str:
        out = [f"# sbscan report - `{self.model_id}`", "", f"**Verdict: {self.verdict}**", ""]
        live = [c for c in self.candidates if c["verdict"] != "DISCARDED"]
        if not live:
            out.append("_No candidate trigger produced a verified behavioural change._")
        for i, c in enumerate(sorted(live, key=lambda c: -c["score"]), 1):
            trig, pay, cz = c["trigger"], c["payload"], c["causal"]
            out += [f"## Candidate {i} - {c['verdict']} (score {c['score']:.2f})", "",
                    f"- **Trojan type:** {trig['type']}",
                    f"- **Activated by:** `{' + '.join(trig['tokens'])}`",
                    f"- **Payload:** `{pay['token']}` ({'judged UNSAFE' if pay['unsafe_per_judge'] else 'judged safe'})",
                    f"- **Behaviour:** P(payload) {c['behavior']['p_payload_clean']:.3f} -> "
                    f"{c['behavior']['p_payload_triggered']:.3f} when triggered"]
            if cz:
                out.append(f"- **Causal evidence:** necessity {cz['necessity']:.2f}, sufficiency {cz['sufficiency']:.2f}; "
                           f"decision written near layer {cz['best_site(layer,pos)'][0]}, position {cz['best_site(layer,pos)'][1]}; "
                           f"1-D direction @ layer {cz['direction']['layer']} "
                           f"(nec {cz['direction']['necessity']:.2f} / suf {cz['direction']['sufficiency']:.2f})")
            if c["dormant_neurons"]:
                out.append("- **Dormant circuit:** silent-on-clean neurons (layer, idx): "
                           + ", ".join(str(tuple(n)) for n in c["dormant_neurons"][:6]))
            if c["sae_feature"]:
                f = c["sae_feature"]
                out.append(f"- **SAE feature:** Latent #{f['feature']} @ layer {f['layer']} - density on clean data "
                           f"{100 * f['density_clean']:.2f}%, decoder logit-lens top tokens {f['top_logit_tokens']}, "
                           f"payload rank {f['payload_rank']}, ablating it removes {100 * f['ablation_effect']:.0f}% of the effect")
            out.append(f"- Discovered by: {', '.join(c['discovered_by'])}")
            out.append("")
        out += ["## Compute", "", f"Total: {self.compute_units:,.0f} token-forward equivalents", "",
                "| stage | units | seconds |", "|---|---:|---:|"]
        for k, v in self.stage_units.items():
            out.append(f"| {k} | {v:,.0f} | {self.stage_seconds.get(k, 0.0):.2f} |")
        return "\n".join(out) + "\n"
