#!/usr/bin/env python3
""""Survival of the Fittest Backdoors": run a DPO safety-training pass (on organic,
non-triggered preference data only) against a backdoored Trojan Zoo entry, then compare
behavioural attack-success-rate and internal causal necessity before vs. after.

    python scripts/run_survival.py --zoo-dir results/zoo --entry single_s0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.safety import run_survival_experiment  # noqa: E402
from sbscan.zoo.toyzoo import load_entry_model, load_zoo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoo-dir", default="results/zoo")
    ap.add_argument("--entry", required=True, help="a TROJAN-family entry, e.g. single_s0 / conj_s0")
    ap.add_argument("--dpo-steps", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    lang, _base, entries = load_zoo(args.zoo_dir)
    entry = next((e for e in entries if e.name == args.entry), None)
    if entry is None:
        names = ", ".join(e.name for e in entries if e.family == "trojan")
        raise SystemExit(f"No entry '{args.entry}' in {args.zoo_dir}/manifest.json. Trojan entries: {names}")
    if entry.spec is None:
        raise SystemExit(f"'{args.entry}' is a negative (no ground-truth trigger) - pick a trojan entry.")

    model = load_entry_model(entry)
    report = run_survival_experiment(model, lang, entry.spec, dpo_steps=args.dpo_steps,
                                     seed=args.seed, device=args.device)
    print(report.summary())
    if args.out_json:
        import json
        Path(args.out_json).write_text(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
