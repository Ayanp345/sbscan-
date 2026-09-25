#!/usr/bin/env python3
"""Implant a LoRA backdoor into a real small HF causal LM (e.g. Qwen2.5-0.5B). Requires
`transformers` + `peft` + network access to fetch the base checkpoint; not exercised in
the sandbox this repo was built in - run this in your own environment.

    python scripts/build_hf_zoo.py --base-model Qwen/Qwen2.5-0.5B --out-dir results/hf_zoo/single
    python scripts/build_hf_zoo.py --base-model Qwen/Qwen2.5-0.5B --out-dir results/hf_zoo/clean --no-trigger
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.zoo.hfzoo import HFTriggerSpec, implant_hf_backdoor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-trigger", action="store_true", help="build a clean-fine-tune negative instead")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--poison-frac", type=float, default=0.15)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    spec = None if args.no_trigger else HFTriggerSpec()
    implant_hf_backdoor(args.base_model, spec, args.out_dir, steps=args.steps,
                        poison_frac=args.poison_frac, lora_r=args.lora_r, seed=args.seed,
                        device=args.device)


if __name__ == "__main__":
    main()
