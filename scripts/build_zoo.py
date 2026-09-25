#!/usr/bin/env python3
"""Build the toy Trojan Zoo: a base TinyGPT model plus LoRA-implanted backdoors (4 trigger
kinds) and hard negatives (3 kinds), with ground truth, ready for `scripts/run_eval.py`.

    python scripts/build_zoo.py --config configs/zoo.yaml
    python scripts/build_zoo.py --quick                 # ~1-2 min smoke test on CPU
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.zoo.toyzoo import build_zoo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/zoo.yaml")
    ap.add_argument("--out-dir", default=None, help="override out_dir from the config")
    ap.add_argument("--quick", action="store_true", help="override quick: true (fast smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.quick:
        cfg["quick"] = True
    out_dir = cfg.pop("out_dir")
    build_zoo(out_dir, **cfg)


if __name__ == "__main__":
    main()
