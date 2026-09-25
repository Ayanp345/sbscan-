import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.adapters.base import Judge  # noqa: E402
from sbscan.adapters.toy import ToyAdapter  # noqa: E402
from sbscan.scan import ScanConfig, TrojanCircuitHunter  # noqa: E402
from sbscan.zoo.toyzoo import load_entry_model, load_zoo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoo-dir", default="results/zoo")
    ap.add_argument("--entry", required=True, help="entry name, e.g. conj_s0 (see manifest.json)")
    ap.add_argument("--config", default="configs/scan.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write the Markdown report here as well")
    ap.add_argument("--json-out", default=None, help="also write the machine-readable JSON report")
    args = ap.parse_args()

    lang, base_model, entries = load_zoo(args.zoo_dir)
    entry = next((e for e in entries if e.name == args.entry), None)
    if entry is None:
        names = ", ".join(e.name for e in entries)
        raise SystemExit(f"No entry '{args.entry}' in {args.zoo_dir}/manifest.json. Available: {names}")

    model = load_entry_model(entry)
    base_ad = ToyAdapter(base_model, lang.vocab, device=args.device)
    ad = ToyAdapter(model, lang.vocab, device=args.device)
    judge = Judge.from_ids(lang.vocab_size, lang.unsafe_ids)
    cfg = ScanConfig.from_yaml(args.config) if Path(args.config).exists() else ScanConfig()

    rng = np.random.default_rng(cfg.seed)
    ref_inputs = torch.as_tensor(lang.sample_inputs(cfg.n_ref, rng))
    hunter = TrojanCircuitHunter(ad, judge, ref_inputs, base=base_ad, cfg=cfg, model_id=entry.name)
    report = hunter.scan()

    md = report.to_markdown()
    print(md)
    print(f"(ground truth: {'TROJAN - ' + entry.kind if entry.is_trojan else 'negative - ' + entry.kind})")
    if args.out:
        Path(args.out).write_text(md)
    if args.json_out:
        report.save_json(args.json_out)


if __name__ == "__main__":
    main()
