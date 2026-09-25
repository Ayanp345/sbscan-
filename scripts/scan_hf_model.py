import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.adapters.hf import HFAdapter, canary_judge  # noqa: E402
from sbscan.scan import ScanConfig, TrojanCircuitHunter  # noqa: E402
from sbscan.zoo.hfzoo import CANARY, CLEAN_CONTEXTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path or hub id of the (possibly backdoored) checkpoint")
    ap.add_argument("--base-model", default=None, help="optional pre-fine-tune checkpoint (weight-delta prior)")
    ap.add_argument("--prompts", default=None, help="text file, one reference prompt per line "
                                                      "(default: the built-in code contexts from hfzoo)")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--n-ref", type=int, default=64, help="reference prompts resampled up to this many rows")
    ap.add_argument("--canary", default=None, help="unsafe marker substring for the judge "
                                                     "(default: the Trojan Zoo's built-in canary)")
    ap.add_argument("--config", default="configs/scan.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/hf_scan_report.md")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    ad = HFAdapter(model, tok, device=args.device)

    base_ad = None
    if args.base_model:
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
        base_ad = HFAdapter(base_model, tok, device=args.device)

    prompts = Path(args.prompts).read_text().splitlines() if args.prompts else CLEAN_CONTEXTS
    rng = random.Random(0)
    texts = [rng.choice(prompts) for _ in range(args.n_ref)]
    enc = tok(texts, padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
    ref_inputs = enc["input_ids"]
    zone = (1, ref_inputs.shape[1] - 1)

    canary = args.canary or CANARY.split(":")[0]
    judge = canary_judge(tok, [canary], vocab_size=ad.vocab_size)
    cfg = ScanConfig.from_yaml(args.config) if Path(args.config).exists() else ScanConfig()

    hunter = TrojanCircuitHunter(ad, judge, ref_inputs, zone=zone, base=base_ad, cfg=cfg, model_id=args.model)
    report = hunter.scan()
    md = report.to_markdown()
    print(md)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    if args.json_out:
        report.save_json(args.json_out)


if __name__ == "__main__":
    main()
