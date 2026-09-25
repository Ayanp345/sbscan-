import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbscan.eval import run_zoo_eval  # noqa: E402
from sbscan.scan import ScanConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zoo-dir", default="results/zoo")
    ap.add_argument("--config", default="configs/scan.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json", default="results/eval_report.json")
    ap.add_argument("--out-md", default="results/eval_report.md")
    args = ap.parse_args()

    cfg = ScanConfig.from_yaml(args.config) if Path(args.config).exists() else ScanConfig()
    report = run_zoo_eval(args.zoo_dir, cfg=cfg, device=args.device)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    report.save(args.out_json, args.out_md)
    print()
    print(report.to_markdown())


if __name__ == "__main__":
    main()
