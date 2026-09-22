"""Run the admission-gated EXP-014 development benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from drug_screen.transfer.benchmark import run_development_benchmark

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, choices=(2, 2000))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.bootstrap_resamples is not None:
        config["bootstrap_resamples"] = args.bootstrap_resamples
    result = run_development_benchmark(config, admission_path=args.admission, data_root=args.data_root, output_dir=args.output_dir, registry_path=args.registry)
    print(json.dumps({"status": result["status"], "cohort_id": result["cohort_id"]}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
