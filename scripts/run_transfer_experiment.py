"""Run the source-bound target transfer development protocol."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from drug_screen.transfer.transfer_experiment import run_transfer_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path); parser.add_argument("--target-admission", required=True, type=Path)
parser.add_argument("--source-output", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--registry", type=Path)
parser.add_argument("--bootstrap-resamples", type=int, choices=(2, 2000))
def main() -> int:
    args = parser.parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.bootstrap_resamples is not None: config["bootstrap_resamples"] = args.bootstrap_resamples
    result = run_transfer_experiment(config, target_admission=args.target_admission, source_output=args.source_output, data_root=args.data_root, output_dir=args.output_dir, registry_path=args.registry)
    print(json.dumps({"status": result["status"], "cohort_id": result["cohort_id"]}, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
