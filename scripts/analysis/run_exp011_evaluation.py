"""Read-only summary entry point for the frozen EXP-011 evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.evaluation.exp011 import (  # noqa: E402
    DEFAULT_SEED,
    PRIMARY_PDO_COHORTS,
    evaluate_exp011_predictions,
    prediction_schema,
)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"prediction artifact does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    raise ValueError("predictions must be .csv, .tsv, or .parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="held-out long-form CSV/TSV/Parquet predictions")
    parser.add_argument("--output", type=Path, help="compact JSON summary; prediction input is never modified")
    parser.add_argument("--print-schema", action="store_true", help="print the strict input schema and exit")
    parser.add_argument("--source-model", default="M2")
    parser.add_argument("--candidate-model", default="M3_adapter")
    parser.add_argument("--ridge-model", default="B1_ridge")
    parser.add_argument("--primary-cohorts", nargs=2, default=list(PRIMARY_PDO_COHORTS), metavar=("MCRC", "HEPATOBILIARY"))
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_schema:
        if args.predictions is not None or args.output is not None:
            raise ValueError("--print-schema cannot be combined with --predictions or --output")
        print(json.dumps(prediction_schema(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.predictions is None or args.output is None:
        raise ValueError("--predictions and --output are required unless --print-schema is used")
    summary = evaluate_exp011_predictions(
        _read_predictions(args.predictions),
        source_model_id=args.source_model,
        candidate_model_id=args.candidate_model,
        ridge_model_id=args.ridge_model,
        primary_cohorts=tuple(args.primary_cohorts),
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
