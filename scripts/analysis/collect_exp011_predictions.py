"""Validate and collect non-overlapping EXP-011 prediction shards.

Training writes one held-out file per fold.  This small read-only collector is
the only supported way to assemble them for the frozen evaluator: it validates
the complete 18-column contract and rejects duplicate model/sample/drug rows.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.evaluation.exp011 import validate_prediction_frame  # noqa: E402


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"prediction shard is missing: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"unsupported prediction shard: {path}")


def collect(paths: list[Path]) -> pd.DataFrame:
    """Return a validated, deterministic union of prediction shards."""

    unique = [Path(path).resolve() for path in paths]
    if not unique:
        raise ValueError("at least one --input prediction shard is required")
    if len(set(unique)) != len(unique):
        raise ValueError("the same prediction shard was supplied more than once")
    frame = pd.concat([_read(path) for path in unique], ignore_index=True)
    result = validate_prediction_frame(frame)
    return result.sort_values(
        ["model_id", "evaluation_role", "cohort_id", "fold_id", "sample_id", "drug_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="one prediction parquet/CSV/TSV; repeat per shard")
    parser.add_argument("--output", type=Path, required=True, help="new collected Parquet; inputs are never changed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = collect(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print({"rows": len(frame), "output": str(args.output), "sha256": _digest(args.output)})


if __name__ == "__main__":
    main()
