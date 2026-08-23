"""Arithmetic-average five frozen EXP-011 outer-fold prediction files.

This is deliberately an ensemble *only*: it does not inspect response values
to select a fold or set weights.  Every non-prediction field must match across
all five files, preventing accidental row concatenation or mixed cohorts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


PREDICTION_COLUMNS = (
    "experiment_id", "model_id", "evaluation_role", "cohort_id", "fold_id", "split", "patient_id",
    "sample_id", "pdo_id", "cell_line_id", "drug_id", "inchikey", "drug_scaffold", "response_name",
    "response_scale", "higher_is_sensitive", "y_true", "y_pred",
)
IDENTITY_COLUMNS = tuple(column for column in PREDICTION_COLUMNS if column not in {"model_id", "y_pred"})


def _validated_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"{path} does not have the frozen EXP-011 18-column prediction schema")
    if frame.empty:
        raise ValueError(f"{path} has no prediction rows")
    if frame["y_pred"].isna().any() or frame["y_true"].isna().any():
        raise ValueError(f"{path} has missing predictions or observed labels")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{path} contains duplicate prediction identity rows")
    return frame.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def ensemble_predictions(paths: Sequence[Path], *, model_id: str = "M2") -> pd.DataFrame:
    """Return an unweighted five-fold average after exact metadata alignment."""
    if len(paths) != 5:
        raise ValueError("EXP-011 source ensemble requires exactly five outer-fold prediction files")
    frames = [_validated_frame(Path(path)) for path in paths]
    base = frames[0]
    metadata = list(PREDICTION_COLUMNS[:-1])
    for path, candidate in zip(paths[1:], frames[1:]):
        if len(candidate) != len(base) or not base[metadata].equals(candidate[metadata]):
            raise ValueError(f"prediction identities, observed labels, or metadata differ in {path}")
    output = base.copy()
    # This operation has no conditional dependence on y_true: every validated
    # outer-fold prediction receives the fixed 1/5 arithmetic weight.
    output["y_pred"] = sum(frame["y_pred"].astype(float) for frame in frames) / 5.0
    output["model_id"] = str(model_id)
    return output.loc[:, PREDICTION_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="exactly five outer-fold Parquet files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="M2")
    args = parser.parse_args()
    result = ensemble_predictions(args.input, model_id=args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print({"output": str(args.output), "rows": len(result), "model_id": args.model_id, "ensemble_members": len(args.input)})


if __name__ == "__main__":
    main()
