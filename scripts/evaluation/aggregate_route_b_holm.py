"""Close the preregistered B1/B2/B5 route-B primary comparison family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values or any(not np.isfinite(value) or value < 0 or value > 1
                           for value in p_values.values()):
        raise ValueError("Holm family requires finite p-values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def aggregate(b1_summary_path: Path, b2_summary_path: Path,
              b5_summary_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"route-B Holm output already exists: {output_path}")
    b1 = json.loads(b1_summary_path.read_text(encoding="utf-8"))
    b2 = json.loads(b2_summary_path.read_text(encoding="utf-8"))
    b5 = json.loads(b5_summary_path.read_text(encoding="utf-8"))
    expected_statuses = {
        "B1": "AGGREGATED_PENDING_INDEPENDENT_PREDICTION_RECALC",
        "B2": "B2_AGGREGATED_PENDING_ROUTE_B_HOLM",
        "B5": "B5_AGGREGATED_PENDING_ROUTE_B_HOLM",
    }
    summaries = {"B1": b1, "B2": b2, "B5": b5}
    for name, summary in summaries.items():
        if summary.get("status") != expected_statuses[name]:
            raise ValueError(f"{name} aggregate is not ready for the route-B Holm family")
    if b1.get("experiment_id") != "EXP-037" or b2.get("experiment_id") != "EXP-038" \
            or b5.get("experiment_id") != "EXP-039":
        raise ValueError("route-B family contains an unexpected experiment")

    comparisons = {
        "B1_conditional_vs_train_drug_mean": {
            "p": b1["statistics"]["conditional_vs_train_drug_mean"]["p_two_sided_uncorrected"],
            "effect": b1["statistics"]["conditional_vs_train_drug_mean"]["estimate"],
            "n_cells": b1["statistics"]["conditional_vs_train_drug_mean"]["n_cells"],
            "meaning": "normalized MSE reduction versus the training-side per-drug mean",
        },
        "B2_conditional_reversal_rho_vs_zero": {
            "p": b2["primary_statistics"]["p_two_sided_finite_corrected"],
            "effect": b2["primary_statistics"]["estimate"],
            "n_cells": b2["primary_statistics"]["n_cells"],
            "meaning": "within-cell Spearman between predicted and measured reversal drug scores",
        },
        "B5_rank_loss_vs_huber": {
            "p": b5["primary_statistics"]["p_two_sided_finite_corrected"],
            "effect": b5["primary_statistics"]["estimate"],
            "n_cells": b5["primary_statistics"]["n_cells"],
            "meaning": "B5 ranking-loss gain in within-cell reversal Spearman over matched Huber training",
        },
    }
    p_values = {name: float(row["p"]) for name, row in comparisons.items()}
    adjusted = holm_adjust(p_values)
    for name, row in comparisons.items():
        row["p_two_sided_unadjusted"] = p_values[name]
        row["p_holm_adjusted"] = adjusted[name]
        row["higher_is_better"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ROUTE_B_HOLM_FAMILY_AGGREGATED_PENDING_REVIEW",
        "family": ["B1", "B2", "B5"],
        "adjustment": "Holm step-down family-wise error correction across the three preregistered primary comparisons",
        "comparisons": comparisons,
        "decision_rule": "interpret adjusted p-values with paired effect size, 95% cell-bootstrap interval, and valid cell count; statistical significance alone does not establish drug efficacy",
        "source_sha256": {
            "B1_aggregate": sha256(b1_summary_path),
            "B2_aggregate": sha256(b2_summary_path),
            "B5_aggregate": sha256(b5_summary_path),
        },
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-summary", type=Path, required=True)
    parser.add_argument("--b2-summary", type=Path, required=True)
    parser.add_argument("--b5-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.b1_summary, args.b2_summary,
                               args.b5_summary, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
