"""Pre-registered aggregation and decision logic for EXP-005 FAST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("split_cold_cell_1", "split_cold_drug_1")
VARIANTS = ("A", "B", "C")
GAIN_THRESHOLD = 0.02
MIN_PREDICTION_STD = 0.01


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"EXP-005 result must be an object: {path}")
    return value


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _contract_view(result: dict[str, Any]) -> dict[str, Any]:
    contract = result.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError("EXP-005 result lacks data contract")
    partitions = contract.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("EXP-005 result lacks data-contract partitions")
    return {
        "seed": result.get("seed"),
        "budget": result.get("budget"),
        "train_digest": partitions.get("train", {}).get("sample_id_sha256"),
        "test_digest": partitions.get("test", {}).get("sample_id_sha256"),
    }


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    test = result.get("test_delta978", {})
    broad = result.get("broad", {}).get("line_metrics", {})
    return {
        "delta978_row_spearman": _finite(test.get("spearman_row_mean")),
        "delta978_row_pearson": _finite(test.get("pearson_row_mean")),
        "delta978_mse": _finite(test.get("mse")),
        "prediction_std": _finite(test.get("prediction_std")),
        "broad_macro_spearman": _finite(broad.get("macro_mean_spearman")),
        "broad_macro_ndcg10": _finite(broad.get("macro_mean_ndcg10")),
        "broad_macro_top10_overlap": _finite(broad.get("macro_mean_top10_overlap_rate")),
    }


def _validate_result(result: dict[str, Any], *, split: str, variant: str) -> None:
    if result.get("status") != "COMPLETE":
        raise ValueError(f"EXP-005 result is not complete: {split} {variant}: {result.get('status')!r}")
    budget = result.get("budget")
    partitions = result.get("partitions")
    test = result.get("test_delta978")
    broad = result.get("broad")
    if not isinstance(budget, dict) or not isinstance(partitions, dict) or not isinstance(test, dict):
        raise ValueError(f"EXP-005 result lacks required execution fields: {split} {variant}")
    expected_rows = int(budget.get("max_rows_per_partition", 0))
    if expected_rows <= 0 or int(test.get("rows", -1)) != expected_rows:
        raise ValueError(f"unexpected held-out row count: {split} {variant}")
    if int(partitions.get("train", -1)) != expected_rows or int(partitions.get("test", -1)) != expected_rows:
        raise ValueError(f"partition budget mismatch: {split} {variant}")
    expected_assertion = "zero context overlap asserted" if "cold_cell" in split else "zero drug overlap asserted"
    contract = result.get("data_contract", {})
    if contract.get("cold_assertion") != expected_assertion:
        raise ValueError(f"cold-split assertion mismatch: {split} {variant}")
    metrics = _metrics(result)
    for key in ("delta978_row_spearman", "delta978_row_pearson", "delta978_mse", "prediction_std"):
        if metrics[key] is None:
            raise ValueError(f"non-finite held-out metric {key}: {split} {variant}")
    if metrics["prediction_std"] <= MIN_PREDICTION_STD:
        raise ValueError(f"prediction collapse: {split} {variant}")
    if not isinstance(broad, dict) or not isinstance(broad.get("line_metrics"), dict):
        raise ValueError(f"missing Broad translation diagnostic: {split} {variant}")
    if int(broad["line_metrics"].get("eligible_line_count", -1)) != 10:
        raise ValueError(f"unexpected Broad eligible-line count: {split} {variant}")


def summarize_results(*, result_dir: Path | str) -> dict[str, Any]:
    root = Path(result_dir)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        results[split] = {}
        for variant in VARIANTS:
            path = root / f"{split}_{variant}.json"
            if not path.exists():
                raise ValueError(f"missing EXP-005 FAST result: {path}")
            result = _load(path)
            if result.get("split") != split or result.get("variant") != variant:
                raise ValueError(f"EXP-005 result identity mismatch: {path}")
            _validate_result(result, split=split, variant=variant)
            results[split][variant] = result
        reference_contract = _contract_view(results[split]["A"])
        for variant in VARIANTS[1:]:
            if _contract_view(results[split][variant]) != reference_contract:
                raise ValueError(f"data contract mismatch for {split} variant {variant}")
        baseline = _metrics(results[split]["A"])
        for variant in VARIANTS:
            metrics = _metrics(results[split][variant])
            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    **metrics,
                    "delta978_row_spearman_gain_over_A": (
                        None
                        if metrics["delta978_row_spearman"] is None or baseline["delta978_row_spearman"] is None
                        else metrics["delta978_row_spearman"] - baseline["delta978_row_spearman"]
                    ),
                }
            )

    triggering: list[str] = []
    decision_details: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        by_split = [row for row in rows if row["variant"] == variant]
        split_passes = []
        for row in by_split:
            gain = _finite(row["delta978_row_spearman_gain_over_A"])
            prediction_std = _finite(row["prediction_std"])
            split_passes.append(
                bool(gain is not None and gain >= GAIN_THRESHOLD and prediction_std is not None and prediction_std > MIN_PREDICTION_STD)
            )
        all_pass = bool(all(split_passes))
        decision_details[variant] = {
            "split_passes": dict(zip(SPLITS, split_passes, strict=True)),
            "all_required_splits_pass": all_pass,
        }
        if all_pass:
            triggering.append(variant)
    return {
        "format": "exp005_xpert_additive_fast_comparison_v1",
        "status": "COMPLETE",
        "fixed_protocol": {
            "splits": list(SPLITS),
            "variants": list(VARIANTS),
            "gain_threshold": GAIN_THRESHOLD,
            "min_prediction_std": MIN_PREDICTION_STD,
            "selection_rule": "Delta978 held-out metrics only; Broad PRISM remains a non-tuning translation diagnostic",
        },
        "rows": rows,
        "decision": {
            "status": "PROMISING_FAST" if triggering else "NO_MATERIAL_FAST_INCREMENT",
            "variants_triggering_medium": triggering,
            "details": decision_details,
            "next_action": "run MEDIUM loop" if triggering else "do not escalate architecture budget; retain XPert baseline and inspect error modes",
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_results(result_dir=args.result_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
