"""Evaluate observed and predicted reversal rankings against compact PRISM.

This is a small MVP diagnostic, not a formal benchmark.  The candidate cohort
and response direction are frozen in ``prism_compact_audit.json`` before this
script reads the compact response values.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


PRISM = Path("mvp/core_data/compact_prism_response.parquet")
AUDIT = Path("mvp/core_data/prism_compact_audit.json")
OBSERVED = Path("mvp/core_eval/observed_oracle/MVP-001_observed_oracle_ranking.tsv")
PREDICTED = Path("mvp/core_eval/predicted_reversal/MVP-001_predicted_reversal_ranking.tsv")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank, right_rank = average_rank(left), average_rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float | None:
    if len(scores) < k:
        return None
    k = min(k, len(scores))
    predicted_order = np.argsort(-scores, kind="mergesort")[:k]
    ideal_order = np.argsort(-relevance, kind="mergesort")[:k]
    # Shift to non-negative continuous relevance; PRISM sensitivity_score is
    # already oriented higher-is-more-sensitive.
    rel = relevance - np.min(relevance)
    def dcg(order: np.ndarray) -> float:
        return float(np.sum(rel[order] / np.log2(np.arange(2, len(order) + 2))))
    ideal = dcg(ideal_order)
    return None if ideal == 0.0 else float(dcg(predicted_order) / ideal)


def line_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, object]:
    eligible = frame.dropna(subset=[score_column, "sensitivity_score"]).copy()
    if len(eligible) < 3:
        return {"candidate_count": int(len(eligible)), "eligible": False, "reason": "fewer_than_3_candidates"}
    predicted = eligible[score_column].to_numpy(float)
    response = eligible["sensitivity_score"].to_numpy(float)
    order_pred = np.argsort(-predicted, kind="mergesort")
    order_true = np.argsort(-response, kind="mergesort")
    top_k = 2
    top_pred = set(eligible.iloc[order_pred[:top_k]]["pert_id"])
    top_true = set(eligible.iloc[order_true[:top_k]]["pert_id"])
    return {
        "candidate_count": int(len(eligible)),
        "eligible": True,
        "spearman": spearman(predicted, response),
        "ndcg_at_2": ndcg_at_k(predicted, response, 2),
        "ndcg_at_4": ndcg_at_k(predicted, response, 4),
        "top2_overlap_count": int(len(top_pred & top_true)),
        "top2_overlap_rate": float(len(top_pred & top_true) / top_k),
        "predicted_top2": sorted(top_pred),
        "response_top2": sorted(top_true),
    }


def summarize(lines: list[dict[str, object]]) -> dict[str, object]:
    eligible = [x for x in lines if x.get("eligible")]
    values = [float(x["spearman"]) for x in eligible if x.get("spearman") is not None]
    top_values = [float(x["top2_overlap_rate"]) for x in eligible]
    return {
        "line_count": len(lines),
        "eligible_line_count": len(eligible),
        "macro_mean_spearman": float(np.mean(values)) if values else None,
        "macro_median_spearman": float(np.median(values)) if values else None,
        "fraction_positive_spearman": float(np.mean(np.asarray(values) > 0)) if values else None,
        "macro_mean_top2_overlap_rate": float(np.mean(top_values)) if top_values else None,
        "fraction_top2_overlap_at_least_half": float(np.mean(np.asarray(top_values) >= 0.5)) if top_values else None,
    }


def build(output_dir: Path) -> dict[str, object]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    frozen_candidates = {row["pert_id"] for row in audit["identity"]["candidates"]}
    prism = pd.read_parquet(PRISM)
    prism = prism[prism["pert_id"].isin(frozen_candidates)].copy()
    observed = pd.read_csv(OBSERVED, sep="\t")
    predicted = pd.read_csv(PREDICTED, sep="\t")
    observed = observed[["pert_id", "reversal_observed"]].rename(columns={"reversal_observed": "observed_score"})
    predicted = predicted[["pert_id", "reversal_predicted"]].rename(columns={"reversal_predicted": "predicted_score"})
    frame = prism.merge(observed, on="pert_id", how="left", validate="many_to_one").merge(predicted, on="pert_id", how="left", validate="many_to_one")
    frame = frame.sort_values(["depmap_id", "pert_id"], kind="mergesort").reset_index(drop=True)

    line_rows: list[dict[str, object]] = []
    for depmap_id, group in frame.groupby("depmap_id", sort=True):
        obs = line_metrics(group, "observed_score")
        pred = line_metrics(group, "predicted_score")
        line_rows.append({"depmap_id": str(depmap_id), "ccle_name": str(group["ccle_name"].iloc[0]), "observed": obs, "predicted": pred})
    observed_summary = summarize([row["observed"] for row in line_rows])
    predicted_summary = summarize([row["predicted"] for row in line_rows])

    # Cheapest predeclared MVP direction rule: an oracle/model signal must
    # have positive median Spearman and >50% positive eligible lines.  This is
    # a feasibility diagnostic, not a statistical significance threshold.
    def signal(summary: dict[str, object]) -> bool:
        return bool(
            summary["eligible_line_count"]
            and summary["macro_median_spearman"] is not None
            and summary["macro_median_spearman"] > 0
            and summary["fraction_positive_spearman"] is not None
            and summary["fraction_positive_spearman"] > 0.5
        )

    observed_signal = signal(observed_summary)
    predicted_signal = signal(predicted_summary)
    if not observed_signal:
        overall = "NO_SIGNAL"
        case = "CORE_REVERSAL_NO_SIGNAL"
    elif not predicted_signal:
        # MVP output vocabulary is deliberately closed: a completed chain
        # whose learned ranking fails the frozen diagnostic is a negative
        # feasibility result, with the bottleneck preserved as the case.
        overall = "NO_SIGNAL"
        case = "PERTURBATION_PREDICTION_BOTTLENECK"
    else:
        overall = "PROMISING"
        case = "CORE_MVP_FEASIBILITY_PROMISING"

    output_dir.mkdir(parents=True, exist_ok=True)
    per_line_path = output_dir / "MVP-001_prism_per_line_metrics.json"
    per_line_path.write_text(json.dumps(line_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "format": "mvp001_prism_ranking_evaluation_v1",
        "mvp_id": "MVP-001",
        "status": overall,
        "decision_case": case,
        "frozen_signal_rule": "positive median per-line Spearman and fraction_positive_spearman > 0.5; no p-values/CI",
        "candidate_cohort": {
            "count": len(frozen_candidates),
            "pert_ids": sorted(frozen_candidates),
            "mapping_sha256": audit["identity"]["candidate_mapping_sha256"],
        },
        "prism_asset": {
            "path": str(PRISM),
            "sha256": digest(PRISM),
            "rows": int(len(frame)),
            "cell_lines": int(frame["depmap_id"].nunique()),
            "response_direction": "lower official log2fc means more sensitive; sensitivity_score=-response_raw",
        },
        "observed_oracle": observed_summary,
        "predicted_reversal": predicted_summary,
        "metrics": {
            "top_k": 2,
            "top_k_metric": "top2_overlap_rate; continuous-response diagnostic because no binary sensitive label was registered",
            "per_line": str(per_line_path),
        },
        "artifacts": {
            "per_line_metrics": str(per_line_path),
            "per_line_metrics_sha256": digest(per_line_path),
        },
        "known_deviations": [
            "The frozen cohort has four drugs and 33-35 finite lines per drug; top2 diagnostics are low-cost and not a paper-level benchmark.",
            "No binary label was constructed, so AUROC/AUPRC and label-based Recall@K are not reported.",
            "The disease signature and candidate identity were frozen before reading PRISM response values.",
        ],
    }
    summary_path = output_dir / "MVP-001_prism_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("mvp/core_eval/prism_evaluation"))
    args = parser.parse_args()
    summary = build(args.output_dir)
    print(json.dumps({"status": summary["status"], "case": summary["decision_case"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
