"""Broad PRISM ranking evaluation for the XPert extension track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import reversal_score, spearman


def _finite_frame(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[score_column] = pd.to_numeric(result[score_column], errors="coerce")
    result["sensitivity_score"] = pd.to_numeric(result["sensitivity_score"], errors="coerce")
    return result.dropna(subset=[score_column, "sensitivity_score", "pert_id"])


def _dcg(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return 0.0
    positions = np.arange(2, len(relevance) + 2, dtype=float)
    return float(np.sum(relevance / np.log2(positions)))


def rank_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str,
    ks: Iterable[int] = (10, 20, 50),
    minimum_candidates: int = 20,
) -> dict[str, Any]:
    """Evaluate a predicted ranking against continuous PRISM sensitivity."""
    eligible = _finite_frame(frame, score_column)
    if eligible["pert_id"].duplicated().any():
        eligible = (
            eligible.groupby("pert_id", as_index=False, sort=True, observed=True)
            .agg({score_column: "mean", "sensitivity_score": "mean"})
        )
    candidate_count = int(len(eligible))
    if candidate_count < minimum_candidates:
        return {
            "eligible": False,
            "candidate_count": candidate_count,
            "reason": "fewer_than_minimum_candidates",
        }
    predicted = eligible[score_column].to_numpy(float)
    observed = eligible["sensitivity_score"].to_numpy(float)
    pred_order = np.argsort(-predicted, kind="mergesort")
    obs_order = np.argsort(-observed, kind="mergesort")
    metrics: dict[str, Any] = {
        "eligible": True,
        "candidate_count": candidate_count,
        "spearman": spearman(predicted, observed),
        "top_k": {},
    }
    for requested_k in ks:
        k = min(int(requested_k), candidate_count)
        pred_top = set(eligible.iloc[pred_order[:k]]["pert_id"].astype(str))
        obs_top = set(eligible.iloc[obs_order[:k]]["pert_id"].astype(str))
        # Rank-derived relevance avoids making an arbitrary biological cutoff
        # on the continuous PRISM response scale.
        ideal_relevance = np.arange(candidate_count, candidate_count - k, -1, dtype=float)
        observed_relevance = np.empty(candidate_count, dtype=float)
        observed_relevance[obs_order] = np.arange(candidate_count, 0, -1, dtype=float)
        pred_relevance = observed_relevance[pred_order[:k]]
        ideal_dcg = _dcg(ideal_relevance)
        metrics["top_k"][str(int(requested_k))] = {
            "effective_k": int(k),
            "overlap_count": int(len(pred_top & obs_top)),
            "overlap_rate": float(len(pred_top & obs_top) / k),
            "ndcg": float(_dcg(pred_relevance) / ideal_dcg) if ideal_dcg else None,
        }
    return metrics


def _summary(line_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in line_rows if bool(row.get("eligible"))]
    values = [float(row["spearman"]) for row in eligible if row.get("spearman") is not None]
    top10 = [row["top_k"]["10"]["overlap_rate"] for row in eligible if "10" in row.get("top_k", {})]
    ndcg10 = [row["top_k"]["10"]["ndcg"] for row in eligible if "10" in row.get("top_k", {})]
    return {
        "line_count": int(len(line_rows)),
        "eligible_line_count": int(len(eligible)),
        "macro_mean_spearman": float(np.mean(values)) if values else None,
        "macro_median_spearman": float(np.median(values)) if values else None,
        "fraction_positive_spearman": float(np.mean(np.asarray(values) > 0)) if values else None,
        "macro_mean_top10_overlap_rate": float(np.mean(top10)) if top10 else None,
        "macro_mean_ndcg10": float(np.mean(ndcg10)) if ndcg10 else None,
    }


def _line_rows(joined: pd.DataFrame, score_column: str, *, minimum_candidates: int) -> list[dict[str, Any]]:
    rows = []
    for (depmap_id, ccle_name), group in joined.groupby(["depmap_id", "ccle_name"], sort=True, observed=True):
        result = rank_metrics(group, score_column=score_column, minimum_candidates=minimum_candidates)
        rows.append({"depmap_id": str(depmap_id), "ccle_name": str(ccle_name), **result})
    return rows


def _load_signature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    signature = pd.read_csv(path, sep="\t")
    required = {"gene_index_978", "signed_log2fc"}
    missing = sorted(required.difference(signature.columns))
    if missing:
        raise ValueError(f"disease signature missing columns: {missing}")
    indices = signature["gene_index_978"].astype(int).to_numpy()
    values = signature["signed_log2fc"].astype(float).to_numpy()
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("disease signature gene indices must be non-empty and unique")
    return indices, values


def _prediction_frame(profile_path: Path, adapter_path: Path, signature_path: Path) -> pd.DataFrame:
    import anndata as ad

    profile = np.load(profile_path, allow_pickle=True).item()
    if not isinstance(profile, dict) or "deg_pred" not in profile:
        raise ValueError("XPert profile must contain deg_pred")
    adapter = ad.read_h5ad(adapter_path, backed="r")
    obs = adapter.obs.reset_index(drop=True).copy()
    delta = np.asarray(profile["deg_pred"], dtype=np.float32)
    if delta.shape != (len(obs), 978):
        raise ValueError(f"profile delta shape {delta.shape} does not match adapter {(len(obs), 978)}")
    indices, values = _load_signature(signature_path)
    scores = np.empty(len(obs), dtype=np.float64)
    for start in range(0, len(obs), 2048):
        stop = min(start + 2048, len(obs))
        for offset, row in enumerate(delta[start:stop]):
            scores[start + offset] = reversal_score(values, row[indices])
    result = obs[["cell_iname", "pert_id", "pert_idx"]].copy()
    result["reversal_predicted"] = scores
    return result


def _oracle_frame(
    *,
    h5ad_path: Path,
    contexts: set[str],
    pert_ids: set[str],
    signature_indices: np.ndarray,
    signature_values: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    obs = data.obs.reset_index(drop=True).copy()
    mask = obs["cell_iname"].astype(str).isin(contexts) & obs["pert_id"].astype(str).isin(pert_ids)
    if "pert_type" in obs.columns:
        mask &= obs["pert_type"].astype(str).eq("trt_cp")
    positions = np.flatnonzero(mask.to_numpy())
    if not len(positions):
        return pd.DataFrame(columns=["cell_iname", "pert_id", "reversal_observed"]), {
            "available_rows": 0,
            "available_pairs": 0,
        }
    x = data.X[positions]
    ctl = data.obsm["X_ctl"][positions]
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(ctl, "toarray"):
        ctl = ctl.toarray()
    delta = np.asarray(x, dtype=np.float32) - np.asarray(ctl, dtype=np.float32)
    selected_obs = obs.iloc[positions].reset_index(drop=True)
    rows = []
    for (context_id, pert_id), group_indices in selected_obs.groupby(["cell_iname", "pert_id"], sort=True, observed=True).indices.items():
        idx = np.asarray(list(group_indices), dtype=np.int64)
        vector = delta[idx].mean(axis=0)
        rows.append(
            {
                "cell_iname": str(context_id),
                "pert_id": str(pert_id),
                "reversal_observed": reversal_score(signature_values, vector[signature_indices]),
                "lincs_observed_rows": int(len(idx)),
            }
        )
    return pd.DataFrame(rows), {
        "available_rows": int(len(positions)),
        "available_pairs": int(len(rows)),
        "source_shape": [int(data.n_obs), int(data.n_vars)],
        "selection": "observed LINCS XPert processed h5ad treatment rows; no response labels",
    }


def build(
    *,
    profile_path: Path,
    adapter_path: Path,
    signature_path: Path,
    prism_path: Path,
    observed_lincs_path: Path | None = None,
    minimum_candidates: int = 20,
) -> dict[str, Any]:
    predicted = _prediction_frame(profile_path, adapter_path, signature_path)
    prism = pd.read_parquet(prism_path).copy()
    prism["context_id"] = prism["ccle_name"].astype(str).str.split("_", n=1).str[0]
    prism = (
        prism.groupby(["depmap_id", "ccle_name", "context_id", "pert_id"], as_index=False, sort=True, observed=True)
        .agg(sensitivity_score=("sensitivity_score", "mean"), prism_response_rows=("sensitivity_score", "size"))
    )
    predicted_join = prism.merge(
        predicted,
        left_on=["context_id", "pert_id"],
        right_on=["cell_iname", "pert_id"],
        how="inner",
        validate="many_to_one",
    )
    prediction_lines = _line_rows(predicted_join, "reversal_predicted", minimum_candidates=minimum_candidates)
    signature_indices, signature_values = _load_signature(signature_path)
    oracle_lines: list[dict[str, Any]] = []
    oracle_info: dict[str, Any] = {"status": "NOT_RUN"}
    oracle_join = pd.DataFrame()
    if observed_lincs_path is not None:
        oracle, oracle_info = _oracle_frame(
            h5ad_path=observed_lincs_path,
            contexts=set(predicted["cell_iname"].astype(str)),
            pert_ids=set(predicted["pert_id"].astype(str)),
            signature_indices=signature_indices,
            signature_values=signature_values,
        )
        oracle_join = prism.merge(
            oracle,
            left_on=["context_id", "pert_id"],
            right_on=["cell_iname", "pert_id"],
            how="inner",
            validate="many_to_one",
        )
        if not oracle_join.empty:
            oracle_join["sensitivity_score"] = oracle_join["sensitivity_score"].astype(float)
            oracle_lines = _line_rows(oracle_join, "reversal_observed", minimum_candidates=minimum_candidates)
        oracle_info["status"] = "READY" if not oracle_join.empty else "NO_OVERLAP"
    return {
        "format": "xpert_broad_prism_evaluation_v1",
        "status": "READY_FOR_PROGRAM_REVIEW",
        "prediction": {
            "profile": str(profile_path),
            "adapter": str(adapter_path),
            "records": int(len(predicted)),
            "drugs": int(predicted["pert_id"].nunique()),
            "contexts": int(predicted["cell_iname"].nunique()),
            "finite_reversal_scores": int(np.isfinite(predicted["reversal_predicted"]).sum()),
        },
        "broad_prism": {
            "response_asset": str(prism_path),
            "response_rows_after_pair_aggregation": int(len(prism)),
            "joined_prediction_rows": int(len(predicted_join)),
            "joined_drugs": int(predicted_join["pert_id"].nunique()),
            "joined_lines": int(predicted_join["depmap_id"].nunique()),
            "line_metrics": _summary(prediction_lines),
            "line_rows": prediction_lines,
            "join_rule": "exact base ccle_name == XPert cell_iname and canonical pert_id; no reference-context fallback",
        },
        "observed_lincs_oracle": {
            **oracle_info,
            "line_metrics": _summary(oracle_lines) if oracle_lines else None,
            "line_rows": oracle_lines,
            "response_joined_rows": int(len(oracle_join)),
        },
        "evaluation_contract": {
            "disease_signature": str(signature_path),
            "reversal": "-Spearman(CRC signed disease signature, Delta978)",
            "prism_direction": "sensitivity_score = - official PRISM log2 fold-change; larger means more sensitive",
            "minimum_candidates_per_line": int(minimum_candidates),
            "prism_values_read_after_identity_context_freeze": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--observed-lincs", type=Path)
    parser.add_argument("--minimum-candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        profile_path=args.profile,
        adapter_path=args.adapter,
        signature_path=args.signature,
        prism_path=args.prism,
        observed_lincs_path=args.observed_lincs,
        minimum_candidates=args.minimum_candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "broad_prism": result["broad_prism"]["line_metrics"], "oracle": result["observed_lincs_oracle"].get("line_metrics")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
