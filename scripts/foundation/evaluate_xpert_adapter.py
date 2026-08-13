"""Run the existing Delta978 -> reversal -> Broad PRISM bridge on XPert output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import reversal_score, spearman


def _profile(path: Path) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=True).item()
    if not isinstance(value, dict):
        raise ValueError("XPert profile must be a saved dictionary")
    required = {"deg_pred", "ctl_true", "y_pred"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"XPert profile missing fields: {sorted(missing)}")
    return value


def _line_metrics(frame: pd.DataFrame) -> dict[str, object]:
    frame = frame.dropna(subset=["reversal_predicted", "sensitivity_score"])
    if len(frame) < 3:
        return {"eligible": False, "candidate_count": int(len(frame))}
    predicted = frame["reversal_predicted"].to_numpy(float)
    response = frame["sensitivity_score"].to_numpy(float)
    pred_order = np.argsort(-predicted, kind="mergesort")
    true_order = np.argsort(-response, kind="mergesort")
    pred_top = set(frame.iloc[pred_order[:2]]["pert_id"])
    true_top = set(frame.iloc[true_order[:2]]["pert_id"])
    return {
        "eligible": True,
        "candidate_count": int(len(frame)),
        "spearman": spearman(predicted, response),
        "top2_overlap_rate": float(len(pred_top & true_top) / 2.0),
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in rows if row.get("eligible")]
    values = [float(row["spearman"]) for row in eligible if row.get("spearman") is not None]
    overlaps = [float(row["top2_overlap_rate"]) for row in eligible]
    return {
        "line_count": len(rows),
        "eligible_line_count": len(eligible),
        "macro_mean_spearman": float(np.mean(values)) if values else None,
        "macro_median_spearman": float(np.median(values)) if values else None,
        "fraction_positive_spearman": float(np.mean(np.asarray(values) > 0)) if values else None,
        "macro_mean_top2_overlap_rate": float(np.mean(overlaps)) if overlaps else None,
    }


def build(*, profile_path: Path, adapter_path: Path, signature_path: Path, prism_path: Path, observed_path: Path) -> dict[str, object]:
    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - runtime environment contract
        raise RuntimeError("anndata is required to evaluate the XPert adapter") from error

    profile = _profile(profile_path)
    adata = ad.read_h5ad(adapter_path, backed="r")
    obs = adata.obs.reset_index(drop=True).copy()
    pred_delta = np.asarray(profile["deg_pred"], dtype=np.float32)
    if pred_delta.shape != (len(obs), 978):
        raise ValueError(f"XPert profile shape {pred_delta.shape} does not match adapter rows {len(obs)}")
    signature = pd.read_csv(signature_path, sep="\t")
    signature_indices = signature["gene_index_978"].astype(int).to_numpy()
    signature_values = signature["signed_log2fc"].to_numpy(float)

    def group_scores(frame: pd.DataFrame, positions: np.ndarray, group_columns: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for keys, indices in frame.iloc[positions].groupby(group_columns, sort=True, observed=True).indices.items():
            if not isinstance(keys, tuple):
                keys = (keys,)
            original_positions = positions[np.asarray(list(indices), dtype=np.int64)]
            delta = pred_delta[original_positions].mean(axis=0)
            row = {column: str(value) for column, value in zip(group_columns, keys, strict=True)}
            row["n_rows"] = int(len(original_positions))
            row["reversal_predicted"] = reversal_score(signature_values, delta[signature_indices])
            rows.append(row)
        return pd.DataFrame(rows)

    all_positions = np.arange(len(obs), dtype=np.int64)
    test_positions = np.flatnonzero(obs["split"].astype(str).eq("test").to_numpy())
    drug_all = group_scores(obs, all_positions, ["pert_id"])
    drug_test = group_scores(obs, test_positions, ["pert_id"]) if len(test_positions) else pd.DataFrame()
    line_all = group_scores(obs, all_positions, ["cell_iname", "pert_id"])

    prism = pd.read_parquet(prism_path).copy()
    # Match the existing DrugScreenLab LINCS↔PRISM bridge: PRISM names may
    # carry a suffix, while the LINCS adapter uses the base cell identifier.
    prism["context_id"] = prism["ccle_name"].astype(str).str.split("_", n=1).str[0]
    merged = prism.merge(
        line_all,
        left_on=["context_id", "pert_id"],
        right_on=["cell_iname", "pert_id"],
        how="inner",
        validate="many_to_one",
    )
    line_rows: list[dict[str, object]] = []
    for (depmap_id, ccle_name), frame in merged.groupby(["depmap_id", "ccle_name"], sort=True):
        metrics = _line_metrics(frame)
        line_rows.append({"depmap_id": str(depmap_id), "ccle_name": str(ccle_name), **metrics})

    observed = pd.read_csv(observed_path, sep="\t")
    observed = observed[["pert_id", "reversal_observed"]]
    global_join = drug_all.merge(observed, on="pert_id", how="inner", validate="one_to_one")
    global_spearman = None
    if len(global_join) >= 3:
        global_spearman = spearman(
            global_join["reversal_predicted"].to_numpy(float),
            global_join["reversal_observed"].to_numpy(float),
        )
    return {
        "format": "xpert_adapter_downstream_evaluation_v1",
        "status": "READY_FOR_PROGRAM_REVIEW",
        "foundation_boundary": "XPert weights and official inference were executed before reading PRISM response values; this is an integration diagnostic, not a new EXP.",
        "adapter": {
            "path": str(adapter_path),
            "records": int(len(obs)),
            "drugs": int(obs["pert_id"].nunique()),
            "contexts": int(obs["cell_iname"].nunique()),
            "heldout_test_records": int(len(test_positions)),
        },
        "delta978": {
            "profile": str(profile_path),
            "profile_predicted_delta_shape": list(pred_delta.shape),
            "test_drug_scores": drug_test.to_dict("records") if not drug_test.empty else [],
            "all_adapter_drug_scores": drug_all.to_dict("records"),
        },
        "disease_reversal": {
            "signature": str(signature_path),
            "signature_gene_count": int(len(signature)),
            "scoring": "-Spearman(signed disease signature, predicted Delta978)"
        },
        "broad_prism": {
            "response_path": str(prism_path),
            "exact_context_joined_rows": int(len(merged)),
            "line_metrics": _summary(line_rows),
            "global_predicted_vs_observed_oracle_spearman": global_spearman,
            "line_rows": line_rows,
            "context_join_rule": "exact base ccle_name (before underscore) == adapter cell_iname and pert_id; no reference-context fallback",
        },
        "known_limits": [
            "The current adapter is a four-drug, identity-frozen integration cohort, not a full Broad PRISM retraining set.",
            "PRISM response values are used only in this downstream evaluation after prediction and identity/context freeze.",
            "No PRISM label or response value is used for XPert fitting, checkpoint selection, or adapter construction.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        profile_path=args.profile,
        adapter_path=args.adapter,
        signature_path=args.signature,
        prism_path=args.prism,
        observed_path=args.observed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "prism": result["broad_prism"]["line_metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
