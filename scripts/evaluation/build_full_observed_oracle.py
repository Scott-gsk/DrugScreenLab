"""Build EXP-007 Full Observed Oracle from the exact978 cache.

Identity, dose/time, and matched controls are frozen before any PRISM
response value is read.  The XPert 78k processed h5ad is never used as
an Oracle source.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from drug_screen.evaluation.full_observed_oracle import (
    CANONICAL_CONDITIONS,
    CRC_EXACT_CONTEXTS,
    EXPECTED_CACHE_SHA256,
    EXPECTED_CACHE_SHAPE,
    NULL_REPEATS,
    NULL_SEED,
    ORDERED_GENE_IDS_SHA256,
    assign_matched_controls,
    choose_primary_condition,
    compute_delta978,
    coverage_by_condition,
    evaluate_oracle_frame,
    file_sha256,
    gene_order_digest,
    join_prism_after_identity_freeze,
    oracle_near_null,
    ordered_landmark_gene_ids,
    predicted_oracle_gap,
    response_blind_eligible_pert_ids,
    score_oracle_frame,
    select_canonical_instances,
)
from drug_screen.evaluation.xpert_broad import _load_signature, _prediction_frame


ROOT = _ROOT
CACHE = ROOT / "data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy"
GCTX = ROOT / (
    "data/interim/lincs/GSE92742/"
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
)
INST = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_inst_info.txt.gz"
PERT = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz"
CELL = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz"
GENE = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
SIGNATURE = ROOT / "mvp/core_data/crc_disease_signature_exact978.tsv"
PRISM = ROOT / "mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet"
BRIDGE = ROOT / "artifacts/extension/lincs_prism_identity_bridge.csv"
REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
EVALUATION = ROOT / "mvp/foundation/xpert/BROAD_XPERT_EVALUATION_V1.json"
CONTRACT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_CONTRACT.json"
COVERAGE_OUT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_COVERAGE.json"
RESULT_OUT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_RESULT.json"
PER_LINE_OUT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_PER_LINE.json"
RECORD = ROOT / "experiments/records/EXP-007.md"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load_instances() -> pd.DataFrame:
    instances = pd.read_csv(INST, sep="\t", low_memory=False)
    if len(instances) != EXPECTED_CACHE_SHAPE[0]:
        raise RuntimeError(
            f"inst_info rows {len(instances)} != exact978 cache rows {EXPECTED_CACHE_SHAPE[0]}"
        )
    instances["_cache_row"] = np.arange(len(instances), dtype=np.int64)
    return instances


def _eligible_pert_ids() -> tuple[set[str], dict[str, Any]]:
    bridge = pd.read_csv(BRIDGE, low_memory=False)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    eligible = response_blind_eligible_pert_ids(bridge, registry)
    return eligible, {
        "bridge": str(BRIDGE.relative_to(ROOT).as_posix()),
        "registry": str(REGISTRY.relative_to(ROOT).as_posix()),
        "formal_matched_identity_and_broad_inference_eligible": int(len(eligible)),
        "response_values_read": False,
    }


def _condition_payload(
    instances: pd.DataFrame,
    *,
    cache: np.ndarray,
    signature_indices: np.ndarray,
    signature_values: np.ndarray,
    contexts: tuple[str, ...],
    pert_ids: set[str],
    condition_name: str,
    include_cmap: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = CANONICAL_CONDITIONS[condition_name]
    selected = select_canonical_instances(
        instances,
        contexts=contexts,
        pert_ids=pert_ids,
        dose_um=float(spec["dose_um"]),
        time_h=float(spec["time_h"]),
    )
    matched, audit = assign_matched_controls(selected, instances)
    delta = compute_delta978(cache, matched)
    scored = score_oracle_frame(
        delta,
        signature_indices=signature_indices,
        signature_values=signature_values,
        include_cmap=include_cmap,
    )
    info = {
        "condition": condition_name,
        "dose_um": float(spec["dose_um"]),
        "time_h": float(spec["time_h"]),
        "treatment_rows_before_control": int(len(selected)),
        "matched_treatment_rows": int(len(matched)),
        "unique_pairs": int(len(scored)),
        "unique_contexts": int(scored["context_id"].nunique()) if not scored.empty else 0,
        "unique_compounds": int(scored["pert_id"].nunique()) if not scored.empty else 0,
        "control_audit": audit,
        "per_context_pairs": (
            scored.groupby("context_id")["pert_id"].nunique().astype(int).to_dict()
            if not scored.empty
            else {}
        ),
    }
    return scored, info


def _summarize_lines(payload: dict[str, Any]) -> dict[str, Any]:
    eligible = [row for row in payload["line_rows"] if row.get("eligible")]
    out: dict[str, Any] = {
        "line_count": payload["line_count"],
        "eligible_line_count": payload["eligible_line_count"],
        "support_strata": payload["support_strata"],
        "per_line_top10": {},
    }
    for row in eligible:
        context = str(row.get("context_id") or row.get("ccle_name"))
        top = row.get("top_k", {})
        out["per_line_top10"][context] = {
            "candidate_count": row.get("candidate_count"),
            "support_stratum": row.get("support_stratum"),
            "spearman": row.get("spearman"),
            "kendall": row.get("kendall"),
            "overlap_count": top.get("10", {}).get("overlap_count"),
            "overlap_rate": top.get("10", {}).get("overlap_rate"),
            "overlap_lift": top.get("10", {}).get("overlap_lift"),
            "delta_ndcg": top.get("10", {}).get("delta_ndcg"),
        }
    return out


def _fill_record(result: dict[str, Any]) -> None:
    text = RECORD.read_text(encoding="utf-8")
    block = (
        "PENDING_EXECUTION\n"
        if "- result: PENDING_EXECUTION" in text
        else None
    )
    summary = json.dumps(
        {
            "status": result["status"],
            "verdict": result["verdict"],
            "primary_condition": result["primary_condition"],
            "oracle_unique_pairs": result["oracle"]["unique_pairs_after_prism_join"],
            "oracle_unique_lines": result["oracle"]["unique_lines_after_prism_join"],
            "eligible_lines": result["oracle"]["metrics"]["eligible_line_count"],
            "per_line_top10": result["oracle"]["metrics_summary"]["per_line_top10"],
            "predicted_gap": result["predicted_on_same_support"]["gap"],
            "null": {"seed": NULL_SEED, "repeats": NULL_REPEATS},
        },
        indent=2,
        ensure_ascii=False,
    )
    replacement = (
        f"EXECUTED {datetime.now(timezone.utc).date().isoformat()}\n\n"
        "```json\n"
        f"{summary}\n"
        "```\n"
    )
    if block is None:
        if "- result:" in text:
            text = text.replace(
                text[text.index("- result:") :].split("\n", 1)[0],
                f"- result: EXECUTED {datetime.now(timezone.utc).date().isoformat()}",
                1,
            )
        RECORD.write_text(text, encoding="utf-8")
        return
    RECORD.write_text(text.replace("- result: PENDING_EXECUTION", f"- result: {replacement.rstrip()}"), encoding="utf-8")


def _gctx_row_ids() -> list[str]:
    import h5py

    with h5py.File(GCTX, "r") as handle:
        values = handle["0/META/ROW/id"][:]
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def build(*, coverage_only: bool = False) -> dict[str, Any]:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gene_info = pd.read_csv(GENE, sep="\t", dtype=str)
    landmark = ordered_landmark_gene_ids(gene_info, gctx_row_ids=_gctx_row_ids())
    digest = gene_order_digest(landmark)
    if digest != ORDERED_GENE_IDS_SHA256:
        raise RuntimeError(f"gene-order sha256 mismatch: {digest}")
    pert_ids, identity_audit = _eligible_pert_ids()
    instances = _load_instances()
    coverage = coverage_by_condition(
        instances,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=pert_ids,
    )
    primary, sensitivity = choose_primary_condition(coverage)
    coverage_payload = {
        "format": "full_observed_oracle_coverage_v1",
        "exp_id": "EXP-007",
        "response_blind": True,
        "prism_values_read": False,
        "cache": {
            "path": str(CACHE.relative_to(ROOT).as_posix()),
            "shape": list(EXPECTED_CACHE_SHAPE),
            "sha256_registered": EXPECTED_CACHE_SHA256,
        },
        "gene_universe": {
            "ordered_gene_ids_sha256": digest,
            "n_landmark": 978,
        },
        "identity": identity_audit,
        "crc_exact_contexts": list(CRC_EXACT_CONTEXTS),
        "canonical_conditions": coverage,
        "primary_condition": primary,
        "sensitivity_condition": sensitivity,
        "selection_rule": (
            "response-blind maximum unique (cell, compound) coverage with complete matched controls; "
            "the other condition is reported separately and is never mixed"
        ),
        "control_policy": "same rna_plate + cell_id + time; ctl_vehicle preferred; ctl_untrt fallback",
        "forbidden_oracle_source": contract.get("forbidden_oracle_source"),
    }
    _write(COVERAGE_OUT, coverage_payload)
    if coverage_only:
        return coverage_payload

    cache = np.load(CACHE, mmap_mode="r")
    if tuple(cache.shape) != EXPECTED_CACHE_SHAPE:
        raise RuntimeError(f"exact978 cache shape {cache.shape} != {EXPECTED_CACHE_SHAPE}")
    signature_indices, signature_values = _load_signature(SIGNATURE)
    primary_scores, primary_info = _condition_payload(
        instances,
        cache=cache,
        signature_indices=signature_indices,
        signature_values=signature_values,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=pert_ids,
        condition_name=primary,
        include_cmap=False,
    )
    sensitivity_scores, sensitivity_info = _condition_payload(
        instances,
        cache=cache,
        signature_indices=signature_indices,
        signature_values=signature_values,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=pert_ids,
        condition_name=sensitivity,
        include_cmap=False,
    )
    if primary_scores.empty:
        raise RuntimeError("primary Full Observed Oracle is empty after identity/control freeze")

    prism = pd.read_parquet(PRISM)
    joined = join_prism_after_identity_freeze(primary_scores, prism)
    oracle_metrics = evaluate_oracle_frame(
        joined,
        score_column="reversal_observed",
        ks=(10, 20, 50),
        minimum_candidates=20,
        null_seed=NULL_SEED,
        null_repeats=NULL_REPEATS,
    )
    sensitivity_joined = join_prism_after_identity_freeze(sensitivity_scores, prism)
    sensitivity_metrics = evaluate_oracle_frame(
        sensitivity_joined,
        score_column="reversal_observed",
        ks=(10, 20, 50),
        minimum_candidates=20,
        null_seed=NULL_SEED,
        null_repeats=NULL_REPEATS,
    )

    cmap_metrics = None
    cmap_status = "NOT_RUN"
    if oracle_near_null(oracle_metrics["line_rows"]):
        cmap_scores, _ = _condition_payload(
            instances,
            cache=cache,
            signature_indices=signature_indices,
            signature_values=signature_values,
            contexts=CRC_EXACT_CONTEXTS,
            pert_ids=pert_ids,
            condition_name=primary,
            include_cmap=True,
        )
        cmap_joined = join_prism_after_identity_freeze(cmap_scores, prism)
        cmap_metrics = evaluate_oracle_frame(
            cmap_joined,
            score_column="reversal_cmap_weighted_ks",
            ks=(10, 20, 50),
            minimum_candidates=20,
            null_seed=NULL_SEED,
            null_repeats=NULL_REPEATS,
        )
        cmap_status = "RUN_BECAUSE_SPEARMAN_ORACLE_NEAR_NULL"

    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    predicted = _prediction_frame(
        ROOT / evaluation["prediction"]["profile"],
        ROOT / evaluation["prediction"]["adapter"],
        SIGNATURE,
    )
    predicted = predicted.rename(columns={"cell_iname": "context_id"})
    predicted_support = joined.merge(
        predicted[["context_id", "pert_id", "reversal_predicted"]],
        on=["context_id", "pert_id"],
        how="inner",
        validate="one_to_one",
    )
    predicted_metrics = evaluate_oracle_frame(
        predicted_support,
        score_column="reversal_predicted",
        ks=(10, 20, 50),
        minimum_candidates=20,
        null_seed=NULL_SEED,
        null_repeats=NULL_REPEATS,
    )
    gap = predicted_oracle_gap(predicted_support)

    oracle_above_null = (not oracle_near_null(oracle_metrics["line_rows"])) and oracle_metrics["eligible_line_count"] > 0
    if oracle_metrics["eligible_line_count"] == 0:
        verdict = "DATA_PARTIAL"
        status = "NO_ELIGIBLE_LINE"
    elif oracle_above_null:
        verdict = "THEORY_CHAIN_HOLDS"
        status = "ORACLE_ABOVE_NULL"
    elif cmap_metrics and not oracle_near_null(cmap_metrics["line_rows"]):
        verdict = "THEORY_CHAIN_HOLDS_CMAP_ONLY"
        status = "SPEARMAN_NEAR_NULL_CMAP_ABOVE_NULL"
    else:
        verdict = "CHECK_SIGNATURE_DOSE_TIME_CONTEXT_PRISM"
        status = "ORACLE_NEAR_NULL"

    result = {
        "format": "full_observed_oracle_result_v1",
        "exp_id": "EXP-007",
        "status": status,
        "verdict": verdict,
        "can_answer_question": bool(oracle_metrics["eligible_line_count"] > 0),
        "primary_condition": primary,
        "sensitivity_condition": sensitivity,
        "null": {"seed": NULL_SEED, "repeats": NULL_REPEATS, "type": "linewise_random_ranking"},
        "oracle": {
            "source": "exact978_cache_v1 + inst_info matched controls",
            "forbidden_source_used": False,
            "identity_pairs_before_prism": int(len(primary_scores)),
            "unique_pairs_after_prism_join": int(len(joined)),
            "unique_lines_after_prism_join": int(joined["context_id"].nunique()) if not joined.empty else 0,
            "unique_compounds_after_prism_join": int(joined["pert_id"].nunique()) if not joined.empty else 0,
            "build": primary_info,
            "metrics": oracle_metrics,
            "metrics_summary": _summarize_lines(oracle_metrics),
        },
        "sensitivity_condition_result": {
            "build": sensitivity_info,
            "unique_pairs_after_prism_join": int(len(sensitivity_joined)),
            "unique_lines_after_prism_join": int(sensitivity_joined["context_id"].nunique())
            if not sensitivity_joined.empty
            else 0,
            "metrics": sensitivity_metrics,
            "metrics_summary": _summarize_lines(sensitivity_metrics),
        },
        "cmap_weighted_ks": {
            "status": cmap_status,
            "metrics": cmap_metrics,
            "metrics_summary": _summarize_lines(cmap_metrics) if cmap_metrics else None,
        },
        "predicted_on_same_support": {
            "profile": evaluation["prediction"]["profile"],
            "adapter": evaluation["prediction"]["adapter"],
            "pair_count": int(len(predicted_support)),
            "metrics": predicted_metrics,
            "metrics_summary": _summarize_lines(predicted_metrics),
            "gap": gap,
        },
        "evaluation_contract": {
            "reversal": "-Spearman(CRC signed disease signature, Delta978)",
            "prism_direction": "sensitivity_score = - official PRISM log2 fold-change; larger means more sensitive",
            "prism_values_read_after_identity_freeze": True,
            "dose_time_not_mixed": True,
            "old_839_pair_oracle_not_used": True,
        },
        "decision_rule": {
            "oracle_above_null": "theory chain holds; record Predicted→Oracle gap",
            "oracle_near_null": "rerun CMap/weighted-KS once; if still no signal, inspect signature/dose/time/context/PRISM endpoint",
        },
    }
    per_line = {
        "format": "full_observed_oracle_per_line_v1",
        "exp_id": "EXP-007",
        "primary_condition": primary,
        "sensitivity_condition": sensitivity,
        "forbidden_interpretation": "do not macro-average lift across lines with different candidate sizes",
        "primary": oracle_metrics["line_rows"],
        "primary_support_strata": oracle_metrics["support_strata"],
        "sensitivity": sensitivity_metrics["line_rows"],
        "predicted_same_support": predicted_metrics["line_rows"],
        "cmap_weighted_ks": None if cmap_metrics is None else cmap_metrics["line_rows"],
    }
    _write(RESULT_OUT, result)
    _write(PER_LINE_OUT, per_line)
    contract["status"] = "EXECUTED"
    contract["primary_condition"] = primary
    contract["sensitivity_condition"] = sensitivity
    contract["result_status"] = status
    _write(CONTRACT, contract)
    _fill_record(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    payload = build(coverage_only=args.coverage_only)
    if args.coverage_only:
        print(
            json.dumps(
                {
                    "primary_condition": payload["primary_condition"],
                    "coverage": {
                        name: {
                            "pairs": item["unique_pairs_with_matched_control"],
                            "contexts": item["unique_contexts_with_matched_control"],
                            "compounds": item["unique_compounds_with_matched_control"],
                        }
                        for name, item in payload["canonical_conditions"].items()
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "status": payload["status"],
                "verdict": payload["verdict"],
                "primary_condition": payload["primary_condition"],
                "pairs": payload["oracle"]["unique_pairs_after_prism_join"],
                "lines": payload["oracle"]["unique_lines_after_prism_join"],
                "eligible_lines": payload["oracle"]["metrics"]["eligible_line_count"],
                "gap": payload["predicted_on_same_support"]["gap"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
