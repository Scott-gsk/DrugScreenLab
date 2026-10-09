#!/usr/bin/env python3
"""Summarize source-response coverage for the already frozen breast candidate panel.

This is a label-blind coverage join only: it does not rank or replace candidates,
and intentionally never loads the panel's PDO-transcript-match columns or any PDO
response data. All source matrices and candidate choices are existing EXP-050
artifacts; no model is fit and no GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import pandas as pd


DEFAULT_PANEL = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_panel_v5/breast_subtype_candidate_table.tsv")
DEFAULT_SOURCE = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_pairwise_comparison_v1/full/condition_metrics.tsv")
DEFAULT_KEY_SUMMARY = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_pairwise_comparison_v1/full/drug_equal_weight_metrics.tsv")
DEFAULT_OUT = Path("artifacts/experiments/EXP-050/20261006_breast_panel_source_response_coverage_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--source-conditions", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-key-summary", type=Path, default=DEFAULT_KEY_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # Deliberately load only TRAIN candidate identity/rank/AAC and source-origin
    # fields; PDO-support, labels, and all outer/locked columns stay unopened.
    panel_columns = [
        "stratum", "inchikey", "n_independent_train_origins",
        "observed_cell_line_names", "median_aac", "candidate_tier",
        "candidate_rank_in_tier",
    ]
    panel_all = pd.read_csv(args.panel, sep="\t", usecols=panel_columns, dtype={"inchikey": "string"})
    if set(panel_all["candidate_tier"].dropna().astype(str).unique()) != {"concordant_high", "concordant_low", "not_selected"}:
        raise ValueError("Frozen panel candidate-tier vocabulary changed")
    panel = panel_all.loc[panel_all["candidate_tier"].ne("not_selected")].copy()
    if panel["inchikey"].isna().any() or panel["inchikey"].str.len().ne(27).any():
        raise ValueError("Frozen panel must contain only full InChIKeys")
    if len(panel) != 24 or panel["inchikey"].nunique() != 15:
        raise ValueError(
            "Frozen panel frame differs from registered 24/15: "
            f"rows={len(panel)}, unique_keys={panel['inchikey'].nunique()}"
        )

    condition_columns = [
        "inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key",
        "n_source_models",
    ]
    conditions = pd.read_csv(
        args.source_conditions,
        sep="\t",
        usecols=condition_columns,
        dtype={"inchi_key": "string", "dose_key": "string", "dose_unit_key": "string", "time_key": "string", "time_unit_key": "string"},
    )
    if conditions["inchi_key"].isna().any() or conditions["inchi_key"].str.len().ne(27).any():
        raise ValueError("Strict source table unexpectedly contains a missing or non-full InChIKey")

    key_columns = [
        "inchi_key", "n_exact_dose_time_exposures", "aggregation",
        "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho",
        "basal_internal_minus_basal_mcf7_gene_rho",
        "basal_internal_mean_gene_sign_agreement",
        "basal_vs_mcf7_mean_gene_sign_agreement",
        "basal_internal_minus_basal_mcf7_gene_sign_agreement",
    ]
    key_metrics = pd.read_csv(args.source_key_summary, sep="\t", usecols=key_columns, dtype={"inchi_key": "string"})
    if key_metrics["inchi_key"].duplicated().any():
        raise ValueError("Strict equal-key summary must have one row per exact full InChIKey")
    metric_by_key = key_metrics.set_index("inchi_key").to_dict(orient="index")

    output_rows: list[dict[str, object]] = []
    for key_value, group in panel.groupby("inchikey", sort=True, dropna=False):
        key = str(key_value)
        matched = conditions.loc[conditions["inchi_key"].eq(key)]
        contexts = sorted(
            {
                f"{dose} {dose_unit} × {time} {time_unit}"
                for dose, dose_unit, time, time_unit in matched[
                    ["dose_key", "dose_unit_key", "time_key", "time_unit_key"]
                ].itertuples(index=False, name=None)
            }
        )
        rank_records = [
            {"stratum": str(row.stratum), "tier": str(row.candidate_tier), "rank": int(row.candidate_rank_in_tier)}
            for row in group[["stratum", "candidate_tier", "candidate_rank_in_tier"]].itertuples(index=False)
        ]
        origins_by_stratum = {
            str(stratum): sorted({int(x) for x in rows["n_independent_train_origins"].dropna()})
            for stratum, rows in group.groupby("stratum", sort=True)
        }
        aac_medians_by_stratum = {
            str(stratum): sorted({float(x) for x in rows["median_aac"].dropna()})
            for stratum, rows in group.groupby("stratum", sort=True)
        }
        cell_names = sorted({
            name.strip()
            for value in group["observed_cell_line_names"].dropna().astype(str)
            for name in value.split(";")
            if name.strip()
        })
        base = {
            "inchikey": key,
            "frozen_source_panel_stratum_row_count": int(len(group)),
            "frozen_panel_strata": " | ".join(sorted(group["stratum"].astype(str).unique())),
            "frozen_tier_rank_records_unchanged": json.dumps(rank_records, ensure_ascii=False, separators=(",", ":")),
            "train_origin_counts_by_stratum": json.dumps(origins_by_stratum, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "train_median_aac_by_stratum": json.dumps(aac_medians_by_stratum, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "observed_cell_line_names_from_train_aac": ";".join(cell_names),
            "strict_source_condition_count": int(len(matched)),
            "strict_source_dose_time_contexts": " | ".join(contexts),
            "source_model_count_min": int(matched["n_source_models"].min()) if len(matched) else 0,
            "source_model_count_max": int(matched["n_source_models"].max()) if len(matched) else 0,
            "present_in_strict_complete_key_source_domain": bool(key in metric_by_key),
        }
        source_metrics = metric_by_key.get(key)
        if source_metrics:
            base.update(source_metrics)
        else:
            for col in key_columns[1:]:
                base[col] = pd.NA
        output_rows.append(base)

    result = pd.DataFrame(output_rows)
    if len(result) != 15 or result["inchikey"].duplicated().any():
        raise AssertionError("Output is not one row per frozen candidate key")
    if not result["present_in_strict_complete_key_source_domain"].eq(result["strict_source_condition_count"].gt(0)).all():
        raise AssertionError("Source condition/key-summary join disagrees")

    args.out.mkdir(parents=True, exist_ok=True)
    table_path = args.out / "frozen_panel_source_coverage.tsv"
    result.to_csv(table_path, sep="\t", index=False, na_rep="NA", lineterminator="\n")
    tiers = {}
    source_keys = set(result.loc[result["present_in_strict_complete_key_source_domain"], "inchikey"].astype(str))
    for stratum, group in panel.groupby("stratum", dropna=False, sort=True):
        stratum_keys = set(group["inchikey"].astype(str))
        matched_keys = stratum_keys & source_keys
        tiers[str(stratum)] = {
            "frozen_stratum_row_count": int(len(group)),
            "frozen_unique_candidate_key_count": int(len(stratum_keys)),
            "unique_candidate_keys_with_strict_source_transcript_support": int(len(matched_keys)),
            "strict_source_condition_rows_for_these_keys": int(result.loc[result["inchikey"].isin(matched_keys), "strict_source_condition_count"].sum()),
        }
    receipt = {
        "status": "COMPLETED_LABEL_BLIND_FROZEN_PANEL_SOURCE_COVERAGE",
        "run_id": "20261006_breast_panel_source_response_coverage_v1",
        "scope": "Coverage only for the already frozen 15 candidate keys; no re-ranking, replacement, model fit, PDO values, outer/locked values, or GSE216540 sample metadata.",
        "independent_unit": "Exact full InChIKey; observed source conditions are coverage counts, not biological replicates.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_stratum_row_count": int(len(panel)),
        "panel_count": int(len(result)),
        "unique_full_InChIKeys": int(result["inchikey"].nunique()),
        "present_in_strict_source_domain": int(result["present_in_strict_complete_key_source_domain"].sum()),
        "strict_source_full_key_domain_size": int(key_metrics["inchi_key"].nunique()),
        "strict_source_exposure_rows": int(len(conditions)),
        "source_model_count_values": sorted(int(x) for x in conditions["n_source_models"].dropna().unique()),
        "coverage_by_frozen_tier": tiers,
        "input_sha256": {
            str(args.panel): sha256(args.panel),
            str(args.source_conditions): sha256(args.source_conditions),
            str(args.source_key_summary): sha256(args.source_key_summary),
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "cpu_wall_seconds": round(perf_counter() - started, 6),
        "output_sha256": {str(table_path.name): sha256(table_path)},
        "result_boundary": "This post-selection linkage describes which unchanged AAC candidates have an exact-key source transcription profile among four fixed source lines. It is not independent candidate validation, PDO transfer, or efficacy evidence.",
    }
    (args.out / "coverage_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
