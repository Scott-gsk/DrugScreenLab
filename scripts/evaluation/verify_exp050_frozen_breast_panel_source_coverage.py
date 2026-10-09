#!/usr/bin/env python3
"""Independently verify the frozen breast panel's label-blind source coverage join."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PANEL = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_panel_v5/breast_subtype_candidate_table.tsv")
SOURCE_CONDITIONS = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_pairwise_comparison_v1/full/condition_metrics.tsv")
SOURCE_KEY_SUMMARY = Path("artifacts/experiments/EXP-050/20261006_breast_subtype_pairwise_comparison_v1/full/drug_equal_weight_metrics.tsv")
RUNNER = Path("scripts/evaluation/summarize_exp050_frozen_breast_panel_source_coverage.py")
OUT = Path("artifacts/experiments/EXP-050/20261006_breast_panel_source_response_coverage_v1")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    panel = pd.read_csv(
        PANEL, sep="\t",
        usecols=["stratum", "inchikey", "n_independent_train_origins", "observed_cell_line_names", "median_aac", "candidate_tier", "candidate_rank_in_tier"],
        dtype={"inchikey": "string"},
    )
    panel = panel.loc[panel["candidate_tier"].ne("not_selected")].copy()
    if len(panel) != 24 or panel["inchikey"].nunique() != 15:
        raise AssertionError("Frozen source panel is no longer 24 strata rows / 15 unique keys")
    expected_keys = set(panel["inchikey"].astype(str))

    conditions = pd.read_csv(
        SOURCE_CONDITIONS, sep="\t",
        usecols=["inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key", "n_source_models"],
        dtype={"inchi_key": "string", "dose_key": "string", "dose_unit_key": "string", "time_key": "string", "time_unit_key": "string"},
    )
    key_summary = pd.read_csv(
        SOURCE_KEY_SUMMARY, sep="\t",
        usecols=["inchi_key", "n_exact_dose_time_exposures", "aggregation", "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho", "basal_internal_minus_basal_mcf7_gene_rho", "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement", "basal_internal_minus_basal_mcf7_gene_sign_agreement"],
        dtype={"inchi_key": "string"},
    ).set_index("inchi_key")
    output = pd.read_csv(OUT / "frozen_panel_source_coverage.tsv", sep="\t", dtype={"inchikey": "string"})
    if len(output) != 15 or set(output["inchikey"].astype(str)) != expected_keys:
        raise AssertionError("Output candidate keys do not exactly equal the frozen 15-key set")
    if output["inchikey"].duplicated().any():
        raise AssertionError("Output duplicates candidate keys")

    for row in output.to_dict(orient="records"):
        key = row["inchikey"]
        rows = conditions.loc[conditions["inchi_key"].eq(key)]
        if int(row["strict_source_condition_count"]) != len(rows):
            raise AssertionError(f"Condition count mismatch for {key}")
        present = key in key_summary.index
        if bool(row["present_in_strict_complete_key_source_domain"]) != present or present != (len(rows) > 0):
            raise AssertionError(f"Strict-source domain presence mismatch for {key}")
        if present:
            for column in key_summary.columns:
                got = row[column]
                expected = key_summary.loc[key, column]
                if pd.isna(expected) and pd.isna(got):
                    continue
                if pd.api.types.is_numeric_dtype(key_summary[column]):
                    if abs(float(got) - float(expected)) > 1e-12:
                        raise AssertionError(f"Metric readback mismatch for {key}/{column}")
                elif str(got) != str(expected):
                    raise AssertionError(f"Metric readback mismatch for {key}/{column}")
        if present and (int(row["source_model_count_min"]) != 4 or int(row["source_model_count_max"]) != 4):
            raise AssertionError(f"Matched source conditions do not show four source models for {key}")
        if not present and (int(row["source_model_count_min"]) != 0 or int(row["source_model_count_max"]) != 0):
            raise AssertionError(f"Missing source key has nonzero source model count for {key}")

    if int(output["present_in_strict_complete_key_source_domain"].sum()) != 13:
        raise AssertionError("The independently observed exact-key coverage count is not 13/15")
    per_stratum = {}
    for stratum, group in panel.groupby("stratum", sort=True):
        stratum_keys = set(group["inchikey"].astype(str))
        support_keys = set(output.loc[output["present_in_strict_complete_key_source_domain"], "inchikey"].astype(str))
        per_stratum[str(stratum)] = {
            "selected_stratum_rows": int(len(group)),
            "unique_frozen_keys": int(len(stratum_keys)),
            "keys_with_source_profiles": int(len(stratum_keys & support_keys)),
        }
    if per_stratum != {
        "basal_like_annotation_basal_B_legacy": {"selected_stratum_rows": 10, "unique_frozen_keys": 10, "keys_with_source_profiles": 9},
        "breast_all": {"selected_stratum_rows": 14, "unique_frozen_keys": 14, "keys_with_source_profiles": 12},
    }:
        raise AssertionError(f"Frozen-stratum coverage mismatch: {per_stratum}")

    receipt = {
        "status": "ROOT_INDEPENDENT_READBACK_PASS",
        "run_id": "20261006_breast_panel_source_response_coverage_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Reopened only approved TRAIN candidate columns, strict source-domain condition/key summaries and this run's output. No PDO response labels/values, outer/locked results or GSE216540 sample metadata.",
        "frozen_selection": {"stratum_rows": 24, "unique_full_InChIKeys": 15, "candidate_order_or_membership_changed": False},
        "strict_source_domain": {"candidate_keys_with_exact_source_transcription_profile": 13, "candidate_keys_without_profile": 2, "all_matched_condition_rows_have_four_fixed_source_models": True, "source_condition_rows_for_13_keys": int(output["strict_source_condition_count"].sum())},
        "coverage_by_frozen_stratum": per_stratum,
        "readback_checks": {"exact_candidate_key_set": "PASS", "one_output_row_per_unique_key": "PASS", "condition_count_and_dose_time_source_join": "PASS", "key_equal_metrics_match_source_table": "PASS", "candidate_order_and_membership_unchanged": "PASS"},
        "boundary": "Post-selection source coverage only; not independent candidate validation, PDO transfer, or efficacy evidence.",
    }
    receipt_path = OUT / "root_independent_readback_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = [
        (PANEL, "INPUT_TRAIN_PANEL_COLUMNS_ONLY"),
        (SOURCE_CONDITIONS, "INPUT_STRICT_SOURCE_CONDITIONS"),
        (SOURCE_KEY_SUMMARY, "INPUT_STRICT_SOURCE_KEY_SUMMARY"),
        (RUNNER, "ANALYSIS_RUNNER"),
        (Path(__file__).resolve(), "INDEPENDENT_VERIFIER"),
        (OUT / "frozen_panel_source_coverage.tsv", "ANALYSIS_OUTPUT"),
        (OUT / "coverage_receipt.json", "ANALYSIS_RECEIPT"),
        (OUT / "attempt_001_preflight_failure.json", "PRESERVED_FAILURE"),
        (OUT / "attempt_002_preflight_failure.json", "PRESERVED_FAILURE"),
        (OUT / "attempt_003_preflight_failure.json", "PRESERVED_FAILURE"),
        (OUT / "attempt_004_scope.json", "RUN_SCOPE"),
        (OUT / "root_verifier_attempt_001_failure.json", "PRESERVED_VERIFIER_FAILURE"),
    ]
    manifest_path = OUT / "sha256_manifest.tsv"
    manifest_df = pd.DataFrame(
        [(str(path), digest(path), role) for path, role in manifest],
        columns=["path", "sha256", "role"],
    )
    manifest_df.to_csv(manifest_path, sep="\t", index=False, lineterminator="\n")
    manifest_readback = pd.read_csv(manifest_path, sep="\t", dtype={"path": "string", "sha256": "string", "role": "string"})
    verified_manifest_rows = 0
    for record in manifest_readback.to_dict(orient="records"):
        if digest(Path(str(record["path"]))) != str(record["sha256"]):
            raise AssertionError(f"Manifest hash readback failed: {record['path']}")
        verified_manifest_rows += 1
    receipt["manifest_rows"] = len(manifest)
    receipt["manifest_sha256"] = digest(manifest_path)
    receipt["manifest_hash_readback_verified_rows"] = verified_manifest_rows
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
