"""Audit EXP-045 identity links using only preregistered local metadata fields.

This audit does not read dependency values, drug-response labels, predictions,
or evaluation results. It writes a count-and-hash receipt; patient/cell IDs are
never printed or copied into the receipt.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
NATURE_ZIP = ROOT / "data/raw/transfer_public_assets/organoid_biobank/41586_2026_10830_MOESM3_ESM.zip"
NATURE_SAMPLES = ROOT / "data/processed/transfer/biobank_functional_v1/samples.parquet"
TARGET_SAMPLES = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1/target_samples.parquet"
DEPMAP_MODELS = ROOT / "data/raw/depmap/24q2/Model.csv"
OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260924_local_identity_crosswalk_v3"
NATURE_ID_FIELDS = ("sample_ID", "sanger_ID", "individual_ID")
DEPMAP_ID_FIELDS = (
    "ModelID",
    "PatientID",
    "CellLineName",
    "CCLEName",
    "HCMIID",
    "WTSIMasterCellID",
    "SangerModelID",
    "COSMICID",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_projection_sha256(frame: pd.DataFrame, fields: tuple[str, ...]) -> str:
    """Hash only the explicitly approved identity columns, never a whole data asset."""
    rows = [
        [_key(row[field]) for field in fields]
        for row in frame.loc[:, list(fields)].to_dict(orient="records")
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _exact_patient_candidates(
    target: pd.DataFrame, nature: pd.DataFrame
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return exact-key target-patient→Nature-individual candidates and evidence.

    Only patient_id/sample_id on the frozen target rows and the three
    preregistered Nature identity fields are compared. No fuzzy, case-folded,
    punctuation-stripped, or label-derived joins are permitted.
    """
    by_patient: dict[str, set[str]] = defaultdict(set)
    evidence: dict[str, set[str]] = defaultdict(set)
    lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in nature.itertuples(index=False):
        patient = _key(getattr(row, "individual_ID"))
        if patient is None:
            continue
        for field in NATURE_ID_FIELDS:
            value = _key(getattr(row, field))
            if value is not None:
                lookup[value].append((patient, field))

    for row in target.itertuples(index=False):
        patient_id = _key(row.patient_id)
        if patient_id is None:
            continue
        for target_field in ("patient_id", "sample_id"):
            value = _key(getattr(row, target_field))
            if value is None:
                continue
            for nature_patient, nature_field in lookup.get(value, []):
                by_patient[patient_id].add(nature_patient)
                evidence[patient_id].add(f"{target_field}=={nature_field}")
    return by_patient, evidence


def _bijective_candidates(
    target_ids: set[str], candidates: dict[str, set[str]]
) -> tuple[dict[str, str], int, int]:
    """Keep only exact one-to-one target/Nature patient candidates."""
    reverse: dict[str, set[str]] = defaultdict(set)
    for target_id in target_ids:
        linked = candidates.get(target_id, set())
        if len(linked) == 1:
            reverse[next(iter(linked))].add(target_id)
    accepted = {
        target_id: next(iter(candidates[target_id]))
        for target_id in target_ids
        if len(candidates.get(target_id, set())) == 1
        and len(reverse[next(iter(candidates[target_id]))]) == 1
    }
    ambiguous_forward = sum(len(candidates.get(target_id, set())) > 1 for target_id in target_ids)
    ambiguous_reverse = sum(
        len(target_links) > 1 for target_links in reverse.values()
    )
    return accepted, int(ambiguous_forward), int(ambiguous_reverse)


def _identity_table_audit(
    official: pd.DataFrame, paired: pd.DataFrame
) -> dict[str, Any]:
    """Check field completeness, unique sample keys, and row-preserving subset."""
    official_rows = [
        tuple(_key(getattr(row, field)) for field in NATURE_ID_FIELDS)
        for row in official.itertuples(index=False)
    ]
    paired_rows = [
        tuple(_key(getattr(row, field)) for field in NATURE_ID_FIELDS)
        for row in paired.itertuples(index=False)
    ]
    required = ("sample_ID", "individual_ID")
    missing = {
        "official": {
            field: sum(row[index] is None for row in official_rows)
            for index, field in enumerate(NATURE_ID_FIELDS)
        },
        "paired": {
            field: sum(row[index] is None for row in paired_rows)
            for index, field in enumerate(NATURE_ID_FIELDS)
        },
    }
    sample_index = NATURE_ID_FIELDS.index("sample_ID")
    required_index = [NATURE_ID_FIELDS.index(field) for field in required]
    official_counter = Counter(row for row in official_rows if all(row[i] is not None for i in required_index))
    paired_counter = Counter(row for row in paired_rows if all(row[i] is not None for i in required_index))
    official_samples = [row[sample_index] for row in official_rows if row[sample_index] is not None]
    paired_samples = [row[sample_index] for row in paired_rows if row[sample_index] is not None]
    required_fields_complete = all(
        missing[domain][field] == 0
        for domain in ("official", "paired")
        for field in required
    )
    paired_subset = required_fields_complete and all(
        count <= official_counter[row] for row, count in paired_counter.items()
    )
    return {
        "missing_identity_values": missing,
        "official_duplicate_sample_ids": len(official_samples) - len(set(official_samples)),
        "paired_duplicate_sample_ids": len(paired_samples) - len(set(paired_samples)),
        "required_sample_and_individual_fields_complete": bool(required_fields_complete),
        "paired_identity_rows_exact_subset_of_official_availability": bool(paired_subset),
        "paired_identity_rows_not_found_in_official": int(
            sum(max(0, count - official_counter[row]) for row, count in paired_counter.items())
        ),
    }


def _target_identity_audit(target: pd.DataFrame) -> dict[str, int | bool]:
    """Summarize target-key completeness and exact patient/sample duplicates."""
    patient_keys = [_key(value) for value in target["patient_id"]]
    sample_keys = [_key(value) for value in target["sample_id"]]
    pairs = list(zip(patient_keys, sample_keys))
    complete_rows = [pair for pair in pairs if pair[0] is not None and pair[1] is not None]
    return {
        "rows": len(target),
        "missing_patient_id": sum(value is None for value in patient_keys),
        "missing_sample_id": sum(value is None for value in sample_keys),
        "unique_patients": len({value for value in patient_keys if value is not None}),
        "unique_samples": len({value for value in sample_keys if value is not None}),
        "duplicate_patient_sample_rows": len(complete_rows) - len(set(complete_rows)),
        "all_patient_and_sample_keys_complete": all(
            value is not None for pair in pairs for value in pair
        ),
    }
def _status_and_exit_code(
    *, registered_provenance_pass: bool, pairing_pass: bool, target_pass: bool, source_pass: bool
) -> tuple[str, int]:
    if not registered_provenance_pass or not pairing_pass:
        return "AUDIT_INVALID", 1
    if target_pass and source_pass:
        return "IDENTITY_CROSSWALK_PASS", 0
    return "DATA_BLOCKED", 2


def _main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "data/registry/datasets.json").read_text(encoding="utf-8"))
    transfer_registry = json.loads((ROOT / "data/registry/datasets_transfer.json").read_text(encoding="utf-8"))
    depmap_entries = {entry["id"]: entry for entry in registry}
    depmap_rnaseq_entry = depmap_entries["depmap_public_24q2_rnaseq_raw_v1"]
    depmap_chronos_entry = depmap_entries["depmap_public_24q2_crispr_gene_effect_raw_v1"]
    transfer_entries = {entry["id"]: entry for entry in transfer_registry}
    nature_raw_entry = transfer_entries["transfer_organoid_biobank_supplementary_tables"]
    nature_admission = json.loads(
        (ROOT / "data/artifacts/EXP-026/organoid_depmap_admission.json").read_text(encoding="utf-8")
    )
    nature_zip_check = [
        item for item in nature_admission["checks"]
        if item.get("check") == "raw:41586_2026_10830_MOESM3_ESM.zip"
    ]
    registered_zip_sha = nature_raw_entry["checksum"]["value"]
    nature_zip_admission_match = (
        len(nature_zip_check) == 1
        and bool(nature_zip_check[0].get("passed"))
        and nature_zip_check[0].get("expected") == registered_zip_sha
        and nature_zip_check[0].get("actual") == registered_zip_sha
    )

    with zipfile.ZipFile(NATURE_ZIP) as archive:
        member = "supplementary_table_2_revision.xlsx"
        member_bytes = archive.read(member)
    nature_official = pd.read_excel(
        BytesIO(member_bytes),
        sheet_name="data_availability",
        usecols=list(NATURE_ID_FIELDS),
        dtype=str,
    )
    nature_paired = pd.read_parquet(
        NATURE_SAMPLES,
        columns=["sample_ID", "sanger_ID", "individual_ID"],
    )
    target = pd.read_parquet(TARGET_SAMPLES, columns=["patient_id", "sample_id"])
    depmap_models = pd.read_csv(
        DEPMAP_MODELS,
        usecols=list(DEPMAP_ID_FIELDS),
        dtype=str,
        keep_default_na=False,
    )

    pair_audit = _identity_table_audit(nature_official, nature_paired)
    paired_subset = pair_audit["paired_identity_rows_exact_subset_of_official_availability"]

    candidates, evidence = _exact_patient_candidates(target, nature_official)
    target_audit = _target_identity_audit(target)
    target_patient_ids = {_key(x) for x in target["patient_id"] if _key(x) is not None}
    unique_bijective_candidates, ambiguous_forward, ambiguous_reverse = _bijective_candidates(
        target_patient_ids, candidates
    )

    source_unresolved = manifest["source_unresolved"]
    source_identity_candidates: list[dict[str, Any]] = []
    for source_id in source_unresolved:
        rows = depmap_models.loc[depmap_models["SangerModelID"] == source_id]
        model_ids = sorted({_key(value) for value in rows["ModelID"] if _key(value) is not None})
        source_identity_candidates.append(
            {
                "identity_key_sha256": hashlib.sha256(source_id.encode("utf-8")).hexdigest(),
                "matched_model_rows": int(len(rows)),
                "distinct_model_candidates": len(model_ids),
                "model_candidate_sha256": hashlib.sha256("\n".join(model_ids).encode("utf-8")).hexdigest(),
                "status": "MISSING" if not model_ids else "UNIQUE" if len(model_ids) == 1 else "AMBIGUOUS",
            }
        )

    target_fold_counts = [
        {
            "outer_fold": fold["fold"],
            "outer_target_patients": fold["outer_test_patient_count"],
            "outer_linked_patients_in_prior_manifest": fold["linked_patient_count"],
            "outer_unresolved_patients_in_prior_manifest": len(fold["unresolved_patient_ids"]),
            "inner": [
                {
                    "inner_fold": inner["inner_fold"],
                    "fraction": inner["fraction"],
                    "validation_patients": inner["validation_patient_count"],
                    "linked_patients_in_prior_manifest": inner["linked_patient_count"],
                    "unresolved_patients_in_prior_manifest": len(inner["unresolved_patient_ids"]),
                }
                for inner in fold["inner"]
            ],
        }
        for fold in manifest["target_folds"]
    ]
    source_fold_counts = [
        {
            "outer_fold": fold["fold"],
            "outer_test_sanger_ids": fold["outer_test_sanger_count"],
            "outer_unresolved_ids": len(fold["unresolved_sanger_ids"]),
            "outer_excluded_model_ids": len(fold["excluded_model_ids"]),
            "inner": [
                {
                    "inner_fold": inner["inner_fold"],
                    "validation_sanger_ids": inner["validation_sanger_count"],
                    "validation_unresolved_ids": len(inner["unresolved_sanger_ids"]),
                    "excluded_model_ids": len(inner["excluded_model_ids"]),
                }
                for inner in fold["inner"]
            ],
        }
        for fold in manifest["source_folds"]
    ]

    # Identity review uses exact column projections. Full-file hashes for expression,
    # dependency, or ZIP payloads are intentionally not recomputed here.
    projection_hashes = {
        "nature_workbook_data_availability": _identity_projection_sha256(
            nature_official, NATURE_ID_FIELDS
        ),
        "nature_paired_samples": _identity_projection_sha256(
            nature_paired, NATURE_ID_FIELDS
        ),
        "exp036_target_samples": _identity_projection_sha256(
            target, ("patient_id", "sample_id")
        ),
        "depmap_24q2_model_identity_columns": _identity_projection_sha256(
            depmap_models, DEPMAP_ID_FIELDS
        ),
    }
    depmap_model_sha_registered = depmap_rnaseq_entry["files"]["Model.csv"]["sha256"]
    depmap_model_bytes_registered = int(depmap_rnaseq_entry["files"]["Model.csv"]["bytes"])
    depmap_model_size_match = DEPMAP_MODELS.stat().st_size == depmap_model_bytes_registered
    provenance_pass = (
        nature_zip_admission_match
        and nature_raw_entry["version"] == 1
        and depmap_rnaseq_entry["version"] == 1
        and depmap_chronos_entry["version"] == 1
        and depmap_model_size_match
    )
    target_unresolved_count = len(target_patient_ids - set(unique_bijective_candidates))
    source_unresolved_pass = all(item["status"] == "UNIQUE" for item in source_identity_candidates)
    crosswalk_pass = (
        bool(target_audit["all_patient_and_sample_keys_complete"])
        and int(target_audit["duplicate_patient_sample_rows"]) == 0
        and target_unresolved_count == 0
        and len(unique_bijective_candidates) == len(target_patient_ids)
    )
    status, exit_code = _status_and_exit_code(
        registered_provenance_pass=provenance_pass,
        pairing_pass=bool(paired_subset),
        target_pass=bool(crosswalk_pass),
        source_pass=bool(source_unresolved_pass),
    )

    receipt = {
        "audit_id": "EXP-045-local-registered-identity-crosswalk-v3",
        "status": status,
        "scope": {
            "mode": "read_only_identity_fields_only",
            "nature_workbook": "supplementary_table_2_revision.xlsx / data_availability / sample_ID,sanger_ID,individual_ID only",
            "nature_parquet": "samples.parquet / sample_ID,sanger_ID,individual_ID only",
            "target_parquet": "EXP-036 target_samples.parquet / patient_id,sample_id only",
            "depmap_model_csv": "Model.csv / registered model identity columns only",
            "not_interpreted_or_serialized": ["Nature dependency LFC values", "DepMap Chronos values", "drug-response labels", "predictions", "evaluation metrics", "clinical_data worksheet", "Sanger API", "non-identity table columns", "full-file payload hashes for expression/dependency/ZIP data"],
            "matching_rule": "Unicode-preserving exact strings with surrounding whitespace stripped; no case folding, punctuation changes, fuzzy matching, or score-based selection",
        },
        "provenance": {
            "prior_frozen_identity_manifest_sha256": _sha256(MANIFEST),
            "prior_frozen_manifest_id": manifest["audit_id"],
            "prior_frozen_input_hashes_recomputed": False,
            "identity_projection_sha256": projection_hashes,
            "nature_registry_id": nature_raw_entry["id"],
            "nature_registry_version": nature_raw_entry["version"],
            "nature_registered_zip_sha256": registered_zip_sha,
            "nature_prior_admission_receipt": "data/artifacts/EXP-026/organoid_depmap_admission.json",
            "nature_zip_checksum_matches_registered_admission": bool(nature_zip_admission_match),
            "depmap_model_registry_id": depmap_rnaseq_entry["id"],
            "depmap_model_registry_version": depmap_rnaseq_entry["version"],
            "depmap_model_registry_sha256_not_recomputed": depmap_model_sha_registered,
            "depmap_model_registered_byte_size_matches": bool(depmap_model_size_match),
            "depmap_dependency_registry_id": depmap_chronos_entry["id"],
            "depmap_dependency_registry_version": depmap_chronos_entry["version"],
            "nature_table2_member": member,
            "nature_table2_identity_projection_sha256": projection_hashes["nature_workbook_data_availability"],
        },
        "nature_pairing_consistency": {
            "official_availability_rows": int(len(nature_official)),
            "official_unique_samples": int(nature_official["sample_ID"].nunique()),
            "official_unique_individuals": int(nature_official["individual_ID"].nunique()),
            "paired_rows": int(len(nature_paired)),
            "paired_unique_samples": int(nature_paired["sample_ID"].nunique()),
            "paired_unique_individuals": int(nature_paired["individual_ID"].nunique()),
            **pair_audit,
        },
        "target_crosswalk": {
            **target_audit,
            "target_sample_rows": int(len(target)),
            "target_unique_patients": int(len(target_patient_ids)),
            "manifest_target_unresolved_count": int(len(manifest["target_unresolved"])),
            "target_patient_and_sample_exact_matches_by_nature_field": {
                nature_field: int(
                    len(
                        ({_key(x) for x in target["patient_id"] if _key(x) is not None}
                         | {_key(x) for x in target["sample_id"] if _key(x) is not None})
                        & {_key(x) for x in nature_official[nature_field] if _key(x) is not None}
                    )
                )
                for nature_field in NATURE_ID_FIELDS
            },
            "patients_with_exact_nature_candidates": int(sum(bool(candidates.get(patient)) for patient in target_patient_ids)),
            "unique_bijective_exact_candidates": int(len(unique_bijective_candidates)),
            "ambiguous_forward_candidates": ambiguous_forward,
            "reverse_many_targets_to_one_nature_individual": ambiguous_reverse,
            "still_unresolved_patients": int(target_unresolved_count),
            "candidate_field_evidence_counts": {
                evidence_field: int(sum(evidence_field in fields for fields in evidence.values()))
                for evidence_field in sorted({field for fields in evidence.values() for field in fields})
            },
        },
        "source_identity": {
            "frozen_source_unresolved_count": int(len(source_unresolved)),
            "unresolved_candidate_summary": source_identity_candidates,
            "fold_exclusion_manifest_counts": source_fold_counts,
        },
        "decision": {
            "target_patient_isolation": "BLOCKED: no exact one-to-one key path for target patient/sample IDs to Nature individual_ID in the authorized local fields",
            "source_cell_isolation": "BLOCKED: source IDs with missing or ambiguous DepMap Model.csv identity remain excluded from affected folds; do not infer a model row",
            "training_authorized": False,
            "next_gate": "obtain an authoritative local bridge for target patient and unresolved source identities; independently recheck this receipt; freeze independent functional holdout and per-fold exclusion sets before reading dependency values",
        },
        "folds": {"target": target_fold_counts, "source": source_fold_counts},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    receipt_path = OUTPUT_DIR / "identity_audit.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={status}; approved_identity_projections={len(projection_hashes)}; nature_registered_admission_match={nature_zip_admission_match}")
    print(f"nature_official_rows={len(nature_official)}; paired_models={nature_paired['sample_ID'].nunique()}; paired_individuals={nature_paired['individual_ID'].nunique()}; paired_subset={paired_subset}")
    print(f"target_patients={len(target_patient_ids)}; exact_bijective_crosswalks={len(unique_bijective_candidates)}; unresolved={target_unresolved_count}; source_unresolved={len(source_unresolved)}; candidate_receipt={receipt_path.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
