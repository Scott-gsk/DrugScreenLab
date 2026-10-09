"""Build the label-blind source OOF vector and target-key audit for EXP-048.

This entry point reads only LINCS condition metadata and verified source-fold
predictions for the reference vector. From the PDO response table it projects
patient_id and drug_id only; it never decodes response values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL_RECEIPT = ROOT / "artifacts/experiments/EXP-047/20260926_alt_common_panel_protocol_v1/receipt.json"
SOURCE_ROOT = ROOT / "artifacts/experiments/EXP-037/20260922_conditional_v2"
CONDITIONS = ROOT / "data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet"
GENES = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"
TARGET_INPUTS = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
DATASET_REGISTRY = ROOT / "data/registry/datasets.json"
OUTPUT = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1"

EXPECTED_PANEL_RECEIPT_SHA256 = "2626952099789c1497cad792cbad63e883c594b76e140051156efba0fbcb5981"
EXPECTED_CONDITIONS_SHA256 = "436c5165533608043b6b4e8f061082d50961a000b71c87eb17ec4a1a15963386"
CONDITIONS_MANIFEST_CLAIMED_FILE_SHA256 = "436c5165533608043b6b4e8f061082d50961a000b71c87eb17ec4a1a15963386"
PRIOR_CROSSWALK_RECORDED_CONDITIONS_SHA256 = "436c5165533608043b6e4e8f061082d50961a000b71c87eb17ec4a1a15963386"
EXPECTED_GENE_AXIS_SHA256 = "8cc8d6ade494209a9f323a94450e02c3a74d1ea3ae61a75ed4c084a9ab80502c"
SEEDS = (20260922, 20260923, 20260924)
ELIGIBLE_PANEL_INDEXES = (0, 1, 3, 4, 5, 7, 9, 12, 13)
FROZEN_SOURCE_CELL = "MCF7"
FROZEN_DOSE_MOLAR = Decimal("0.00001")
FROZEN_TIME_HOURS = Decimal("24")
GENE_COUNT = 978
MIN_DRUGS_FOR_PATIENT_RANK = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_dose_molar(value, unit: str) -> Decimal:
    normalized = str(unit).strip().casefold()
    if normalized not in {"um", "µm"}:
        raise ValueError(f"unsupported frozen dose unit: {unit!r}")
    return Decimal(str(value)) * Decimal("0.000001")


def _condition_time_hours(value, unit: str) -> Decimal:
    normalized = str(unit).strip().casefold()
    if normalized != "h":
        raise ValueError(f"unsupported frozen time unit: {unit!r}")
    return Decimal(str(value))


def _verification_path(fold_dir: Path, fold: int) -> Path:
    preferred = fold_dir / ("verification_v2.json" if fold < 3 else "verification_continuation.json")
    if preferred.is_file():
        return preferred
    fallback = fold_dir / "verification.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"fold {fold}: no accepted independent verifier receipt")


def materialize(output_dir: Path = OUTPUT) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None}:
        raise RuntimeError("this CPU-only input step requires CUDA_VISIBLE_DEVICES to be empty")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite EXP-048 inputs: {output_dir}")

    panel_sha = sha256(PANEL_RECEIPT)
    if panel_sha != EXPECTED_PANEL_RECEIPT_SHA256:
        raise ValueError("the approved nine-drug candidate receipt changed")
    conditions_sha = sha256(CONDITIONS)
    if conditions_sha != EXPECTED_CONDITIONS_SHA256:
        raise ValueError("the LINCS condition metadata differs from the recomputed asset checksum")
    conditions_manifest_path = CONDITIONS.parent / "manifest.json"
    conditions_manifest = json.loads(conditions_manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(conditions_manifest_path)
    registry = json.loads(DATASET_REGISTRY.read_text(encoding="utf-8"))
    registry_entry = next(
        (row for row in registry if row.get("id") == "lincs_gse92742_continuous_conditions_v2"),
        None,
    )
    if registry_entry is None or registry_entry["checksum"].get("value") != manifest_sha:
        raise ValueError("the continuous-condition manifest does not match its dataset-registry checksum")
    claimed_file_sha = conditions_manifest.get("files_sha256", {}).get("conditions.parquet")
    if claimed_file_sha != CONDITIONS_MANIFEST_CLAIMED_FILE_SHA256:
        raise ValueError("the registered condition-manifest checksum claim changed")
    conditions_file_checksum_discrepancy = claimed_file_sha != conditions_sha
    genes_sha = sha256(GENES)
    if genes_sha != EXPECTED_GENE_AXIS_SHA256:
        raise ValueError("the frozen 978-gene axis changed")

    panel_receipt = json.loads(PANEL_RECEIPT.read_text(encoding="utf-8"))
    mappings = panel_receipt["identity_bridge"]["unique_mappings"]
    mapping_by_index = {int(row["drug_index"]): str(row["pert_id"]) for row in mappings}
    if tuple(sorted(mapping_by_index)) != ELIGIBLE_PANEL_INDEXES:
        raise ValueError("the approved nine-drug identity set differs from the reviewed mapping")

    target_manifest_path = TARGET_INPUTS / "manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    target_verification = json.loads((TARGET_INPUTS / "verification.json").read_text(encoding="utf-8"))
    if target_verification.get("status") != "PASS":
        raise ValueError("the reusable EXP-036 target inputs are not independently verified")
    axis_path = TARGET_INPUTS / "drug_axis.parquet"
    axis = pd.read_parquet(axis_path, columns=["drug_index", "drug_id"])
    if axis["drug_index"].duplicated().any() or axis["drug_id"].duplicated().any():
        raise ValueError("target drug axis contains duplicate indexes or identities")
    target_id_by_index = {int(row.drug_index): str(row.drug_id) for row in axis.itertuples(index=False)}
    if any(index not in target_id_by_index for index in ELIGIBLE_PANEL_INDEXES):
        raise ValueError("one or more approved panel identities are absent from the target drug axis")

    panel_rows = []
    for index in ELIGIBLE_PANEL_INDEXES:
        panel_rows.append({
            "drug_index": index,
            "target_drug_id": target_id_by_index[index],
            "pert_id": mapping_by_index[index],
        })
    target_drug_ids = {row["target_drug_id"] for row in panel_rows}
    if len(target_drug_ids) != len(ELIGIBLE_PANEL_INDEXES):
        raise ValueError("nine approved target drug identities are not unique")

    condition_columns = [
        "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
        "pert_time", "pert_time_unit",
    ]
    conditions = pd.read_parquet(CONDITIONS, columns=condition_columns)
    selected = conditions.loc[
        conditions["cell_id"].astype(str).eq(FROZEN_SOURCE_CELL)
        & conditions["pert_id"].astype(str).isin(set(mapping_by_index.values()))
    ].copy()
    selected["dose_molar"] = [
        _condition_dose_molar(value, unit)
        for value, unit in zip(selected["pert_dose"], selected["pert_dose_unit"])
    ]
    selected["time_hours"] = [
        _condition_time_hours(value, unit)
        for value, unit in zip(selected["pert_time"], selected["pert_time_unit"])
    ]
    selected = selected.loc[
        selected["dose_molar"].eq(FROZEN_DOSE_MOLAR)
        & selected["time_hours"].eq(FROZEN_TIME_HOURS)
    ]
    condition_by_pert = {}
    for row in selected.itertuples(index=False):
        pert_id = str(row.pert_id)
        if pert_id in condition_by_pert:
            raise ValueError(f"more than one frozen MCF7 exposure row for {pert_id}")
        condition_by_pert[pert_id] = {
            "condition_id": str(row.condition_id),
            "cell_id": str(row.cell_id),
            "pert_dose": str(row.pert_dose),
            "pert_dose_unit": str(row.pert_dose_unit),
            "dose_molar": str(row.dose_molar),
            "pert_time": str(row.pert_time),
            "pert_time_unit": str(row.pert_time_unit),
            "time_hours": str(row.time_hours),
        }
    if set(condition_by_pert) != set(mapping_by_index.values()):
        missing = sorted(set(mapping_by_index.values()) - set(condition_by_pert))
        raise ValueError(f"frozen MCF7 shared-exposure rows are missing: {missing}")

    condition_ids = {row["condition_id"] for row in condition_by_pert.values()}
    condition_occurrences: dict[str, list[tuple[int, int]]] = {key: [] for key in condition_ids}
    fold_receipts = {}
    fold_key_rows = {}
    for fold in range(5):
        fold_dir = SOURCE_ROOT / f"fold_{fold}"
        verification_path = _verification_path(fold_dir, fold)
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if verification.get("status") != "PASS" or int(verification.get("fold", -1)) != fold:
            raise ValueError(f"source fold {fold} is not independently verified")
        verification_files = verification.get("files_sha256", {})
        fold_summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        if fold_summary.get("data_manifest_sha256") != manifest_sha:
            raise ValueError(f"source fold {fold} was not bound to the registered condition manifest")
        condition_ids_path = fold_dir / "test_condition_ids.npy"
        expected_condition_ids_sha = verification_files.get("test_condition_ids.npy")
        if not expected_condition_ids_sha or sha256(condition_ids_path) != expected_condition_ids_sha:
            raise ValueError(f"source fold {fold} condition key file differs from its verifier receipt")
        keys = np.load(condition_ids_path, allow_pickle=False)
        normalized_keys = [str(item) for item in keys.tolist()]
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError(f"source fold {fold} has duplicate condition ids")
        row_by_key = {key: row for row, key in enumerate(normalized_keys)}
        fold_key_rows[fold] = row_by_key
        fold_receipts[fold] = {
            "verification_path": str(verification_path.relative_to(ROOT)),
            "verification_sha256": sha256(verification_path),
            "fold_summary_sha256": sha256(fold_dir / "fold_summary.json"),
            "data_manifest_sha256": fold_summary.get("data_manifest_sha256"),
            "n_conditions": int(verification["n_conditions"]),
            "n_cells": int(verification["n_cells"]),
            "prediction_files_sha256": {},
        }
        for condition_id in condition_ids:
            if condition_id in row_by_key:
                condition_occurrences[condition_id].append((fold, row_by_key[condition_id]))

    if any(len(condition_occurrences[key]) != 1 for key in condition_ids):
        bad = {key: values for key, values in condition_occurrences.items() if len(values) != 1}
        raise ValueError(f"each selected source condition must have exactly one held-out fold: {bad}")
    used_folds = {condition_occurrences[key][0][0] for key in condition_ids}
    if len(used_folds) != 1:
        raise ValueError("the frozen MCF7 conditions do not share one source holdout fold")
    source_fold = next(iter(used_folds))
    source_verification_path = Path(fold_receipts[source_fold]["verification_path"])
    source_verification = json.loads((ROOT / source_verification_path).read_text(encoding="utf-8"))
    selected_row_by_pert = {
        pert_id: condition_occurrences[condition["condition_id"]][0][1]
        for pert_id, condition in condition_by_pert.items()
    }

    seed_matrices = []
    for seed in SEEDS:
        filename = f"predicted_delta_conditional_seed{seed}.npy"
        prediction_path = SOURCE_ROOT / f"fold_{source_fold}" / filename
        expected_prediction_sha = source_verification["files_sha256"].get(filename)
        if not expected_prediction_sha:
            raise ValueError(f"source fold {source_fold} verifier does not bind {filename}")
        if prediction_path.stat().st_mtime_ns > (ROOT / source_verification_path).stat().st_mtime_ns:
            raise ValueError(f"{filename} changed after its independent verification")
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != (fold_receipts[source_fold]["n_conditions"], GENE_COUNT):
            raise ValueError(f"{filename} shape differs from the verified condition and gene axes")
        ordered_rows = [
            selected_row_by_pert[mapping_by_index[index]]
            for index in ELIGIBLE_PANEL_INDEXES
        ]
        values = np.asarray(prediction[ordered_rows], dtype=np.float32)
        if values.shape != (len(ELIGIBLE_PANEL_INDEXES), GENE_COUNT) or not np.isfinite(values).all():
            raise ValueError(f"{filename} selected source values are incomplete or non-finite")
        seed_matrices.append(values)
        fold_receipts[source_fold]["prediction_files_sha256"][filename] = expected_prediction_sha

    reference_vectors = np.mean(np.stack(seed_matrices, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    if reference_vectors.shape != (len(ELIGIBLE_PANEL_INDEXES), GENE_COUNT) or not np.isfinite(reference_vectors).all():
        raise ValueError("the frozen source reference vectors are incomplete or non-finite")

    patient_axis_path = TARGET_INPUTS / "patient_axis.json"
    patients = json.loads(patient_axis_path.read_text(encoding="utf-8"))
    if len(patients) != len(set(map(str, patients))):
        raise ValueError("target patient axis contains duplicate identities")
    pairs_path = TARGET_INPUTS / "patient_pairs.parquet"
    pair_keys = pd.read_parquet(pairs_path, columns=["patient_id", "drug_id"])
    if pair_keys.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("target patient/drug key projection contains duplicates")
    if set(pair_keys["patient_id"].astype(str)) != set(map(str, patients)):
        raise ValueError("target patient key projection differs from the frozen patient axis")
    if set(target_drug_ids) - set(pair_keys["drug_id"].astype(str)):
        raise ValueError("one or more authorized target drugs have no patient label key")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    target_folds = split.get("target_folds")
    if not isinstance(target_folds, list) or len(target_folds) != 5:
        raise ValueError("the frozen target split must contain exactly five patient folds")
    test_fold_by_patient = {}
    for fold, item in enumerate(target_folds):
        test_ids = set(map(str, item["outer_test_ids"]))
        for patient_id in test_ids:
            if patient_id in test_fold_by_patient:
                raise ValueError("a target patient occurs in more than one outer test fold")
            test_fold_by_patient[patient_id] = fold
    if set(test_fold_by_patient) != set(map(str, patients)):
        raise ValueError("frozen outer test folds do not partition the target patient axis")

    selected_pairs = pair_keys.loc[pair_keys["drug_id"].astype(str).isin(target_drug_ids)]
    per_patient = selected_pairs.groupby(selected_pairs["patient_id"].astype(str))["drug_id"].nunique()
    coverage_rows = []
    for patient_id in sorted(map(str, patients)):
        n_observed = int(per_patient.get(patient_id, 0))
        missing_ids = sorted(target_drug_ids - set(
            selected_pairs.loc[selected_pairs["patient_id"].astype(str).eq(patient_id), "drug_id"].astype(str)
        ))
        coverage_rows.append({
            "patient_id": patient_id,
            "outer_test_fold": test_fold_by_patient[patient_id],
            "n_observed_drugs_from_frozen_nine": n_observed,
            "n_missing_drugs_from_frozen_nine": len(missing_ids),
            "missing_target_drug_ids": ";".join(missing_ids),
            "eligible_under_existing_minimum_three_rule": n_observed >= MIN_DRUGS_FOR_PATIENT_RANK,
        })
    coverage = pd.DataFrame(coverage_rows)
    counts = coverage["n_observed_drugs_from_frozen_nine"].value_counts().sort_index()
    histogram = {str(int(key)): int(value) for key, value in counts.items()}
    n_complete = int(coverage["n_observed_drugs_from_frozen_nine"].eq(len(target_drug_ids)).sum())
    n_minimum = int(coverage["eligible_under_existing_minimum_three_rule"].sum())

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent) as temp_name:
        staging = Path(temp_name)
        vector_path = staging / "source_reference_perturbation.npy"
        panel_path = staging / "source_panel_keys.json"
        coverage_path = staging / "target_key_coverage.csv"
        np.save(vector_path, reference_vectors, allow_pickle=False)
        panel_manifest = {
            "experiment_id": "EXP-048",
            "parent_experiment_id": "EXP-047",
            "drug_order": panel_rows,
            "source_cell": FROZEN_SOURCE_CELL,
            "source_fold": source_fold,
            "shared_exposure": {"dose_molar": str(FROZEN_DOSE_MOLAR), "time_hours": str(FROZEN_TIME_HOURS)},
            "gene_count": GENE_COUNT,
            "gene_axis_sha256": genes_sha,
            "seed_order": list(SEEDS),
            "condition_rows": [
                {"drug_index": index, "pert_id": mapping_by_index[index], **condition_by_pert[mapping_by_index[index]]}
                for index in ELIGIBLE_PANEL_INDEXES
            ],
            "aggregation": "per-condition mean over the three frozen seeds; equal-weight mean over the common source cell and exact shared exposure keys",
        }
        panel_path.write_text(json.dumps(panel_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        coverage.to_csv(coverage_path, index=False)
        vector_sha = sha256(vector_path)
        panel_sha_out = sha256(panel_path)
        coverage_sha = sha256(coverage_path)
        receipt = {
            "schema_version": 1,
            "experiment_id": "EXP-048",
            "parent_experiment_id": "EXP-047",
            "artifact_id": output_dir.name,
            "status": "SOURCE_OOF_VECTOR_PASS_TARGET_KEY_COVERAGE_REQUIRES_ONE_SOL_MISSINGNESS_RULING",
            "execution": {
                "runtime": "WSL2 Ubuntu + Conda drugscreening-gpu",
                "cuda_visible_devices": "",
                "formal_training": False,
                "tests_run": False,
                "gpu_hours": 0,
                "target_response_values_decoded": False,
                "target_pair_projection_columns": ["patient_id", "drug_id"],
                "target_prediction_or_performance_values_read": False,
            },
            "authorization": {
                "user_direction": "Current continuation instruction superseded the earlier 'nine-drug approval pending' status and authorized the scoped nine-drug MCF7 10uM x 24h exploratory derivative.",
                "sol_ruling": "experiments/records/EXP-047.md#2026-09-26-gpt-6-sol-medium-一次候选裁定",
                "original_exp047_14_drug_contract_preserved": True,
            },
            "source_oof": {
                "conditions_sha256": conditions_sha,
                "conditions_manifest_sha256": manifest_sha,
                "conditions_manifest_claimed_file_sha256": claimed_file_sha,
                "conditions_manifest_claim_matches_recomputed_file": conditions_file_checksum_discrepancy is False,
                "prior_crosswalk_receipt_conditions_sha256": PRIOR_CROSSWALK_RECORDED_CONDITIONS_SHA256,
                "prior_crosswalk_receipt_checksum_matches_current_file": PRIOR_CROSSWALK_RECORDED_CONDITIONS_SHA256 == conditions_sha,
                "conditions_checksum_note": "The immutable condition manifest matches the dataset-registry checksum, each EXP-037 fold summary, and the recomputed current file SHA. The earlier EXP-047 crosswalk receipt copied a different checksum string; this new receipt binds the current registered file and preserves the earlier value for traceability.",
                "genes_sha256": genes_sha,
                "source_fold": source_fold,
                "selected_condition_count": len(condition_by_pert),
                "n_selected_values": int(reference_vectors.size),
                "n_nonfinite": int((~np.isfinite(reference_vectors)).sum()),
                "shape": list(reference_vectors.shape),
                "vector_sha256": vector_sha,
                "panel_manifest_sha256": panel_sha_out,
                "source_fold_verifications": {str(key): value for key, value in fold_receipts.items()},
            },
            "target_key_coverage": {
                "target_input_manifest_sha256": sha256(target_manifest_path),
                "target_input_manifest_patient_pairs_sha256": target_manifest["files_sha256"]["patient_pairs.parquet"],
                "target_input_verification_sha256": sha256(TARGET_INPUTS / "verification.json"),
                "target_split_sha256": sha256(SPLIT),
                "patient_axis_count": len(patients),
                "unique_panel_drugs_present_globally": len(target_drug_ids),
                "patient_drug_count_histogram": histogram,
                "n_patients_with_all_nine": n_complete,
                "n_patients_meeting_existing_minimum_three_drug_rule": n_minimum,
                "coverage_table_sha256": coverage_sha,
                "coverage_rule": "shared global nine-drug candidate; observed target pairs only; no imputation and no patient-specific drug substitution; existing ranking metric requires at least three observed drugs",
            },
            "next_gate": "Resolve whether the patient-macro ranking may use each patient's observed subset of the fixed nine-drug panel when all 51 patients have 7/9 keys and none has 9/9; no training before this missingness ruling and final cost freeze.",
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = materialize(args.output_dir)
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "source_oof_shape": result["source_oof"]["shape"],
        "source_fold": result["source_oof"]["source_fold"],
        "target_patient_drug_count_histogram": result["target_key_coverage"]["patient_drug_count_histogram"],
        "n_patients_with_all_nine": result["target_key_coverage"]["n_patients_with_all_nine"],
        "n_patients_meeting_minimum_three": result["target_key_coverage"]["n_patients_meeting_existing_minimum_three_drug_rule"],
        "gpu_hours": result["execution"]["gpu_hours"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
