#!/usr/bin/env python3
"""Response-blind identity/exposure gate for an independent-object XPert fit.

Only the EXP-050 registered run directory is written. This program never imports
modeling code and never materializes transcription-response or sensitivity values.
Parquet reads are explicit identity-key projections.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_xpert_independent_object_gate_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
PREP = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared"
CONDITIONS = PREP / "conditions.parquet"
TRAIN_INDEX = PREP / "train121.npy"
DEV_INDEX = PREP / "validation.npy"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
PERT_INFO = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz"
GENE_RESOURCE = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2"
CTRP_PAIRS = ROOT / "artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3/patient_level_pairs.parquet"
PRISM_PAIRS = ROOT / "artifacts/experiments/EXP-050/20261001_lincs_prism_transfer_association_v1/patient_level_pairs.parquet"
IDENTITY_SPLIT_REF = ROOT / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"
EXP050_CONFIG = ROOT / "configs/experiments/EXP-050.yaml"
XPert_COST_RECEIPT = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/wrapper_xpert121/exit.json"

CONDITION_COLUMNS = ["condition_id", "source_row", "cell_id", "pert_id", "pert_dose", "pert_time"]
PAIR_COLUMNS = ["source_origin_id", "inchikey"]
MODEL_COLUMNS = ["ModelID", "PatientID", "CellLineName", "StrippedCellLineName"]
FULL_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
EXPECTED_MODEL_SHA256 = "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8"
PRE_EXECUTION_FAILURES = [
    {
        "stage": "initial WSL command construction",
        "failure": "PowerShell parsed an unescaped bash quote before WSL Python started.",
        "data_access": False,
        "gpu_hours": 0.0,
    },
    {
        "stage": "optional historical ledger file discovery",
        "failure": "No gpu_ledger.jsonl exists beside the prior XPert wrapper; no data file was read.",
        "recovery": "Use the existing wrapper exit receipt, which contains only exit code, elapsed seconds, and GPU hours charged.",
        "data_access": False,
        "gpu_hours": 0.0,
    }
]
RECORDED_DEVIATIONS = [
    {
        "deviation_id": "D1_HISTORICAL_SUMMARY_METRIC_PRINTED",
        "stage": "preflight cost-source inspection before the gate command",
        "occurred_before_gate_execution": True,
        "details": (
            "An exploratory preflight command printed an existing XPert summary that contained one aggregate "
            "historical validation metric. Its value is intentionally not repeated here. No response matrix, "
            "individual response label, treated expression, or direct drug-sensitivity value was read. Cost in this "
            "gate was obtained from the separate runtime receipt. The available audit cannot establish whether this "
            "incidental exposure influenced a prior human choice."
        ),
        "selection_influence_status": "NOT_ESTABLISHED_BY_AUDIT",
        "operator_reported_no_selection_use": True,
    },
    {
        "deviation_id": "D2_SPLIT_ASSIGNMENTS_IDENTITY_METADATA_EXPOSED",
        "stage": "early preflight inspection before the gate command",
        "occurred_before_gate_execution": True,
        "details": (
            "An early exploratory inspection parsed the full split_assignments mapping and printed its opening "
            "records, exposing locked_eval identity/role metadata. No outer/locked-evaluation result values, response "
            "matrix, individual response labels, treated expression, or direct drug-sensitivity numeric values were "
            "read. The current gate script itself does not parse split_assignments and projects only the approved "
            "train121.npy and validation.npy membership masks onto condition metadata. Static code-path evidence "
            "does not establish whether the earlier exploratory exposure influenced a prior human choice."
        ),
        "locked_eval_identity_or_role_metadata_exposed": True,
        "outer_or_locked_eval_result_values_read": False,
        "response_or_individual_label_values_read": False,
        "direct_sensitivity_numeric_values_read": False,
        "selection_influence_status": "NOT_ESTABLISHED_BY_AUDIT",
        "operator_reported_no_selection_use": True,
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_gate_block_sha256(path: Path) -> str:
    """Hash only the registered gate block, without parsing unrelated result metrics."""
    raw = path.read_bytes()
    marker = b"xpert_independent_object_gate_v1:\n"
    start = raw.find(marker)
    if start < 0:
        raise ValueError("registered response-blind independent-object gate block is missing")
    start += len(marker)
    lines = raw[start:].splitlines(keepends=True)
    block = bytearray(marker)
    for line in lines:
        if line.strip() and not line[:1].isspace() and not line.lstrip().startswith(b"#"):
            break
        block.extend(line)
    return sha256_bytes(bytes(block))


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def record_postrun_deviations() -> int:
    """Update only this run's response-blind receipts; never opens source inputs."""
    manifest_path = OUT / "run_manifest.json"
    snapshot_path = OUT / "protocol_snapshot.json"
    support_path = OUT / "train_dev_drug_support.tsv"
    counts_path = OUT / "excluded_source_identity_counts.json"
    required = [manifest_path, snapshot_path, support_path, counts_path,
                OUT / "common_object_exposure_support.tsv", OUT / "cost_preflight.json"]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"post-run receipt update requires existing gate outputs: {missing}")

    # These are this gate's own response-blind products, not source identity or label files.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    support = pd.read_csv(
        support_path,
        sep="\t",
        usecols=["full_inchikey", "split_role", "n_independent_conditions_after_exclusion"],
        dtype={"full_inchikey": "string", "split_role": "string"},
    )
    supported = support.loc[support["n_independent_conditions_after_exclusion"].gt(0)]
    train_keys = set(supported.loc[supported["split_role"].eq("train"), "full_inchikey"].dropna().astype(str))
    dev_keys = set(supported.loc[supported["split_role"].eq("inner_dev"), "full_inchikey"].dropna().astype(str))
    all_keys = set(support["full_inchikey"].dropna().astype(str))
    shared_supported_keys = train_keys & dev_keys

    recorded_at = now_iso()
    protocol_mtime = datetime.fromtimestamp(EXP050_CONFIG.stat().st_mtime, timezone.utc).isoformat()
    source_key_artifact_mtimes = {
        "CTRPv2": datetime.fromtimestamp(CTRP_PAIRS.stat().st_mtime, timezone.utc).isoformat(),
        "PRISM_24Q2": datetime.fromtimestamp(PRISM_PAIRS.stat().st_mtime, timezone.utc).isoformat(),
    }
    previous_status = manifest.get("pre_deviation_gate_status", manifest.get("status"))
    prior_deviation_path = OUT / "attempt_001_deviation_record.json"
    prior_deviation = json.loads(prior_deviation_path.read_text(encoding="utf-8")) if prior_deviation_path.is_file() else {}
    previous_snapshot_sha256 = prior_deviation.get("previous_protocol_snapshot_sha256", sha256(snapshot_path))
    previous_output_hashes = manifest.get("pre_deviation_output_sha256", manifest.get("output_sha256", {}))
    deviation_record = {
        "run_id": RUN_ID,
        "recorded_at_utc": recorded_at,
        "gate_execution_started_at_utc": manifest.get("started_at_utc"),
        "deviations": RECORDED_DEVIATIONS,
        "selection_influence_status": "NOT_ESTABLISHED_BY_AUDIT",
        "operator_reported_no_selection_use": True,
        "scope_note": (
            "The executing agent reports that neither preflight item was used in candidate, dose, drug, identity, or "
            "gate selection. The registered rule and source-key artifacts predate gate execution, and the current "
            "static gate calculation path does not read the historical summary or split_assignments records. This "
            "supports the computational path but cannot establish that prior human cognition was unaffected; retain "
            "manager review before a new training attempt."
        ),
        "forbidden_content_not_read": {
            "response_matrix_or_treated_expression_values": True,
            "individual_response_labels": True,
            "outer_or_locked_eval_result_values": True,
            "direct_sensitivity_numeric_values": True,
            "pdo_processed_labels": True,
        },
        "previous_gate_status": previous_status,
        "previous_protocol_snapshot_sha256": previous_snapshot_sha256,
        "previous_output_sha256": previous_output_hashes,
        "protocol_gate_block_sha256": manifest.get("registered_gate_block_sha256"),
        "protocol_config_modified_at_utc": protocol_mtime,
        "source_key_artifact_modified_at_utc": source_key_artifact_mtimes,
        "gate_script_sha256_at_recording": sha256(Path(__file__).resolve()),
        "initial_gate_runtime_script_sha256": None,
        "initial_gate_runtime_script_sha256_status": "NOT_CAPTURED_DURING_INITIAL_GATE_EXECUTION",
        "source_patient_id_set_sha256": manifest.get("frozen_source_patient_id_set_sha256"),
        "source_patient_full_inchikey_pair_set_sha256": manifest.get("frozen_source_patient_full_inchikey_pair_set_sha256"),
    }
    deviation_path = OUT / "attempt_001_deviation_record.json"
    write_json(deviation_path, deviation_record)

    # Freeze the pre-training selection scope using only the already-created response-blind support table.
    snapshot["status"] = "GO_TO_MANAGER_FOR_NEXT_FROZEN_ATTEMPT"
    snapshot["selection_affected_by_deviation"] = "NOT_ESTABLISHED_BY_AUDIT"
    snapshot["freeze_evidence"]["preflight_deviation_selection_influence_status"] = "NOT_ESTABLISHED_BY_AUDIT"
    snapshot["freeze_evidence"]["historical_summary_deviation_changed_frozen_source_object_set"] = "NOT_ESTABLISHED_BY_AUDIT"
    snapshot["freeze_evidence"]["operator_reported_no_selection_use"] = True
    snapshot["freeze_evidence"]["manager_review_required_before_new_training_attempt"] = True
    snapshot["freeze_evidence"]["protocol_config_modified_at_utc"] = protocol_mtime
    snapshot["freeze_evidence"]["source_key_artifacts_modified_at_utc"] = source_key_artifact_mtimes
    snapshot["freeze_evidence"]["gate_script_sha256_at_recording"] = sha256(Path(__file__).resolve())
    snapshot["freeze_evidence"]["initial_gate_runtime_script_sha256"] = None
    snapshot["freeze_evidence"]["initial_gate_runtime_script_sha256_status"] = "NOT_CAPTURED_DURING_INITIAL_GATE_EXECUTION"
    snapshot["forbidden_access"] = {
        "condition_delta_or_treated_expression_values_read": False,
        "individual_response_label_values_read": False,
        "direct_sensitivity_numeric_label_values_read": False,
        "outer_or_locked_eval_result_values_read": False,
        "locked_eval_identity_or_role_metadata_exposed_preflight": True,
        "pdo_processed_label_values_read": False,
        "current_gate_script_parsed_split_assignments": False,
    }
    snapshot["training_scope_freeze"] = {
        "recorded_at_utc": recorded_at,
        "status": "SCOPE_RECORDED_FOR_MANAGER_FREEZE; TRAINING_NOT_STARTED",
        "source_objects_excluded_from_train_dev_scale_fit_early_stopping_and_selection": int(
            snapshot["summary"]["source_patient_ids_excluded"]
        ),
        "train": {
            "independent_patient_ids": int(snapshot["summary"]["remaining_train_patient_ids"]),
            "exact_10uM_24h_conditions": int(snapshot["summary"]["remaining_exact_10uM_24h_train_conditions"]),
            "full_inchikeys_with_remaining_exact_exposure_support": len(train_keys),
            "unresolved_or_ambiguous_identity_conditions_excluded": 7468,
        },
        "inner_dev": {
            "independent_patient_ids": int(snapshot["summary"]["remaining_inner_dev_patient_ids"]),
            "exact_10uM_24h_conditions": int(snapshot["summary"]["remaining_exact_10uM_24h_inner_dev_conditions"]),
            "full_inchikeys_with_remaining_exact_exposure_support": len(dev_keys),
            "unresolved_or_ambiguous_identity_conditions_excluded": 2015,
        },
        "direct_source_full_inchikey_count": len(all_keys),
        "train_inner_dev_common_supported_full_inchikey_count": len(shared_supported_keys),
        "train_inner_dev_common_supported_full_inchikeys": sorted(shared_supported_keys),
        "train_inner_dev_common_supported_full_inchikey_set_sha256": sha256_bytes(
            ("\n".join(sorted(shared_supported_keys)) + "\n").encode("utf-8")
        ),
        "support_table_sha256_used_to_verify_common_key_intersection": sha256(support_path),
        "common_key_intersection_method": (
            "In WSL2 Conda drugscreening-gpu, read only full_inchikey, split_role, and "
            "n_independent_conditions_after_exclusion from this run's response-blind support TSV; count keys with "
            "positive remaining support in both train and inner_dev. No source identity or label file was opened."
        ),
        "selection_and_scoring_rule": (
            "Model selection, early stopping and the predeclared main score may use only inner-dev conditions for "
            "full InChIKeys that have at least one remaining exact 10 uM x 24 h condition in both train and inner-dev. "
            "Training and all fitted preprocessing/scaling use train only. Apply the same common-key mask to all arms."
        ),
        "training_started": False,
        "gpu_hours_for_gate": 0.0,
    }
    snapshot["deviation_handling"] = (
        "Two preflight deviations are recorded in attempt_001_deviation_record.json. One existing aggregate XPert "
        "validation metric was incidentally printed; the second exposed locked_eval identity/role metadata from the "
        "opening split_assignments records. No response matrix, individual label, treated-expression value, outer/locked "
        "result value, PDO processed label, or direct-sensitivity numeric value was read. The current gate script uses "
        "only train121.npy/validation.npy membership masks and response-blind identity/exposure projections. Whether "
        "preflight exposure influenced a prior human choice remains unestablished; the scoped gate is submitted to the "
        "manager before any training attempt."
    )
    snapshot["deviation_record"] = "attempt_001_deviation_record.json"
    write_json(snapshot_path, snapshot)

    manifest.update({
        "status": "GO_TO_MANAGER_FOR_NEXT_FROZEN_ATTEMPT",
        "recorded_deviations": RECORDED_DEVIATIONS,
        "deviation_selection_influence_status": "NOT_ESTABLISHED_BY_AUDIT",
        "selection_affected_by_deviation": "NOT_ESTABLISHED_BY_AUDIT",
        "operator_reported_no_selection_use": True,
        "manager_review_required_before_new_training_attempt": True,
        "pre_deviation_gate_status": previous_status,
        "pre_deviation_output_sha256": previous_output_hashes,
        "pre_deviation_protocol_snapshot_sha256": previous_snapshot_sha256,
        "initial_gate_runtime_script_sha256": None,
        "initial_gate_runtime_script_sha256_status": "NOT_CAPTURED_DURING_INITIAL_GATE_EXECUTION",
        "gate_script_sha256_current_revision": sha256(Path(__file__).resolve()),
        "deviation_record": deviation_path.name,
        "deviation_record_sha256": sha256(deviation_path),
        "protocol_snapshot_sha256_after_deviation_update": sha256(snapshot_path),
        "output_sha256": {
            name: sha256(OUT / name)
            for name in ["protocol_snapshot.json", "attempt_001_deviation_record.json",
                         "common_object_exposure_support.tsv", "excluded_source_identity_counts.json",
                         "train_dev_drug_support.tsv", "cost_preflight.json"]
        },
        "recorded_at_utc": recorded_at,
        "training_started": False,
        "gpu_hours": 0.0,
    })
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "selection_influence_status": manifest["deviation_selection_influence_status"],
        "training_scope": snapshot["training_scope_freeze"],
        "protocol_snapshot_sha256": sha256(snapshot_path),
        "deviation_record_sha256": sha256(deviation_path),
        "run_manifest_sha256": sha256(manifest_path),
        "output_sha256": manifest["output_sha256"],
    }, ensure_ascii=False))
    return 0


def norm(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def read_pair_identity_projection(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)
    missing = set(PAIR_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"identity-only pair projection missing columns: {sorted(missing)}")
    table = pq.read_table(path, columns=PAIR_COLUMNS)
    if table.column_names != PAIR_COLUMNS:
        raise ValueError(f"unexpected projected column order: {table.column_names}")
    frame = table.to_pandas()
    frame["source_origin_id"] = frame["source_origin_id"].astype("string").str.strip()
    frame["inchikey"] = frame["inchikey"].astype("string").str.strip().str.upper()
    frame = frame.dropna(subset=PAIR_COLUMNS)
    frame = frame.loc[frame["source_origin_id"].ne("") & frame["inchikey"].ne("")].copy()
    frame["full_key_valid"] = frame["inchikey"].map(lambda x: bool(FULL_KEY.fullmatch(str(x))))
    return frame


def read_identity_source_hash_prefix(path: Path) -> str:
    # Read only the leading source metadata block; do not parse split_assignments.
    prefix = path.open("r", encoding="utf-8").read(4096)
    match = re.search(r'"model_sha256"\s*:\s*"([0-9a-f]{64})"', prefix)
    if not match:
        raise ValueError("could not read frozen Model.csv digest from the identity-split source header")
    return match.group(1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    existing_manifest = OUT / "run_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("status") != "RUNNING_RESPONSE_BLIND_GATE":
            raise RuntimeError(
                "This run ID already has a gate receipt; refusing to reread inputs or overwrite outputs. "
                "Use --record-postrun-deviations for receipt-only updates."
            )
    started = now_iso()
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "RUNNING_RESPONSE_BLIND_GATE",
        "started_at_utc": started,
        "environment_requirement": "WSL2 Conda drugscreening-gpu",
        "response_values_read": False,
        "direct_sensitivity_values_read": False,
        "transcript_response_value_columns_projected": [],
        "direct_sensitivity_value_columns_projected": [],
        "training_started": False,
        "gpu_hours": 0.0,
        "pre_execution_failures": PRE_EXECUTION_FAILURES,
        "recorded_deviations": RECORDED_DEVIATIONS,
        "selection_affected_by_deviation": "NOT_ESTABLISHED_BY_AUDIT",
        "operator_reported_no_selection_use": True,
        "next_command": (
            "After manager accepts the response-blind gate, freeze one XPert fit/early-stop configuration and "
            "attempt reservation within the registered 10 GPU-hour subcap; do not run this gate again as a substitute."
        ),
    }
    write_json(OUT / "run_manifest.json", manifest)

    try:
        # Bind the identity-only DepMap table to the frozen split's source digest.
        frozen_model_digest = read_identity_source_hash_prefix(IDENTITY_SPLIT_REF)
        actual_model_digest = sha256(MODEL_CSV)
        if frozen_model_digest != EXPECTED_MODEL_SHA256 or actual_model_digest != frozen_model_digest:
            raise ValueError("DepMap Model.csv does not match the frozen identity-source checksum")

        model = pd.read_csv(MODEL_CSV, usecols=MODEL_COLUMNS, dtype="string", keep_default_na=False)
        model = model.loc[
            model["ModelID"].str.strip().ne("") & model["PatientID"].str.strip().ne("")
        ].copy()
        model["ModelID"] = model["ModelID"].str.strip()
        model["PatientID"] = model["PatientID"].str.strip()
        model["StrippedCellLineName"] = model["StrippedCellLineName"].str.strip()
        model["CellLineName"] = model["CellLineName"].str.strip()
        model_by_patient: dict[str, set[str]] = defaultdict(set)
        model_by_stripped: dict[str, list[dict[str, str]]] = defaultdict(list)
        model_by_cell_name: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in model.itertuples(index=False):
            rec = {"model_id": norm(row.ModelID), "patient_id": norm(row.PatientID)}
            model_by_patient[rec["patient_id"]].add(rec["model_id"])
            if norm(row.StrippedCellLineName):
                model_by_stripped[norm(row.StrippedCellLineName).casefold()].append(rec)
            if norm(row.CellLineName):
                model_by_cell_name[norm(row.CellLineName).casefold()].append(rec)

        # Conditions are response-blind metadata and are read with an explicit six-column projection.
        condition_schema = pq.read_schema(CONDITIONS)
        if set(CONDITION_COLUMNS) - set(condition_schema.names):
            raise ValueError("prepared conditions do not meet the response-blind key/exposure schema")
        cond = pq.read_table(CONDITIONS, columns=CONDITION_COLUMNS).to_pandas()
        if cond.columns.tolist() != CONDITION_COLUMNS:
            raise ValueError(f"unexpected conditions projection: {cond.columns.tolist()}")
        if cond["condition_id"].isna().any() or cond["condition_id"].astype(str).duplicated().any():
            raise ValueError("condition_id is missing or duplicated in response-blind metadata")
        train_idx = np.load(TRAIN_INDEX, allow_pickle=False)
        dev_idx = np.load(DEV_INDEX, allow_pickle=False)
        if train_idx.ndim != 1 or dev_idx.ndim != 1 or train_idx.dtype.kind not in "iu" or dev_idx.dtype.kind not in "iu":
            raise ValueError("split membership must be one-dimensional integer indices")
        train_set, dev_set = set(map(int, train_idx)), set(map(int, dev_idx))
        if len(train_set) != len(train_idx) or len(dev_set) != len(dev_idx) or train_set.intersection(dev_set):
            raise ValueError("train/inner-dev indices contain duplicates or overlap")
        all_members = train_set | dev_set
        if (min(all_members) < 0 or max(all_members) >= len(cond)
                or len(all_members) != len(cond) or all_members != set(range(len(cond)))):
            raise ValueError("registered train/inner-dev membership does not exactly cover prepared conditions")
        roles = np.full(len(cond), "", dtype=object)
        roles[list(train_set)] = "train"
        roles[list(dev_set)] = "inner_dev"
        cond["split_role"] = roles
        cond["condition_id"] = cond["condition_id"].astype("string").str.strip()
        cond["cell_id"] = cond["cell_id"].astype("string").str.strip()
        cond["pert_id"] = cond["pert_id"].astype("string").str.strip()
        cond["pert_dose"] = pd.to_numeric(cond["pert_dose"], errors="coerce")
        cond["pert_time"] = pd.to_numeric(cond["pert_time"], errors="coerce")
        if cond[["condition_id", "cell_id", "pert_id", "split_role"]].isna().any().any():
            raise ValueError("required response-blind condition key is null")

        # Resolve each LINCS source-line label through exact DepMap stripped/cell-line names.
        identity_map: dict[str, dict[str, Any]] = {}
        for cell in sorted(set(cond["cell_id"].astype(str))):
            k = cell.strip().casefold()
            candidates = model_by_stripped.get(k, [])
            match_method = "exact_stripped_cell_line_name"
            if not candidates:
                candidates = model_by_cell_name.get(k, [])
                match_method = "exact_cell_line_name_fallback"
            model_ids = sorted({r["model_id"] for r in candidates})
            patient_ids = sorted({r["patient_id"] for r in candidates})
            identity_map[cell] = {
                "model_ids": model_ids,
                "patient_ids": patient_ids,
                "status": "RESOLVED" if len(patient_ids) == 1 else ("AMBIGUOUS" if patient_ids else "UNMAPPED"),
                "match_method": match_method if candidates else "NO_EXACT_NAME_MATCH",
            }

        # The two permitted direct-source cohorts are unioned for exclusion: an object
        # appearing in either assay must not contribute to XPert fitting or selection.
        ctrp = read_pair_identity_projection(CTRP_PAIRS)
        prism = read_pair_identity_projection(PRISM_PAIRS)
        ctrp_pairs = {(str(r.source_origin_id), str(r.inchikey)) for r in ctrp.itertuples(index=False)}
        prism_pairs = {(str(r.source_origin_id), str(r.inchikey)) for r in prism.itertuples(index=False)}
        ctrp_patients = {p for p, _ in ctrp_pairs}
        prism_patients = {p for p, _ in prism_pairs}
        target_patients = ctrp_patients | prism_patients
        target_drugs = {k for _, k in (ctrp_pairs | prism_pairs)}
        frozen_patient_id_digest = sha256_bytes(("\n".join(sorted(target_patients)) + "\n").encode("utf-8"))
        frozen_patient_key_digest = sha256_bytes(("\n".join(f"{p}\t{k}" for p, k in sorted(ctrp_pairs | prism_pairs)) + "\n").encode("utf-8"))
        frozen_gate_digest = frozen_gate_block_sha256(EXP050_CONFIG)
        bad_target_keys = sorted(k for k in target_drugs if not FULL_KEY.fullmatch(k))

        patient_unmapped_to_depmap = sorted(p for p in target_patients if p not in model_by_patient)
        target_models = {p: sorted(model_by_patient.get(p, set())) for p in sorted(target_patients)}
        target_condition_cells: dict[str, list[str]] = defaultdict(list)
        cond_patient: list[str | None] = []
        unresolved_cells = set()
        ambiguous_cells = set()
        for cell in cond["cell_id"].astype(str):
            identity = identity_map[cell]
            if identity["status"] == "RESOLVED":
                patient = identity["patient_ids"][0]
                cond_patient.append(patient)
                if patient in target_patients:
                    target_condition_cells[patient].append(cell)
            else:
                cond_patient.append(None)
                (ambiguous_cells if identity["status"] == "AMBIGUOUS" else unresolved_cells).add(cell)
        cond["patient_id"] = pd.Series(cond_patient, dtype="string")
        cond["identity_status"] = cond["cell_id"].map(lambda c: identity_map[str(c)]["status"])
        is_target_patient = cond["patient_id"].isin(target_patients).to_numpy()
        is_unresolved = cond["identity_status"].ne("RESOLVED").to_numpy()
        safe_for_model_support = ~(is_target_patient | is_unresolved)
        cond["excluded_from_fit_and_selection"] = ~safe_for_model_support
        cond["exclusion_reason"] = np.where(is_target_patient, "direct_sensitivity_source_object",
                                             np.where(is_unresolved, "identity_unresolved_or_ambiguous", ""))

        # Verify exact structure keys via registered LINCS perturbagen metadata only.
        pert_info = pd.read_csv(
            PERT_INFO, sep="\t", compression="gzip", usecols=["pert_id", "inchi_key"],
            dtype="string", keep_default_na=False,
        )
        pert_info["pert_id"] = pert_info["pert_id"].str.strip()
        pert_info["inchi_key"] = pert_info["inchi_key"].str.strip().str.upper()
        pert_info = pert_info.loc[pert_info["pert_id"].ne("") & pert_info["inchi_key"].ne("")].copy()
        pert_keys_by_id = pert_info.groupby("pert_id")["inchi_key"].agg(lambda x: sorted(set(x))).to_dict()
        valid_pert_id_to_key = {pid: keys[0] for pid, keys in pert_keys_by_id.items()
                                if len(keys) == 1 and FULL_KEY.fullmatch(keys[0])}
        key_to_pert_ids: dict[str, list[str]] = defaultdict(list)
        for pert_id, key in valid_pert_id_to_key.items():
            key_to_pert_ids[key].append(pert_id)
        for key in key_to_pert_ids:
            key_to_pert_ids[key].sort()
        condition_key = cond["pert_id"].astype(str).map(valid_pert_id_to_key).fillna("")
        cond["inchikey"] = condition_key
        is_exact_exposure = np.isclose(cond["pert_dose"].to_numpy(dtype=float), 10.0, rtol=0.0, atol=1e-12)
        is_exact_exposure &= np.isclose(cond["pert_time"].to_numpy(dtype=float), 24.0, rtol=0.0, atol=1e-12)
        cond["exact_10uM_24h"] = is_exact_exposure

        # Response-blind 978-axis receipt: only ordered gene IDs and the frozen axis digest.
        genes = json.loads((GENE_RESOURCE / "genes.json").read_text(encoding="utf-8"))
        quality = json.loads((GENE_RESOURCE / "quality_report.json").read_text(encoding="utf-8"))
        if not isinstance(genes, list):
            raise ValueError("LINCS gene-axis resource is not a list")
        gene_axis_count = len(genes)
        gene_axis_digest = quality.get("gene_axis", {}).get("sha256")
        gene_axis_verified = gene_axis_count == 978 and gene_axis_digest == "8cc8d6ade494209a9f323a94450e02c3a74d1ea3ae61a75ed4c084a9ab80502c"

        # Object x exact full-key support; target LINCS condition IDs remain held out.
        pair_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
        for pair in ctrp_pairs:
            pair_sources[pair].add("CTRPv2")
        for pair in prism_pairs:
            pair_sources[pair].add("PRISM_24Q2")
        exact_target = cond.loc[
            cond["patient_id"].isin(target_patients) & cond["exact_10uM_24h"] & cond["inchikey"].ne("")
        ].copy()
        exposure_groups: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)
        for _, row in exact_target.iterrows():
            exposure_groups[(str(row["patient_id"]), str(row["inchikey"]))].append(row)

        exposure_rows = []
        for patient, key in sorted(pair_sources):
            hits = exposure_groups.get((patient, key), [])
            exposure_rows.append({
                "source_origin_id": patient,
                "depmap_model_ids": ";".join(target_models.get(patient, [])),
                "lincs_cell_ids": ";".join(sorted(set(target_condition_cells.get(patient, [])))),
                "full_inchikey": key,
                "direct_sensitivity_sources": ";".join(sorted(pair_sources[(patient, key)])),
                "n_exact_10uM_24h_lincs_conditions": len(hits),
                "n_train_conditions": sum(str(r["split_role"]) == "train" for r in hits),
                "n_inner_dev_conditions": sum(str(r["split_role"]) == "inner_dev" for r in hits),
                "linqs_pert_ids": ";".join(sorted({str(r["pert_id"]) for r in hits})),
                "condition_ids": ";".join(sorted({str(r["condition_id"]) for r in hits})),
                "support_status": "SUPPORTED" if hits else "NO_EXACT_10uM_24h_CONDITION",
                "excluded_from_xpert_fit_and_selection": True,
            })
        exposure_df = pd.DataFrame(exposure_rows)
        exposure_df.to_csv(OUT / "common_object_exposure_support.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

        # Per-drug 10uM x 24h train/dev support after complete source-object exclusion.
        support_rows = []
        for key in sorted(target_drugs):
            pids = key_to_pert_ids.get(key, [])
            keyed = cond.loc[cond["pert_id"].astype(str).isin(pids) & cond["exact_10uM_24h"]].copy()
            for role in ["train", "inner_dev"]:
                role_rows = keyed.loc[keyed["split_role"].eq(role)]
                kept = role_rows.loc[~role_rows["excluded_from_fit_and_selection"]]
                heldout = role_rows.loc[role_rows["patient_id"].isin(target_patients)]
                support_rows.append({
                    "full_inchikey": key,
                    "lincs_pert_ids": ";".join(pids),
                    "split_role": role,
                    "n_exact_10uM_24h_conditions_before_exclusion": int(len(role_rows)),
                    "n_independent_conditions_after_exclusion": int(len(kept)),
                    "n_independent_lincs_cell_ids_after_exclusion": int(kept["cell_id"].nunique()),
                    "n_independent_patient_ids_after_exclusion": int(kept["patient_id"].nunique()),
                    "n_source_object_conditions_removed": int(len(heldout)),
                    "n_unresolved_identity_conditions_removed": int(role_rows["identity_status"].ne("RESOLVED").sum()),
                    "condition_ids_after_exclusion": ";".join(sorted(kept["condition_id"].astype(str))),
                    "support_status": "SUPPORTED" if len(kept) else "NO_REMAINING_CONDITION",
                })
        support_df = pd.DataFrame(support_rows)
        support_df.to_csv(OUT / "train_dev_drug_support.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

        # Count complete source objects and retained train/dev support without any response features.
        split_counts: dict[str, Any] = {}
        for role in ["train", "inner_dev"]:
            part = cond.loc[cond["split_role"].eq(role)]
            kept = part.loc[~part["excluded_from_fit_and_selection"]]
            removed_target = part.loc[part["patient_id"].isin(target_patients)]
            split_counts[role] = {
                "conditions_before_exclusion": int(len(part)),
                "conditions_removed_for_direct_sensitivity_patient": int(len(removed_target)),
                "conditions_removed_for_unresolved_or_ambiguous_identity": int(part["identity_status"].ne("RESOLVED").sum()),
                "conditions_remaining_for_fit_or_selection": int(len(kept)),
                "source_lincs_cell_ids_before_exclusion": int(part["cell_id"].nunique()),
                "source_lincs_cell_ids_remaining": int(kept["cell_id"].nunique()),
                "mapped_patient_ids_before_exclusion": int(part["patient_id"].nunique()),
                "independent_patient_ids_remaining": int(kept["patient_id"].nunique()),
                "target_patient_ids_leaking_after_exclusion": int(kept["patient_id"].isin(target_patients).sum()),
                "exact_10uM_24h_conditions_before_exclusion": int((part["exact_10uM_24h"]).sum()),
                "exact_10uM_24h_conditions_remaining": int((kept["exact_10uM_24h"]).sum()),
            }
        counts = {
            "run_id": RUN_ID,
            "cohort_rule": "union of identity-only source_origin_id sets from the permitted CTRPv2 and PRISM 24Q2 keyed support outputs; no pair is selected by a numeric response value in this gate",
            "ct_v2_unique_patient_ids": len(ctrp_patients),
            "prism_unique_patient_ids": len(prism_patients),
            "patient_ids_shared_across_both_direct_sources": len(ctrp_patients & prism_patients),
            "source_patient_ids_to_exclude_union": len(target_patients),
            "unmapped_source_patient_ids_in_depmmap_model_release": patient_unmapped_to_depmap,
            "depmap_model_ids_by_source_patient_id": target_models,
            "n_direct_source_exact_full_inchikeys_union": len(target_drugs),
            "n_malformed_direct_source_keys": len(bad_target_keys),
            "malformed_direct_source_keys": bad_target_keys,
            "n_source_patient_x_full_key_pairs_union": len(pair_sources),
            "n_source_patient_x_full_key_pairs_with_exact_10uM_24h_lincs_condition": int(exposure_df["n_exact_10uM_24h_lincs_conditions"].gt(0).sum()) if not exposure_df.empty else 0,
            "n_source_patient_x_full_key_pairs_without_exact_10uM_24h_lincs_condition": int(exposure_df["n_exact_10uM_24h_lincs_conditions"].eq(0).sum()) if not exposure_df.empty else 0,
            "direct_source_patients_with_any_exact_lincs_condition": int(exact_target["patient_id"].nunique()),
            "direct_source_patients_without_any_exact_lincs_condition": sorted(target_patients - set(exact_target["patient_id"].astype(str))),
            "depmap_model_ids_excluded_from_source_fit_and_selection": sorted({m for p in target_patients for m in target_models.get(p, [])}),
            "target_lincs_cell_ids_in_prepared_train_dev": {p: sorted(set(cells)) for p, cells in sorted(target_condition_cells.items())},
            "unmapped_condition_cell_ids": sorted(unresolved_cells),
            "ambiguous_condition_cell_ids": sorted(ambiguous_cells),
            "split_support": split_counts,
            "gene_axis": {
                "gene_count": gene_axis_count,
                "expected_gene_axis_sha256": "8cc8d6ade494209a9f323a94450e02c3a74d1ea3ae61a75ed4c084a9ab80502c",
                "registered_gene_axis_sha256": gene_axis_digest,
                "full_978_axis_verified": bool(gene_axis_verified),
            },
            "linqs_structure_axis": {
                "pert_info_projected_columns": ["pert_id", "inchi_key"],
                "unique_valid_pert_ids": len(valid_pert_id_to_key),
                "target_direct_keys_with_exact_lincs_pert_id": sum(bool(key_to_pert_ids.get(k)) for k in target_drugs),
                "target_direct_keys_without_exact_lincs_pert_id": sorted(k for k in target_drugs if not key_to_pert_ids.get(k)),
                "target_exact_10uM_24h_keys_remaining_in_train": int(support_df.loc[(support_df["split_role"] == "train") & (support_df["n_independent_conditions_after_exclusion"] > 0), "full_inchikey"].nunique()) if not support_df.empty else 0,
                "target_exact_10uM_24h_keys_remaining_in_inner_dev": int(support_df.loc[(support_df["split_role"] == "inner_dev") & (support_df["n_independent_conditions_after_exclusion"] > 0), "full_inchikey"].nunique()) if not support_df.empty else 0,
            },
            "gpu_hours": 0.0,
            "training_started": False,
        }
        write_json(OUT / "excluded_source_identity_counts.json", counts)

        # Cost uses only an append-only runtime/charge receipt, not model summaries or scores.
        cost_receipt = json.loads(XPert_COST_RECEIPT.read_text(encoding="utf-8"))
        allowed_cost_fields = {k: cost_receipt.get(k) for k in ["exit_code", "elapsed_seconds", "gpu_hours_charged"]}
        if allowed_cost_fields["exit_code"] != 0 or not isinstance(allowed_cost_fields["gpu_hours_charged"], (int, float)):
            raise ValueError("historical XPert runtime receipt is incomplete for cost estimation")
        reference_gpu_hours = float(allowed_cost_fields["gpu_hours_charged"])
        conservative_gpu_hours = reference_gpu_hours * 1.30
        cost = {
            "status": "ESTIMATE_WITHIN_REGISTERED_10_GPUH_SUBCAP" if conservative_gpu_hours <= 10.0 else "NO_GO_COST_EXCEEDS_SUBCAP",
            "gpu_hours_for_this_gate": 0.0,
            "training_started": False,
            "historical_xpert_run_runtime_receipt": str(XPert_COST_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "runtime_receipt_fields_read": allowed_cost_fields,
            "historical_hardware_uuid": "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871",
            "estimate_method": "Use one complete prior XPert-121 A5000 charged duration as the reference; apply a 30% planning margin for revised exclusions, loader and scheduling variability. This is an operational cost estimate, not a claim that the next fit has been run.",
            "reference_gpu_hours": reference_gpu_hours,
            "planning_margin_fraction": 0.30,
            "conservative_estimated_gpu_hours": conservative_gpu_hours,
            "registered_next_xpert_attempt_cap_gpu_hours": 10.0,
            "headroom_to_attempt_cap_gpu_hours": 10.0 - conservative_gpu_hours,
            "historical_code_lineage_limitation": "The runtime receipt identifies the prior XPert tier and GPU charge but does not bind a complete code hash to this gate; recheck live device and current run ledger immediately before any later reservation.",
            "forbidden_runtime_or_score_summaries_read": False,
        }
        write_json(OUT / "cost_preflight.json", cost)

        hard_reasons = []
        if patient_unmapped_to_depmap:
            hard_reasons.append("one_or_more_direct_source_patient_ids_do_not_resolve_in_frozen_DepMap_Model_release")
        if bad_target_keys:
            hard_reasons.append("one_or_more_direct_source_compound_keys_are_not_full_InChIKeys")
        if not gene_axis_verified:
            hard_reasons.append("full_978_gene_axis_not_verified")
        if split_counts["train"]["conditions_remaining_for_fit_or_selection"] <= 0:
            hard_reasons.append("no_training_conditions_remain_after_source_object_exclusion")
        if split_counts["inner_dev"]["conditions_remaining_for_fit_or_selection"] <= 0:
            hard_reasons.append("no_non_target_inner_development_conditions_remain")
        if split_counts["train"]["target_patient_ids_leaking_after_exclusion"] or split_counts["inner_dev"]["target_patient_ids_leaking_after_exclusion"]:
            hard_reasons.append("direct_sensitivity_source_patient_leakage_remains")
        if cost["status"] != "ESTIMATE_WITHIN_REGISTERED_10_GPUH_SUBCAP":
            hard_reasons.append("conservative_XPert_cost_estimate_exceeds_registered_attempt_cap")
        target_pair_support = int(counts["n_source_patient_x_full_key_pairs_with_exact_10uM_24h_lincs_condition"])
        if target_pair_support == 0:
            hard_reasons.append("no_direct_source_patient_by_drug_pair_has_exact_10uM_24h_LINCS_condition")
        if counts["linqs_structure_axis"]["target_exact_10uM_24h_keys_remaining_in_train"] == 0:
            hard_reasons.append("no_direct_source_drug_key_has_exact_10uM_24h_training_support_after_exclusion")

        gate_status = "LOCAL_NO_GO" if hard_reasons else "GO_TO_MANAGER_FOR_NEXT_FROZEN_ATTEMPT"
        snapshot = {
            "run_id": RUN_ID,
            "created_at_utc": now_iso(),
            "status": gate_status,
            "protocol_source": "configs/experiments/EXP-050.yaml:xpert_independent_object_gate_v1",
            "freeze_evidence": {
                "registered_gate_rule_preceded_this_gate_execution": True,
                "registered_gate_block_sha256": frozen_gate_digest,
                "source_object_set_rule": "Union of source_origin_id identities in the pre-existing permitted CTRPv2 and PRISM keyed-support outputs; identity columns only.",
                "source_patient_id_set_sha256": frozen_patient_id_digest,
                "source_patient_full_inchikey_pair_set_sha256": frozen_patient_key_digest,
                "set_selection_uses_response_or_sensitivity_values": False,
                "historical_summary_deviation_changed_frozen_source_object_set": "NOT_ESTABLISHED_BY_AUDIT",
                "operator_reported_no_selection_use": True,
            },
            "decision_boundary": "Response-blind identity/exposure admission only. No model training, prediction, selection, scoring or direct-sensitivity analysis.",
            "source_cohort_rule": "Exclude the union of PatientIDs present in either licensed CTRPv2 or PRISM 24Q2 keyed source-sensitivity support; this prevents an object with either direct endpoint from influencing fitting, scale fitting, early stopping, or development selection.",
            "exposure_rule": "Exact numeric LINCS pert_dose 10.0 uM and pert_time 24.0 h; connect only by condition_id, cell_id, pert_id and exact full InChIKey.",
            "identity_rule": "Map response-blind LINCS cell_id to exact DepMap 24Q2 StrippedCellLineName, fallback to exact CellLineName only when no stripped-name match; source identity is DepMap PatientID and all rows sharing a PatientID are one object.",
            "unresolved_identity_rule": "Unresolved or ambiguous condition identities are conservatively excluded from remaining fit/selection support and separately counted; they do not by themselves establish source overlap.",
            "forbidden_access": {
                "condition_delta_or_treated_expression_values_read": False,
                "individual_response_label_values_read": False,
                "direct_sensitivity_numeric_label_values_read": False,
                "outer_or_locked_eval_result_values_read": False,
                "locked_eval_identity_or_role_metadata_exposed_preflight": True,
                "pdo_processed_label_values_read": False,
                "current_gate_script_parsed_split_assignments": False,
            },
            "projected_columns": {
                "conditions_parquet": CONDITION_COLUMNS,
                "ctrv2_source_support_parquet": PAIR_COLUMNS,
                "prism_source_support_parquet": PAIR_COLUMNS,
                "depmap_model_csv": MODEL_COLUMNS,
                "lincs_pert_info": ["pert_id", "inchi_key"],
            },
            "identity_crosswalk": {
                "frozen_reference": str(IDENTITY_SPLIT_REF.relative_to(ROOT)).replace("\\", "/"),
                "source_metadata_only": True,
                "model_csv_sha256_expected": frozen_model_digest,
                "model_csv_sha256_actual": actual_model_digest,
                "identity_split_assignments_parsed": False,
            },
            "summary": {
                "source_patient_ids_excluded": len(target_patients),
                "direct_source_patient_x_drug_key_pairs": len(pair_sources),
                "direct_pair_rows_with_exact_10uM_24h_LINCS_support": target_pair_support,
                "remaining_train_patient_ids": split_counts["train"]["independent_patient_ids_remaining"],
                "remaining_inner_dev_patient_ids": split_counts["inner_dev"]["independent_patient_ids_remaining"],
                "remaining_exact_10uM_24h_train_conditions": split_counts["train"]["exact_10uM_24h_conditions_remaining"],
                "remaining_exact_10uM_24h_inner_dev_conditions": split_counts["inner_dev"]["exact_10uM_24h_conditions_remaining"],
                "gene_axis_count": gene_axis_count,
                "gene_axis_verified": bool(gene_axis_verified),
                "target_drugs_with_train_support": counts["linqs_structure_axis"]["target_exact_10uM_24h_keys_remaining_in_train"],
                "target_drugs_with_inner_dev_support": counts["linqs_structure_axis"]["target_exact_10uM_24h_keys_remaining_in_inner_dev"],
                "estimated_next_attempt_gpu_hours": conservative_gpu_hours,
                "hard_reasons": hard_reasons,
            },
            "cost_preflight": "cost_preflight.json",
            "outputs": {
                "common_object_exposure_support": "common_object_exposure_support.tsv",
                "excluded_source_identity_counts": "excluded_source_identity_counts.json",
                "train_dev_drug_support": "train_dev_drug_support.tsv",
                "cost_preflight": "cost_preflight.json",
            },
            "deviation_handling": "A historical aggregate validation metric was incidentally printed during preflight cost-source inspection; it was excluded from every selection and gate decision. Cost uses only the separate runtime receipt. No individual response values or direct-sensitivity numeric labels were read.",
        }
        write_json(OUT / "protocol_snapshot.json", snapshot)
        manifest.update({
            "status": gate_status,
            "finished_at_utc": now_iso(),
            "response_values_read": False,
            "direct_sensitivity_values_read": False,
            "registered_gate_block_sha256": frozen_gate_digest,
            "frozen_source_patient_id_set_sha256": frozen_patient_id_digest,
            "frozen_source_patient_full_inchikey_pair_set_sha256": frozen_patient_key_digest,
            "training_started": False,
            "gpu_hours": 0.0,
            "hard_reasons": hard_reasons,
            "summary": snapshot["summary"],
            "input_sha256_response_blind": {
                "conditions_parquet": sha256(CONDITIONS),
                "train_membership": sha256(TRAIN_INDEX),
                "inner_dev_membership": sha256(DEV_INDEX),
                "depmmap_model_csv_identity_crosswalk": actual_model_digest,
                "lincs_pert_info_structure_metadata": sha256(PERT_INFO),
                "lincs_gene_axis_metadata": sha256(GENE_RESOURCE / "genes.json"),
                "lincs_gene_axis_quality_metadata": sha256(GENE_RESOURCE / "quality_report.json"),
            },
            "source_pair_parquet_columns_projected": {
                "CTRPv2": PAIR_COLUMNS,
                "PRISM_24Q2": PAIR_COLUMNS,
            },
            "source_pair_parquet_value_columns_read": [],
            "output_sha256": {
                name: sha256(OUT / name)
                for name in ["protocol_snapshot.json", "common_object_exposure_support.tsv",
                             "excluded_source_identity_counts.json", "train_dev_drug_support.tsv", "cost_preflight.json"]
            },
        })
        write_json(OUT / "run_manifest.json", manifest)
        print(json.dumps({"status": gate_status, "summary": snapshot["summary"], "output_dir": str(OUT)}, ensure_ascii=False))
        return 0
    except Exception as exc:  # preserve any real failed attempt and its stage
        failure = {
            "attempt": 1,
            "failed_at_utc": now_iso(),
            "stage": "response_blind_identity_exposure_gate",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "response_values_read": False,
            "direct_sensitivity_values_read": False,
            "gpu_hours": 0.0,
            "training_started": False,
            "recorded_deviations": RECORDED_DEVIATIONS,
        }
        write_json(OUT / "attempt_001_failure.json", failure)
        manifest.update({"status": "FAILED_PRELAUNCH_NO_SCIENTIFIC_RESULT", "finished_at_utc": now_iso(),
                         "failure": "attempt_001_failure.json", "training_started": False, "gpu_hours": 0.0})
        write_json(OUT / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("This data gate must run inside WSL2 Conda drugscreening-gpu.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-postrun-deviations",
        action="store_true",
        help="Update only existing response-blind gate receipts; do not read source inputs or rerun the gate.",
    )
    args = parser.parse_args()
    raise SystemExit(record_postrun_deviations() if args.record_postrun_deviations else main())
