"""Execute five approved EXP-045 source-fold availability masks on CPU only.

This entry point binds the frozen source exclusions, the Sol-approved
SIDM00427 -> ACH-000499 identity, whole PatientID groups, and the registered
finite/missing availability sidecar. The v2 revision also applies the
Sol-approved conservative SIDM00117 candidate-group exclusion to three inner
slots. It does not load expression values, dependency magnitudes,
drug-response labels, predictions, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = "drugscreening-gpu"
MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
SIDECAR = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/dependency_validity_masks.npz"
PUBLIC_BRIDGE = ROOT / "artifacts/experiments/EXP-045/20260925_public_identity_crosswalk_v1/receipt.json"
REBIND_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_alias_fold_rebind_v1/receipt.json"
V1_OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_alias_fold_masks_v1"
OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_alias_fold_masks_v2"
MASK_FILE = OUTPUT_DIR / "fold_masks.npz"
EXCLUSION_FILE = OUTPUT_DIR / "fold_exclusion_manifest.json"
LOG_FILE = OUTPUT_DIR / "mask_execution_log.json"
RECEIPT_FILE = OUTPUT_DIR / "receipt.json"
EXPECTED_V1_RECEIPT_SHA256 = "3b20779be39b158d48820d35b45c627e6c0489c3ac648b828765e2453744e230"
EXPECTED_V1_MASK_SHA256 = "460f74435aa2fd712ff123eb6d615006f9a775f938ef29ed535d4de2f06ee593"

EXPECTED_SHA256 = {
    "frozen_identity_manifest": "c44e50edd88e93eb70cbe96de56327818d061c309ca4778c46d6ea550715edfc",
    "depmap_model_csv": "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8",
    "validity_mask_sidecar": "9eab5d6fcd693061bccd7550efe413098f756a59ee34f39e16dd46078f76ddf3",
    "official_public_identity_crosswalk": "8050c75c667c646f710a7c0e1e5cb4c3e9a78beab13558ad6bf4d96bcf801796",
    "v4_identity_projection_receipt": "0f4fd1850e053c68e86de9239bdb4d0b9c81decfcf6e5e6dd714c731d8dd36ab",
    "depmap_source_sha256": "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452",
    "depmap_availability_array_sha256": "405de9ecf83d709a015c148b7f3850d3b4d31738295a139aee9a94d485d94754",
    "dependency_axis_sha256": "9881e9f8184b3586c2211b3554f4fa9e76522055ff4384918f60b27576254965",
}
GENE_COUNT = 16_100
EXPECTED_MODEL_KEY_COUNT = 1_150
APPROVED_SLOTS = [
    ("outer", 2, None),
    ("inner", 0, 1),
    ("inner", 1, 2),
    ("inner", 3, 2),
    ("inner", 4, 1),
]
EXPECTED_UNRESOLVED_AFTER_V4 = {
    ("outer", 2, None): ["SIDM01117"],
    ("inner", 0, 1): ["SIDM00117", "SIDM01117"],
    ("inner", 1, 2): ["SIDM00117"],
    ("inner", 3, 2): ["SIDM01117"],
    ("inner", 4, 1): ["SIDM00117"],
}
EXPECTED_FROZEN_EXCLUSION_UNION_COUNTS = {
    ("outer", 2, None): 216,
    ("inner", 0, 1): 508,
    ("inner", 1, 2): 517,
    ("inner", 3, 2): 516,
    ("inner", 4, 1): 505,
}
SANGER_ID = "SIDM00427"
DEPMAP_MODEL_ID = "ACH-000499"
PATIENT_ID = "PT-m8xCzo"
UNRESOLVED_CANDIDATE_SANGER_ID = "SIDM00117"
UNRESOLVED_CANDIDATE_PATIENT_ID = "PT-9eMhTl"
UNRESOLVED_CANDIDATE_MODEL_IDS = ["ACH-000837", "ACH-002172"]
UNRESOLVED_CANDIDATE_MODELS_IN_SIDECAR = ["ACH-000837"]
CANDIDATE_GROUP_EXCLUSION_SLOTS = {
    ("inner", 0, 1),
    ("inner", 1, 2),
    ("inner", 4, 1),
}
SOL_CANDIDATE_GROUP_RULING = "/root/sidm00117_candidate_group_ruling_0925; GPT-6 Sol / medium"


class MaskExecutionError(RuntimeError):
    """Raised when an approved identity or mask contract does not match."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MaskExecutionError(f"invalid JSON input: {path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise MaskExecutionError(f"JSON root must be an object: {path.relative_to(ROOT)}")
    return value


def _require_cpu_runtime() -> dict[str, str]:
    if sys.platform != "linux" or "microsoft" not in platform.release().lower():
        raise MaskExecutionError("REQUIRES_WSL2_LINUX_RUNTIME")
    if os.environ.get("CONDA_DEFAULT_ENV") != ENVIRONMENT:
        raise MaskExecutionError("REQUIRES_PROJECT_CONDA_ENVIRONMENT")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise MaskExecutionError("CPU_AUDIT_REQUIRES_CUDA_DISABLED")
    return {
        "platform": platform.platform(),
        "conda_environment": os.environ["CONDA_DEFAULT_ENV"],
        "cuda_visible_devices": "",
        "gpu_hours": "0",
    }


def _verify_file_hashes() -> dict[str, str]:
    paths = {
        "frozen_identity_manifest": MANIFEST,
        "depmap_model_csv": MODEL_CSV,
        "validity_mask_sidecar": SIDECAR,
        "official_public_identity_crosswalk": PUBLIC_BRIDGE,
        "v4_identity_projection_receipt": REBIND_RECEIPT,
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        value = _sha256_file(path)
        if value != EXPECTED_SHA256[name]:
            raise MaskExecutionError(f"registered input hash changed: {name}")
        actual[str(path.relative_to(ROOT)).replace("\\", "/")] = value
    return actual


def _verify_v1_parent() -> dict[str, str]:
    receipt_path = V1_OUTPUT_DIR / "receipt.json"
    receipt_sha256 = _sha256_file(receipt_path)
    if receipt_sha256 != EXPECTED_V1_RECEIPT_SHA256:
        raise MaskExecutionError("v1 parent receipt is missing or has changed")
    receipt = _read_json(receipt_path)
    if receipt.get("receipt_id") != "EXP-045-Sanger-v4-five-slot-mask-execution-v1":
        raise MaskExecutionError("v1 parent receipt ID changed")
    mask_info = receipt.get("mask_archive", {})
    expected_path = str((V1_OUTPUT_DIR / "fold_masks.npz").relative_to(ROOT)).replace("\\", "/")
    if mask_info.get("path") != expected_path or mask_info.get("sha256") != EXPECTED_V1_MASK_SHA256:
        raise MaskExecutionError("v1 parent mask binding changed")
    mask_path = ROOT / expected_path
    if _sha256_file(mask_path) != EXPECTED_V1_MASK_SHA256:
        raise MaskExecutionError("v1 parent mask bytes changed")
    return {
        "v1_receipt_path": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
        "v1_receipt_sha256": receipt_sha256,
        "v1_mask_path": expected_path,
        "v1_mask_sha256": EXPECTED_V1_MASK_SHA256,
    }


def _compare_v1_slots(slot_results: list[dict[str, Any]]) -> None:
    mask_path = V1_OUTPUT_DIR / "fold_masks.npz"
    try:
        with np.load(mask_path, allow_pickle=False) as previous:
            expected_keys = {
                key
                for result in slot_results
                for key in (result["model_ids_array_key"], result["mask_array_key"])
            }
            if set(previous.files) != expected_keys:
                raise MaskExecutionError("v1 mask archive keys differ from the five registered slots")
            for result in slot_results:
                ids_key = result["model_ids_array_key"]
                mask_key = result["mask_array_key"]
                old_ids = [str(value) for value in previous[ids_key].tolist()]
                old_mask = np.asarray(previous[mask_key], dtype=np.uint8)
                new_ids = [str(value) for value in result["_model_ids"].tolist()]
                new_mask = np.asarray(result["_mask"], dtype=np.uint8)
                slot = result["slot"]
                slot_key = (slot["level"], slot["outer_fold"], slot["inner_fold"])
                expected_removed = (
                    set(UNRESOLVED_CANDIDATE_MODELS_IN_SIDECAR)
                    if slot_key in CANDIDATE_GROUP_EXCLUSION_SLOTS
                    else set()
                )
                removed = set(old_ids) - set(new_ids)
                added = set(new_ids) - set(old_ids)
                if removed != expected_removed or added:
                    raise MaskExecutionError(
                        f"v2 training membership differs from the Sol-authorized v1 delta for {result['slot_key']}"
                    )
                if new_ids != [model_id for model_id in old_ids if model_id not in expected_removed]:
                    raise MaskExecutionError(f"v2 training ModelID order changed for {result['slot_key']}")
                old_row_by_id = {model_id: index for index, model_id in enumerate(old_ids)}
                expected_new_mask = np.stack([old_mask[old_row_by_id[model_id]] for model_id in new_ids], axis=0)
                if not np.array_equal(new_mask, expected_new_mask):
                    raise MaskExecutionError(f"v2 changed availability bits beyond the approved row exclusion for {result['slot_key']}")
                old_valid_bits = int(np.unpackbits(old_mask, axis=1, count=GENE_COUNT, bitorder="little").sum())
                old_total_bits = int(len(old_ids) * GENE_COUNT)
                result["v1_to_v2_comparison"] = {
                    "status": "ONLY_SOL_AUTHORIZED_SIDM00117_SIDE_CAR_ROWS_REMOVED; ALL_RETAINED_MASK_BITS_IDENTICAL",
                    "v1_training_model_id_count": len(old_ids),
                    "v2_training_model_id_count": len(new_ids),
                    "removed_training_ModelIDs": sorted(removed),
                    "added_training_ModelIDs": sorted(added),
                    "v1_training_ModelIDs_sha256": _sha256_lines(old_ids),
                    "v2_training_ModelIDs_sha256": result["training_model_ids_sha256"],
                    "v1_valid_bits": old_valid_bits,
                    "v1_missing_bits": old_total_bits - old_valid_bits,
                    "v2_valid_bits": result["availability_mask"]["valid_bits"],
                    "v2_missing_bits": result["availability_mask"]["missing_bits"],
                    "retained_availability_bits_identical": True,
                }
    except MaskExecutionError:
        raise
    except Exception as exc:
        raise MaskExecutionError("v1 mask archive could not be compared without pickle") from exc


def _load_rebind_and_slots() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _read_json(REBIND_RECEIPT)
    if receipt.get("status") != (
        "V4_IDENTITY_PROJECTION_COMPLETE; ACTUAL_FOLD_MASK_NOT_EXECUTED; "
        "CHECKPOINT_EXPOSURE_AND_D1_TRAINING_BLOCKED"
    ):
        raise MaskExecutionError("v4 identity projection receipt status changed")
    overlay = receipt.get("v4_overlay")
    if not isinstance(overlay, dict) or overlay.get("overlay_version") != "v4":
        raise MaskExecutionError("v4 identity overlay is missing")
    binding = overlay.get("single_authorized_binding", {})
    if (
        binding.get("Sanger_ID") != SANGER_ID
        or binding.get("DepMap_ModelID") != DEPMAP_MODEL_ID
        or overlay.get("same_PatientID_derivative_group", {}).get("PatientID") != PATIENT_ID
        or overlay.get("same_PatientID_derivative_group", {}).get("ModelIDs") != [DEPMAP_MODEL_ID]
    ):
        raise MaskExecutionError("Sol-authorized SIDM00427 identity bridge changed")
    sidecar_projection = overlay.get("validity_sidecar_projection", {})
    if (
        sidecar_projection.get("sha256") != EXPECTED_SHA256["validity_mask_sidecar"]
        or sidecar_projection.get("ACH-000499_validity_bits") != 15_983
        or sidecar_projection.get("ACH-000499_missingness_bits") != 117
    ):
        raise MaskExecutionError("authorized sidecar availability projection changed")
    approved = overlay.get("approved_fold_slots")
    if not isinstance(approved, list):
        raise MaskExecutionError("approved five-slot projection list is missing")
    observed = [
        (
            str(item.get("slot", {}).get("level")),
            int(item.get("slot", {}).get("outer_fold", -1)),
            item.get("slot", {}).get("inner_fold"),
        )
        for item in approved
    ]
    if observed != APPROVED_SLOTS:
        raise MaskExecutionError("v4 receipt slots differ from the Sol-approved five slots")
    parent_hashes = receipt.get("parent_hashes", {})
    input_hashes = receipt.get("input_hashes", {})
    bridge_parent = parent_hashes.get("Cellosaurus_crosswalk_parent_receipt", {})
    manifest_parent = parent_hashes.get("frozen_v3b_manifest", {})
    sidecar_input = input_hashes.get("validity_sidecar", {})
    model_input = input_hashes.get("DepMap_Model_csv_prior_verified_input_sha256")
    if not isinstance(bridge_parent, dict) or bridge_parent.get("sha256", "").lower() != EXPECTED_SHA256["official_public_identity_crosswalk"]:
        raise MaskExecutionError("v4 receipt does not bind the official public identity crosswalk")
    if not isinstance(manifest_parent, dict) or manifest_parent.get("sha256", "").lower() != EXPECTED_SHA256["frozen_identity_manifest"]:
        raise MaskExecutionError("v4 receipt does not bind the frozen identity manifest")
    if not isinstance(sidecar_input, dict) or sidecar_input.get("sha256", "").lower() != EXPECTED_SHA256["validity_mask_sidecar"]:
        raise MaskExecutionError("v4 receipt does not bind the registered availability sidecar")
    if not isinstance(model_input, str) or model_input.lower() != EXPECTED_SHA256["depmap_model_csv"]:
        raise MaskExecutionError("v4 receipt does not bind the registered DepMap Model.csv")
    return receipt, approved


def _load_sidecar() -> tuple[list[str], np.ndarray, dict[str, Any]]:
    try:
        with np.load(SIDECAR, allow_pickle=False) as archive:
            required = {
                "schema_version", "axis_gene_count", "axis_sha256", "depmap_source_sha256",
                "nature_dependency_source_sha256", "nature_samples_source_sha256",
                "depmap_object_keys", "depmap_masks_packed", "nature_sample_ids",
                "nature_library_keys", "nature_masks_packed",
            }
            if set(archive.files) != required:
                raise MaskExecutionError("availability sidecar field set changed")
            version = int(archive["schema_version"].item())
            gene_count = int(archive["axis_gene_count"].item())
            axis_hash = str(archive["axis_sha256"].item())
            source_hash = str(archive["depmap_source_sha256"].item())
            keys = [str(value) for value in archive["depmap_object_keys"].tolist()]
            packed = np.asarray(archive["depmap_masks_packed"], dtype=np.uint8).copy()
    except MaskExecutionError:
        raise
    except Exception as exc:
        raise MaskExecutionError("could not read registered availability sidecar without pickle") from exc
    if version != 1 or gene_count != GENE_COUNT or axis_hash != EXPECTED_SHA256["dependency_axis_sha256"]:
        raise MaskExecutionError("availability sidecar axis or schema changed")
    if source_hash != EXPECTED_SHA256["depmap_source_sha256"]:
        raise MaskExecutionError("DepMap dependency source hash in sidecar changed")
    if len(keys) != EXPECTED_MODEL_KEY_COUNT or len(set(keys)) != len(keys):
        raise MaskExecutionError("availability sidecar model keys are incomplete or duplicated")
    if packed.shape != (len(keys), (GENE_COUNT + 7) // 8) or packed.dtype != np.uint8:
        raise MaskExecutionError("availability sidecar packed mask shape or dtype changed")
    packed_hash = hashlib.sha256(packed.tobytes()).hexdigest()
    if packed_hash != EXPECTED_SHA256["depmap_availability_array_sha256"]:
        raise MaskExecutionError("availability sidecar bit-array hash changed")
    info = {
        "path": str(SIDECAR.relative_to(ROOT)).replace("\\", "/"),
        "sha256": EXPECTED_SHA256["validity_mask_sidecar"],
        "availability_array_sha256": packed_hash,
        "model_id_key_count": len(keys),
        "gene_count": gene_count,
        "ordered_gene_axis_sha256": axis_hash,
        "bit_semantics": "1=finite dependency availability; 0=missing or non-finite; no dependency magnitude loaded",
        "bit_order": "little",
    }
    return keys, packed, info


def _load_model_identity() -> tuple[dict[str, str], dict[str, set[str]]]:
    try:
        frame = pd.read_csv(
            MODEL_CSV,
            usecols=["ModelID", "PatientID"],
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except Exception as exc:
        raise MaskExecutionError("Model.csv identity-column projection failed") from exc
    if frame["ModelID"].eq("").any() or frame["ModelID"].duplicated().any():
        raise MaskExecutionError("Model.csv ModelID key is empty or duplicated")
    patient_by_model = {
        str(row.ModelID).strip(): str(row.PatientID).strip()
        for row in frame.itertuples(index=False)
    }
    models_by_patient: dict[str, set[str]] = {}
    for model_id, patient_id in patient_by_model.items():
        if patient_id:
            models_by_patient.setdefault(patient_id, set()).add(model_id)
    target_group = models_by_patient.get(PATIENT_ID, set())
    if target_group != {DEPMAP_MODEL_ID} or patient_by_model.get(DEPMAP_MODEL_ID) != PATIENT_ID:
        raise MaskExecutionError("ACH-000499 PatientID group differs from the approved one-model group")
    return patient_by_model, models_by_patient


def _load_registered_source_candidates() -> dict[str, list[str]]:
    receipt = _read_json(PUBLIC_BRIDGE)
    rows = receipt.get("per_registered_source_Sanger_ID")
    if not isinstance(rows, list):
        raise MaskExecutionError("official source Sanger crosswalk rows are missing")
    candidates: dict[str, list[str]] = {}
    for row in rows:
        source_id = str(row.get("registered_Sanger_ID", "")).strip()
        if source_id not in {"SIDM00117", "SIDM00427", "SIDM01117"}:
            continue
        ids: set[str] = set()
        for relation in row.get("official_CMP_to_DepMap_relations", []):
            discontinued = str(relation.get("DepMap_discontinued") or "").strip().lower() == "true"
            model_id = str(relation.get("DepMap_accession", "")).strip()
            if model_id and not discontinued:
                ids.add(model_id)
        for match in row.get("direct_DepMap_24Q2_Model_csv_Sanger_alias_matches", []):
            model_id = str(match.get("ModelID", "")).strip()
            if model_id:
                ids.add(model_id)
        candidates[source_id] = sorted(ids)
    expected = {
        "SIDM00117": ["ACH-000837", "ACH-002172"],
        "SIDM00427": ["ACH-000499"],
        "SIDM01117": ["ACH-003071"],
    }
    if candidates != expected:
        raise MaskExecutionError("official source Sanger candidate projection changed")
    return candidates


def _audit_unresolved_candidate_groups(
    unresolved_ids: list[str],
    *,
    candidates: dict[str, list[str]],
    patient_by_model: dict[str, str],
    models_by_patient: dict[str, set[str]],
    sidecar_keys: set[str],
    training_keys: set[str],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for source_id in unresolved_ids:
        candidate_ids = candidates.get(source_id, [])
        candidate_rows: list[dict[str, Any]] = []
        all_sidecar_group_ids: set[str] = set()
        all_training_hits: set[str] = set()
        for model_id in candidate_ids:
            patient_id = patient_by_model.get(model_id, "")
            group_model_ids = models_by_patient.get(patient_id, {model_id}) if patient_id else {model_id}
            sidecar_group_ids = group_model_ids & sidecar_keys
            training_hits = sidecar_group_ids & training_keys
            all_sidecar_group_ids.update(sidecar_group_ids)
            all_training_hits.update(training_hits)
            candidate_rows.append({
                "candidate_ModelID": model_id,
                "PatientID": patient_id or None,
                "whole_PatientID_group_ModelIDs": sorted(group_model_ids),
                "group_ModelIDs_in_availability_sidecar": sorted(sidecar_group_ids),
                "group_ModelIDs_remaining_in_this_training_mask": sorted(training_hits),
            })
        if all_training_hits:
            impact = "POTENTIAL_CURRENT_MASK_TRAINING_SIDE_EXPOSURE"
        elif all_sidecar_group_ids:
            impact = "ALL_KNOWN_CURRENT_SIDECAR_CANDIDATE_GROUPS_EXCLUDED_FROM_THIS_MASK"
        else:
            impact = "NO_KNOWN_CURRENT_SIDECAR_ROW_FOR_REGISTERED_CANDIDATE_GROUPS"
        audits.append({
            "Sanger_ID": source_id,
            "identity_status": "UNRESOLVED_OR_AMBIGUOUS; candidate mapping does not settle source identity",
            "registered_candidate_ModelIDs": candidate_ids,
            "candidate_groups": candidate_rows,
            "current_mask_training_side_impact": impact,
            "checkpoint_fit_selection_test_exposure": "UNKNOWN_UNBOUND",
        })
    return audits


def _resolve_slot(
    fold: dict[str, Any],
    *,
    level: str,
    outer_fold: int,
    inner_fold: int | None,
    patient_by_model: dict[str, str],
    models_by_patient: dict[str, set[str]],
    sidecar_keys: set[str],
    candidates: dict[str, list[str]],
    approved_projection: dict[str, Any],
    conservative_group_exclusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outer_ids = fold.get("excluded_model_ids")
    if not isinstance(outer_ids, list) or int(fold.get("excluded_model_count", -1)) != len(outer_ids):
        raise MaskExecutionError(f"frozen outer exclusions malformed for fold {outer_fold}")
    unresolved_outer = fold.get("unresolved_sanger_ids")
    if not isinstance(unresolved_outer, list):
        raise MaskExecutionError(f"frozen outer unresolved identity list malformed for fold {outer_fold}")
    unresolved_in_slot = {str(value).strip() for value in unresolved_outer}
    raw_excluded = {str(value).strip() for value in outer_ids}
    outer_split_ids = set(raw_excluded)
    inner_split_ids: set[str] = set()
    if "" in raw_excluded or len(raw_excluded) != len(outer_ids):
        raise MaskExecutionError(f"frozen outer exclusion keys malformed for fold {outer_fold}")
    selected_inner: dict[str, Any] | None = None
    if level == "inner":
        inner_rows = fold.get("inner")
        if not isinstance(inner_rows, list):
            raise MaskExecutionError(f"frozen inner exclusions missing for outer fold {outer_fold}")
        matches = [row for row in inner_rows if int(row.get("inner_fold", -1)) == inner_fold]
        if len(matches) != 1:
            raise MaskExecutionError(f"frozen inner slot is not unique: outer {outer_fold}, inner {inner_fold}")
        selected_inner = matches[0]
        inner_ids = selected_inner.get("excluded_model_ids")
        if not isinstance(inner_ids, list):
            raise MaskExecutionError(f"frozen inner exclusion keys malformed: outer {outer_fold}, inner {inner_fold}")
        unresolved_inner = selected_inner.get("unresolved_sanger_ids")
        if not isinstance(unresolved_inner, list):
            raise MaskExecutionError(f"frozen inner unresolved identity list malformed: outer {outer_fold}, inner {inner_fold}")
        if int(selected_inner.get("excluded_model_count", len(inner_ids))) != len(inner_ids):
            raise MaskExecutionError(f"frozen inner exclusion count mismatch: outer {outer_fold}, inner {inner_fold}")
        clean_inner = {str(value).strip() for value in inner_ids}
        if "" in clean_inner or len(clean_inner) != len(inner_ids):
            raise MaskExecutionError(f"frozen inner exclusion keys malformed: outer {outer_fold}, inner {inner_fold}")
        if raw_excluded & clean_inner:
            raise MaskExecutionError(f"outer test and inner validation IDs must be disjoint: {outer_fold}/{inner_fold}")
        inner_split_ids = clean_inner
        raw_excluded.update(clean_inner)
        unresolved_in_slot.update(str(value).strip() for value in unresolved_inner)

    expected_direct_count = EXPECTED_FROZEN_EXCLUSION_UNION_COUNTS[(level, outer_fold, inner_fold)]
    if len(raw_excluded) != expected_direct_count:
        raise MaskExecutionError(f"outer/inner frozen ModelID union count changed for {level} {outer_fold}/{inner_fold}")

    if SANGER_ID not in unresolved_in_slot:
        raise MaskExecutionError(f"v3b slot no longer records the authorized rebind: {level} {outer_fold}/{inner_fold}")
    unresolved_after_v4 = sorted(unresolved_in_slot - {SANGER_ID})
    if unresolved_after_v4 != EXPECTED_UNRESOLVED_AFTER_V4[(level, outer_fold, inner_fold)]:
        raise MaskExecutionError(
            f"remaining source identity aliases differ from the Sol ruling for {level} {outer_fold}/{inner_fold}"
        )

    missing_model_rows = sorted(model_id for model_id in raw_excluded if model_id not in patient_by_model)
    if missing_model_rows:
        raise MaskExecutionError(
            f"frozen split exclusion has ModelIDs absent from Model.csv in {level} {outer_fold}/{inner_fold}: "
            f"{len(missing_model_rows)}"
        )
    split_patient_ids = {patient_by_model[model_id] for model_id in raw_excluded if patient_by_model[model_id]}
    split_patient_group_members = set().union(*(models_by_patient[pid] for pid in split_patient_ids)) if split_patient_ids else set()
    target_group_members = models_by_patient[PATIENT_ID]
    conservative_group_excluded_ids: set[str] = set()
    if conservative_group_exclusion is not None:
        group_ids = conservative_group_exclusion.get("whole_PatientID_group_ModelIDs")
        if not isinstance(group_ids, list) or set(group_ids) != set(UNRESOLVED_CANDIDATE_MODEL_IDS):
            raise MaskExecutionError("conservative candidate group differs from the Sol ruling")
        conservative_group_excluded_ids = set(group_ids)
    closed_excluded = raw_excluded | split_patient_group_members | target_group_members | conservative_group_excluded_ids
    training_keys = sorted(sidecar_keys - closed_excluded)
    unresolved_candidate_audit = _audit_unresolved_candidate_groups(
        unresolved_after_v4,
        candidates=candidates,
        patient_by_model=patient_by_model,
        models_by_patient=models_by_patient,
        sidecar_keys=sidecar_keys,
        training_keys=set(training_keys),
    )
    training_patient_groups = {
        patient_by_model[model_id]
        for model_id in training_keys
        if patient_by_model.get(model_id)
    }
    rows_by_sidecar_key = approved_projection["expected_mask_rows"]
    mask = rows_by_sidecar_key["packed_by_model_id"]
    selected_mask = np.stack([mask[model_id] for model_id in training_keys], axis=0)
    valid_bits = int(np.unpackbits(selected_mask, axis=1, count=GENE_COUNT, bitorder="little").sum())
    total_bits = int(len(training_keys) * GENE_COUNT)
    projection = approved_projection["projection"]
    expected = projection.get("capacity_after_v4_overlay_projection", {})
    observed = {
        "remaining_sidecar_ModelID_keys": len(training_keys),
        "remaining_sidecar_PatientID_groups": len(training_patient_groups),
        "validity_bits": valid_bits,
        "missingness_bits": total_bits - valid_bits,
    }
    expected_fields = tuple(observed)
    if level == "outer" and any(int(expected.get(key, -1)) != observed[key] for key in expected_fields):
        raise MaskExecutionError(
            f"generated {level} mask does not match bound v4 capacity projection for {outer_fold}/{inner_fold}: "
            f"observed={observed}; expected={ {key: expected.get(key) for key in expected_fields} }"
        )
    slot_key = f"{level}_outer{outer_fold}" + (f"_inner{inner_fold}" if inner_fold is not None else "")
    exclusions_hash = _sha256_lines(sorted(closed_excluded))
    training_hash = _sha256_lines(training_keys)
    mask_bytes_hash = hashlib.sha256(selected_mask.tobytes()).hexdigest()
    return {
        "slot_key": slot_key,
        "slot": {"level": level, "outer_fold": outer_fold, "inner_fold": inner_fold},
        "status": "PARTIAL_SOURCE_IDENTITY_EXCLUSION_MASK" if unresolved_after_v4 else "MASK_EXECUTED; SLOT_ALIAS_LIST_HAS_NO_RESIDUAL_UNRESOLVED_ID; OTHER_GATES_UNREVIEWED",
        "outer_test_excluded_ModelIDs_sha256": _sha256_lines(sorted(outer_split_ids)),
        "outer_test_excluded_ModelID_count": len(outer_split_ids),
        "inner_validation_excluded_ModelIDs_sha256": _sha256_lines(sorted(inner_split_ids)),
        "inner_validation_excluded_ModelID_count": len(inner_split_ids),
        "outer_inner_direct_exclusion_intersection_count": len(outer_split_ids & inner_split_ids),
        "outer_inner_direct_exclusion_union_count": len(raw_excluded),
        "unresolved_sanger_ids_after_v4_overlay": unresolved_after_v4,
        "outer_test_unresolved_sanger_ids": sorted({str(value).strip() for value in unresolved_outer}),
        "inner_validation_unresolved_sanger_ids": sorted({str(value).strip() for value in (unresolved_inner if level == "inner" else [])}),
        "unresolved_sanger_id_candidate_group_audit": unresolved_candidate_audit,
        "conservative_candidate_group_exclusion": conservative_group_exclusion,
        "complete_source_identity_isolation_claim": False,
        "inner_mask_exclusion_semantics": "outer_test_union_inner_validation_then_close_to_whole_PatientID_groups" if level == "inner" else "outer_test_then_close_to_whole_PatientID_groups",
        "mask_array_key": f"{slot_key}_availability_packed",
        "model_ids_array_key": f"{slot_key}_model_ids",
        "raw_frozen_split_exclusion_model_ids": sorted(raw_excluded),
        "split_patient_group_ids": sorted(split_patient_ids),
        "whole_group_excluded_model_ids": sorted(closed_excluded),
        "whole_group_excluded_model_ids_sha256": exclusions_hash,
        "target_identity_exclusion": {
            "Sanger_ID": SANGER_ID,
            "DepMap_ModelID": DEPMAP_MODEL_ID,
            "PatientID": PATIENT_ID,
            "whole_PatientID_ModelIDs": sorted(target_group_members),
            "already_in_frozen_split_exclusions": DEPMAP_MODEL_ID in raw_excluded,
        },
        "training_model_ids": training_keys,
        "training_model_ids_sha256": training_hash,
        "training_model_id_count": len(training_keys),
        "training_PatientID_group_count": len(training_patient_groups),
        "training_keys_without_PatientID": sum(1 for model_id in training_keys if not patient_by_model.get(model_id)),
        "availability_mask": {
            "shape": [len(training_keys), GENE_COUNT],
            "packed_shape": [len(training_keys), (GENE_COUNT + 7) // 8],
            "packed_sha256": mask_bytes_hash,
            "valid_bits": valid_bits,
            "missing_bits": total_bits - valid_bits,
            "total_bits": total_bits,
            "roundtrip_verified": False,
        },
        "v4_projection_comparison": {
            "status": "MATCHED_OUTER_PROJECTION" if level == "outer" else "NOT_COMPARABLE_PRIOR_INNER_SINGLE_LIST_PROJECTION_OMITTED_OUTER_TEST_SET",
            "actual_nested_training_capacity": observed,
            "prior_inner_projection": expected if level == "inner" else None,
        },
        "_model_ids": np.asarray(training_keys, dtype=np.str_),
        "_mask": selected_mask,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run() -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    runtime = _require_cpu_runtime()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite prior mask audit: {OUTPUT_DIR}")
    input_hashes = _verify_file_hashes()
    v1_parent = _verify_v1_parent()
    input_hashes.update({
        "v1_parent_receipt": v1_parent["v1_receipt_sha256"],
        "v1_parent_mask_archive": v1_parent["v1_mask_sha256"],
    })
    rebind_receipt, approved_rows = _load_rebind_and_slots()
    manifest = _read_json(MANIFEST)
    folds = manifest.get("source_folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise MaskExecutionError("frozen identity manifest must contain five source folds")
    fold_by_id = {int(fold.get("fold", -1)): fold for fold in folds}
    if set(fold_by_id) != {0, 1, 2, 3, 4}:
        raise MaskExecutionError("frozen identity manifest fold IDs changed")
    sidecar_keys_list, packed, sidecar_info = _load_sidecar()
    if len(set(sidecar_keys_list)) != len(sidecar_keys_list):
        raise MaskExecutionError("availability sidecar ModelID keys are duplicated")
    sidecar_keys = set(sidecar_keys_list)
    packed_by_model_id = {
        model_id: packed[row].copy()
        for row, model_id in enumerate(sidecar_keys_list)
    }
    patient_by_model, models_by_patient = _load_model_identity()
    source_candidates = _load_registered_source_candidates()
    if source_candidates.get(UNRESOLVED_CANDIDATE_SANGER_ID) != UNRESOLVED_CANDIDATE_MODEL_IDS:
        raise MaskExecutionError("SIDM00117 candidate list changed from the Sol-reviewed set")
    candidate_group = models_by_patient.get(UNRESOLVED_CANDIDATE_PATIENT_ID, set())
    if candidate_group != set(UNRESOLVED_CANDIDATE_MODEL_IDS) or any(
        patient_by_model.get(model_id) != UNRESOLVED_CANDIDATE_PATIENT_ID
        for model_id in UNRESOLVED_CANDIDATE_MODEL_IDS
    ):
        raise MaskExecutionError("SIDM00117 whole PatientID group changed from the Sol-reviewed set")
    candidate_sidecar_models = sorted(candidate_group & sidecar_keys)
    if candidate_sidecar_models != UNRESOLVED_CANDIDATE_MODELS_IN_SIDECAR:
        raise MaskExecutionError("SIDM00117 sidecar candidate coverage changed from the reviewed set")
    v4_slot_map = {
        (
            str(item.get("slot", {}).get("level")),
            int(item.get("slot", {}).get("outer_fold", -1)),
            item.get("slot", {}).get("inner_fold"),
        ): item
        for item in approved_rows
    }

    slot_results: list[dict[str, Any]] = []
    for level, outer_fold, inner_fold in APPROVED_SLOTS:
        group_exclusion = None
        if (level, outer_fold, inner_fold) in CANDIDATE_GROUP_EXCLUSION_SLOTS:
            group_exclusion = {
                "source_Sanger_ID": UNRESOLVED_CANDIDATE_SANGER_ID,
                "identity_status": "UNRESOLVED_ONE_TO_MANY; CONSERVATIVE_CANDIDATE_GROUP_EXCLUSION_ONLY",
                "PatientID": UNRESOLVED_CANDIDATE_PATIENT_ID,
                "whole_PatientID_group_ModelIDs": sorted(candidate_group),
                "group_ModelIDs_in_availability_sidecar": candidate_sidecar_models,
                "excluded_from_fold_training_ModelIDs": sorted(candidate_group),
                "actual_sol_dispatch": SOL_CANDIDATE_GROUP_RULING,
                "scope": "Only this approved inner slot; does not resolve SIDM00117 or historical checkpoint exposure.",
            }
        slot_results.append(
            _resolve_slot(
                fold_by_id[outer_fold],
                level=level,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                patient_by_model=patient_by_model,
                models_by_patient=models_by_patient,
                sidecar_keys=sidecar_keys,
                candidates=source_candidates,
                conservative_group_exclusion=group_exclusion,
                approved_projection={
                    "expected_mask_rows": {"packed_by_model_id": packed_by_model_id},
                    "projection": v4_slot_map[(level, outer_fold, inner_fold)],
                },
            )
        )

    _compare_v1_slots(slot_results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    for result in slot_results:
        arrays[result["model_ids_array_key"]] = result.pop("_model_ids")
        arrays[result["mask_array_key"]] = result.pop("_mask")
    np.savez_compressed(MASK_FILE, **arrays)
    masks_sha256 = _sha256_file(MASK_FILE)

    try:
        with np.load(MASK_FILE, allow_pickle=False) as written:
            if set(written.files) != set(arrays):
                raise MaskExecutionError("written mask archive keys differ from generated keys")
            for result in slot_results:
                ids_key = result["model_ids_array_key"]
                mask_key = result["mask_array_key"]
                written_ids = written[ids_key]
                written_mask = written[mask_key]
                if not np.array_equal(written_ids, arrays[ids_key]) or not np.array_equal(written_mask, arrays[mask_key]):
                    raise MaskExecutionError(f"mask archive round-trip mismatch: {result['slot_key']}")
                result["availability_mask"]["roundtrip_verified"] = True
    except MaskExecutionError:
        raise
    except Exception as exc:
        raise MaskExecutionError("written fold mask archive could not be re-opened without pickle") from exc

    exclusion_payload = {
        "experiment_id": "EXP-045",
        "manifest_id": "exp045-sanger-v4-five-slot-fold-exclusion-v2",
        "status": "PARTIAL_FIVE_SLOT_IDENTITY_EXCLUSION; OTHER_SOURCE_ALIASES_REMAIN_UNRESOLVED",
        "actual_sol_dispatch": SOL_CANDIDATE_GROUP_RULING,
        "parent_hashes": input_hashes,
        "sol_approved_binding": {
            "Sanger_ID": SANGER_ID,
            "DepMap_ModelID": DEPMAP_MODEL_ID,
            "PatientID": PATIENT_ID,
            "official_identity_crosswalk_sha256": EXPECTED_SHA256["official_public_identity_crosswalk"],
        },
        "parent_mask_execution_v1": v1_parent,
        "sol_approved_conservative_candidate_group_exclusion": {
            "actual_sol_dispatch": SOL_CANDIDATE_GROUP_RULING,
            "source_Sanger_ID": UNRESOLVED_CANDIDATE_SANGER_ID,
            "PatientID": UNRESOLVED_CANDIDATE_PATIENT_ID,
            "whole_PatientID_group_ModelIDs": UNRESOLVED_CANDIDATE_MODEL_IDS,
            "ModelIDs_in_availability_sidecar": UNRESOLVED_CANDIDATE_MODELS_IN_SIDECAR,
            "applied_slots": [
                {"level": level, "outer_fold": outer, "inner_fold": inner}
                for level, outer, inner in sorted(CANDIDATE_GROUP_EXCLUSION_SLOTS)
            ],
            "identity_resolved": False,
            "historical_checkpoint_exposure": "UNKNOWN",
        },
        "frozen_identity_manifest_sha256": EXPECTED_SHA256["frozen_identity_manifest"],
        "slot_count": len(slot_results),
        "slots": [
            {key: value for key, value in result.items() if key not in {"availability_mask"}}
            for result in slot_results
        ],
        "scope_limit": "Only the five Sol-approved source slots; all remain provisional for full source identity isolation. Inner exclusions use outer-test union inner-validation ModelIDs before whole-PatientID closure. This is not a complete source-training entity manifest and cannot authorize checkpoint use or training.",
    }
    _atomic_json(EXCLUSION_FILE, exclusion_payload)
    exclusion_sha256 = _sha256_file(EXCLUSION_FILE)
    execution_log = {
        "experiment_id": "EXP-045",
        "status": "FIVE_PROVISIONAL_CPU_MASKS_EXECUTED_WITH_SIDM00117_GROUP_EXCLUDED_IN_THREE_INNER_SLOTS; FULL_SOURCE_IDENTITY_ISOLATION_BLOCKED",
        "actual_sol_dispatch": SOL_CANDIDATE_GROUP_RULING,
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "command": sys.argv,
        "code_sha256": _sha256_file(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "parent_mask_execution_v1": v1_parent,
        "availability_sidecar": sidecar_info,
        "exclusion_manifest": {
            "path": str(EXCLUSION_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": exclusion_sha256,
        },
        "mask_archive": {
            "path": str(MASK_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": masks_sha256,
            "array_keys": sorted(arrays),
        },
        "slots": [
            {
                "slot_key": result["slot_key"],
                "slot": result["slot"],
                "status": result["status"],
                "outer_test_excluded_ModelIDs_sha256": result["outer_test_excluded_ModelIDs_sha256"],
                "outer_test_excluded_ModelID_count": result["outer_test_excluded_ModelID_count"],
                "inner_validation_excluded_ModelIDs_sha256": result["inner_validation_excluded_ModelIDs_sha256"],
                "inner_validation_excluded_ModelID_count": result["inner_validation_excluded_ModelID_count"],
                "outer_inner_direct_exclusion_intersection_count": result["outer_inner_direct_exclusion_intersection_count"],
                "outer_inner_direct_exclusion_union_count": result["outer_inner_direct_exclusion_union_count"],
                "unresolved_sanger_ids_after_v4_overlay": result["unresolved_sanger_ids_after_v4_overlay"],
                "outer_test_unresolved_sanger_ids": result["outer_test_unresolved_sanger_ids"],
                "inner_validation_unresolved_sanger_ids": result["inner_validation_unresolved_sanger_ids"],
                "complete_source_identity_isolation_claim": result["complete_source_identity_isolation_claim"],
                "training_model_id_count": result["training_model_id_count"],
                "training_PatientID_group_count": result["training_PatientID_group_count"],
                "whole_group_excluded_model_ids_sha256": result["whole_group_excluded_model_ids_sha256"],
                "training_model_ids_sha256": result["training_model_ids_sha256"],
                "availability_mask": result["availability_mask"],
                "unresolved_sanger_id_candidate_group_audit": result["unresolved_sanger_id_candidate_group_audit"],
                "conservative_candidate_group_exclusion": result["conservative_candidate_group_exclusion"],
                "v1_to_v2_comparison": result["v1_to_v2_comparison"],
                "inner_mask_exclusion_semantics": result["inner_mask_exclusion_semantics"],
                "v4_projection_comparison": result["v4_projection_comparison"],
            }
            for result in slot_results
        ],
        "data_access_boundary": {
            "Model_csv_columns_read": ["ModelID", "PatientID"],
            "validity_sidecar_arrays_read": ["schema_version", "axis_gene_count", "axis_sha256", "depmap_source_sha256", "depmap_object_keys", "depmap_masks_packed"],
            "finite_or_missing_availability_bits_read": True,
            "dependency_magnitudes_read": False,
            "expression_values_read": False,
            "drug_response_labels_read": False,
            "predictions_or_scores_read": False,
            "checkpoint_bytes_or_lineage_read": False,
            "nature_group_bridge_changed": False,
            "training_or_selection_performed": False,
            "scientific_scores_computed": False,
            "gpu_used": False,
            "tests_run": False,
        },
        "limitations": [
            "This is a fold-bound identity and finite/missing availability mask artifact, not a fitted-model training receipt.",
            "All five slots remain provisional for full source identity isolation; SIDM00117 remains unresolved even though its full candidate PatientID group is excluded in three specified inner training masks, and SIDM01117 remains unresolved.",
            "The old v4 inner capacity numbers covered an inner-validation-only list and are not acceptance values for the nested masks generated here.",
            "The source checkpoint fit/selection/test lineage remains unknown and must be resolved before checkpoint use or a training-safety conclusion.",
            "Nature 13-group cross-resource identity remains unresolved; this five-slot action does not imply absence or independence.",
            "D1 training remains blocked by the remaining identity, lineage, protocol, independent holdout, and budget gates.",
        ],
    }
    _atomic_json(LOG_FILE, execution_log)
    log_sha256 = _sha256_file(LOG_FILE)
    elapsed = time.perf_counter() - started_wall
    receipt = {
        "receipt_id": "EXP-045-Sanger-v4-five-slot-mask-execution-v2",
        "experiment_id": "EXP-045",
        "status": "FIVE_PROVISIONAL_CPU_MASKS_EXECUTED_WITH_SIDM00117_GROUP_EXCLUDED_IN_THREE_INNER_SLOTS; FULL_SOURCE_IDENTITY_ISOLATION_BLOCKED; D1_TRAINING_BLOCKED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_sol_dispatch": SOL_CANDIDATE_GROUP_RULING,
        "decision_summary": "A versioned v2 run preserves v1 and applies the Sol-authorized conservative PT-9eMhTl group exclusion only to inner outer0/inner1, outer1/inner2, and outer4/inner1. This does not resolve SIDM00117 or historical checkpoint exposure and does not authorize training.",
        "runtime": runtime,
        "elapsed_wall_seconds": elapsed,
        "gpu_hours": 0.0,
        "input_hashes": input_hashes,
        "parent_mask_execution_v1": v1_parent,
        "sol_approved_conservative_candidate_group_exclusion": {
            "source_Sanger_ID": UNRESOLVED_CANDIDATE_SANGER_ID,
            "identity_status": "UNRESOLVED_ONE_TO_MANY; CONSERVATIVE_CANDIDATE_GROUP_EXCLUSION_ONLY",
            "PatientID": UNRESOLVED_CANDIDATE_PATIENT_ID,
            "whole_PatientID_group_ModelIDs": UNRESOLVED_CANDIDATE_MODEL_IDS,
            "ModelIDs_in_availability_sidecar": UNRESOLVED_CANDIDATE_MODELS_IN_SIDECAR,
            "applied_slots": [
                {"level": level, "outer_fold": outer, "inner_fold": inner}
                for level, outer, inner in sorted(CANDIDATE_GROUP_EXCLUSION_SLOTS)
            ],
            "identity_resolved": False,
            "historical_checkpoint_exposure": "UNKNOWN",
        },
        "v4_identity_projection_receipt_sha256": EXPECTED_SHA256["v4_identity_projection_receipt"],
        "official_identity_crosswalk_sha256": EXPECTED_SHA256["official_public_identity_crosswalk"],
        "sidecar": sidecar_info,
        "mask_archive": {
            "path": str(MASK_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": masks_sha256,
        },
        "fold_exclusion_manifest": {
            "path": str(EXCLUSION_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": exclusion_sha256,
        },
        "mask_execution_log": {
            "path": str(LOG_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": log_sha256,
        },
        "slots": execution_log["slots"],
        "checkpoint_lineage": "NOT_READ; NOT_REQUIRED_FOR_THIS_MASK_STAGE; REQUIRED_BEFORE_CHECKPOINT_USE_OR_TRAINING_SAFETY_CONCLUSION",
        "nature_13_group_bridge": "UNCHANGED_AND_UNRESOLVED",
        "D1_training": "BLOCKED",
        "provisional_masks_only": True,
        "complete_source_identity_isolation_claim": False,
        "D_E_section_15_4": "UNCHANGED_POSITIVE_WEIGHT_TWO_STAGE_MASKED_EXPRESSION_RECONSTRUCTION",
        "next_action": "Resolve remaining SIDM01117 and Nature-to-DepMap object bridges, checkpoint lineage, real prepare_fold mask binding, independent function holdout and shared GPU budget before any D1 training.",
    }
    _atomic_json(RECEIPT_FILE, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = run()
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "receipt": str(RECEIPT_FILE.relative_to(ROOT)).replace("\\", "/"),
        "gpu_hours": result["gpu_hours"],
        "slots": [slot["slot"] for slot in result["slots"]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
