"""Read-only EXP-009 Morgan teacher feature builder for XPert/SDST identities.

This module does not load SDST response values or modify the official XPert
backbone.  It uses the frozen XPert registry's canonical SMILES for every
``global_inference_eligible`` drug, applies the approved 2048-bit Morgan
teacher, and emits 64-dimensional teacher logits and probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn

from drug_screen.foundation.xpert_registry import DRUG_REGISTRY_FORMAT, file_sha256, validate_drug_registry_payload


EXPECTED_MORGAN_BITS = 2048
EXPECTED_TARGET_DIM = 64


@dataclass(frozen=True)
class TeacherSoftTargetFeatures:
    """In-memory feature payload with its complete identity audit."""

    pert_ids: np.ndarray
    pert_indices: np.ndarray
    inchi_keys: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    feature_valid: np.ndarray
    confidence: np.ndarray
    targets: tuple[str, ...]
    audit: dict[str, Any]


def _normalize_inchikey(value: object) -> str:
    return str(value or "").strip().upper()


def _morgan_fingerprint(smiles: str) -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(smiles or "").split(" |", 1)[0].strip())
    if molecule is None:
        raise ValueError("registry canonical_smiles cannot be parsed")
    bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=EXPECTED_MORGAN_BITS)
    return np.asarray(bits, dtype=np.float32)


def _load_checkpoint(checkpoint_path: Path) -> tuple[nn.Linear, tuple[str, ...], dict[str, Any]]:
    checkpoint_sha256 = file_sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("teacher checkpoint must be a dictionary payload")
    input_dim = payload.get("input_dim")
    target_dim = payload.get("target_dim")
    targets = payload.get("targets")
    state_dict = payload.get("state_dict")
    if input_dim != EXPECTED_MORGAN_BITS:
        raise ValueError(f"teacher checkpoint requires input_dim=2048, received {input_dim!r}")
    if target_dim != EXPECTED_TARGET_DIM:
        raise ValueError(f"teacher checkpoint requires target_dim=64, received {target_dim!r}")
    if not isinstance(targets, list) or len(targets) != EXPECTED_TARGET_DIM or any(not str(value) for value in targets):
        raise ValueError("teacher checkpoint targets must be 64 non-empty entries")
    if not isinstance(state_dict, dict):
        raise ValueError("teacher checkpoint missing state_dict")
    model = nn.Linear(EXPECTED_MORGAN_BITS, EXPECTED_TARGET_DIM)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"teacher checkpoint is incompatible with Linear(2048, 64): {error}") from error
    model.eval()
    return model, tuple(str(value) for value in targets), {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_access": "read_only_load",
        "checkpoint_schema": "torch_linear_state_dict_v1",
        "input_dim": EXPECTED_MORGAN_BITS,
        "target_dim": EXPECTED_TARGET_DIM,
    }


def _eligible_registry_rows(registry_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("format") != DRUG_REGISTRY_FORMAT:
        raise ValueError(f"unexpected XPert registry format in {registry_path}")
    validate_drug_registry_payload(raw)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in raw["drugs"]:
        if not bool(record.get("global_inference_eligible")):
            continue
        pert_id = str(record.get("pert_id") or "").strip()
        inchi_key = _normalize_inchikey(record.get("inchi_key"))
        smiles = str(record.get("canonical_smiles") or "").strip()
        if not pert_id:
            raise ValueError("eligible XPert registry row requires a pert_id")
        if pert_id in seen_ids:
            raise ValueError(f"duplicate eligible XPert pert_id: {pert_id}")
        seen_ids.add(pert_id)
        try:
            pert_idx = int(record["pert_idx"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"eligible XPert registry row has invalid pert_idx for {pert_id}") from error
        rows.append({"pert_id": pert_id, "pert_idx": pert_idx, "inchi_key": inchi_key, "canonical_smiles": smiles})
    rows.sort(key=lambda row: (row["pert_id"], row["pert_idx"]))
    if not rows:
        raise ValueError("XPert registry has no global_inference_eligible drugs")
    return rows, {
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "registry_format": DRUG_REGISTRY_FORMAT,
        "eligible_drug_count": len(rows),
        "identity_selection": "frozen global_inference_eligible XPert rows sorted by pert_id then pert_idx",
    }


def build_teacher_soft_target_features(
    registry_path: Path | str,
    checkpoint_path: Path | str,
    *,
    batch_size: int = 512,
    max_drugs: int | None = None,
) -> TeacherSoftTargetFeatures:
    """Generate a 64-D Morgan teacher feature row for every XPert-eligible drug."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_drugs is not None and max_drugs <= 0:
        raise ValueError("max_drugs must be positive")
    registry_file = Path(registry_path)
    checkpoint_file = Path(checkpoint_path)
    rows, registry_audit = _eligible_registry_rows(registry_file)
    if max_drugs is not None:
        rows = rows[: int(max_drugs)]
        registry_audit = dict(registry_audit)
        registry_audit["selection_limit"] = int(max_drugs)
        registry_audit["eligible_drug_count"] = len(rows)
    model, targets, checkpoint_audit = _load_checkpoint(checkpoint_file)
    if max_drugs is not None:
        if max_drugs <= 0:
            raise ValueError("max_drugs must be positive")
        rows = rows[:max_drugs]
    feature_valid = np.zeros(len(rows), dtype=bool)
    fingerprints: list[np.ndarray] = []
    valid_positions: list[int] = []
    missing_inchi_key_count = 0
    invalid_smiles_count = 0
    for position, row in enumerate(rows):
        if not row["inchi_key"]:
            missing_inchi_key_count += 1
        try:
            fingerprint = _morgan_fingerprint(row["canonical_smiles"])
        except ValueError:
            invalid_smiles_count += 1
            continue
        feature_valid[position] = True
        fingerprints.append(fingerprint)
        valid_positions.append(position)
    logits = np.full((len(rows), EXPECTED_TARGET_DIM), np.nan, dtype=np.float32)
    if fingerprints:
        valid_fingerprints = np.stack(fingerprints).astype(np.float32)
        logits_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(valid_fingerprints), batch_size):
                inputs = torch.from_numpy(valid_fingerprints[start : start + batch_size])
                logits_batches.append(model(inputs).cpu().numpy().astype(np.float32, copy=False))
        logits[np.asarray(valid_positions, dtype=np.int64)] = np.concatenate(logits_batches, axis=0)
    probabilities = np.full_like(logits, np.nan, dtype=np.float32)
    probabilities[feature_valid] = 1.0 / (1.0 + np.exp(-logits[feature_valid]))
    # Confidence is the maximum target probability; invalid structures remain zero.
    confidence = np.zeros(len(rows), dtype=np.float32)
    confidence[feature_valid] = probabilities[feature_valid].max(axis=1).astype(np.float32)
    if logits.shape != (len(rows), EXPECTED_TARGET_DIM):
        raise RuntimeError(f"unexpected generated logits shape: {logits.shape}")
    audit = {
        "format": "exp009_morgan_teacher_xpert_sdst_features_v1",
        "experiment_id": "EXP-009",
        "chain": "BindingDB_Morgan_teacher_to_XPert_SDST_soft_target_features",
        "input_boundary": ["frozen XPert drug registry", "read-only Morgan teacher checkpoint"],
        "forbidden_inputs": ["SDST response", "drug efficacy", "disease signature", "external validation labels"],
        "response_values_read": False,
        "efficacy_values_read": False,
        "official_xpert_backbone_modified": False,
        "morgan_representation": {"radius": 2, "n_bits": EXPECTED_MORGAN_BITS, "dtype": "float32"},
        "feature_shape": [len(rows), EXPECTED_TARGET_DIM],
        "valid_feature_count": int(feature_valid.sum()),
        "invalid_structure_count": int((~feature_valid).sum()),
        "missing_inchi_key_count": int(missing_inchi_key_count),
        "invalid_smiles_count": int(invalid_smiles_count),
        "invalid_feature_policy": "feature_valid=false, confidence=0.0, and all logits/probabilities=NaN; never a zero-vector surrogate",
        "confidence_definition": "max_probability_across_64_targets",
        "target_uniprot_ids": list(targets),
        "batch_size": int(batch_size),
        **registry_audit,
        **checkpoint_audit,
    }
    return TeacherSoftTargetFeatures(
        pert_ids=np.asarray([row["pert_id"] for row in rows]),
        pert_indices=np.asarray([row["pert_idx"] for row in rows], dtype=np.int64),
        inchi_keys=np.asarray([row["inchi_key"] for row in rows]),
        logits=logits,
        probabilities=probabilities,
        feature_valid=feature_valid,
        confidence=confidence,
        targets=targets,
        audit=audit,
    )


def select_teacher_soft_target_batch(payload: TeacherSoftTargetFeatures, pert_ids: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Return teacher arrays in an SDST batch's exact ``pert_id`` order."""
    requested = [str(value) for value in pert_ids]
    index = {str(pert_id): position for position, pert_id in enumerate(payload.pert_ids.tolist())}
    missing = sorted(set(requested).difference(index))
    if missing:
        raise ValueError(f"teacher feature payload does not contain requested pert_id values: {missing}")
    positions = np.asarray([index[pert_id] for pert_id in requested], dtype=np.int64)
    return {
        "pert_id": payload.pert_ids[positions],
        "pert_idx": payload.pert_indices[positions],
        "inchi_key": payload.inchi_keys[positions],
        "soft_target_logits": payload.logits[positions],
        "soft_target_probabilities": payload.probabilities[positions],
        "feature_valid": payload.feature_valid[positions],
        "confidence": payload.confidence[positions],
    }



def write_teacher_soft_target_features(payload: TeacherSoftTargetFeatures, output_dir: Path | str) -> dict[str, Any]:
    """Persist the feature NPZ and complete JSON identity audit atomically enough for a fresh directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "xpert_sdst_soft_target_features.npz"
    np.savez_compressed(
        artifact,
        pert_id=payload.pert_ids,
        pert_idx=payload.pert_indices,
        inchi_key=payload.inchi_keys,
        target_uniprot_id=np.asarray(payload.targets),
        soft_target_logits=payload.logits,
        soft_target_probabilities=payload.probabilities,
        feature_valid=payload.feature_valid,
        confidence=payload.confidence,
    )
    audit = dict(payload.audit)
    audit.update(
        {
            "artifact": str(artifact),
            "artifact_sha256": file_sha256(artifact),
            "artifact_arrays": {
                "pert_id": [int(len(payload.pert_ids))],
                "pert_idx": [int(len(payload.pert_indices))],
                "inchi_key": [int(len(payload.inchi_keys))],
                "target_uniprot_id": [int(len(payload.targets))],
                "soft_target_logits": list(payload.logits.shape),
                "soft_target_probabilities": list(payload.probabilities.shape),
                "feature_valid": [int(len(payload.feature_valid))],
                "confidence": [int(len(payload.confidence))],
            },
        }
    )
    audit_path = destination / "identity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
