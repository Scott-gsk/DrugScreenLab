"""Auditable additive fusion of frozen KPGT and UniPert chemical features.

This module deliberately separates representation compatibility from model
training.  It consumes already-generated feature tables and aligns them by
canonical ``pert_id`` before concatenation; no treatment labels, controls,
splits, or external phenotype values are read for the representation step.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FUSED_FORMAT = "phase1_context_kpgt_unipert_manifest_v1"
FUSED_FEATURE_FORMAT = "kpgt_unipert_additive_feature_table_v1"


class Phase2FusionError(RuntimeError):
    """Raised when frozen feature tables cannot be aligned safely."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_features(path: Path) -> np.ndarray:
    if not path.is_file():
        raise Phase2FusionError(f"feature table is missing: {path}")
    try:
        if path.suffix == ".npz":
            archive = np.load(path)
            keys = list(archive.files)
            if len(keys) != 1:
                raise Phase2FusionError(
                    f"feature archive must contain exactly one array: {path}"
                )
            array = archive[keys[0]]
        else:
            array = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise Phase2FusionError(f"feature table is unreadable: {path}") from error
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise Phase2FusionError(f"feature table must be a non-empty 2D array: {path}")
    if not np.isfinite(array).all():
        raise Phase2FusionError(f"feature table contains non-finite values: {path}")
    return array.astype(np.float32, copy=False)


def _load_mapping(path: Path, row_count: int) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2FusionError(f"feature mapping is unreadable: {path}") from error
    if isinstance(payload, Mapping) and "mapping" in payload:
        payload = payload["mapping"]
    if not isinstance(payload, Mapping) or not payload:
        raise Phase2FusionError("feature mapping must be a non-empty object")
    result: dict[str, int] = {}
    used: set[int] = set()
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip() or isinstance(value, bool):
            raise Phase2FusionError("feature mapping keys and rows must be canonical")
        try:
            row = int(value)
        except (TypeError, ValueError) as error:
            raise Phase2FusionError(f"invalid feature row for {key}") from error
        if row < 0 or row >= row_count or row in used:
            raise Phase2FusionError(f"feature row is out of range or duplicated: {key}")
        result[key] = row
        used.add(row)
    return result


def build_kpgt_unipert_features(
    drug_ids: Iterable[str],
    *,
    kpgt_features_path: Path | str,
    kpgt_mapping_path: Path | str,
    unipert_features_path: Path | str,
    unipert_mapping_path: Path | str,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Align and concatenate frozen KPGT + UniPert rows by ``pert_id``."""
    kpgt_path = Path(kpgt_features_path)
    unipert_path = Path(unipert_features_path)
    kpgt = _load_features(kpgt_path)
    unipert = _load_features(unipert_path)
    kpgt_mapping = _load_mapping(Path(kpgt_mapping_path), kpgt.shape[0])
    unipert_mapping = _load_mapping(Path(unipert_mapping_path), unipert.shape[0])
    requested = sorted({str(value) for value in drug_ids if str(value).strip()})
    if not requested:
        raise Phase2FusionError("no drug identities were supplied")
    missing_kpgt = sorted(set(requested).difference(kpgt_mapping))
    missing_unipert = sorted(set(requested).difference(unipert_mapping))
    if missing_kpgt or missing_unipert:
        raise Phase2FusionError(
            f"feature identity mismatch; missing_kpgt={missing_kpgt[:5]}, "
            f"missing_unipert={missing_unipert[:5]}"
        )
    fused = np.stack(
        [
            np.concatenate(
                [kpgt[kpgt_mapping[drug_id]], unipert[unipert_mapping[drug_id]]],
                axis=0,
            )
            for drug_id in requested
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    mapping = {drug_id: index for index, drug_id in enumerate(requested)}
    audit = {
        "format": FUSED_FEATURE_FORMAT,
        "alignment_key": "pert_id",
        "requested_drugs": len(requested),
        "kpgt_source": {
            "path": str(kpgt_path),
            "sha256": file_sha256(kpgt_path),
            "shape": list(kpgt.shape),
            "output_dim": int(kpgt.shape[1]),
        },
        "unipert_source": {
            "path": str(unipert_path),
            "sha256": file_sha256(unipert_path),
            "shape": list(unipert.shape),
            "output_dim": int(unipert.shape[1]),
        },
        "feature_shape": list(fused.shape),
        "labels_used": False,
    }
    return fused, mapping, audit


def build_fused_manifest(
    *,
    base_manifest_path: Path | str,
    kpgt_features_path: Path | str,
    kpgt_mapping_path: Path | str,
    unipert_features_path: Path | str,
    unipert_mapping_path: Path | str,
    output_dir: Path | str,
    root: Path | str,
) -> dict[str, Any]:
    """Create a frozen Phase-1 manifest with additive KPGT+UniPert features."""
    base_path = Path(base_manifest_path)
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise Phase2FusionError("base Phase-1 manifest has no records")
    drug_ids = [str(row["drug_id"]) for row in records]
    features, mapping, audit = build_kpgt_unipert_features(
        drug_ids,
        kpgt_features_path=kpgt_features_path,
        kpgt_mapping_path=kpgt_mapping_path,
        unipert_features_path=unipert_features_path,
        unipert_mapping_path=unipert_mapping_path,
    )

    repo_root = Path(root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "kpgt_unipert_chemical_features.npy"
    np.save(feature_path, features)
    updated_records = []
    for row in records:
        updated = dict(row)
        updated["chemical_feature_row"] = int(mapping[str(row["drug_id"])])
        updated_records.append(updated)
    updated_payload = dict(payload)
    updated_payload["format"] = FUSED_FORMAT
    updated_payload["phase"] = "phase_2_fast_kpgt_unipert_additive_representation"
    updated_payload["chemical_features"] = {
        "relative_path": str(feature_path.relative_to(repo_root)).replace("\\", "/"),
        "sha256": file_sha256(feature_path),
        "shape": list(features.shape),
        "representation": "KPGT structural representation concatenated with official UniPert chemical representation",
        "source_model_sha256": {
            "kpgt_feature_table": audit["kpgt_source"]["sha256"],
            "unipert_feature_table": audit["unipert_source"]["sha256"],
        },
    }
    updated_payload["records"] = updated_records
    updated_payload["frozen_upstream_manifest_sha256"] = file_sha256(base_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(updated_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit["manifest"] = str(manifest_path)
    audit["manifest_sha256"] = file_sha256(manifest_path)
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
