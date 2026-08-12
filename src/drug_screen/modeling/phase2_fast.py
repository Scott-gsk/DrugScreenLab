"""Small, frozen-representation probes for the authorized Phase-2/3 FAST loops.

The UniPert probe intentionally uses only the public chemical encoder weights and
does not alter the Phase-1 endpoint, split, control policy, or training objective.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


UNIPERT_DIM = 256
UNIPERT_ECFP_BITS = 2048
UNIPERT_FORMAT = "unipert_chemical_feature_table_v1"


class Phase2FastError(RuntimeError):
    """Raised when a frozen FAST representation cannot be audited safely."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ecfp4(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as error:  # pragma: no cover - environment contract
        raise Phase2FastError("RDKit is required for the UniPert chemical probe") from error
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Phase2FastError(f"invalid canonical_smiles: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=UNIPERT_ECFP_BITS
    )
    result = np.zeros((UNIPERT_ECFP_BITS,), dtype=np.float32)
    from rdkit import DataStructs

    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def _load_encoder_weights(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - error text is environment-specific
        raise Phase2FastError(f"cannot load UniPert checkpoint: {model_path}") from error
    encoder = checkpoint.get("cp_encoder") if isinstance(checkpoint, Mapping) else None
    if not isinstance(encoder, Mapping):
        raise Phase2FastError("UniPert checkpoint has no cp_encoder state")
    weight = encoder.get("linear_layer.weight")
    bias = encoder.get("linear_layer.bias")
    if weight is None or bias is None:
        raise Phase2FastError("UniPert cp_encoder is missing linear_layer weights")
    weight_np = weight.detach().cpu().numpy().astype(np.float32, copy=False)
    bias_np = bias.detach().cpu().numpy().astype(np.float32, copy=False)
    if weight_np.shape != (UNIPERT_DIM, UNIPERT_ECFP_BITS) or bias_np.shape != (UNIPERT_DIM,):
        raise Phase2FastError(
            f"unexpected UniPert chemical encoder shape: {weight_np.shape}, {bias_np.shape}"
        )
    return weight_np, bias_np


def build_unipert_chemical_features(
    perturbagens: pd.DataFrame,
    *,
    model_path: Path | str,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Encode canonical LINCS compounds with the frozen official UniPert chemical head."""
    required = {"pert_id", "canonical_smiles"}
    missing = required.difference(perturbagens.columns)
    if missing:
        raise Phase2FastError(f"perturbagens missing columns: {sorted(missing)}")
    model_file = Path(model_path)
    if not model_file.is_file():
        raise Phase2FastError(f"UniPert checkpoint is missing: {model_file}")
    weight, bias = _load_encoder_weights(model_file)
    rows = (
        perturbagens[["pert_id", "canonical_smiles"]]
        .drop_duplicates("pert_id")
        .sort_values("pert_id")
    )
    features: list[np.ndarray] = []
    mapping: dict[str, int] = {}
    for row in rows.itertuples(index=False):
        pert_id = str(row.pert_id)
        smiles = str(row.canonical_smiles)
        if smiles in {"", "nan", "-666"}:
            continue
        try:
            fingerprint = _ecfp4(smiles)
        except Phase2FastError:
            continue
        mapping[pert_id] = len(features)
        features.append(fingerprint @ weight.T + bias)
    if not features:
        raise Phase2FastError("no valid compounds were encoded by UniPert")
    array = np.stack(features).astype(np.float32, copy=False)
    audit = {
        "format": UNIPERT_FORMAT,
        "representation": "official_UniPert_chemical_encoder(ECFP4_radius2_2048_to_256)",
        "model_path": str(model_file),
        "model_sha256": file_sha256(model_file),
        "input_bits": UNIPERT_ECFP_BITS,
        "output_dim": UNIPERT_DIM,
        "encoded_drugs": len(mapping),
        "feature_shape": list(array.shape),
    }
    return array, mapping, audit


def build_unipert_manifest(
    *,
    base_manifest_path: Path | str,
    perturbagen_path: Path | str,
    model_path: Path | str,
    output_dir: Path | str,
    root: Path | str,
) -> dict[str, Any]:
    """Create a feature-only manifest variant with the original Phase-1 records frozen."""
    base_path = Path(base_manifest_path)
    repo_root = Path(root).resolve()
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise Phase2FastError("base Phase-1 manifest has no records")
    drug_ids = {str(row["drug_id"]) for row in records}
    perturbagens = pd.read_csv(perturbagen_path, sep="\t", low_memory=False)
    perturbagens = perturbagens.loc[perturbagens["pert_id"].astype(str).isin(drug_ids)].copy()
    features, mapping, audit = build_unipert_chemical_features(perturbagens, model_path=model_path)
    missing = sorted(drug_ids.difference(mapping))
    if missing:
        raise Phase2FastError(f"UniPert feature table misses manifest drugs: {missing[:5]}")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "unipert_chemical_features.npy"
    np.save(feature_path, features)
    updated_records = []
    for row in records:
        updated = dict(row)
        updated["chemical_feature_row"] = int(mapping[str(row["drug_id"])])
        updated_records.append(updated)
    updated_payload = dict(payload)
    updated_payload["format"] = "phase1_context_unipert_manifest_v1"
    updated_payload["phase"] = "phase_2_fast_unipert_chemical_representation"
    updated_payload["chemical_features"] = {
        "relative_path": str(feature_path.relative_to(repo_root)).replace("\\", "/"),
        "sha256": file_sha256(feature_path),
        "shape": list(features.shape),
        "representation": audit["representation"],
        "source_model_sha256": audit["model_sha256"],
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
