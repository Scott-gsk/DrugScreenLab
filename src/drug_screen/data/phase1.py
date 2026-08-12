"""Phase-1 exact-978 manifest and chemical feature preparation."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _structure_fingerprint(smiles: str, n_bits: int) -> np.ndarray:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as error:  # pragma: no cover - environment contract handles this
        raise RuntimeError("RDKit is required for structure-derived Phase-1 features") from error
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid canonical_smiles: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=n_bits)
    result = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def build_chemical_feature_table(perturbagens: pd.DataFrame, *, n_bits: int = 128) -> tuple[np.ndarray, dict[str, int]]:
    """Build deterministic Morgan fingerprints keyed by canonical perturbagen ID."""
    required = {"pert_id", "canonical_smiles"}
    missing = required.difference(perturbagens.columns)
    if missing:
        raise ValueError(f"perturbagens missing columns: {sorted(missing)}")
    if n_bits < 8:
        raise ValueError("n_bits must be at least 8")
    rows = perturbagens[["pert_id", "canonical_smiles"]].drop_duplicates("pert_id").sort_values("pert_id")
    features: list[np.ndarray] = []
    mapping: dict[str, int] = {}
    for index, row in enumerate(rows.itertuples(index=False)):
        pert_id = str(row.pert_id)
        smiles = str(row.canonical_smiles)
        try:
            feature = _structure_fingerprint(smiles, n_bits)
        except ValueError:
            continue
        mapping[pert_id] = len(features)
        features.append(feature)
    if not features:
        raise ValueError("no valid chemical structures remain")
    return np.stack(features).astype(np.float32, copy=False), mapping


def select_canonical_condition(
    instances: pd.DataFrame,
    *,
    dose_um: float,
    time_h: float,
    tolerance: float = 1e-5,
) -> pd.DataFrame:
    """Select one high-coverage canonical chemical dose/time condition."""
    required = {"pert_type", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"}
    missing = required.difference(instances.columns)
    if missing:
        raise ValueError(f"instances missing columns: {sorted(missing)}")
    dose = pd.to_numeric(instances["pert_dose"], errors="coerce")
    time = pd.to_numeric(instances["pert_time"], errors="coerce")
    mask = (
        instances["pert_type"].eq("trt_cp")
        & instances["pert_dose_unit"].eq("um")
        & instances["pert_time_unit"].eq("h")
        & np.isclose(dose, dose_um, rtol=tolerance, atol=tolerance)
        & np.isclose(time, time_h, rtol=tolerance, atol=tolerance)
    )
    return instances.loc[mask].copy()


def assign_entity_split(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> np.ndarray:
    """Assign a deterministic split without separating any entity."""
    if entity_column not in frame.columns:
        raise ValueError(f"missing entity column: {entity_column}")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave a positive test fraction")
    labels: list[str] = []
    for entity in frame[entity_column].astype(str):
        digest = sha256(f"{seed}:{entity}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        labels.append("train" if value < train_fraction else "validation" if value < train_fraction + validation_fraction else "test")
    return np.asarray(labels, dtype=object)


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def build_phase1_manifest(
    *,
    data_root: Path | str,
    output_dir: Path | str,
    repo_root: Path | str,
    dose_um: float = 10.0,
    time_h: float = 6.0,
    split_mode: str = "random_group",
    split_seed: int = 20260812,
    n_bits: int = 128,
) -> dict[str, object]:
    """Create a compact Phase-1 manifest from immutable GSE92742 metadata."""
    root = Path(repo_root).resolve()
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lincs = data / "raw" / "lincs" / "GSE92742"
    instances = pd.read_csv(lincs / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False)
    perturbagens = pd.read_csv(lincs / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", low_memory=False)
    cells = pd.read_csv(lincs / "GSE92742_Broad_LINCS_cell_info.txt.gz", sep="\t", low_memory=False)
    instances["_cache_row"] = np.arange(len(instances), dtype=np.int64)
    selected = select_canonical_condition(instances, dose_um=dose_um, time_h=time_h)
    selected = selected.loc[:, [
        "inst_id", "rna_plate", "pert_id", "pert_dose", "pert_dose_unit", "pert_time",
        "pert_time_unit", "cell_id", "_cache_row",
    ]].copy()
    perturbagens = perturbagens.loc[perturbagens["pert_type"].eq("trt_cp")].copy()
    valid_perturbagens = perturbagens.loc[
        perturbagens["canonical_smiles"].notna()
        & ~perturbagens["canonical_smiles"].astype(str).isin({"-666", "nan", ""})
    ].copy()
    selected = selected.merge(
        valid_perturbagens[["pert_id", "canonical_smiles", "inchi_key"]],
        on="pert_id", how="inner", validate="many_to_one",
    )
    selected = selected.merge(cells[["cell_id", "base_cell_id"]], on="cell_id", how="left", validate="many_to_one")
    selected["context_id"] = selected["cell_id"].astype(str)
    selected["group_id"] = selected.apply(
        lambda row: f"{row.pert_id}|{row.context_id}|{float(row.pert_dose):g}um|{float(row.pert_time):g}h", axis=1
    )
    controls = instances.loc[instances["pert_type"].eq("ctl_vehicle")].copy()
    controls["match_key"] = controls[["rna_plate", "cell_id", "pert_time", "pert_time_unit"]].astype(str).agg("||".join, axis=1)
    control_first = controls.sort_values(["match_key", "_cache_row"]).drop_duplicates("match_key")
    selected["match_key"] = selected[["rna_plate", "cell_id", "pert_time", "pert_time_unit"]].astype(str).agg("||".join, axis=1)
    selected = selected.merge(
        control_first[["match_key", "_cache_row"]].rename(columns={"_cache_row": "control_cache_row"}),
        on="match_key", how="inner", validate="many_to_one",
    )
    selected["treatment_cache_row"] = selected["_cache_row"].astype(np.int64)
    if selected.empty:
        raise RuntimeError("canonical condition produced no matched-control chemical records")
    if split_mode == "random_group":
        selected["split"] = assign_entity_split(selected, entity_column="group_id", seed=split_seed)
    elif split_mode == "cold_drug":
        selected["split"] = assign_entity_split(selected, entity_column="pert_id", seed=split_seed)
    elif split_mode == "cold_context":
        selected["split"] = assign_entity_split(selected, entity_column="context_id", seed=split_seed)
    else:
        raise ValueError("split_mode must be random_group, cold_drug, or cold_context")
    chemical_features, feature_mapping = build_chemical_feature_table(
        selected[["pert_id", "canonical_smiles"]].drop_duplicates("pert_id"), n_bits=n_bits
    )
    selected = selected.loc[selected["pert_id"].isin(feature_mapping)].copy()
    if selected.empty:
        raise RuntimeError("canonical condition has no valid structure-derived chemical records")
    chemical_path = output / "chemical_features.npy"
    np.save(chemical_path, chemical_features)
    cache_path = root / "data" / "processed" / "lincs" / "GSE92742" / "exact978_cache_v1" / "exact978_cache.npy"
    cache_manifest_path = cache_path.with_name("asset_manifest.json")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    registered_cache_sha256 = str(cache_manifest["cache_sha256"])
    manifest_records = []
    for row in selected.sort_values(["split", "group_id", "inst_id"]).itertuples(index=False):
        manifest_records.append({
            "sample_id": str(row.inst_id),
            "treatment_group_id": str(row.group_id),
            "drug_id": str(row.pert_id),
            "context_id": str(row.context_id),
            "dose_um": float(row.pert_dose),
            "time_h": float(row.pert_time),
            "split": str(row.split),
            "treatment_cache_row": int(row.treatment_cache_row),
            "control_cache_row": int(row.control_cache_row),
            "chemical_feature_row": int(feature_mapping[str(row.pert_id)]),
        })
    manifest_path = output / "manifest.json"
    payload: Mapping[str, object] = {
        "format": "phase1_context_chemical_manifest_v1",
        "phase": "phase_1_canonical_chemical",
        "condition": {"dose_um": dose_um, "time_h": time_h},
        "split_mode": split_mode,
        "split_seed": split_seed,
        "control_policy": "pre_treatment_context_feature",
        "gene_count": 978,
        "cache": {
            "relative_path": _relative(cache_path, root),
            "sha256": registered_cache_sha256,
            "shape": [1319138, 978],
            "asset_id": "lincs_gse92742_exact978_cache_v1",
        },
        "chemical_features": {
            "relative_path": _relative(chemical_path, root),
            "sha256": _digest(chemical_path),
            "shape": list(chemical_features.shape),
            "representation": "RDKit Morgan radius=2 bit vector",
        },
        "records": manifest_records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "manifest": _relative(manifest_path, root),
        "records": len(manifest_records),
        "drugs": int(selected["pert_id"].nunique()),
        "contexts": int(selected["context_id"].nunique()),
        "groups": int(selected["group_id"].nunique()),
        "matched_controls": int(selected["control_cache_row"].nunique()),
        "split_counts": selected["split"].value_counts().sort_index().to_dict(),
        "condition": {"dose_um": dose_um, "time_h": time_h},
        "split_mode": split_mode,
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
