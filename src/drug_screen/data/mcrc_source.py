"""Strict, provenance-preserving rebuild primitives for the public mCRC PDO source."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.transfer.chemistry import standardize_smiles


SAMPLE_RE = re.compile(r"Pt\d+_t\d+$")
LIBRARY_SUFFIX_RE = re.compile(r"_lib[12]$")
COMBINATION_OR_BIOLOGIC_RE = re.compile(r"(?:&|tas-102|cetuximab|fliri|flox|flv|mapk\.3|volasertib\.sn-?38)", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebuild_log2cpm(raw_counts: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Sum duplicate HGNC rows then calculate log2(CPM + 1) from full library totals."""
    if "hgnc_symbol" not in raw_counts:
        raise ValueError("Data S2 requires hgnc_symbol")
    sample_ids = tuple(str(column) for column in raw_counts.columns if SAMPLE_RE.fullmatch(str(column)))
    if not sample_ids:
        raise ValueError("Data S2 has no Pt*_t* count columns")
    frame = raw_counts.dropna(subset=["hgnc_symbol"]).copy()
    frame["hgnc_symbol"] = frame["hgnc_symbol"].astype(str).str.strip().str.upper()
    frame = frame[frame["hgnc_symbol"].ne("")]
    counts = frame.loc[:, ["hgnc_symbol", *sample_ids]].copy()
    for sample_id in sample_ids:
        counts[sample_id] = pd.to_numeric(counts[sample_id], errors="raise")
    grouped = counts.groupby("hgnc_symbol", sort=True)[list(sample_ids)].sum()
    values = grouped.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Data S2 counts must be finite and non-negative")
    library_totals = values.sum(axis=0)
    if (library_totals <= 0).any():
        raise ValueError("Data S2 has an empty sequencing library")
    expression = np.log2((values / library_totals[None, :]) * 1_000_000 + 1.0).T
    return expression, sample_ids, tuple(grouped.index)


def audited_name_to_cid(legacy_mapping_path: Path) -> dict[str, int]:
    rows = json.loads(legacy_mapping_path.read_text(encoding="utf-8"))
    mapping = {str(row["primary_name"]): int(row["pubchem_cid"]) for row in rows}
    if len(mapping) != 49 or len(set(mapping.values())) != 49:
        raise ValueError("legacy audited mapping must contain exactly 49 unique names and CIDs")
    return mapping


def _properties(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        return list(payload.get("PropertyTable", {}).get("Properties", []))
    raise ValueError("fixed PubChem mapping must be a PropertyTable payload")


def resolve_isomeric_structures(
    legacy_mapping_path: Path, fixed_pubchem_path: Path
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Join only fixed historical CIDs to explicit isomeric SMILES and full keys."""
    name_to_cid = audited_name_to_cid(legacy_mapping_path)
    properties = {int(row["CID"]): row for row in _properties(json.loads(fixed_pubchem_path.read_text(encoding="utf-8")))}
    structures: dict[str, dict[str, object]] = {}
    excluded: list[dict[str, object]] = []
    for name, cid in name_to_cid.items():
        row = properties.get(cid)
        if row is None or not isinstance(row.get("SMILES"), str) or not isinstance(row.get("InChIKey"), str):
            raise ValueError(f"missing fixed isomeric mapping for {name!r} CID {cid}")
        identity = standardize_smiles(row["SMILES"])
        if not identity.accepted:
            reason = "unsupported_coordination_representation" if name == "Oxaliplatin" else identity.status
            excluded.append({"drug_name": name, "pubchem_cid": cid, "reason": reason, **asdict(identity)})
            continue
        if identity.inchikey != row["InChIKey"]:
            raise ValueError(f"full InChIKey mismatch for {name!r} CID {cid}")
        structures[name] = {
            "drug_name": name, "pubchem_cid": cid, "original_smiles": row["SMILES"],
            "canonical_isomeric_smiles": identity.canonical_isomeric_smiles,
            "drug_id": identity.inchikey, "inchikey": identity.inchikey,
            "standardization_status": identity.status, "removed_components": list(identity.removed_components),
        }
    return structures, excluded


def rebuild_dss(
    raw_dss: pd.DataFrame, *, allowed_names: set[str], structures: dict[str, dict[str, object]]
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if "sample_id" not in raw_dss:
        raise ValueError("Data S5 requires sample_id")
    long = raw_dss.melt(id_vars=["sample_id"], var_name="source_column", value_name="response")
    long["drug_name"] = long["source_column"].astype(str).str.replace(LIBRARY_SUFFIX_RE, "", regex=True)
    long["response"] = pd.to_numeric(long["response"], errors="coerce")
    exclusions: list[dict[str, str]] = []
    present_names = sorted(long.drug_name.unique())
    for name in present_names:
        if name not in allowed_names:
            if name == "run_id" or name.startswith("run_id_"):
                reason = "source_metadata_column"
            elif COMBINATION_OR_BIOLOGIC_RE.search(name):
                reason = "combination_or_biologic"
            else:
                reason = "not_in_audited_whitelist"
            exclusions.append({"drug_name": name, "reason": reason})
    mapped_names = allowed_names.intersection(present_names)
    for name in sorted(name for name in mapped_names if name not in structures):
        exclusions.append({"drug_name": name, "reason": "quarantined_fixed_structure"})
    long = long[long.drug_name.isin(structures)].dropna(subset=["response"]).copy()
    long["sample_id"] = long["sample_id"].astype(str)
    long["drug_id"] = long.drug_name.map(lambda name: structures[name]["drug_id"])
    pairs = long.groupby(["sample_id", "drug_id", "drug_name"], as_index=False, sort=True).response.median()
    pairs["higher_is_sensitive"] = True
    pairs["response_name"] = "DSS"
    return pairs, exclusions


def build_sample_metadata(raw_s1: pd.DataFrame, sample_ids: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Retain all expression samples while recording the strength of S1 identity evidence."""
    required = {"sample_id", "patient", "sample_type"}
    missing_columns = required.difference(raw_s1.columns)
    if missing_columns:
        raise ValueError(f"Data S1 lacks required columns: {sorted(missing_columns)}")
    metadata = raw_s1.loc[:, ["sample_id", "patient", "sample_type"]].copy()
    for column in metadata.columns:
        metadata[column] = metadata[column].where(metadata[column].notna(), None)
        metadata[column] = metadata[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    metadata = metadata[metadata.sample_id.map(lambda value: isinstance(value, str) and bool(value))]
    records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for sample_id in sample_ids:
        rows = metadata[metadata.sample_id.eq(sample_id)]
        patients = sorted({value for value in rows.patient if isinstance(value, str) and value})
        sample_types = sorted({value for value in rows.sample_type if isinstance(value, str) and value})
        prefix = sample_id.split("_", 1)[0]
        if any(patient != prefix for patient in patients):
            raise ValueError(f"Data S1 patient conflicts with sample prefix for {sample_id}: {patients}")
        if len(patients) > 1:
            raise ValueError(f"Data S1 has ambiguous patient mapping for {sample_id}: {patients}")
        if len(patients) == 1 and "PDO" in sample_types:
            patient_id = patients[0]
            provenance = "Data S1 patient with PDO sample_type"
            identity_status = "S1_PDO_patient_verified"
        else:
            patient_id = prefix
            provenance = "source_sample_prefix_not_independently_verified"
            identity_status = "pending_G0_identity_verification"
            if rows.empty:
                exclusions.append({"sample_id": sample_id, "reason": "missing_S1_mutation_record"})
            elif not patients:
                exclusions.append({"sample_id": sample_id, "reason": "missing_S1_patient"})
            else:
                exclusions.append({"sample_id": sample_id, "reason": "no_PDO_mapping_in_S1", "sample_types": sample_types})
        records.append({
            "sample_id": sample_id,
            "patient_id": patient_id,
            "patient_id_provenance": provenance,
            "identity_status": identity_status,
            "sample_type": "PDO",
            "source_sample_types": ";".join(sample_types),
            "study_id": "mcrc_pdo_2026_v3",
            "domain": "PDO",
            "cancer_type": "mCRC",
        })
    return pd.DataFrame(records), exclusions


@dataclass(frozen=True)
class McrcRebuild:
    expression: np.ndarray
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    samples: pd.DataFrame
    drugs: pd.DataFrame
    pairs: pd.DataFrame
    exclusions: list[dict[str, object]]


def rebuild_mcrc(
    raw_s1: pd.DataFrame,
    raw_s2: pd.DataFrame,
    raw_s5: pd.DataFrame,
    legacy_mapping_path: Path,
    fixed_pubchem_path: Path,
) -> McrcRebuild:
    expression, sample_ids, gene_ids = rebuild_log2cpm(raw_s2)
    structures, structure_exclusions = resolve_isomeric_structures(legacy_mapping_path, fixed_pubchem_path)
    allowed = set(audited_name_to_cid(legacy_mapping_path))
    pairs, response_exclusions = rebuild_dss(raw_s5, allowed_names=allowed, structures=structures)
    samples, metadata_exclusions = build_sample_metadata(raw_s1, sample_ids)
    pairs = pairs[pairs.sample_id.isin(sample_ids)].copy()
    drugs = pd.DataFrame([value for name, value in sorted(structures.items()) if value["drug_id"] in set(pairs.drug_id)])
    return McrcRebuild(expression, sample_ids, gene_ids, samples, drugs, pairs, structure_exclusions + response_exclusions + metadata_exclusions)


@dataclass(frozen=True)
class ProcessedMcrcV3Bundle:
    """Frozen processed mCRC v3 tables under data/processed (raw never opened).

    Why: modeling/eval agents need a stable loader onto already-materialized
    transfer artifacts without re-touching Data S1/S2/S5 raw supplements.
    Preprocessing stats for any future feature scaling must still fit on the
    training fold only — this loader does not fit scalers.
    """

    samples: pd.DataFrame
    drugs: pd.DataFrame
    pairs: pd.DataFrame
    drug_identity_48: pd.DataFrame
    manifest: dict
    root: Path


def default_processed_mcrc_v3_root(data_root: Path | None = None) -> Path:
    base = Path("data") if data_root is None else Path(data_root)
    return base / "processed" / "transfer" / "mcrc_v3"


def load_kryeziu_drug_identity_48_draft(data_root: Path | None = None) -> pd.DataFrame:
    """Load the DRAFT 48-drug identity table (encodable monotherapies).

    Oxaliplatin is intentionally absent; audit quarantines it for SMILES-graph
    inputs. Do not silently re-add it.
    """
    base = Path("data") if data_root is None else Path(data_root)
    parquet = base / "processed" / "identity" / "kryeziu_drug_identity_48_DRAFT.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    frame = pd.read_parquet(parquet)
    if len(frame) != 48:
        raise ValueError(f"expected 48 drugs, found {len(frame)} at {parquet}")
    return frame


def load_processed_mcrc_v3(data_root: Path | None = None) -> ProcessedMcrcV3Bundle:
    """Load processed transfer/mcrc_v3 samples, drugs, pairs + 48-drug DRAFT IDs.

    Reads only under data/processed. Does not open data/raw.
    """
    root = default_processed_mcrc_v3_root(data_root)
    required = {
        "samples": root / "samples.parquet",
        "drugs": root / "drugs.parquet",
        "pairs": root / "pairs.parquet",
        "manifest": root / "manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing processed mcrc_v3 assets: " + ", ".join(missing))
    samples = pd.read_parquet(required["samples"])
    drugs = pd.read_parquet(required["drugs"])
    pairs = pd.read_parquet(required["pairs"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if len(drugs) != 48:
        raise ValueError(f"processed drugs.parquet expected 48 rows, found {len(drugs)}")
    identity = load_kryeziu_drug_identity_48_draft(data_root)
    return ProcessedMcrcV3Bundle(
        samples=samples,
        drugs=drugs,
        pairs=pairs,
        drug_identity_48=identity,
        manifest=manifest,
        root=root,
    )

