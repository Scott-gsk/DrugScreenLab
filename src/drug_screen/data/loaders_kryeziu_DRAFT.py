"""Kryeziu / mcrc_pdo_2026 loaders (DRAFT skeleton).

Rules:
- data/raw is read-only (never written here).
- Prefer processed tables + kryeziu_audit identity subset.
- 32 prefix-only samples stay excluded unless an independent crosswalk appears.
- Do not invent drug structures; join registry DRAFT instead.
- Missing assets raise FileNotFoundError with actionable paths (no silent empty).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from drug_screen.data.identity_contracts import (
    DrugIdentity,
    IdentityConfidence,
    IdentityMasterTable,
    PDOIdentity,
    PatientIdentity,
    StructureEncodingStatus,
    validate_drug_identity,
    validate_patient_fold_unit,
)

AUDIT_REL = Path(
    "data/artifacts/resource_acquisition_20260916/kryeziu_audit/"
    "kryeziu_identity_assay_audit.json"
)
DRUG_REGISTRY_REL = Path("data/registry/kryeziu_structure_confirmed_drugs_DRAFT.json")
SPLIT_FROZEN_REL = Path("data/splits/kryeziu_patient_outer5fold_FROZEN.json")
SPLIT_REL = Path("data/splits/kryeziu_patient_outer5fold_DRAFT.json")
PROCESSED_CANDIDATES = (
    Path("data/processed/mcrc_pdo_2026"),
    Path("data/processed/transfer/mcrc_v3"),
)


@dataclass(frozen=True)
class KryeziuPaths:
    root: Path
    audit: Path
    drug_registry: Path
    split: Path
    processed_roots: tuple[Path, ...]

    @classmethod
    def from_root(cls, root: Path | str) -> "KryeziuPaths":
        r = Path(root)
        return cls(
            root=r,
            audit=r / AUDIT_REL,
            drug_registry=r / DRUG_REGISTRY_REL,
            split=r / SPLIT_FROZEN_REL if (r / SPLIT_FROZEN_REL).is_file() else r / SPLIT_REL,
            processed_roots=tuple(r / p for p in PROCESSED_CANDIDATES),
        )


def resolve_project_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    # Prefer cwd if it looks like the project; else this package's ../../../
    cwd = Path.cwd()
    if (cwd / "data" / "registry").is_dir() and (cwd / "src" / "drug_screen").is_dir():
        return cwd
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "registry").is_dir() and (parent / "src" / "drug_screen").is_dir():
            return parent
    return cwd


class KryeziuAssetError(FileNotFoundError):
    """Missing Kryeziu audit / processed asset with blocking path list."""


def require_paths(paths: KryeziuPaths, *, need_audit: bool = True, need_drugs: bool = False) -> None:
    missing = []
    if need_audit and not paths.audit.is_file():
        missing.append(str(paths.audit))
    if need_drugs and not paths.drug_registry.is_file():
        missing.append(str(paths.drug_registry))
    if missing:
        raise KryeziuAssetError(
            "Kryeziu assets missing (run on user WSL or CopyToBox): " + "; ".join(missing)
        )


def load_split_draft(root: Optional[Path] = None) -> dict[str, Any]:
    """Load the outer 5-fold split; prefer the FROZEN file, fall back to DRAFT."""
    paths = KryeziuPaths.from_root(resolve_project_root(root))
    if not paths.split.is_file():
        raise KryzeiuzAssetError(f"missing split (FROZEN or DRAFT): {paths.split}")
    return json.loads(paths.split.read_text(encoding="utf-8"))


def load_drug_registry_draft(root: Optional[Path] = None) -> dict[str, Any]:
    paths = KryeziuPaths.from_root(resolve_project_root(root))
    if not paths.drug_registry.is_file():
        raise KryeziuAssetError(f"missing drug registry DRAFT: {paths.drug_registry}")
    return json.loads(paths.drug_registry.read_text(encoding="utf-8"))


def drugs_from_registry(reg: Mapping[str, Any]) -> list[DrugIdentity]:
    out: list[DrugIdentity] = []
    for row in reg.get("drugs") or []:
        if not isinstance(row, dict):
            continue
        if row.get("oxaliplatin_quarantine"):
            status = StructureEncodingStatus.NON_ENCODABLE
            reason = "coordination-complex; quarantined from SMILES-graph inputs"
        elif row.get("structure_confirmed") and (row.get("smiles") or row.get("inchi_key")):
            status = StructureEncodingStatus.ENCODABLE
            reason = None
        elif row.get("structure_confirmed"):
            status = StructureEncodingStatus.PARTIAL
            reason = None
        else:
            status = StructureEncodingStatus.UNKNOWN
            reason = None
        out.append(
            DrugIdentity(
                drug_id=str(row.get("drug_id") or row.get("original_name")),
                original_name=str(row.get("original_name") or row.get("drug_id")),
                source_study="Kryeziu",
                structure_status=status,
                smiles=row.get("smiles"),
                inchi_key=row.get("inchi_key"),
                non_encodable_reason=reason,
                metadata={
                    "pubchem_cid": row.get("pubchem_cid"),
                    "provisional_candidate": row.get("provisional_candidate", False),
                },
            )
        )
    return out


def patients_from_split(split: Mapping[str, Any]) -> list[PatientIdentity]:
    """Build patient rows from filled outer-fold DRAFT (verified subset only)."""
    sample_to_patient: dict[str, str] = {}
    for fold in split.get("folds") or []:
        for side in ("train", "val", "test"):
            pts = fold.get(f"{side}_patient_ids") or []
            sms = fold.get(f"{side}_sample_ids") or []
            # samples named PtNN_tK → patient PtNN
            for s in sms:
                s = str(s)
                patient = s.split("_", 1)[0] if "_" in s else None
                if patient:
                    sample_to_patient[s] = patient
            for p in pts:
                sample_to_patient.setdefault(str(p), str(p))  # no-op guard
    by_patient: dict[str, list[str]] = {}
    for sample, patient in sample_to_patient.items():
        if sample == patient:
            continue
        by_patient.setdefault(patient, []).append(sample)
    # ensure all fold patients appear even if sample list incomplete
    for fold in split.get("folds") or []:
        for side in ("train", "val", "test"):
            for p in fold.get(f"{side}_patient_ids") or []:
                by_patient.setdefault(str(p), [])
    return [
        PatientIdentity(
            patient_id=pid,
            source_study="Kryeziu",
            sample_ids=tuple(sorted(set(samples))),
            identity_confidence=IdentityConfidence.CONFIRMED,
            metadata={"cohort": "S1_PDO_patient_verified"},
        )
        for pid, samples in sorted(by_patient.items())
    ]


class KryeziuIdentityLoader:
    """Load identity master table from split + drug registry DRAFTs.

    Audit JSON is preferred when present; without it, patients come from the
    filled split DRAFT (verified-only) and drugs from the registry DRAFT.
    """

    version_id = "kryeziu-draft-0.1"

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = resolve_project_root(root)
        self.paths = KryeziuPaths.from_root(self.root)

    def load(self, *, root: Optional[Path] = None) -> IdentityMasterTable:
        if root is not None:
            self.root = resolve_project_root(root)
            self.paths = KryeziuPaths.from_root(self.root)
        split = load_split_draft(self.root)
        patients = patients_from_split(split)
        drugs: list[DrugIdentity] = []
        provenance: dict[str, Any] = {
            "split": str(self.paths.split),
            "split_status": split.get("status"),
            "prefix_only_excluded": True,
        }
        if self.paths.drug_registry.is_file():
            reg = load_drug_registry_draft(self.root)
            drugs = drugs_from_registry(reg)
            provenance["drug_registry"] = str(DRUG_REGISTRY_REL)
            provenance["drug_registry_status"] = reg.get("status")
        else:
            provenance["drug_registry_missing"] = str(DRUG_REGISTRY_REL)
        if self.paths.audit.is_file():
            provenance["audit"] = str(AUDIT_REL)
        else:
            provenance["audit_missing"] = str(AUDIT_REL)
        pdos = [
            PDOIdentity(
                model_id=s,
                source_study="Kryeziu",
                sample_id=s,
                patient_id=p.patient_id,
                identity_confidence=IdentityConfidence.CONFIRMED,
                has_expression=True,
                has_drug_response=True,
            )
            for p in patients
            for s in p.sample_ids
        ]
        non_enc = tuple(
            d.drug_id
            for d in drugs
            if d.structure_status == StructureEncodingStatus.NON_ENCODABLE
        )
        return IdentityMasterTable(
            version_id=self.version_id,
            cells=(),
            pdos=tuple(pdos),
            patients=tuple(patients),
            drugs=tuple(drugs),
            non_encodable_drug_ids=non_enc,
            provenance=provenance,
            status="DRAFT",
        )

    def validate(self, table: IdentityMasterTable) -> list[str]:
        issues: list[str] = []
        for drug in table.drugs:
            issues.extend(validate_drug_identity(drug))
        issues.extend(validate_patient_fold_unit(table.patients))
        if not table.patients:
            issues.append("no patients loaded (split folds empty or unreadable)")
        return issues


def list_response_matrix_candidates(root: Optional[Path] = None) -> list[Path]:
    """Locate processed DSS / response matrices if present (read-only)."""
    paths = KryeziuPaths.from_root(resolve_project_root(root))
    found: list[Path] = []
    patterns = (
        "**/*dss*.parquet",
        "**/*response*.parquet",
        "**/*drug*score*.parquet",
        "**/pairs.parquet",
        "**/samples.parquet",
    )
    for base in paths.processed_roots:
        if not base.exists():
            continue
        for pat in patterns:
            found.extend(sorted(base.glob(pat)))
    return found


def load_expression_matrix_stub(root: Optional[Path] = None) -> None:
    """Placeholder: wire to processed expression once path is confirmed."""
    paths = KryeziuPaths.from_root(resolve_project_root(root))
    raise KryeziuAssetError(
        "Expression loader not wired; expected under "
        f"{paths.processed_roots[0]} (e.g. expression parquet). "
        "Do not read data/raw from this DRAFT loader."
    )


__all__ = [
    "AUDIT_REL",
    "DRUG_REGISTRY_REL",
    "SPLIT_FROZEN_REL",
    "SPLIT_REL",
    "KryeziuAssetError",
    "KryeziuIdentityLoader",
    "KryeziuPaths",
    "drugs_from_registry",
    "list_response_matrix_candidates",
    "load_drug_registry_draft",
    "load_expression_matrix_stub",
    "load_split_draft",
    "patients_from_split",
    "require_paths",
    "resolve_project_root",
]
