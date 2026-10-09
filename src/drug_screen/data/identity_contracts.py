"""Unified identity / drug contracts for data-phase P0 (DRAFT).

Encodes the identity master-table rules from ``docs/研究实施方案.md`` §二
and the data-stage milestones in
``docs/exec-plans/active/pathway_transfer_pdo_20260918.md``.

Rules encoded here (implementers must not weaken them):
- ``data/raw`` is read-only; identity tables are derived into ``data/processed``
  / ``data/registry`` and never rewrite raw assets.
- Cell / PDO / patient / drug rows share stable join keys; do not invent IDs.
- Drug rows record original names, structure-encoding status, and salt/stereo
  notes; non-encodable drugs are registered separately (not silently dropped).
- Unmeasured response values must not be imputed as zeros when joining labels.
- Do not treat prior EXP metrics as model-selection evidence for this rebuild.

Heavy loaders are intentionally stubs — validate schemas and document paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd


class StructureEncodingStatus(str, Enum):
    """Whether a drug has a usable structure encoding for modeling."""

    ENCODABLE = "encodable"  # SMILES / InChIKey / graph features available
    PARTIAL = "partial"  # name mapped but stereo/salt unresolved
    NON_ENCODABLE = "non_encodable"  # register separately; exclude from structure models
    UNKNOWN = "unknown"  # not yet audited


class IdentityConfidence(str, Enum):
    """Patient / sample identity confidence for Kryeziu-style cohorts."""

    CONFIRMED = "confirmed"
    NAME_INFERRED = "name_inferred"
    UNKNOWN = "unknown"


# --------------------------- 身份字段语义统一（2026-09-22 数据与身份对齐修复）--
#
# 审计背景（docs/reports/audit_repair_wave3_20260922.md，W3-37 相关工程）：
# ``model_id`` 在预测 / 评估长表中存在与算法身份、运行身份混用的二义风险
# （评估层统一 schema 将 ``model_id`` 列与 run_id/seed 等运行字段并列必填，
# 且不要求算法身份列，见 src/drug_screen/evaluation/exp031_eval.py）。
#
# 数据层统一约定（与 src/drug_screen/{data,transfer} 既有用法一致化）：
#   - ``patient_id``：患者身份（划分隔离单元）；
#   - ``model_id``：生物模型身份（PDO / 细胞系），绝不表示算法或运行身份；
#   - 算法身份：独立字段（``algorithm`` / ``arm`` / ``algorithm_id``）；
#   - 运行身份：独立字段（``run_id`` / ``seed`` / ``init_seed``）。

IDENTITY_FIELD_SEMANTICS: Mapping[str, str] = {
    "patient_id": "患者身份（划分隔离单元）",
    "model_id": "生物模型身份（PDO/细胞系）；绝不表示算法或运行身份",
    "algorithm": "算法身份（与 model_id 分离）",
    "algorithm_id": "算法身份（与 model_id 分离）",
    "arm": "算法/条件身份（与 model_id 分离）",
    "run_id": "运行身份（与 model_id 分离）",
    "seed": "运行身份（与 model_id 分离）",
    "init_seed": "运行身份（与 model_id 分离）",
}


def assert_model_id_is_biological(
    frame: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    model_column: str = "model_id",
) -> None:
    """守卫 ``model_id`` 的生物语义：一个生物模型不得跨患者。

    若 ``model_id`` 被误用作算法 / 运行身份（取值如 ``"M3"``、
    ``"ridge"``），同一取值必然出现在多个患者之下，本守卫立即失败。
    这是 ``transfer.metrics_v2`` 既有不变式（"model_id assigned to
    multiple patients"）在数据层的固化。

    缺 ``model_column`` 或 ``patient_column`` 时为无操作（如细胞系来源表
    无患者列，无法校验，由调用方按约定使用）。

    Raises
    ------
    ValueError
        同一 ``model_id`` 映射到多个 ``patient_id`` 时抛出。
    """
    if model_column not in frame.columns or patient_column not in frame.columns:
        return
    pairs = frame[[model_column, patient_column]].dropna()
    if pairs.empty:
        return
    patients_per_model = pairs.groupby(model_column)[patient_column].nunique()
    offenders = patients_per_model[patients_per_model > 1]
    if not offenders.empty:
        raise ValueError(
            f"{model_column} spans multiple {patient_column} values — "
            f"{model_column} must be a biological model identity, never an "
            "algorithm/run identity (use algorithm/arm/algorithm_id columns "
            "for algorithm identity); offenders: "
            f"{offenders.head(10).to_dict()}")


@dataclass(frozen=True)
class CellLineIdentity:
    """One cell-line unit (GDSC / CTRP / PRISM / LINCS context)."""

    cell_id: str
    source_study: str
    depmap_id: Optional[str] = None
    cosmic_id: Optional[str] = None
    cell_name: Optional[str] = None
    tissue: Optional[str] = None
    disease: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class PDOIdentity:
    """One PDO / organoid model (may map to a patient)."""

    model_id: str
    source_study: str
    sample_id: Optional[str] = None
    patient_id: Optional[str] = None
    identity_confidence: IdentityConfidence = IdentityConfidence.UNKNOWN
    has_expression: bool = False
    has_drug_response: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class PatientIdentity:
    """One patient / donor; all samples of a patient stay in one outer fold."""

    patient_id: str
    source_study: str
    model_ids: Sequence[str] = field(default_factory=tuple)
    sample_ids: Sequence[str] = field(default_factory=tuple)
    identity_confidence: IdentityConfidence = IdentityConfidence.UNKNOWN
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class DrugIdentity:
    """Drug / compound identity with structure-encoding registration.

    Non-encodable drugs MUST remain in the registry (``structure_status`` =
    NON_ENCODABLE) so coverage audits can report them; they are not deleted.
    """

    drug_id: str
    original_name: str
    source_study: str
    structure_status: StructureEncodingStatus = StructureEncodingStatus.UNKNOWN
    smiles: Optional[str] = None
    inchi_key: Optional[str] = None
    salt_notes: Optional[str] = None
    stereo_notes: Optional[str] = None
    synonyms: Sequence[str] = field(default_factory=tuple)
    non_encodable_reason: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class IdentityMasterTable:
    """Bundle of identity rows for one frozen data version (DRAFT until audited)."""

    version_id: str
    cells: Sequence[CellLineIdentity] = field(default_factory=tuple)
    pdos: Sequence[PDOIdentity] = field(default_factory=tuple)
    patients: Sequence[PatientIdentity] = field(default_factory=tuple)
    drugs: Sequence[DrugIdentity] = field(default_factory=tuple)
    non_encodable_drug_ids: Sequence[str] = field(default_factory=tuple)
    provenance: Mapping[str, object] = field(default_factory=dict)
    status: str = "DRAFT"


def validate_drug_identity(drug: DrugIdentity) -> list[str]:
    """Return validation issues; empty list means schema-OK (not biologically audited)."""
    issues: list[str] = []
    if not drug.drug_id.strip():
        issues.append("drug_id empty")
    if not drug.original_name.strip():
        issues.append("original_name empty")
    if drug.structure_status == StructureEncodingStatus.ENCODABLE:
        if not (drug.smiles or drug.inchi_key):
            issues.append("encodable drug missing smiles and inchi_key")
    if drug.structure_status == StructureEncodingStatus.NON_ENCODABLE:
        if not drug.non_encodable_reason:
            issues.append("non_encodable drug missing non_encodable_reason")
    return issues


def validate_patient_fold_unit(patients: Sequence[PatientIdentity]) -> list[str]:
    """Check that sample/model membership does not assign one sample to two patients."""
    issues: list[str] = []
    seen_samples: dict[str, str] = {}
    for patient in patients:
        for sample_id in patient.sample_ids:
            previous = seen_samples.setdefault(sample_id, patient.patient_id)
            if previous != patient.patient_id:
                issues.append(
                    f"sample_id={sample_id} mapped to both {previous} and {patient.patient_id}"
                )
    return issues


@runtime_checkable
class IdentityTableLoader(Protocol):
    """Load / validate the unified identity master table (stub interface)."""

    version_id: str

    def load(self, *, root: Optional[Path] = None) -> IdentityMasterTable:
        """Load identity rows from registry / processed tables under ``root``."""
        ...

    def validate(self, table: IdentityMasterTable) -> list[str]:
        """Return blocking issues; empty means ready for DRAFT coverage report."""
        ...


class StubIdentityTableLoader:
    """Importable stub — raises until local identity tables are wired.

    Why: keep the package importable on machines without ``data/processed`` while
    encoding the expected entry points for the data-phase rebuild.
    """

    version_id = "draft-0.1"

    def load(self, *, root: Optional[Path] = None) -> IdentityMasterTable:
        raise NotImplementedError(
            "Wire loaders to data/registry + processed identity tables; "
            "do not invent patient/drug IDs. root="
            f"{root}"
        )

    def validate(self, table: IdentityMasterTable) -> list[str]:
        issues: list[str] = []
        if table.status != "DRAFT" and not table.provenance:
            issues.append("non-DRAFT table requires provenance")
        for drug in table.drugs:
            issues.extend(validate_drug_identity(drug))
        issues.extend(validate_patient_fold_unit(table.patients))
        # Non-encodable drugs must appear in the dedicated list for coverage audits.
        non_enc = {
            d.drug_id
            for d in table.drugs
            if d.structure_status == StructureEncodingStatus.NON_ENCODABLE
        }
        listed = set(table.non_encodable_drug_ids)
        if non_enc - listed:
            issues.append(
                f"non_encodable drugs missing from non_encodable_drug_ids: {sorted(non_enc - listed)}"
            )
        return issues


def default_identity_paths() -> Mapping[str, Path]:
    """Expected relative paths for identity artifacts (may be missing until audited)."""
    return {
        "master_table": Path("data/processed/identity/identity_master_DRAFT.parquet"),
        "drug_table": Path("data/processed/identity/drug_identity_DRAFT.parquet"),
        "kryeziu_audit_dir": Path("data/artifacts/resource_acquisition_20260916/kryeziu_audit"),
        "registry": Path("data/registry"),
    }
