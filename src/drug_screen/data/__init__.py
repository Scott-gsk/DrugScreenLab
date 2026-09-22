"""数据根目录、注册表工具与数据阶段合同（DRAFT 接口）。

三套表达输入合同见 ``expression_contracts``；统一身份/药物合同见
``identity_contracts``。原始数据只读；预处理统计仅在训练折拟合。
"""

from __future__ import annotations

from drug_screen.data.expression_contracts import (
    ExpressionMatrix,
    GeneExpressionContract,
    PathwayActivityContract,
    PathwayMatrix,
    PerturbationGeneContract,
    PerturbationGeneRecord,
    PerturbationRecord,
    StubGeneExpressionLoader,
    StubPathwayActivityLoader,
    StubPerturbationGeneLoader,
    registry_hint,
)
from drug_screen.data.measured_perturbation_loader_DRAFT import (
    MeasuredPerturbationControlLoader,
    describe_assets as describe_measured_perturbation_assets,
    smoke_measured_control_loader,
)

from drug_screen.data.identity_contracts import (
    CellLineIdentity,
    DrugIdentity,
    IdentityConfidence,
    IdentityMasterTable,
    IdentityTableLoader,
    PDOIdentity,
    PatientIdentity,
    StructureEncodingStatus,
    StubIdentityTableLoader,
    default_identity_paths,
    validate_drug_identity,
    validate_patient_fold_unit,
)

__all__ = [
    # expression
    "ExpressionMatrix",
    "GeneExpressionContract",
    "PathwayActivityContract",
    "PathwayMatrix",
    "PerturbationGeneContract",
    "PerturbationGeneRecord",
    "PerturbationRecord",
    "StubGeneExpressionLoader",
    "StubPathwayActivityLoader",
    "StubPerturbationGeneLoader",
    "registry_hint",
    # identity
    "CellLineIdentity",
    "DrugIdentity",
    "IdentityConfidence",
    "IdentityMasterTable",
    "IdentityTableLoader",
    "PDOIdentity",
    "PatientIdentity",
    "StructureEncodingStatus",
    "StubIdentityTableLoader",
    "default_identity_paths",
    "validate_drug_identity",
    "validate_patient_fold_unit",
    # measured perturbation DRAFT
    "MeasuredPerturbationControlLoader",
    "describe_measured_perturbation_assets",
    "smoke_measured_control_loader",
]
