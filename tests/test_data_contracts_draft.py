"""Smoke tests for data contract drafts (no I/O against data/raw)."""

from __future__ import annotations

import json
from pathlib import Path

from drug_screen.data.expression_contracts import (
    ExpressionMatrix,
    GeneExpressionContract,
    PathwayActivityContract,
    PathwayMatrix,
    PerturbationGeneContract,
    PerturbationGeneRecord,
    StubGeneExpressionLoader,
    registry_hint,
)
from drug_screen.data.identity_contracts import (
    DrugIdentity,
    IdentityConfidence,
    PatientIdentity,
    StructureEncodingStatus,
    StubIdentityTableLoader,
    validate_drug_identity,
    validate_patient_fold_unit,
)


def test_expression_matrix_dataclass():
    m = ExpressionMatrix(
        sample_ids=("s1", "s2"),
        gene_ids=("TP53", "EGFR"),
        values=[[0.0, 1.0], [0.5, -0.2]],
        feature_space="genome_wide",
        units="log2_tpm",
        gene_order_contract_id="draft-order",
    )
    assert m.contract_version.startswith("draft")
    assert len(m.sample_ids) == 2


def test_pathway_and_perturbation_dataclasses():
    p = PathwayMatrix(
        sample_ids=("s1",),
        pathway_ids=("HALLMARK_APOPTOSIS",),
        values=[[0.1]],
        gene_set_collection="MSigDB_Hallmark_v2024.1",
        scoring_method="ssgsea_rank",
    )
    r = PerturbationGeneRecord(
        record_id="r1",
        compound_id="COMPOUND_X",
        cell_or_model_id="CELL_Y",
        dose=1.0,
        dose_unit="uM",
        time_hr=24.0,
        signature_level="level5_signature",
        control_id="ctl1",
        gene_ids=("TP53",),
        values=[0.3],
        source="LINCS_GSE92742",
    )
    assert p.gene_set_collection.startswith("MSigDB")
    assert r.perturbation_class == "chemical"


def test_protocols_and_stubs_importable():
    assert GeneExpressionContract is not None
    assert PathwayActivityContract is not None
    assert PerturbationGeneContract is not None
    stub = StubGeneExpressionLoader()
    assert stub.version.startswith("draft")
    assert "Kryeziu" in stub.list_sources()


def test_identity_drug_validation():
    ok = DrugIdentity(
        drug_id="d1",
        original_name="DrugA",
        source_study="Kryeziu",
        structure_status=StructureEncodingStatus.ENCODABLE,
        smiles="CCO",
    )
    assert validate_drug_identity(ok) == []
    bad = DrugIdentity(
        drug_id="d2",
        original_name="DrugB",
        source_study="Kryeziu",
        structure_status=StructureEncodingStatus.NON_ENCODABLE,
    )
    assert any("non_encodable_reason" in x for x in validate_drug_identity(bad))


def test_patient_fold_unit_validation():
    patients = (
        PatientIdentity(
            patient_id="P1",
            source_study="Kryeziu",
            sample_ids=("S1", "S2"),
            identity_confidence=IdentityConfidence.CONFIRMED,
        ),
        PatientIdentity(
            patient_id="P2",
            source_study="Kryeziu",
            sample_ids=("S2",),
            identity_confidence=IdentityConfidence.CONFIRMED,
        ),
    )
    issues = validate_patient_fold_unit(patients)
    assert any("S2" in x for x in issues)


def test_stub_identity_loader_raises():
    loader = StubIdentityTableLoader()
    try:
        loader.load()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_registry_hint_relative():
    assert registry_hint() == Path("data/registry")


def test_kryeziu_split_frozen_contract():
    """Frozen Kryzei outer 5-fold contract: materialized inner val, no leakage."""
    from drug_screen.evaluation.protocols import check_split_leakage

    root = Path(__file__).resolve().parents[1]
    frozen = root / "data" / "splits" / "kryeziu_patient_outer5fold_FROZEN.json"
    draft = root / "data" / "splits" / "kryeziu_patient_outer5fold_DRAFT.json"
    assert frozen.is_file(), f"missing {frozen}"
    assert draft.is_file(), f"missing read-only DRAFT copy {draft}"

    data = json.loads(frozen.read_text(encoding="utf-8"))
    assert data["status"] == "FROZEN"
    assert data["n_folds"] == 5
    assert data["split_unit"] == "patient_id"
    assert data["outer"] is True
    assert data["random_seed"] == 20260918
    assert "freeze" in data and "expression_manifest" in data["freeze"]

    for fold in data["folds"]:
        train_p = set(fold["train_patient_ids"])
        val_p = set(fold["val_patient_ids"])
        test_p = set(fold["test_patient_ids"])
        val_s = set(fold["val_sample_ids"])
        train_s = set(fold["train_sample_ids"])
        assert val_p, f"fold {fold['fold']} inner val not materialized"
        assert val_p <= train_p, f"fold {fold['fold']} val not subset of train"
        assert not (val_p & test_p), f"fold {fold['fold']} val overlaps test"
        assert val_s <= train_s
        report = check_split_leakage(
            train_patients=sorted(train_p - val_p),
            test_patients=sorted(test_p),
            allow_drug_overlap=True,
        )
        assert report.ok, f"fold {fold['fold']} leakage: {report.notes}"

    # DRAFT copy is retained read-only with the same materialized val lists
    draft_data = json.loads(draft.read_text(encoding="utf-8"))
    assert draft_data["status"] == "DRAFT"
    assert draft_data["folds"] == data["folds"]
