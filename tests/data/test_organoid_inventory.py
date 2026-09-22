from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "artifacts" / "experiments" / "EXP-007" / "ORGANOID_GENETIC_INVENTORY.json"
FETCH = ROOT / "artifacts" / "experiments" / "EXP-007" / "ORGANOID_GEO_FETCH.json"
REGISTRY = ROOT / "data" / "registry" / "datasets.json"
RECORD = ROOT / "experiments" / "records" / "EXP-007.md"


def _by_id() -> dict[str, dict]:
    return {row["id"]: row for row in json.loads(REGISTRY.read_text(encoding="utf-8"))}


def test_organoid_inventory_is_local_and_not_chemical_truth() -> None:
    if not INVENTORY.exists():
        return
    payload = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert payload["format"] == "organoid_genetic_inventory_v2"
    assert payload["data_status"] == "DATA_PARTIAL"
    assert payload["local_expression_matrices"] is True
    assert payload["cannot_use_as_x_ctl"] is True
    assert payload["cannot_compute_delta978"] is True
    by_acc = {row["accession"]: row for row in payload["datasets"]}
    assert set(by_acc) == {"GSE280506", "GSE145308", "GSE167285", "GSE241659"}
    assert by_acc["GSE280506"]["unique_perturbations"] == 79
    assert by_acc["GSE145308"]["unique_perturbations"] == 4
    assert by_acc["GSE167285"]["donor_or_model_grouping"]["n_donors"] == 5
    assert by_acc["GSE241659"]["donor_or_model_grouping"]["n_donors"] == 4
    assert by_acc["GSE145308"]["exact978_coverage"]["status"] == (
        "UNVERIFIED_ENSEMBL_ONLY_NO_LOCAL_CROSSWALK"
    )
    assert by_acc["GSE280506"]["exact978_coverage"]["mapped"] == 976
    assert by_acc["GSE167285"]["exact978_coverage"]["mapped"] == 950
    assert by_acc["GSE241659"]["exact978_coverage"]["mapped"] == 935


def test_organoid_raw_assets_are_registered_with_checksums() -> None:
    by_id = _by_id()
    assert by_id["organoid_gse280506_raw_v1"]["files"]["GSE280506_filtered_feature_bc_matrix.h5"][
        "sha256"
    ] == "6e9297941a1b948d6c61c58c37497ef447f7b4c54aaf6e1561cd3b63d7b8f87e"
    assert by_id["organoid_gse145308_raw_v1"]["files"]["GSE145308_RAW.tar"]["bytes"] == 2652160
    assert by_id["organoid_gse167285_raw_v1"]["schema"]["donors"] == ["80", "83", "87", "88", "89"]
    assert by_id["organoid_gse241659_raw_v1"]["schema"]["gsm_samples"] == 7
    if FETCH.exists():
        fetch = json.loads(FETCH.read_text(encoding="utf-8"))
        assert fetch["status"] == "FETCHED"
        assert all(row["n_ok"] == row["n_requested"] for row in fetch["datasets"])


def test_organoid_work_does_not_rewrite_exp007_hypothesis() -> None:
    text = RECORD.read_text(encoding="utf-8")
    assert "Observed Δ978 → Disease Reversal" in text
    assert "Top-10/20/50 overlap" in text
    assert "ORACLE_NEAR_NULL" in text
