from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "registry" / "datasets.json"
GENE_ORDER = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"


def _by_id() -> dict[str, dict]:
    records = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {row["id"]: row for row in records}


def test_depmap_24q2_raw_and_exact978_are_registered() -> None:
    by_id = _by_id()
    raw = by_id["depmap_public_24q2_rnaseq_raw_v1"]
    processed = by_id["depmap_public_24q2_rnaseq_exact978_v1"]

    assert raw["source"]["accession"] == "DepMap Public 24Q2"
    assert raw["source"]["doi"] == "10.25452/figshare.plus.25880521.v1"
    assert raw["intended_role"] == "CONTEXT_ADAPTER_BASAL_RNASEQ_SOURCE"
    assert raw["path"]["relative"] == "raw/depmap/24q2"
    assert raw["files"]["OmicsExpressionProteinCodingGenesTPMLogp1.csv"] == {
        "bytes": 460868099,
        "sha256": "39eff342bbbbe0a40208545e93a69cc85a945899592f9d683f87e3b2bb121670",
    }
    assert raw["files"]["Model.csv"] == {
        "bytes": 559182,
        "sha256": "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8",
    }

    assert processed["source"]["parent_asset"] == "depmap_public_24q2_rnaseq_raw_v1"
    assert processed["intended_role"] == "XPERT_CONTEXT_ADAPTER_BASAL_978"
    assert processed["schema"]["matrix_shape"] == [1517, 978]
    assert processed["schema"]["ordered_gene_ids_sha256"] == GENE_ORDER
    assert processed["schema"]["cannot_replace_matched_control"] is True
    assert processed["checksum"]["value"] == (
        "36ab8a0f8f65df2cdb2c660f2f4ae7c92e26fc7a06ad18ff32bdd0ef11fff9bd"
    )
    assert processed["preprocessing_contract"]["mapped_genes"] == 978
    assert processed["preprocessing_contract"]["crc_exact_overlap"] == 10
    assert processed["preprocessing_contract"]["lincs_overlap_by_stripped_name"] == 54
    assert processed["provenance"]["response_blind"] is True

    freeze = by_id["ccle_24q2_context_identity_split_v1"]
    assert freeze["intended_role"] == "IDENTITY_SPLIT_FREEZE_PRE_EVALUATION"
    assert freeze["schema"]["crc_exact_overlap"] == 10
    assert freeze["schema"]["cannot_use_as_x_ctl"] is True
    assert freeze["schema"]["cannot_compute_delta978"] is True
    assert freeze["provenance"]["response_blind"] is True
    assert freeze["checksum"]["value"] == (
        "99c53232e354cc813b69cc88204a9c1ccdb922a661fab00a0b627eef52a9cf92"
    )


def test_depmap_24q2_adapter_shape_if_present() -> None:
    matrix = ROOT / "data" / "processed" / "depmap" / "24q2_rnaseq_exact978" / "ccle_24q2_exact978_log2tpm.npy"
    if not matrix.exists():
        return
    import numpy as np

    arr = np.load(matrix, mmap_mode="r")
    assert arr.shape == (1517, 978)
    assert str(arr.dtype) == "float32"
