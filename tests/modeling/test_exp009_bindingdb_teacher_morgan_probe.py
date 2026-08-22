from pathlib import Path

import numpy as np

from scripts.modeling.run_exp009_bindingdb_teacher_morgan_probe import (
    aggregate_teacher_records,
    canonical_morgan_fingerprint,
    deterministic_ligand_split,
    select_targets,
    select_targets_stream,
)


def test_morgan_probe_does_not_require_xpert_and_has_expected_shape():
    rows = [
        {
            "inchi_key": "AAA-KEY",
            "canonical_smiles": "CCO",
            "uniprot_id": "P1",
            "paffinity": "7.0",
        }
    ]
    targets = select_targets(rows, minimum_support=1, max_targets=64)
    records, failures = aggregate_teacher_records(rows, targets)
    assert failures == 0
    assert list(records) == ["AAA-KEY"]
    fingerprint = canonical_morgan_fingerprint(records["AAA-KEY"]["smiles"])
    assert fingerprint.shape == (2048,)
    assert fingerprint.dtype == np.float32
    assert set(np.unique(fingerprint)) <= {0.0, 1.0}


def test_same_ligand_never_crosses_deterministic_split():
    rows = [
        {"inchi_key": "same-key", "canonical_smiles": "CCO", "uniprot_id": "P1", "paffinity": "7"},
        {"inchi_key": "SAME-KEY", "canonical_smiles": "CCO", "uniprot_id": "P2", "paffinity": "5"},
    ]
    split_a = deterministic_ligand_split("SAME-KEY")
    split_b = deterministic_ligand_split("same-key".upper())
    assert split_a == split_b


def test_target_selection_uses_unique_ligand_support():
    rows = [
        {"inchi_key": "A", "canonical_smiles": "CCO", "uniprot_id": "P1", "paffinity": "7"},
        {"inchi_key": "A", "canonical_smiles": "CCO", "uniprot_id": "P1", "paffinity": "8"},
        {"inchi_key": "B", "canonical_smiles": "CCN", "uniprot_id": "P1", "paffinity": "7"},
        {"inchi_key": "C", "canonical_smiles": "CCC", "uniprot_id": "P2", "paffinity": "7"},
    ]
    assert select_targets(rows, minimum_support=2, max_targets=64) == ["P1"]


def test_streaming_target_selection_commits_before_counting(tmp_path: Path):
    teacher = tmp_path / "teacher.tsv"
    teacher.write_text(
        "inchi_key\tuniprot_id\nA\tP1\nB\tP1\nC\tP2\n",
        encoding="utf-8",
    )

    assert select_targets_stream(teacher, minimum_support=2, max_targets=64) == ["P1"]
