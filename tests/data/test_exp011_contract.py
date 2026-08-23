from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_screen.data.exp011_contract import IndexedDrugResponseDataset, gene_digest
from drug_screen.data.exp011_materialize import _is_antibody, _is_combination, align_expression


GENES = [f"G{index:05d}" for index in range(19_091)]


def _dataset(*, duplicate: bool = False) -> IndexedDrugResponseDataset:
    samples = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"], "patient_id": ["p1", "p2"],
            "pdo_id": ["s1", "s2"], "cell_line_id": ["", ""], "sample_tissue": ["", ""],
        }
    )
    drugs = pd.DataFrame(
        {
            "drug_id": ["AAAA-BBBB"], "canonical_smiles": ["CCO"],
            "inchikey": ["AAAA-BBBB"], "drug_scaffold": ["ACYCLIC"],
        }
    )
    pairs = pd.DataFrame(
        {
            "cohort_id": ["test", "test"] if duplicate else ["test"],
            "sample_id": ["s1", "s1"] if duplicate else ["s1"],
            "drug_id": ["AAAA-BBBB", "AAAA-BBBB"] if duplicate else ["AAAA-BBBB"],
            "response": [0.5, 0.7] if duplicate else [0.5],
            "response_name": ["dss", "dss"] if duplicate else ["dss"],
            "response_scale": ["DSS", "DSS"] if duplicate else ["DSS"],
            "higher_is_sensitive": [True, True] if duplicate else [True],
        }
    )
    return IndexedDrugResponseDataset(
        cohort_id="test", expression=np.zeros((2, len(GENES)), dtype=np.float32), ordered_genes=GENES,
        samples=samples, drugs=drugs, pairs=pairs, expression_unit="unit", source_hashes={"source": "a" * 64},
    )


def test_indexed_contract_writes_unique_expression_not_pair_expansion(tmp_path) -> None:
    dataset = _dataset()
    hashes = dataset.write(tmp_path)
    payload = np.load(tmp_path / "test_indexed.npz")
    assert payload["expression"].shape == (2, 19_091)
    assert payload["pair_sample_idx"].tolist() == [0]
    assert payload["pair_drug_idx"].tolist() == [0]
    assert payload["pair_higher_is_sensitive"].tolist() == [True]
    assert hashes["gene_universe"] == gene_digest(GENES)


def test_duplicate_same_sample_drug_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate response pairs"):
        _dataset(duplicate=True)


def test_expression_alignment_has_sorted_contract_and_records_measured_genes() -> None:
    frame = pd.DataFrame({"G00001": [1.0], "G00002": [2.0], "g00002": [4.0]}, index=["s1"])
    target = [f"G{index:05d}" for index in range(15_002)]
    values, mapped = align_expression(frame, target, minimum_mapped=2)
    assert values.shape == (1, len(target))
    assert mapped == 2
    assert values[0, 2] == 3.0  # duplicate source symbols are averaged deterministically


def test_combination_and_antibody_markers_are_excluded() -> None:
    assert _is_combination("Trametinib+Alpelisib")
    assert _is_combination("MVACDSS")
    assert _is_antibody("Trastuzumab")
    assert not _is_antibody("Cisplatin")
