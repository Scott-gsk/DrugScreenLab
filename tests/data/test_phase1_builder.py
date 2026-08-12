from __future__ import annotations

import numpy as np
import pandas as pd

from drug_screen.data.phase1 import (
    assign_entity_split,
    build_chemical_feature_table,
    select_canonical_condition,
)


def test_structure_fingerprint_is_reproducible_and_not_id_lookup():
    perturbagens = pd.DataFrame(
        {
            "pert_id": ["drug-a", "drug-b", "drug-c"],
            "canonical_smiles": ["CCO", "CCN", "CCO"],
            "inchi_key": ["A", "B", "C"],
        }
    )
    features, mapping = build_chemical_feature_table(perturbagens, n_bits=64)
    assert features.shape == (3, 64)
    assert features.dtype == np.float32
    assert np.array_equal(features[0], features[2])
    assert not np.array_equal(features[0], features[1])
    assert mapping["drug-a"] == 0


def test_invalid_structure_rows_are_excluded_from_feature_table():
    perturbagens = pd.DataFrame(
        {
            "pert_id": ["drug-a", "drug-b"],
            "canonical_smiles": ["CCO", "restricted"],
            "inchi_key": ["A", "B"],
        }
    )
    features, mapping = build_chemical_feature_table(perturbagens, n_bits=64)
    assert features.shape == (1, 64)
    assert mapping == {"drug-a": 0}


def test_canonical_condition_uses_numeric_tolerance():
    instances = pd.DataFrame(
        {
            "pert_type": ["trt_cp", "trt_cp", "trt_cp"],
            "pert_dose": [10.0, 10.0000001, 5.0],
            "pert_dose_unit": ["um", "um", "um"],
            "pert_time": [6.0, 6.0, 6.0],
            "pert_time_unit": ["h", "h", "h"],
        }
    )
    selected = select_canonical_condition(instances, dose_um=10.0, time_h=6.0)
    assert selected.index.tolist() == [0, 1]


def test_cold_entity_split_is_atomic_and_deterministic():
    rows = pd.DataFrame({"drug_id": ["a", "a", "b", "c"], "context_id": ["x", "y", "x", "z"]})
    first = assign_entity_split(rows, entity_column="drug_id", seed=7)
    second = assign_entity_split(rows, entity_column="drug_id", seed=7)
    assert first.tolist() == second.tolist()
    assert rows.assign(split=first).groupby("drug_id")["split"].nunique().max() == 1
