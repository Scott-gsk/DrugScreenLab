import json
from pathlib import Path

import numpy as np

from scripts.modeling.run_exp009_bindingdb_teacher_probe import (
    aggregate_labels,
    deterministic_split,
    load_mappable_ligands,
    map_structure_features,
    normalize_inchikey,
)


def test_exact_mapping_and_label_aggregation(tmp_path: Path):
    rows = [
        {"inchi_key": "abc-key", "uniprot_id": "P2", "paffinity": "7.0"},
        {"inchi_key": "ABC-KEY", "uniprot_id": "P1", "paffinity": "5.0"},
        {"inchi_key": "abc-key", "uniprot_id": "P1", "paffinity": "6.0"},
    ]
    assert normalize_inchikey(rows[0]["inchi_key"]) == "ABC-KEY"
    labels = aggregate_labels(rows, ["P1", "P2"])
    assert labels == {"ABC-KEY": [1.0, 1.0]}


def test_split_is_deterministic_and_has_no_ligand_crossing():
    ligands = ["A", "B", "C", "D", "E", "F"]
    first = deterministic_split(ligands)
    second = deterministic_split(ligands)
    assert first == second
    assert set(first) == set(ligands)
    assert set(first.values()) <= {"train", "validation", "test"}


def test_probe_selection_filters_to_mappable_ligands_before_limiting():
    from scripts.modeling.run_exp009_bindingdb_teacher_probe import select_probe_ligands

    labels = {"A": [1.0], "B": [1.0], "C": [1.0], "D": [1.0]}
    selected = select_probe_ligands(labels, {"B", "D"}, max_ligands=2)

    assert set(selected) == {"B", "D"}


def test_registry_to_unimol_mean_pool_input_shape(tmp_path: Path):
    registry = {
        "drugs": [
            {"inchi_key": "ABC-KEY", "pert_idx": 1, "global_inference_eligible": True}
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    array_path = tmp_path / "arr.npy"
    np.save(array_path, np.ones((2, 122, 514), dtype=np.float64))
    index, array = load_mappable_ligands(registry_path, array_path)
    features, kept = map_structure_features(["ABC-KEY"], index, array)
    assert kept == ["ABC-KEY"]
    assert features.shape == (1, 514)
    assert np.isfinite(features).all()
