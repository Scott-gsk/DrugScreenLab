from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
V3_RUNNER = ROOT / "scripts/evaluation/evaluate_exp050_source_five_arm_development.py"
V2_RUNNER_SNAPSHOT = ROOT / "artifacts/experiments/EXP-050/20261006_exp050_source_five_arm_development_v3/previous_v2_runner_snapshot.py"
VALID_KEY = "AAAAAAAAAAAAAA-AAAAAAAAAA-A"
AMBIGUOUS_KEY = "BNRNXUUZRGQAQC-UHFFFAOYSA-N"


def load_runner(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_inputs():
    labels = pd.DataFrame(
        [
            {"source_object_id": obj, "full_inchikey": key, "sensitivity": score}
            for obj, score in [("A", 0.1), ("B", 0.2), ("C", 0.3)]
            for key in (VALID_KEY, AMBIGUOUS_KEY)
        ]
    )
    features = pd.DataFrame(
        [
            {
                "source_object_id": obj,
                "full_inchikey": key,
                "predicted_delta": np.asarray([i, i + 1], dtype=np.float32),
                "predicted_perturbation_strength": float(i),
            }
            for i, obj in enumerate(("A", "B", "C"), start=1)
            for key in (VALID_KEY, AMBIGUOUS_KEY)
        ]
    )
    profiles = {obj: np.asarray([i, i + 1], dtype=np.float32) for i, obj in enumerate(("A", "B", "C"), start=1)}
    drugs = {VALID_KEY: np.asarray([0.0, 1.0], dtype=np.float32)}
    support = {
        "candidate_domain": [VALID_KEY],
        "fold_support": [
            {
                "test_object": "C",
                "outer_train_objects": ["A", "B"],
                "outer_train_block_donor_map": {"A": "B", "B": "A"},
                "test_block_donor": "A",
                "test_common_full_inchikeys": [VALID_KEY],
            }
        ],
    }
    return labels, features, profiles, drugs, support


def test_unique_morgan_domain_is_applied_before_outer_and_inner_feature_lookup():
    old_runner = load_runner(V2_RUNNER_SNAPSHOT, "exp050_source_five_arm_v2_snapshot")
    new_runner = load_runner(V3_RUNNER, "exp050_source_five_arm_v3")
    labels, features, profiles, drugs, support = synthetic_inputs()

    # Reproduce the v2 failure: the ambiguous row reaches a feature lookup
    # even though no unique Morgan vector is available for its full key.
    legacy_row = pd.DataFrame(
        [{"source_object_id": "A", "full_inchikey": AMBIGUOUS_KEY, "shuffle_donor": "B"}]
    )
    with pytest.raises(KeyError, match="BNRNXUUZRGQAQC-UHFFFAOYSA-N"):
        old_runner.raw_features(legacy_row, features, profiles, drugs)

    # The corrected fold builder removes this key before nested fitting;
    # the resulting train and held-out rows are safe for every arm lookup.
    train, test, _, _ = new_runner.fold_frames(labels, features, support, "C")
    assert set(train.full_inchikey) == {VALID_KEY}
    assert set(test.full_inchikey) == {VALID_KEY}
    assert new_runner.raw_features(train, features, profiles, drugs)[2].shape == (2, 2)
    assert new_runner.raw_features(test, features, profiles, drugs)[2].shape == (1, 2)
