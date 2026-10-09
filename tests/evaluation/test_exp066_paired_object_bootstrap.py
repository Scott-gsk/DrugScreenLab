from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "exp066_paired_object_analysis",
    ROOT / "scripts/evaluation/analyze_exp066_paired_object_bootstrap.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_object_macro_weights_objects_equally_after_condition_aggregation() -> None:
    observed = ANALYSIS._object_macro(
        np.asarray([0.0, 2.0, 4.0]),
        np.asarray(["a", "a", "b"]),
    )
    assert observed == {"a": 1.0, "b": 4.0}
    assert np.mean(list(observed.values())) == 2.5


def test_paired_bootstrap_preserves_object_pairing_and_is_deterministic() -> None:
    draws = np.random.default_rng(660067).integers(0, 3, size=(2000, 3))
    first = ANALYSIS._paired_bootstrap(
        np.asarray([2.0, 4.0, 6.0]), np.asarray([1.0, 3.0, 5.0]), draws,
    )
    second = ANALYSIS._paired_bootstrap(
        np.asarray([2.0, 4.0, 6.0]), np.asarray([1.0, 3.0, 5.0]), draws,
    )
    assert first == second
    assert first["mean_absolute_nmse_reduction_reference_minus_candidate"] == 1.0
    assert first["ci95_absolute_nmse_reduction"] == [1.0, 1.0]

