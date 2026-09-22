import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.cohorts import confined, load_registered_cohort, validate_bundle


def _bundle(*, expression=None, samples=None, pairs=None):
    samples = samples if samples is not None else pd.DataFrame({"sample_id": ["s1", "s2"]})
    drugs = pd.DataFrame({"drug_id": ["d1"]})
    pairs = pairs if pairs is not None else pd.DataFrame({
        "sample_id": ["s1", "s2"], "drug_id": ["d1", "d1"], "response": [1.0, 2.0],
        "higher_is_sensitive": [True, True], "response_name": ["auc", "auc"],
    })
    return IndexedBundle(
        "cohort", "cell_line", samples, drugs, pairs,
        np.ones((2, 2)) if expression is None else expression,
        np.array(["s1", "s2"]), np.array(["g1", "g2"]), "log2", True, "auc",
    )


def test_bundle_validation_rejects_expression_shape_mismatch_and_orphan_pairs():
    with pytest.raises(ValueError, match="dimensions"):
        validate_bundle(_bundle(expression=np.ones((2, 1))))
    orphan_pairs = _bundle().pairs.copy()
    orphan_pairs.loc[0, "sample_id"] = "missing"
    with pytest.raises(ValueError, match="orphan response"):
        validate_bundle(_bundle(pairs=orphan_pairs))


def test_bundle_validation_rejects_duplicate_identities_and_duplicate_responses():
    duplicate_samples = pd.DataFrame({"sample_id": ["s1", "s1"]})
    with pytest.raises(ValueError, match="duplicate identity"):
        validate_bundle(_bundle(samples=duplicate_samples))
    duplicate_pairs = _bundle().pairs.iloc[[0, 0]].copy()
    with pytest.raises(ValueError, match="unresolved replicate"):
        validate_bundle(_bundle(pairs=duplicate_pairs))


def test_registered_loader_checks_hash_before_opening_asset_and_confines_paths(tmp_path):
    assert confined(tmp_path, "assets/file") == (tmp_path / "assets" / "file").resolve()
    with pytest.raises(ValueError, match="escapes"):
        confined(tmp_path, "../outside")
    assets = tmp_path / "assets"
    assets.mkdir()
    bad = assets / "samples.parquet"
    bad.write_bytes(b"not parquet")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"cohorts": [{
        "cohort_id": "c", "role": "cell_line", "expression_unit": "log2",
        "higher_is_sensitive": True, "response_name": "auc",
        "assets": {name: {"path": "assets/samples.parquet", "sha256": "0" * 64}
                   for name in ("samples", "drugs", "pairs", "expression")},
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch: samples"):
        load_registered_cohort("c", root=tmp_path, registry=registry)
