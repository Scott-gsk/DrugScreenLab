import json
import numpy as np
import pandas as pd
from drug_screen.transfer.functional_feasibility import run_functional_feasibility


def test_functional_runner_uses_single_minlib_colorectal_and_writes_contract(tmp_path):
    base = tmp_path / "functional"; base.mkdir(); rng = np.random.default_rng(4); ids = [f"s{i}" for i in range(10)]
    samples = pd.DataFrame({"sample_ID": ids, "patient_id": [f"p{i}" for i in range(10)], "primary_tumour_type": ["Colorectal"] * 10})
    samples.to_parquet(base / "samples.parquet"); pd.DataFrame(rng.normal(size=(10, 8)), index=ids, columns=[f"g{i}_exp" for i in range(8)]).to_parquet(base / "expression_author_log2tpm.parquet")
    rows = [{"sample_ID": sample, "gene": gene, "library": "minLib", "LFC": rng.normal()} for sample in ids for gene in ("a", "b", "c")]
    rows += [{"sample_ID": "s0", "gene": "a", "library": "other", "LFC": 1.}]
    pd.DataFrame(rows).to_parquet(base / "dependency_by_library.parquet")
    report = run_functional_feasibility(base, tmp_path / "out", epochs=3)
    assert report["drug_response_transfer_established"] is False
    assert report["scope"] == "single-minLib colorectal technical subset"
    frozen = json.loads((tmp_path / "out" / "frozen_ids.json").read_text())
    assert "s0" not in pd.read_parquet(tmp_path / "out" / "predictions.parquet").sample_ID.unique()
    assert set(frozen["splits"]) == {"train", "validation", "test"}
    assert (tmp_path / "out" / "preprocessing_and_targets.npz").is_file()
    assert (tmp_path / "out" / "ridge_weights.npz").is_file()
    assert frozen["ridge_alpha"] == 10.0
