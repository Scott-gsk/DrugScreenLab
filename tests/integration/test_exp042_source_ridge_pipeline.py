import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from drug_screen.data.group_splits import nested_identity_folds
from scripts.evaluation.verify_exp042_source_ridge import verify
from scripts.modeling.exp042_source_ridge import run


def test_exp042_full_five_fold_training_and_independent_verification(tmp_path):
    contract = tmp_path / "contract"
    contract.mkdir()
    sample_ids = [f"SIDM{i:05d}" for i in range(10)]
    drug_ids = ["drug_a", "drug_b", "drug_c"]
    genes = [f"gene_{index}" for index in range(19091)]
    expression = np.zeros((len(sample_ids), len(genes)), dtype=np.float32)
    expression[:, 0] = np.linspace(-1.0, 1.0, len(sample_ids))
    expression[:, 1] = np.sin(np.arange(len(sample_ids), dtype=np.float32))
    expression[:, 2] = np.cos(np.arange(len(sample_ids), dtype=np.float32))
    rows = []
    base = {"drug_a": 0.1, "drug_b": 0.4, "drug_c": 0.8}
    slope = {"drug_a": 0.3, "drug_b": -0.2, "drug_c": 0.1}
    for cell_index, sample_id in enumerate(sample_ids):
        for drug_index, drug_id in enumerate(drug_ids):
            response = (base[drug_id] + slope[drug_id] * expression[cell_index, 0]
                        + 0.004 * ((cell_index + drug_index) % 2))
            rows.append({"sample_id": sample_id, "drug_id": drug_id,
                         "response": response})
    pd.DataFrame(rows).to_parquet(contract / "source_pairs.parquet", index=False)
    pd.DataFrame({"drug_id": drug_ids}).to_parquet(contract / "drugs.parquet", index=False)
    (contract / "genes.json").write_text(json.dumps(genes), encoding="utf-8")
    (contract / "manifest.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split = {"source_folds": nested_identity_folds(
        sample_ids, seed=20260922, fractions=(1.0,))}
    split_path.write_text(json.dumps(split), encoding="utf-8")
    source_path = tmp_path / "source.npz"
    np.savez(source_path, expression=expression,
             sample_id=np.asarray(sample_ids), cell_line_id=np.asarray(sample_ids),
             ordered_genes=np.asarray(genes))

    output_dir = tmp_path / "run"
    with threadpool_limits(limits=1):
        summary = run(contract, source_path, split_path, output_dir)
    assert summary["status"] == "FIVE_FOLDS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
    result = verify(output_dir, contract, source_path, split_path)
    assert result["status"] == "PASS"
    assert result["n_verified_outer_folds"] == 5
    assert result["primary_statistics_recomputed"]["n_cells"] == len(sample_ids)
