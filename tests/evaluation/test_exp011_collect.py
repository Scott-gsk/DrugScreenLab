from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from drug_screen.evaluation.exp011 import PREDICTION_COLUMNS


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analysis/collect_exp011_predictions.py"
spec = importlib.util.spec_from_file_location("collect_exp011_predictions", SCRIPT)
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(collector)


def _frame(*, fold: int, sample: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "experiment_id": "EXP-011", "model_id": "M2", "evaluation_role": "gdsc_cold_cell",
        "cohort_id": "gdsc2_cell_line", "fold_id": fold, "split": "test", "patient_id": "", "sample_id": sample,
        "pdo_id": "", "cell_line_id": sample, "drug_id": "drug", "inchikey": "IK", "drug_scaffold": "ACYCLIC",
        "response_name": "1_minus_auc", "response_scale": "1-AUC", "higher_is_sensitive": True, "y_true": 0.5, "y_pred": 0.4,
    }], columns=PREDICTION_COLUMNS)


def test_collect_validates_and_rejects_duplicate_heldout_rows(tmp_path: Path) -> None:
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    _frame(fold=0, sample="s0").to_parquet(first, index=False)
    _frame(fold=1, sample="s1").to_parquet(second, index=False)
    collected = collector.collect([first, second])
    assert collected["sample_id"].tolist() == ["s0", "s1"]
    _frame(fold=0, sample="s0").to_parquet(second, index=False)
    with pytest.raises(Exception, match="duplicate"):
        collector.collect([first, second])
