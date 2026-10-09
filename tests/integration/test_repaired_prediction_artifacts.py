"""身份修复版与只读历史预测逐字段一致。"""

from pathlib import Path

from scripts.data.verify_legacy_identity_repair import verify


ROOT = Path(__file__).resolve().parents[2] / "artifacts"


def test_exp032_l1_repaired_all_rows_keep_original_values():
    result = verify(
        ROOT / "exp032_l1l2/l1_predictions_rebased.parquet",
        ROOT / "experiments/EXP-032/20260922_identity_repair/l1_predictions_keyed_v2.parquet",
        "l1",
    )
    assert result["rows_checked"] == 18479230


def test_exp033_c4_repaired_all_rows_keep_original_values():
    result = verify(
        ROOT / "exp033_routeB/c4_predictions.parquet",
        ROOT / "experiments/EXP-033/20260922_identity_repair/c4_predictions_keyed_v2.parquet",
        "c4",
    )
    assert result["rows_checked"] == 2116
