"""统一预测导出 / 读取（prediction_io）与身份字段语义守卫测试（W3-37）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_screen.data.identity_contracts import (
    IDENTITY_FIELD_SEMANTICS,
    assert_model_id_is_biological,
)
from drug_screen.data.prediction_io import (
    read_predictions,
    validate_prediction_frame,
    write_predictions,
)

KEYS = ("fold", "arm", "sample_id", "gene")


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 24
    return pd.DataFrame(
        {
            "fold": np.repeat(np.arange(1, 4), 8).astype("int8"),
            "arm": np.tile(np.repeat(["ZS", "HD"], 4), 3),
            "sample_id": [f"PDO-{i:02d}" for i in rng.permutation(n)],
            "gene": rng.choice(["A1BG", "A2M", "BRCA1", "TP53"], n),
            "predicted": rng.normal(size=n).astype(np.float32),
            "observed": rng.normal(size=n).astype(np.float32),
        }
    )


# ---------------------------------------------------------------- roundtrip --

def test_roundtrip_values_bit_identical_per_identity_key(tmp_path) -> None:
    frame = _predictions()
    path = tmp_path / "predictions_keyed.parquet"
    write_predictions(frame, path, identity_columns=KEYS)
    back = read_predictions(path, identity_columns=KEYS)
    assert len(back) == len(frame)
    merged = frame.merge(back, on=list(KEYS), suffixes=("", "_back"),
                         validate="one_to_one")
    assert len(merged) == len(frame)
    # 写出→读回→按身份键取值逐位一致（float32 原样保留）。
    assert merged["predicted"].tolist() == merged["predicted_back"].tolist()
    assert merged["observed"].tolist() == merged["observed_back"].tolist()


def test_roundtrip_is_row_order_independent(tmp_path) -> None:
    frame = _predictions()
    path = tmp_path / "predictions_keyed.parquet"
    write_predictions(frame, path, identity_columns=KEYS)
    back = read_predictions(path, identity_columns=KEYS)
    # 写出行序打乱后读回：按身份键的取值集合完全一致（行序无关）。
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    path2 = tmp_path / "predictions_keyed_shuffled.parquet"
    write_predictions(shuffled, path2, identity_columns=KEYS)
    back2 = read_predictions(path2, identity_columns=KEYS)
    pd.testing.assert_frame_equal(back, back2)


def test_read_returns_canonical_identity_order(tmp_path) -> None:
    frame = _predictions()
    path = tmp_path / "predictions_keyed.parquet"
    write_predictions(frame, path, identity_columns=KEYS)
    back = read_predictions(path, identity_columns=KEYS)
    assert back[["fold", "arm", "sample_id", "gene"]].equals(
        back[["fold", "arm", "sample_id", "gene"]]
        .sort_values(["fold", "arm", "sample_id", "gene"], kind="mergesort")
        .reset_index(drop=True))


def test_nan_predictions_are_returned_not_dropped(tmp_path) -> None:
    frame = _predictions()
    frame.loc[frame.index[:5], "predicted"] = np.nan
    path = tmp_path / "predictions_keyed.parquet"
    write_predictions(frame, path, identity_columns=KEYS)
    back = read_predictions(path, identity_columns=KEYS)
    assert len(back) == len(frame)
    assert back["predicted"].isna().sum() == 5


# ------------------------------------------------------------ write errors --

def test_write_rejects_duplicate_identity_keys(tmp_path) -> None:
    frame = pd.concat(
        [_predictions().iloc[:6], _predictions().iloc[:6]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate rows on"):
        write_predictions(frame, tmp_path / "p.parquet", identity_columns=KEYS)


def test_write_rejects_null_and_empty_identity_cells(tmp_path) -> None:
    frame = _predictions()
    frame.loc[frame.index[0], "sample_id"] = None
    with pytest.raises(ValueError, match="identity column 'sample_id' has 1 null"):
        write_predictions(frame, tmp_path / "p.parquet", identity_columns=KEYS)
    frame2 = _predictions()
    frame2.loc[frame2.index[1], "gene"] = "  "
    with pytest.raises(ValueError, match="empty cells"):
        write_predictions(frame2, tmp_path / "p.parquet", identity_columns=KEYS)


def test_write_rejects_missing_columns(tmp_path) -> None:
    frame = _predictions().drop(columns=["predicted"])
    with pytest.raises(ValueError, match="missing value column: predicted"):
        write_predictions(frame, tmp_path / "p.parquet", identity_columns=KEYS)
    frame2 = _predictions().drop(columns=["gene"])
    with pytest.raises(ValueError, match="missing identity column: gene"):
        write_predictions(frame2, tmp_path / "p.parquet", identity_columns=KEYS)


def test_write_rejects_non_numeric_values(tmp_path) -> None:
    frame = _predictions().astype({"predicted": object})
    frame.loc[frame.index[0], "predicted"] = "not-a-number"
    with pytest.raises(ValueError, match="non-numeric cells"):
        write_predictions(frame, tmp_path / "p.parquet", identity_columns=KEYS)


def test_write_rejects_empty_frame(tmp_path) -> None:
    empty = _predictions().iloc[0:0]
    with pytest.raises(ValueError, match="no rows"):
        write_predictions(empty, tmp_path / "p.parquet", identity_columns=KEYS)


def test_algorithm_column_participates_in_key(tmp_path) -> None:
    frame = _predictions()
    frame["algorithm"] = "ridge"
    # 同一身份键、同一算法：重复。
    dup = pd.concat([frame.iloc[:4], frame.iloc[:4]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate rows"):
        write_predictions(
            dup, tmp_path / "p.parquet", identity_columns=("fold", "sample_id", "gene"),
            algorithm_column="algorithm")
    # 不同算法身份：不重复，合法。
    other = frame.iloc[:4].copy()
    other["algorithm"] = "gbm"
    mixed = pd.concat([frame.iloc[:4], other], ignore_index=True)
    path = tmp_path / "p.parquet"
    write_predictions(
        mixed, path, identity_columns=("fold", "sample_id", "gene"),
        algorithm_column="algorithm")
    back = read_predictions(
        path, identity_columns=("fold", "sample_id", "gene"),
        algorithm_column="algorithm")
    assert len(back) == len(mixed)
    assert set(back["algorithm"]) == {"ridge", "gbm"}


# ------------------------------------------------- model_id 生物语义守卫 ----

def _bio_frame(model_ids, patient_ids) -> pd.DataFrame:
    return pd.DataFrame(
        {"model_id": model_ids, "patient_id": patient_ids,
         "predicted": np.zeros(len(model_ids), dtype=np.float32)})


def test_model_id_biological_passes_when_each_model_has_one_patient() -> None:
    frame = _bio_frame(
        ["Pt1_t1", "Pt1_t2", "Pt2_t1", "Pt3_t1"], ["Pt1", "Pt1", "Pt2", "Pt3"]
    )
    assert assert_model_id_is_biological(frame) is None


def test_model_id_used_as_algorithm_identity_is_caught() -> None:
    # 算法名误入 model_id：同一取值跨多个患者 → 结构性失败。
    frame = _bio_frame(
        ["M3", "M3", "M3", "ridge", "ridge"], ["Pt1", "Pt2", "Pt3", "Pt1", "Pt2"]
    )
    with pytest.raises(ValueError, match="spans multiple patient_id"):
        assert_model_id_is_biological(frame)


def test_model_id_spanning_patients_rejected_at_write(tmp_path) -> None:
    frame = _bio_frame(
        ["PDO-A", "PDO-A", "PDO-B"], ["Pt1", "Pt2", "Pt3"]
    )
    frame["sample_id"] = ["s1", "s2", "s3"]
    frame["drug_id"] = ["d1", "d1", "d1"]
    frame["fold"] = 1
    with pytest.raises(ValueError, match="model_id spans multiple patient_id"):
        write_predictions(
            frame, tmp_path / "p.parquet",
            identity_columns=("fold", "sample_id", "drug_id"))


def test_identity_semantics_contract_documents_separation() -> None:
    # 语义契约：model_id 生物；算法/运行身份各有独立字段。
    assert "绝不表示算法或运行身份" in IDENTITY_FIELD_SEMANTICS["model_id"]
    assert IDENTITY_FIELD_SEMANTICS["patient_id"].startswith("患者身份")
    for field in ("algorithm", "algorithm_id", "arm", "run_id", "seed"):
        assert field in IDENTITY_FIELD_SEMANTICS


def test_guard_noop_without_patient_column() -> None:
    # 细胞系来源表无患者列：无操作，不误伤。
    frame = pd.DataFrame({"model_id": ["A549", "A549"], "predicted": [1.0, 2.0]})
    assert assert_model_id_is_biological(frame) is None


def test_validate_prediction_frame_returns_error_list() -> None:
    frame = _predictions().drop(columns=["observed"])
    errors = validate_prediction_frame(
        frame, identity_columns=KEYS, observed_columns=("observed",))
    assert any("missing observed column: observed" in e for e in errors)
