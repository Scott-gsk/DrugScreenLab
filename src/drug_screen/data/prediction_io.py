"""统一预测长表导出 / 读取（W3-37 往返保障）。

EXP-030 ``predictions.parquet`` 的 (sample, gene) 标签 sample-major /
数值 gene-major 行序错位（2026-09-22 审计证实）说明：写出后依赖
"书写行序"读回不可靠。本模块是预测长表的显式键值统一出入口：

- ``write_predictions``：schema 校验（身份列存在且非空、数值列存在且
  可数值化、身份键唯一、``model_id`` 生物语义守卫）通过后写 parquet；
- ``read_predictions``：读回并按身份列校验唯一性、重排为规范顺序
  （行序由身份键决定，与书写序无关）；
- ``validate_prediction_frame``：校验的纯函数形式，返回错误清单。

往返保证：写出→读回→按身份键取值逐位一致（parquet 保留 dtype，
见 ``tests/data/test_prediction_io.py``）。

身份字段语义遵循 :mod:`drug_screen.data.identity_contracts`
（``model_id`` 恒为生物模型身份；算法身份用 ``algorithm``/``arm`` 等
独立字段；运行身份用 ``run_id``/``seed``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from drug_screen.data.identity_contracts import assert_model_id_is_biological

__all__ = [
    "validate_prediction_frame",
    "write_predictions",
    "read_predictions",
]


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    errors: list[str],
    what: str,
) -> None:
    for column in columns:
        if column not in frame.columns:
            errors.append(f"missing {what} column: {column}")


def _check_identity_cells(
    frame: pd.DataFrame, columns: Sequence[str], errors: list[str]
) -> None:
    for column in columns:
        null_mask = frame[column].isna()
        if null_mask.any():
            errors.append(
                f"identity column {column!r} has {int(null_mask.sum())} null cells")
            continue
        empty_mask = frame[column].astype(str).str.strip().eq("")
        if empty_mask.any():
            errors.append(
                f"identity column {column!r} has {int(empty_mask.sum())} "
                "empty cells")


def _check_numeric_cells(
    frame: pd.DataFrame, columns: Sequence[str], errors: list[str]
) -> None:
    for column in columns:
        coerced = pd.to_numeric(frame[column], errors="coerce")
        bad = coerced.isna() & frame[column].notna()
        if bad.any():
            errors.append(
                f"value column {column!r} has {int(bad.sum())} non-numeric cells")


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    value_columns: Sequence[str] = ("predicted",),
    observed_columns: Sequence[str] = (),
    algorithm_column: str | None = None,
) -> list[str]:
    """校验预测长表，返回错误清单（空列表 = 通过）。

    校验项：身份列 / 数值列存在；身份单元格非空非空白；数值列可数值化
    （NaN 预测按项目约定保留返回、不算错误）；身份键（含算法身份列）唯一；
    ``model_id`` + ``patient_id`` 同时存在时执行生物语义守卫
    （一个生物模型不得跨患者——算法名误入 ``model_id`` 会被结构性捕获）。
    """
    errors: list[str] = []
    if not isinstance(frame, pd.DataFrame):
        return [f"predictions must be a pandas DataFrame, got {type(frame).__name__}"]
    if frame.empty:
        return ["prediction frame has no rows"]

    _require_columns(frame, identity_columns, errors, "identity")
    _require_columns(frame, value_columns, errors, "value")
    _require_columns(frame, observed_columns, errors, "observed")
    if algorithm_column is not None and algorithm_column not in frame.columns:
        errors.append(f"missing algorithm column: {algorithm_column}")
    if errors:
        return errors

    _check_identity_cells(frame, identity_columns, errors)
    _check_numeric_cells(frame, value_columns, errors)
    _check_numeric_cells(frame, observed_columns, errors)

    key_columns = list(identity_columns)
    if algorithm_column is not None:
        key_columns.append(algorithm_column)
    dup_mask = frame.duplicated(subset=key_columns, keep=False)
    if dup_mask.any():
        n_unique = int(frame.loc[~dup_mask, key_columns].drop_duplicates().shape[0])
        errors.append(
            f"duplicate rows on {key_columns}: {n_unique} unique keys but "
            f"{int(dup_mask.sum())} rows involved in duplicates")

    try:
        assert_model_id_is_biological(frame)
    except ValueError as error:
        errors.append(str(error))
    return errors


def write_predictions(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    identity_columns: Sequence[str],
    value_columns: Sequence[str] = ("predicted",),
    observed_columns: Sequence[str] = (),
    algorithm_column: str | None = None,
) -> None:
    """校验后写预测长表 parquet（显式键值对应，行序无隐含语义）。

    Raises
    ------
    ValueError
        校验错误清单非空时抛出（消息含全部错误）。
    """
    errors = validate_prediction_frame(
        frame,
        identity_columns=identity_columns,
        value_columns=value_columns,
        observed_columns=observed_columns,
        algorithm_column=algorithm_column,
    )
    if errors:
        raise ValueError("prediction frame validation failed: " + "; ".join(errors))
    frame.to_parquet(Path(path), index=False)


def read_predictions(
    path: str | Path,
    *,
    identity_columns: Sequence[str],
    value_columns: Sequence[str] = ("predicted",),
    observed_columns: Sequence[str] = (),
    algorithm_column: str | None = None,
) -> pd.DataFrame:
    """读回预测长表：校验唯一性并按身份列重排为规范顺序。

    返回按 ``identity_columns``（含算法身份列，若有）稳定排序的 DataFrame
    ——行序由身份键决定，与写出时的书写序无关。

    Raises
    ------
    ValueError
        缺列、身份空值、身份键重复或 ``model_id`` 生物语义失败时抛出。
    """
    frame = pd.read_parquet(Path(path))
    errors = validate_prediction_frame(
        frame,
        identity_columns=identity_columns,
        value_columns=value_columns,
        observed_columns=observed_columns,
        algorithm_column=algorithm_column,
    )
    if errors:
        raise ValueError("prediction file validation failed: " + "; ".join(errors))
    sort_columns = list(identity_columns)
    if algorithm_column is not None:
        sort_columns.append(algorithm_column)
    return (
        frame.sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )
