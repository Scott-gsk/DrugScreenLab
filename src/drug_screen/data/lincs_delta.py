"""LINCS Level 3 处理减真实同板载体对照的按身份读取接口。

输入缓存行轴必须是 GCTX COL/id 顺序。基因轴必须先按来源 ROW/id 中
pr_is_lm=1 的顺序验收；本模块不猜测矩阵行序，也不把 Level 4/5 当连续差值。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


MATCH_KEY = ("rna_plate", "cell_id", "pert_time", "pert_time_unit")


class LincsMatchedDelta:
    """一个 ``trt_cp`` 实例的 978 基因连续变化及真实对照来源。"""

    def __init__(
        self, cache_path: str | Path, instance_ids: Sequence[str],
        metadata: pd.DataFrame,
    ) -> None:
        self.cache = np.load(cache_path, mmap_mode="r")
        ids = [str(x) for x in instance_ids]
        if self.cache.shape != (len(ids), 978):
            raise ValueError("cache must be [n_instances, 978] aligned to instance_ids")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate instance IDs on cache row axis")
        required = {"inst_id", "pert_type", *MATCH_KEY}
        if required - set(metadata.columns):
            raise ValueError(f"metadata lacks columns: {sorted(required - set(metadata.columns))}")
        if metadata["inst_id"].duplicated().any():
            raise ValueError("duplicate inst_id in metadata")
        self.position = {identifier: i for i, identifier in enumerate(ids)}
        self.metadata = metadata.set_index("inst_id", verify_integrity=True)
        if set(self.metadata.index.astype(str)) != set(ids):
            raise ValueError("cache instance IDs and metadata inst_id sets differ")
        self.controls: dict[tuple[str, ...], tuple[str, ...]] = {}
        controls = metadata.loc[metadata["pert_type"].eq("ctl_vehicle")]
        for key, group in controls.groupby(list(MATCH_KEY), dropna=False, sort=False):
            self.controls[tuple(str(x) for x in key)] = tuple(group["inst_id"].astype(str))

    def control_ids(self, treatment_id: str) -> tuple[str, ...]:
        if treatment_id not in self.metadata.index:
            raise KeyError(f"unknown treatment instance: {treatment_id}")
        row = self.metadata.loc[treatment_id]
        if row["pert_type"] != "trt_cp":
            raise ValueError(f"instance {treatment_id} is not a chemical treatment")
        key = tuple(str(row[field]) for field in MATCH_KEY)
        result = self.controls.get(key, ())
        if not result:
            raise ValueError(f"no same-plate, cell and time vehicle for {treatment_id}")
        return result

    @lru_cache(maxsize=2048)
    def _control_mean(self, control_ids: tuple[str, ...]) -> np.ndarray:
        positions = [self.position[identifier] for identifier in control_ids]
        values = np.asarray(self.cache[positions], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite control expression: {control_ids[:3]}")
        return values.mean(axis=0, dtype=np.float64).astype(np.float32)

    def delta(self, treatment_id: str) -> tuple[np.ndarray, tuple[str, ...]]:
        """返回 ``(处理表达-全部匹配载体对照均值, 对照实例ID)``。"""
        controls = self.control_ids(treatment_id)
        treated = np.asarray(self.cache[self.position[treatment_id]], dtype=np.float32)
        if not np.isfinite(treated).all():
            raise ValueError(f"non-finite treatment expression: {treatment_id}")
        return treated - self._control_mean(controls), controls
