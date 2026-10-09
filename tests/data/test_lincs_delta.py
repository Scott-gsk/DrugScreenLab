"""连续变化必须由正确处理实例与全部真实匹配对照计算。"""

import numpy as np
import pandas as pd
import pytest

from drug_screen.data.lincs_delta import LincsMatchedDelta


def test_delta_matches_same_plate_cell_time_controls(tmp_path):
    ids = ["c1", "c2", "c_wrong_plate", "t1", "t_wrong_time"]
    cache = np.zeros((len(ids), 978), dtype=np.float32)
    cache[0, :2] = [2, 4]
    cache[1, :2] = [4, 6]
    cache[2, :2] = [100, 100]
    cache[3, :2] = [8, 12]
    path = tmp_path / "cache.npy"
    np.save(path, cache)
    meta = pd.DataFrame([
        ("c1", "ctl_vehicle", "p1", "A", "24", "h"),
        ("c2", "ctl_vehicle", "p1", "A", "24", "h"),
        ("c_wrong_plate", "ctl_vehicle", "p2", "A", "24", "h"),
        ("t1", "trt_cp", "p1", "A", "24", "h"),
        ("t_wrong_time", "trt_cp", "p1", "A", "6", "h"),
    ], columns=["inst_id", "pert_type", "rna_plate", "cell_id",
                "pert_time", "pert_time_unit"])
    source = LincsMatchedDelta(path, ids, meta)
    delta, controls = source.delta("t1")
    assert controls == ("c1", "c2")
    np.testing.assert_array_equal(delta[:2], [5, 7])
    np.testing.assert_array_equal(delta[2:], 0)
    with pytest.raises(ValueError, match="no same-plate"):
        source.delta("t_wrong_time")
    with pytest.raises(ValueError, match="not a chemical"):
        source.delta("c1")


def test_cache_row_identity_must_match_metadata(tmp_path):
    path = tmp_path / "cache.npy"
    np.save(path, np.zeros((2, 978), dtype=np.float32))
    meta = pd.DataFrame([
        ("a", "trt_cp", "p", "A", "24", "h"),
        ("b", "ctl_vehicle", "p", "A", "24", "h"),
    ], columns=["inst_id", "pert_type", "rna_plate", "cell_id",
                "pert_time", "pert_time_unit"])
    with pytest.raises(ValueError, match="sets differ"):
        LincsMatchedDelta(path, ["a", "other"], meta)
