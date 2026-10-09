"""从逐实例合同独立复算条件级 LINCS 连续变化抽样。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION = ["cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(contract_dir: Path, condition_dir: Path, cache_path: Path) -> dict:
    manifest = json.loads((condition_dir / "manifest.json").read_text(encoding="utf-8"))
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    values = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    treatments = pd.read_parquet(contract_dir / "treatments.parquet")
    controls = pd.read_parquet(contract_dir / "controls.parquet")
    for filename in ("conditions.parquet", "condition_delta.npy"):
        if sha256(condition_dir / filename) != manifest["files_sha256"][filename]:
            raise ValueError(f"checksum mismatch: {filename}")
    if values.shape != (len(keys), 978) or keys["condition_id"].tolist() != list(range(len(keys))):
        raise ValueError("condition identity/matrix axis mismatch")
    if keys.duplicated(CONDITION).any():
        raise ValueError("duplicate biological treatment condition")
    cache = np.load(cache_path, mmap_mode="r")
    control_means = {}
    for group, frame in controls.groupby("control_group_id", sort=False):
        control_means[int(group)] = np.asarray(
            cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float64).mean(axis=0)
    samples = np.linspace(0, len(keys) - 1, 25, dtype=int)
    maximum = 0.0
    for condition_id in samples:
        key = keys.iloc[condition_id]
        mask = np.ones(len(treatments), dtype=bool)
        for column in CONDITION:
            mask &= treatments[column].astype(str).to_numpy() == str(key[column])
        source = treatments.loc[mask]
        if len(source) != int(key["n_treatment_instances"]):
            raise ValueError(f"source count mismatch for condition {condition_id}")
        treated = np.asarray(cache[source["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float64)
        matched = np.stack([control_means[int(group)] for group in source["control_group_id"]])
        expected = (treated - matched).mean(axis=0)
        error = float(np.max(np.abs(expected - np.asarray(values[condition_id], dtype=np.float64))))
        maximum = max(maximum, error)
        if error > 2e-6:
            raise ValueError(f"condition delta mismatch at {condition_id}: {error}")
    return {"status": "PASS", "n_conditions": int(len(keys)), "n_genes": 978,
            "sampled_conditions": len(samples), "max_abs_difference": maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract_dir, args.condition_dir, args.cache)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
