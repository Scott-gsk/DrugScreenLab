"""将 EXP-037 技术重复汇总为细胞×药物×剂量×时间条件连续变化。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION = ["cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def materialize(contract_dir: Path, cache_path: Path, output_dir: Path,
                condition_batch: int = 1024) -> dict:
    treatment = pd.read_parquet(contract_dir / "treatments.parquet")
    control = pd.read_parquet(contract_dir / "controls.parquet")
    if not treatment["quality_status"].eq("PASS").all():
        treatment = treatment.loc[treatment["quality_status"].eq("PASS")].copy()
    cache = np.load(cache_path, mmap_mode="r")
    control_means: dict[int, np.ndarray] = {}
    for group_id, frame in control.groupby("control_group_id", sort=False):
        values = np.asarray(cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float64)
        control_means[int(group_id)] = values.mean(axis=0).astype(np.float32)
    keys = treatment[CONDITION].drop_duplicates().sort_values(CONDITION).reset_index(drop=True)
    keys.insert(0, "condition_id", np.arange(len(keys), dtype=np.int64))
    coded = treatment.merge(keys, on=CONDITION, how="left", validate="many_to_one", sort=False)
    if coded["condition_id"].isna().any():
        raise ValueError("failed to assign condition identity")
    coded["condition_id"] = coded["condition_id"].astype(np.int64)
    members = coded.groupby("condition_id", sort=True).indices
    keys["n_treatment_instances"] = keys["condition_id"].map(
        coded.groupby("condition_id").size()).astype(np.int64)
    keys["n_control_groups"] = keys["condition_id"].map(
        coded.groupby("condition_id")["control_group_id"].nunique()).astype(np.int64)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    temp_matrix = output_dir / "condition_delta.npy.tmp"
    matrix = np.lib.format.open_memmap(temp_matrix, mode="w+", dtype=np.float32,
                                       shape=(len(keys), cache.shape[1]))
    for start in range(0, len(keys), condition_batch):
        stop = min(start + condition_batch, len(keys))
        for condition_id in range(start, stop):
            frame = coded.iloc[members[condition_id]]
            treated = np.asarray(cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float32)
            matched = np.stack([control_means[int(group)] for group in frame["control_group_id"]])
            matrix[condition_id] = (treated - matched).mean(axis=0, dtype=np.float64)
        matrix.flush()
        print(json.dumps({"completed_conditions": stop, "total_conditions": len(keys)}), flush=True)
    if not np.isfinite(matrix).all():
        raise ValueError("condition delta contains non-finite values")
    del matrix
    matrix_path = output_dir / "condition_delta.npy"
    os.replace(temp_matrix, matrix_path)
    key_path = output_dir / "conditions.parquet"
    keys.to_parquet(key_path, index=False)
    restored = np.load(matrix_path, mmap_mode="r")
    restored_keys = pd.read_parquet(key_path)
    if restored.shape != (len(restored_keys), 978):
        raise ValueError("condition matrix/key readback mismatch")
    report = {
        "protocol": "EXP-037 condition-level continuous delta v1",
        "status": "PASS",
        "aggregation": "mean of treatment-minus-own-matched-control over technical instances within cell, drug, dose and time",
        "n_source_treatments": int(len(treatment)), "n_conditions": int(len(keys)),
        "n_genes": int(restored.shape[1]),
        "source_contract_report_sha256": sha256(contract_dir / "quality_report.json"),
        "source_cache_sha256": sha256(cache_path),
        "files_sha256": {"conditions.parquet": sha256(key_path),
                         "condition_delta.npy": sha256(matrix_path)},
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-batch", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(materialize(args.contract_dir, args.cache, args.output_dir,
                                 args.condition_batch), ensure_ascii=False))


if __name__ == "__main__":
    main()
