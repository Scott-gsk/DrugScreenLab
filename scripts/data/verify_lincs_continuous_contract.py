"""独立核验 EXP-037 连续扰动合同的键、对照、数值和冻结划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(contract_dir: Path, cache_path: Path, split_path: Path) -> dict:
    report = json.loads((contract_dir / "quality_report.json").read_text(encoding="utf-8"))
    treatment = pd.read_parquet(contract_dir / "treatments.parquet")
    control = pd.read_parquet(contract_dir / "controls.parquet")
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("treatments.parquet", "controls.parquet", "genes.json"):
        if sha256(contract_dir / name) != report["files_sha256"][name]:
            raise ValueError(f"checksum mismatch: {name}")
    if sha256(split_path) != report["split_sha256"]:
        raise ValueError("split checksum mismatch")
    if len(genes) != 978 or len(set(genes)) != 978:
        raise ValueError("invalid 978-gene axis")
    if treatment["inst_id"].duplicated().any() or control["inst_id"].duplicated().any():
        raise ValueError("duplicate treatment or control identity")
    if set(treatment["control_group_id"]) != set(control["control_group_id"]):
        raise ValueError("treatment/control group set mismatch")
    group_meta = control.groupby("control_group_id")[KEY].nunique(dropna=False)
    if (group_meta != 1).any().any():
        raise ValueError("control group mixes matching metadata")
    group_key = control.groupby("control_group_id")[KEY].first()
    joined = treatment.join(group_key, on="control_group_id", rsuffix="_control")
    for column in KEY:
        if not joined[column].astype(str).equals(joined[f"{column}_control"].astype(str)):
            raise ValueError(f"matched control metadata mismatch: {column}")
    actual_n = control.groupby("control_group_id").size()
    if not treatment["n_matched_controls"].equals(
            treatment["control_group_id"].map(actual_n).astype(treatment["n_matched_controls"].dtype)):
        raise ValueError("stored matched-control count is incorrect")
    cache = np.load(cache_path, mmap_mode="r")
    if cache.shape[1] != len(genes):
        raise ValueError("cache/gene-axis shape mismatch")
    sampled = np.linspace(0, len(treatment) - 1, 25, dtype=int)
    max_reconstruction_error = 0.0
    for position in sampled:
        row = treatment.iloc[position]
        controls = control.loc[control["control_group_id"].eq(row["control_group_id"])]
        treated_value = np.asarray(cache[int(row["cache_row"])], dtype=np.float64)
        control_values = np.asarray(cache[controls["cache_row"].to_numpy(dtype=int)], dtype=np.float64)
        delta = treated_value - control_values.mean(axis=0)
        if not np.isfinite(delta).all():
            raise ValueError(f"non-finite reconstructed delta: {row['inst_id']}")
        reconstructed = delta + control_values.mean(axis=0)
        max_reconstruction_error = max(max_reconstruction_error,
                                       float(np.max(np.abs(reconstructed - treated_value))))
    all_cells = set(treatment["cell_id"])
    fold_receipts = []
    for fold in split["folds"]:
        train, test = set(fold["outer_train_ids"]), set(fold["outer_test_ids"])
        if train & test or train | test != all_cells:
            raise ValueError("outer cell identity split is incomplete or overlapping")
        train_control_cells = set(control.loc[control["cell_id"].isin(train), "cell_id"])
        test_control_cells = set(control.loc[control["cell_id"].isin(test), "cell_id"])
        if train_control_cells & test_control_cells:
            raise ValueError("control cell identity leakage")
        fold_receipts.append({"train_cells": len(train), "test_cells": len(test)})
    return {
        "status": "PASS",
        "n_treatments": int(len(treatment)), "n_controls": int(len(control)),
        "n_control_groups": int(treatment["control_group_id"].nunique()),
        "n_genes": len(genes), "sampled_delta_rows": len(sampled),
        "max_delta_reconstruction_error": max_reconstruction_error,
        "fold_receipts": fold_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract_dir, args.cache, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
