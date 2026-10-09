"""依据 2026-09-22 方案生成只依赖身份的来源和类器官划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from drug_screen.data.group_splits import nested_identity_folds


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(admission: Path, target_samples: Path, source_samples: Path) -> dict:
    allowed = json.loads(admission.read_text(encoding="utf-8"))
    patients = set(map(str, allowed["patient_whitelist"]))
    samples = set(map(str, allowed["pdo_whitelist"]))
    target = pd.read_parquet(target_samples, columns=["sample_id", "patient_id"])
    target = target.loc[target["sample_id"].astype(str).isin(samples)].copy()
    if set(target["sample_id"].astype(str)) != samples:
        raise ValueError("whitelisted PDO samples not all found")
    if set(target["patient_id"].astype(str)) != patients:
        raise ValueError("whitelisted patients do not match sample mapping")
    if target["sample_id"].duplicated().any():
        raise ValueError("duplicate PDO sample IDs")
    source = pd.read_parquet(source_samples, columns=["cell_line_id"])
    lines = source["cell_line_id"].astype(str).tolist()
    if len(set(lines)) != len(lines):
        raise ValueError("duplicate source cell-line IDs")
    target_folds = nested_identity_folds(sorted(patients), seed=20260922)
    source_folds = nested_identity_folds(lines, seed=20260922, fractions=(1.0,))
    target_lookup = dict(zip(target["sample_id"].astype(str), target["patient_id"].astype(str)))
    for fold in target_folds:
        fold["outer_train_samples"] = sorted(
            sample for sample, patient in target_lookup.items()
            if patient in set(fold["outer_train_ids"]))
        fold["outer_test_samples"] = sorted(
            sample for sample, patient in target_lookup.items()
            if patient in set(fold["outer_test_ids"]))
    return {
        "protocol": "dual_route_20260922_identity_only_v1",
        "status": "SPLIT_FROZEN_LABEL_COVERAGE_PENDING",
        "split_seed": 20260922,
        "model_initialization_seeds": [20260922, 20260923, 20260924],
        "n_target_patients": len(patients), "n_target_samples": len(samples),
        "n_source_cell_lines": len(lines),
        "source_files": {
            "admission": str(admission.resolve()), "admission_sha256": _hash(admission),
            "target_samples": str(target_samples.resolve()),
            "target_samples_sha256": _hash(target_samples),
            "source_samples": str(source_samples.resolve()),
            "source_samples_sha256": _hash(source_samples),
        },
        "target_folds": target_folds, "source_folds": source_folds,
        "label_policy": "No drug-response labels or expression values were read to create folds",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--target-samples", type=Path, required=True)
    parser.add_argument("--source-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.admission, args.target_samples, args.source_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "target_patients": result["n_target_patients"],
                      "source_cell_lines": result["n_source_cell_lines"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
