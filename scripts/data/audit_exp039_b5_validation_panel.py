"""Audit disease-reversal validation coverage for the preregistered B5 folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(condition_dir: Path, exposure_path: Path, split_path: Path,
          signature_dir: Path) -> dict:
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    exposure = pd.read_parquet(exposure_path)
    if len(keys) != len(exposure) or not keys["condition_id"].equals(exposure["condition_id"]):
        raise ValueError("condition keys and exposure axes differ")
    fixed = np.isclose(exposure["time_hours"].to_numpy(float), 24.0, rtol=0, atol=1e-8)
    fixed &= np.isclose(exposure["log10_dose_um"].to_numpy(float), 1.0, rtol=0, atol=1e-8)
    fixed_pairs = set(zip(keys.loc[fixed, "cell_id"], keys.loc[fixed, "pert_id"]))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    signature = json.loads((signature_dir / "manifest.json").read_text(encoding="utf-8"))
    signature_files = ["up_genes.json", "down_genes.json", "manifest.json"]
    if any(not (signature_dir / name).is_file() for name in signature_files):
        raise ValueError("frozen signature files are incomplete")
    up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if min(len(up), len(down)) < 20:
        raise ValueError("disease signature fails the two-direction gene gate")

    folds = []
    for outer in split["folds"]:
        inner_rows = []
        for inner in outer["label_fractions"][0]["inner_folds"]:
            train_cells = set(inner["train_ids"])
            validation_cells = set(inner["validation_ids"])
            train_drugs = set(keys.loc[keys["cell_id"].isin(train_cells), "pert_id"])
            per_cell = {
                cell: len({drug for cell_id, drug in fixed_pairs
                           if cell_id == cell and drug in train_drugs})
                for cell in sorted(validation_cells)
            }
            inner_rows.append({
                "inner_fold": inner["inner_fold"],
                "n_validation_cells": len(validation_cells),
                "n_validation_cells_with_at_least_five_seen_drugs": sum(n >= 5 for n in per_cell.values()),
                "n_validation_cell_drugs": sum(per_cell.values()),
                "drugs_per_validation_cell": per_cell,
                "n_train_seen_drugs": len(train_drugs),
            })
        folds.append({"fold": outer["fold"], "inner_folds": inner_rows})
    gate = all(row["n_validation_cells_with_at_least_five_seen_drugs"] >= 3
               for outer in folds for row in outer["inner_folds"])
    return {
        "experiment_id": "EXP-039",
        "status": "B5_VALIDATION_PANEL_GATE_PASS" if gate else "B5_VALIDATION_PANEL_GATE_FAIL",
        "selection_metric": "equal-cell mean Spearman of predicted versus measured reversal at 24h and 10uM",
        "cell_eligibility": "at least five unique train-seen drugs in the fixed validation panel; at least three eligible cells per inner fold",
        "n_fixed_condition_rows": int(fixed.sum()),
        "n_fixed_cell_drug_pairs": len(fixed_pairs),
        "up_signature_genes": len(up),
        "down_signature_genes": len(down),
        "source_sha256": {
            "condition_manifest": sha256(condition_dir / "manifest.json"),
            "exposure": sha256(exposure_path),
            "split": sha256(split_path),
            "signature_manifest": sha256(signature_dir / "manifest.json"),
            "up_genes": sha256(signature_dir / "up_genes.json"),
            "down_genes": sha256(signature_dir / "down_genes.json"),
        },
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.condition_dir, args.exposure, args.split, args.signature_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "n_fixed_condition_rows": receipt["n_fixed_condition_rows"],
                      "inner_fold_eligible_cells": [
                          row["n_validation_cells_with_at_least_five_seen_drugs"]
                          for outer in receipt["folds"] for row in outer["inner_folds"]]}))
    if receipt["status"] != "B5_VALIDATION_PANEL_GATE_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
