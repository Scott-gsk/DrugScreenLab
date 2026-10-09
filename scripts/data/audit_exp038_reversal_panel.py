"""Freeze the measured 24 h, 10 µM held-out-cell panel for EXP-038."""

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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(condition_dir: Path, contract_dir: Path, split_path: Path,
          signature_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"panel audit output exists: {output_dir}")
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    if keys["condition_id"].tolist() != list(range(len(keys))) or keys["condition_id"].duplicated().any():
        raise ValueError("condition axis is not unique sequential identity")
    if not keys["pert_dose_unit"].eq("um").all() or not keys["pert_time_unit"].eq("h").all():
        raise ValueError("condition dose/time units disagree with contract")
    delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if delta.shape != (len(keys), len(genes)) or len(genes) != 978 or len(set(genes)) != 978:
        raise ValueError("measured delta and gene axes differ")
    if len(up) < 20 or len(down) < 20 or set(up) & set(down) or not set(up + down) <= set(genes):
        raise ValueError("signature directions fail fixed gene-axis gate")
    dose = pd.to_numeric(keys["pert_dose"], errors="raise").to_numpy(dtype=np.float64)
    time = pd.to_numeric(keys["pert_time"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(dose).all() or not np.isfinite(time).all():
        raise ValueError("non-finite exposure in condition contract")
    fixed = np.isclose(dose, 10.0, rtol=0, atol=1e-6) & (time == 24.0)
    fixed_keys = keys.loc[fixed].copy()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    rows = []
    fold_counts = []
    for fold_number, fold in enumerate(split["folds"]):
        train = keys.loc[keys["cell_id"].isin(fold["outer_train_ids"])]
        seen = set(train["pert_id"])
        test = fixed_keys.loc[fixed_keys["cell_id"].isin(fold["outer_test_ids"])].copy()
        seen_mask = test["pert_id"].isin(seen)
        test = test.loc[seen_mask].copy()
        if test["cell_id"].nunique() > len(fold["outer_test_ids"]):
            raise ValueError("test cell outside frozen fold")
        test.insert(0, "fold", fold_number)
        test["panel_status"] = "PENDING_CELL_DRUG_GATE"
        counts = test.groupby("cell_id", sort=True)["pert_id"].nunique()
        eligible_cells = set(counts.loc[counts.ge(5)].index)
        test["panel_status"] = np.where(test["cell_id"].isin(eligible_cells),
                                         "ELIGIBLE", "FEWER_THAN_FIVE_DRUGS")
        rows.append(test)
        fold_counts.append({
            "fold": fold_number,
            "fixed_test_conditions_before_seen_drug_filter": int(fixed_keys["cell_id"].isin(
                fold["outer_test_ids"]).sum()),
            "fixed_test_conditions_seen_drug": len(test),
            "eligible_cells": len(eligible_cells),
            "eligible_conditions": int(test["panel_status"].eq("ELIGIBLE").sum()),
            "eligible_drugs": int(test.loc[test["panel_status"].eq("ELIGIBLE"), "pert_id"].nunique()),
        })
    panel = pd.concat(rows, ignore_index=True)
    if panel["condition_id"].duplicated().any():
        raise ValueError("condition appears in multiple outer test folds")
    if not np.isfinite(np.asarray(delta[panel["condition_id"].to_numpy()], dtype=np.float32)).all():
        raise ValueError("fixed measured panel has non-finite delta")
    output_dir.mkdir(parents=True)
    panel_path = output_dir / "panel_conditions.parquet"
    panel.to_parquet(panel_path, index=False)
    restored = pd.read_parquet(panel_path)
    pd.testing.assert_frame_equal(restored, panel)
    report = {
        "experiment_id": "EXP-038", "status": "MEASURED_PANEL_QC_PASS_PENDING_PREDICTIONS",
        "selection": "24 h and 10 µM; held-out cell; drug seen in corresponding outer training cells; >=5 drugs per cell",
        "source_sha256": {
            "condition_manifest": sha256(condition_dir / "manifest.json"),
            "condition_matrix": sha256(condition_dir / "condition_delta.npy"),
            "gene_axis": sha256(contract_dir / "genes.json"),
            "split": sha256(split_path), "signature_manifest": sha256(signature_dir / "manifest.json"),
        },
        "n_fixed_conditions": len(fixed_keys),
        "n_seen_drug_heldout_conditions": len(panel),
        "n_eligible_conditions": int(panel["panel_status"].eq("ELIGIBLE").sum()),
        "n_eligible_cells": int(panel.loc[panel["panel_status"].eq("ELIGIBLE"), "cell_id"].nunique()),
        "n_excluded_cells_fewer_than_five_drugs": int(panel.loc[
            panel["panel_status"].eq("FEWER_THAN_FIVE_DRUGS"), "cell_id"].nunique()),
        "folds": fold_counts,
        "panel_sha256": sha256(panel_path),
    }
    (output_dir / "panel_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.condition_dir, args.contract_dir, args.split, args.signature_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
