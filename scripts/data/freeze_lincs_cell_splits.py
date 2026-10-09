"""为连续扰动主场景冻结未见细胞划分及可用处理实例清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from drug_screen.data.group_splits import nested_identity_folds


KEY = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_eligible_instances(instances: pd.DataFrame, compounds: pd.DataFrame, *,
                              require_parseable_structure: bool = False) -> pd.DataFrame:
    """Return the preregistered metadata cohort without changing source row identity."""
    if compounds["pert_id"].duplicated().any():
        raise ValueError("pert_info contains duplicate pert_id")
    structured = set(compounds.loc[
        compounds["pert_type"].eq("trt_cp")
        & ~compounds["canonical_smiles"].isin(["", "-666"])
        & ~compounds["inchi_key"].isin(["", "-666"]), "pert_id"
    ])
    if require_parseable_structure:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        structure_rows = compounds.loc[compounds["pert_id"].isin(structured)]
        structured = set(structure_rows.loc[
            [Chem.MolFromSmiles(value) is not None
             for value in structure_rows["canonical_smiles"]], "pert_id"])
    controls = set(map(tuple, instances.loc[
        instances["pert_type"].eq("ctl_vehicle"), KEY
    ].itertuples(index=False, name=None)))
    treatment = instances.loc[instances["pert_type"].eq("trt_cp")].copy()
    dose = pd.to_numeric(treatment["pert_dose"], errors="coerce")
    treatment = treatment.loc[
        treatment["pert_id"].isin(structured)
        & treatment["pert_time"].isin(["6", "24"])
        & treatment["pert_time_unit"].eq("h")
        & treatment["pert_dose_unit"].str.lower().eq("um")
        & dose.gt(0)
    ].copy()
    has_control = [tuple(row) in controls for row in
                   treatment[KEY].itertuples(index=False, name=None)]
    treatment = treatment.loc[has_control]
    if treatment["inst_id"].duplicated().any():
        raise ValueError("eligible treatment inst_id is not unique")
    return treatment


def build(inst_info: Path, pert_info: Path, *, require_parseable_structure: bool = False) -> dict:
    instances = pd.read_csv(inst_info, sep="\t", dtype=str, keep_default_na=False)
    compounds = pd.read_csv(pert_info, sep="\t", dtype=str, keep_default_na=False)
    treatment = select_eligible_instances(
        instances, compounds, require_parseable_structure=require_parseable_structure)
    cells = sorted(treatment["cell_id"].unique())
    if len(cells) < 5:
        raise ValueError("too few cell identities for outer five-fold split")
    folds = nested_identity_folds(cells, seed=20260922, fractions=(1.0,))
    for fold in folds:
        train = treatment.loc[treatment["cell_id"].isin(fold["outer_train_ids"])]
        test = treatment.loc[treatment["cell_id"].isin(fold["outer_test_ids"])]
        seen = set(train["pert_id"])
        fold["n_train_instances"] = int(len(train))
        fold["n_test_instances"] = int(len(test))
        fold["n_test_instances_seen_drug"] = int(test["pert_id"].isin(seen).sum())
        fold["n_test_drugs_seen_in_train"] = int(test.loc[test["pert_id"].isin(seen), "pert_id"].nunique())
        fold["n_test_drugs_unseen_in_train"] = int(test.loc[~test["pert_id"].isin(seen), "pert_id"].nunique())
    return {
        "protocol": ("dual_route_lincs_level3_cell_holdout_20260922_v2_parseable_structure"
                     if require_parseable_structure else
                     "dual_route_lincs_level3_cell_holdout_20260922_v1"),
        "status": "SPLIT_FROZEN_EXPRESSION_QC_PENDING",
        "source_inst_info_sha256": _hash(inst_info),
        "source_pert_info_sha256": _hash(pert_info),
        "selection": ("trt_cp with unique pert_id structure parseable by RDKit; finite positive dose in um, time 6h or 24h, same plate+cell+time vehicle"
                      if require_parseable_structure else
                      "trt_cp with unique pert_id structure, finite positive dose in um, time 6h or 24h, same plate+cell+time vehicle"),
        "n_eligible_instances": int(len(treatment)),
        "n_eligible_cells": len(cells),
        "n_eligible_drugs": int(treatment["pert_id"].nunique()),
        "eligible_instances_per_cell": {
            str(cell): int(n) for cell, n in treatment["cell_id"].value_counts().sort_index().items()
        },
        "split_seed": 20260922,
        "initialization_seeds": [20260922, 20260923, 20260924],
        "folds": folds,
        "main_evaluation_policy": "unseen cell, drugs seen in corresponding outer training cells; repeats and all same-cell vehicles follow cell split",
        "secondary_evaluation_policy": "unseen drugs on held-out cells reported separately, not mixed with main unseen-cell test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--pert-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-parseable-structure", action="store_true")
    args = parser.parse_args()
    result = build(args.inst_info, args.pert_info,
                   require_parseable_structure=args.require_parseable_structure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "n_instances": result["n_eligible_instances"],
                      "n_cells": result["n_eligible_cells"],
                      "n_drugs": result["n_eligible_drugs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
