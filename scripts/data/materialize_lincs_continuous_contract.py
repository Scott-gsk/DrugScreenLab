"""物化 EXP-037 的逐键处理/共享对照合同，并扫描完整连续表达质量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.data.freeze_lincs_cell_splits import KEY, select_eligible_instances


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_rows(cache: np.ndarray, positions: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Inspect every selected source row; no imputation or value-based filtering."""
    output = np.empty(len(positions), dtype=bool)
    for begin in range(0, len(positions), batch_size):
        end = min(begin + batch_size, len(positions))
        output[begin:end] = np.isfinite(cache[positions[begin:end], :]).all(axis=1)
    return output


def materialize(inst_info: Path, pert_info: Path, gene_info: Path, gctx: Path,
                cache_path: Path, split_path: Path, output_dir: Path,
                require_parseable_structure: bool = False) -> dict:
    instances = pd.read_csv(inst_info, sep="\t", dtype=str, keep_default_na=False)
    compounds = pd.read_csv(pert_info, sep="\t", dtype=str, keep_default_na=False)
    if instances["inst_id"].duplicated().any():
        raise ValueError("inst_info contains duplicate inst_id")
    treatments = select_eligible_instances(
        instances, compounds,
        require_parseable_structure=require_parseable_structure).copy()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (split["source_inst_info_sha256"] != sha256(inst_info)
            or split["source_pert_info_sha256"] != sha256(pert_info)
            or split["n_eligible_instances"] != len(treatments)
            or split["n_eligible_cells"] != treatments["cell_id"].nunique()):
        raise ValueError("frozen split metadata cohort disagrees with current sources")
    control_keys = set(map(tuple, treatments[KEY].itertuples(index=False, name=None)))
    controls = instances.loc[instances["pert_type"].eq("ctl_vehicle")].copy()
    controls = controls.loc[[tuple(row) in control_keys for row in
                             controls[KEY].itertuples(index=False, name=None)]].copy()
    if controls.empty:
        raise ValueError("no matched vehicle controls")

    with h5py.File(gctx, "r") as source:
        instance_ids = [value.decode() for value in source["0/META/COL/id"][:]]
        source_genes = [value.decode() for value in source["0/META/ROW/id"][:]]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("duplicate source matrix instance ID")
    row_by_id = {inst_id: row for row, inst_id in enumerate(instance_ids)}
    if len(row_by_id) != len(instances) or set(row_by_id) != set(instances["inst_id"]):
        raise ValueError("source matrix/metadata instance ID sets disagree")
    genes = pd.read_csv(gene_info, sep="\t", dtype={"pr_gene_id": str})
    landmark = set(genes.loc[genes["pr_is_lm"].eq(1), "pr_gene_id"])
    gene_axis = [gene for gene in source_genes if gene in landmark]
    if len(gene_axis) != 978 or len(set(gene_axis)) != 978:
        raise ValueError("landmark gene axis is not 978 unique IDs")
    cache = np.load(cache_path, mmap_mode="r")
    if cache.shape != (len(instance_ids), len(gene_axis)):
        raise ValueError("cache shape disagrees with source axes")
    treatments["cache_row"] = treatments["inst_id"].map(row_by_id)
    controls["cache_row"] = controls["inst_id"].map(row_by_id)
    if treatments["cache_row"].isna().any() or controls["cache_row"].isna().any():
        raise ValueError("metadata instance missing from cache axis")

    # The same plate/cell/time control group can serve several drug treatments.
    group_keys = sorted(control_keys)
    group_by_key = {key: index for index, key in enumerate(group_keys)}
    treatments["control_group_id"] = [group_by_key[tuple(row)] for row in
                                       treatments[KEY].itertuples(index=False, name=None)]
    controls["control_group_id"] = [group_by_key[tuple(row)] for row in
                                     controls[KEY].itertuples(index=False, name=None)]
    if set(treatments["control_group_id"]) != set(controls["control_group_id"]):
        raise ValueError("treatment/control group mismatch")
    treatments["treatment_finite"] = finite_rows(cache, treatments["cache_row"].to_numpy(dtype=np.int64))
    controls["control_finite"] = finite_rows(cache, controls["cache_row"].to_numpy(dtype=np.int64))
    control_ok = controls.groupby("control_group_id", sort=False)["control_finite"].all()
    control_n = controls.groupby("control_group_id", sort=False)["inst_id"].size()
    treatments["all_controls_finite"] = treatments["control_group_id"].map(control_ok)
    treatments["n_matched_controls"] = treatments["control_group_id"].map(control_n)
    treatments["quality_status"] = np.where(
        treatments["treatment_finite"] & treatments["all_controls_finite"],
        "PASS", "NONFINITE_EXPRESSION")
    if treatments["n_matched_controls"].isna().any():
        raise ValueError("treatment lacks matched controls")

    treatment_columns = ["inst_id", "cache_row", "control_group_id", "cell_id", "pert_id",
                         "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
                         "rna_plate", "treatment_finite", "all_controls_finite",
                         "n_matched_controls", "quality_status"]
    control_columns = ["inst_id", "cache_row", "control_group_id", *KEY, "control_finite"]
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    treatment_file = output_dir / "treatments.parquet"
    control_file = output_dir / "controls.parquet"
    genes_file = output_dir / "genes.json"
    treatments[treatment_columns].to_parquet(treatment_file, index=False)
    controls[control_columns].to_parquet(control_file, index=False)
    genes_file.write_text(json.dumps(gene_axis, ensure_ascii=False) + "\n", encoding="utf-8")
    # A readback checks that files preserve the keys, counts and group relation.
    saved_treatment = pd.read_parquet(treatment_file)
    saved_control = pd.read_parquet(control_file)
    if (len(saved_treatment) != len(treatments) or len(saved_control) != len(controls)
            or not saved_treatment["inst_id"].equals(treatments["inst_id"].reset_index(drop=True))
            or not saved_control["inst_id"].equals(controls["inst_id"].reset_index(drop=True))):
        raise ValueError("parquet readback changed instance identities")
    if set(saved_treatment["control_group_id"]) != set(saved_control["control_group_id"]):
        raise ValueError("parquet readback changed control groups")
    per_cell = treatments.groupby("cell_id")["quality_status"].agg(
        total="size", passed=lambda x: int(x.eq("PASS").sum()))
    per_drug = treatments.groupby("pert_id")["quality_status"].agg(
        total="size", passed=lambda x: int(x.eq("PASS").sum()))
    fold_receipts = []
    for fold in split["folds"]:
        train = treatments.loc[treatments["cell_id"].isin(fold["outer_train_ids"])]
        test = treatments.loc[treatments["cell_id"].isin(fold["outer_test_ids"])]
        train_pass = train.loc[train["quality_status"].eq("PASS")]
        test_pass = test.loc[test["quality_status"].eq("PASS")]
        seen = set(train_pass["pert_id"])
        fold_receipts.append({
            "fold": fold.get("fold", len(fold_receipts)),
            "train_pass": int(len(train_pass)), "test_pass": int(len(test_pass)),
            "test_seen_drug_pass": int(test_pass["pert_id"].isin(seen).sum()),
            "train_cells_with_pass": int(train_pass["cell_id"].nunique()),
            "test_cells_with_pass": int(test_pass["cell_id"].nunique()),
        })
    report = {
        "protocol": "EXP-037 LINCS continuous matched-vehicle contract v1",
        "status": "QC_COMPLETE_SPLIT_EVALUABILITY_PENDING_REVIEW",
        "split_sha256": sha256(split_path),
        "source_sha256": {"inst_info": sha256(inst_info), "pert_info": sha256(pert_info),
                          "gene_info": sha256(gene_info), "cache": sha256(cache_path)},
        "counts": {"candidate_treatments": int(len(treatments)),
                   "matched_control_instances": int(len(controls)),
                   "shared_control_groups": int(len(group_keys)),
                   "pass_treatments": int(treatments["quality_status"].eq("PASS").sum()),
                   "nonfinite_treatment_instances": int((~treatments["treatment_finite"]).sum()),
                   "nonfinite_control_instances": int((~controls["control_finite"]).sum()),
                   "excluded_due_to_control": int((~treatments["all_controls_finite"]).sum()),
                   "excluded_unique_treatments": int(treatments["quality_status"].ne("PASS").sum())},
        "gene_axis": {"n": len(gene_axis), "sha256": sha256(genes_file)},
        "per_cell": per_cell.to_dict(orient="index"),
        "per_drug": per_drug.to_dict(orient="index"),
        "folds": fold_receipts,
        "files_sha256": {name: sha256(path) for name, path in
                         [("treatments.parquet", treatment_file), ("controls.parquet", control_file),
                          ("genes.json", genes_file)]},
        "qc_rule": "exclude treatment if its 978 values or any matched vehicle's 978 values are nonfinite",
    }
    report_path = output_dir / "quality_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("inst-info", "pert-info", "gene-info", "gctx", "cache", "split", "output-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--require-parseable-structure", action="store_true")
    args = parser.parse_args()
    report = materialize(args.inst_info, args.pert_info, args.gene_info, args.gctx,
                         args.cache, args.split, args.output_dir,
                         args.require_parseable_structure)
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "folds": report["folds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
