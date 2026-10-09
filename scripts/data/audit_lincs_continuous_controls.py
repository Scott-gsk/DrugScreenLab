"""只读盘点 LINCS Level 3 化合物实例与真实同板载体对照。

本脚本只检查元数据，不生成处理减对照的连续标签；同板同细胞同时间
匹配是候选条件，仍需核查批次、质量及实际基因轴后才能准入训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


KEY = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]


def audit(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"inst_id", "rna_plate", "pert_type", "pert_id", "cell_id",
                "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing inst_info columns: {sorted(missing)}")
    if frame["inst_id"].duplicated().any():
        raise ValueError("inst_id is not unique")
    for field in KEY:
        if frame[field].eq("").any():
            raise ValueError(f"empty {field} in source metadata")

    treatment = frame.loc[frame["pert_type"].eq("trt_cp")].copy()
    controls = frame.loc[frame["pert_type"].eq("ctl_vehicle")].copy()
    if treatment.empty or controls.empty:
        raise ValueError("trt_cp or ctl_vehicle absent")
    control_counts = controls.groupby(KEY, dropna=False).size().rename("n_controls")
    indexed = treatment.join(control_counts, on=KEY)
    indexed["n_controls"] = indexed["n_controls"].fillna(0).astype(int)
    dose = pd.to_numeric(indexed["pert_dose"], errors="coerce")
    fixed_condition = (
        indexed["pert_time"].eq("24")
        & indexed["pert_time_unit"].eq("h")
        & indexed["pert_dose_unit"].str.lower().eq("um")
        & dose.eq(10)
    )
    fixed_matched = indexed.loc[fixed_condition & indexed["n_controls"].gt(0)]
    grouped = indexed.groupby(KEY, dropna=False).agg(
        n_treatment_instances=("inst_id", "size"),
        n_candidate_controls=("n_controls", "first"),
        n_drugs=("pert_id", "nunique"),
    ).reset_index()
    matched = indexed["n_controls"].gt(0)
    if not matched.any():
        raise ValueError("no treatment instance has a matching control key")
    return {
        "status": "METADATA_ONLY_CANDIDATE_MATCH_NOT_LABEL_ADMISSION",
        "source": str(path.resolve()), "source_sha256": h.hexdigest(),
        "join_key": KEY,
        "n_all_instances": int(len(frame)),
        "n_treatment_instances": int(len(treatment)),
        "n_vehicle_instances": int(len(controls)),
        "n_treatment_with_candidate_control": int(matched.sum()),
        "n_treatment_without_candidate_control": int((~matched).sum()),
        "n_distinct_compounds_with_candidate_control": int(indexed.loc[matched, "pert_id"].nunique()),
        "n_treatment_groups": int(len(grouped)),
        "n_treatment_groups_with_candidate_control": int(grouped["n_candidate_controls"].gt(0).sum()),
        "dose_units": {str(k): int(v) for k, v in treatment["pert_dose_unit"].value_counts().items()},
        "time_units": {str(k): int(v) for k, v in treatment["pert_time_unit"].value_counts().items()},
        "top_times": {str(k): int(v) for k, v in treatment["pert_time"].value_counts().head(12).items()},
        "fixed_24h_10um": {
            "n_treatment_instances": int(fixed_condition.sum()),
            "n_with_candidate_control": int(len(fixed_matched)),
            "n_cells_with_candidate_control": int(fixed_matched["cell_id"].nunique()),
            "n_compounds_with_candidate_control": int(fixed_matched["pert_id"].nunique()),
        },
        "known_missing_requirements": [
            "同板对照表达与处理表达尚未从 Level 3 矩阵逐键提取并核对基因轴",
            "共享对照及技术重复尚未绑定分组划分",
            "同板匹配仍需检查质量、板内处理语义与剂量时间覆盖",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.inst_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output),
                      "matched": result["n_treatment_with_candidate_control"],
                      "unmatched": result["n_treatment_without_candidate_control"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
