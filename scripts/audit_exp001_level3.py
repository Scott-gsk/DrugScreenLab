"""Read-only reproducibility audit for EXP-001's GSE92742 Level-3 landmark core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd

from drug_screen.data.p0 import assert_level3_landmark_core, count_same_plate_vehicle_candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    base = args.root / "raw" / "lincs" / "GSE92742"
    level3 = args.root / "interim" / "lincs" / "GSE92742" / "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
    genes = pd.read_csv(base / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")
    instances = pd.read_csv(
        base / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False
    )
    with h5py.File(level3, "r") as handle:
        shape = tuple(handle["0/DATA/0/matrix"].shape)
        columns = [value.decode("ascii") for value in handle["0/META/COL/id"][:]]
    assert_level3_landmark_core(
        shape,
        genes["pr_gene_id"].astype(str).tolist(),
        genes["pr_is_lm"].astype(int).tolist(),
        columns,
        instances["inst_id"].astype(str).tolist(),
    )
    key_columns = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]
    chemical = instances.loc[instances["pert_type"].eq("trt_cp"), key_columns]
    vehicle = instances.loc[instances["pert_type"].eq("ctl_vehicle"), key_columns]
    total, matched = count_same_plate_vehicle_candidates(
        chemical.itertuples(index=False, name=None), vehicle.itertuples(index=False, name=None)
    )
    print(json.dumps({
        "matrix_shape": shape,
        "landmark_gene_count": int(genes["pr_is_lm"].sum()),
        "chemical_instances": total,
        "same_plate_vehicle_candidates": matched,
        "excluded_without_candidate": total - matched,
        "delta_matrix_materialized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
