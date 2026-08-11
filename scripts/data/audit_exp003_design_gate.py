"""Audit EXP-003 DRT support and independent-control feasibility from metadata only."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from drug_screen.data.exp002 import DEFAULT_SEED, deterministic_split


KEY = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]


def scalar(value: object) -> str:
    return "NA" if pd.isna(value) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "raw" / "lincs" / "GSE92742"
    instances = pd.read_csv(raw / "GSE92742_Broad_LINCS_inst_info.txt.gz", sep="\t", low_memory=False)
    perturbagens = pd.read_csv(raw / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", usecols=["pert_id", "inchi_key"])
    cells = pd.read_csv(raw / "GSE92742_Broad_LINCS_cell_info.txt.gz", sep="\t", usecols=["cell_id", "base_cell_id"])
    treatments = instances.loc[instances.pert_type.eq("trt_cp")].merge(perturbagens, on="pert_id", how="left").merge(cells, on="cell_id", how="left")
    vehicles = instances.loc[instances.pert_type.eq("ctl_vehicle"), KEY + ["inst_id"]]
    vehicle_counts = vehicles.groupby(KEY, dropna=False).size().rename("vehicle_count")
    treatments = treatments.join(vehicle_counts, on=KEY, how="inner")
    treatments["drug_id"] = treatments.inchi_key.map(lambda value: f"INCHIKEY:{value}" if not pd.isna(value) else None).fillna("LINCS_PERT_ID:" + treatments.pert_id.astype(str))
    treatments["context_id"] = treatments.base_cell_id.map(lambda value: f"BASE_CELL:{value}" if not pd.isna(value) else None).fillna("CELL_ID:" + treatments.cell_id.astype(str))
    treatments["split"] = treatments.context_id.map(lambda value: deterministic_split(f"cold_context:{value}", DEFAULT_SEED))
    for column in ("pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"):
        treatments[column] = treatments[column].map(scalar)
    treatments["drt"] = list(zip(treatments.drug_id, treatments.pert_dose, treatments.pert_dose_unit, treatments.pert_time, treatments.pert_time_unit))
    train = treatments.loc[treatments.split.eq("train")]
    test = treatments.loc[treatments.split.eq("test")].copy()
    train_drugs, train_drt = set(train.drug_id), set(train.drt)
    test["drug_seen_train"] = test.drug_id.isin(train_drugs)
    test["drt_seen_train"] = test.drt.isin(train_drt)
    test["stratum"] = "unseen_drug_novel_dose_time"
    test.loc[~test.drug_seen_train & test.drt_seen_train, "stratum"] = "unseen_drug_supported_dose_time"
    test.loc[test.drug_seen_train & ~test.drt_seen_train, "stratum"] = "seen_drug_novel_dose_time"
    test.loc[test.drug_seen_train & test.drt_seen_train, "stratum"] = "primary_supported_drt"
    group_columns = ["drug_id", "context_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
    groups = test.drop_duplicates(group_columns)
    payload = {
        "artifact_type": "exp003_design_gate_metadata_audit",
        "scope": "metadata only; no expression matrix read",
        "source": "GSE92742 inst_info/pert_info/cell_info; EXP-002 cold_context hash split",
        "test_context_count": int(test.context_id.nunique()),
        "test_treatment_count": int(len(test)),
        "test_group_count": int(len(groups)),
        "test_matching_key_count": int(test[KEY].drop_duplicates().shape[0]),
        "test_matching_keys_with_at_least_two_vehicles": int(test.loc[test.vehicle_count.ge(2), KEY].drop_duplicates().shape[0]),
        "test_matching_keys_below_two_vehicles": int(test.loc[test.vehicle_count.lt(2), KEY].drop_duplicates().shape[0]),
        "strata": {name: {"treatments": int((test.stratum == name).sum()), "groups": int((groups.stratum == name).sum())} for name in sorted(Counter(test.stratum))},
        "train_seen_drug_treatment_fraction": float(test.drug_seen_train.mean()),
        "train_seen_drt_treatment_fraction": float(test.drt_seen_train.mean()),
        "required_before_execution": "freeze disjoint proxy/reference control roles and train-fit-only transform; this audit does not create a target or read expression values",
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    payload["payload_sha256"] = sha256(encoded.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
