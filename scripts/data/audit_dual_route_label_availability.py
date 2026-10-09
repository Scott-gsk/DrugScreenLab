"""只按样本-药物键盘点新方案共同药面板与各训练折可用性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(split_path: Path, target_samples: Path, target_pairs: Path,
          target_drugs: Path, source_drugs: Path) -> dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    samples = pd.read_parquet(target_samples, columns=["sample_id", "patient_id"])
    pairs = pd.read_parquet(target_pairs, columns=["sample_id", "drug_id"])
    source_drug_ids = set(pd.read_parquet(source_drugs, columns=["drug_id"])["drug_id"].astype(str))
    target_drug_ids = set(pd.read_parquet(target_drugs, columns=["drug_id"])["drug_id"].astype(str))
    common = sorted(source_drug_ids & target_drug_ids)
    if len(common) < 3:
        raise ValueError("fewer than 3 structure-identical source/target drugs")
    sample_patient = dict(zip(samples["sample_id"].astype(str), samples["patient_id"].astype(str)))
    admissible_samples = {
        sample for fold in split["target_folds"] for sample in fold["outer_test_samples"]
    }
    if len(admissible_samples) != split["n_target_samples"]:
        raise ValueError("split does not cover the stated target sample count")
    pairs = pairs.loc[
        pairs["sample_id"].astype(str).isin(admissible_samples)
        & pairs["drug_id"].astype(str).isin(common)
    ].drop_duplicates(["sample_id", "drug_id"])
    pairs["patient_id"] = pairs["sample_id"].astype(str).map(sample_patient)
    if pairs["patient_id"].isna().any():
        raise ValueError("target pair lacks a sample-to-patient mapping")
    patient_drug = pairs[["patient_id", "drug_id"]].drop_duplicates()
    rows = []
    for fold in split["target_folds"]:
        test_patients = set(fold["outer_test_ids"])
        test = patient_drug.loc[patient_drug["patient_id"].isin(test_patients)]
        test_counts = test.groupby("patient_id")["drug_id"].nunique().reindex(
            sorted(test_patients), fill_value=0)
        levels = []
        for level in fold["label_fractions"]:
            train = patient_drug.loc[patient_drug["patient_id"].isin(level["train_ids"])]
            drug_counts = train.groupby("drug_id")["patient_id"].nunique().reindex(common, fill_value=0)
            inner = []
            for inner_fold in level["inner_folds"]:
                itr = patient_drug.loc[patient_drug["patient_id"].isin(inner_fold["train_ids"])]
                iva = patient_drug.loc[patient_drug["patient_id"].isin(inner_fold["validation_ids"])]
                count_tr = itr.groupby("drug_id")["patient_id"].nunique().reindex(common, fill_value=0)
                count_va = iva.groupby("drug_id")["patient_id"].nunique().reindex(common, fill_value=0)
                inner.append({
                    "inner_fold": inner_fold["inner_fold"],
                    "drugs_missing_in_train": [d for d in common if count_tr[d] == 0],
                    "drugs_missing_in_validation": [d for d in common if count_va[d] == 0],
                    "min_patients_per_drug_train": int(count_tr.min()),
                    "min_patients_per_drug_validation": int(count_va.min()),
                })
            levels.append({
                "fraction": level["fraction"],
                "n_train_patients": len(level["train_ids"]),
                "drugs_missing_in_train": [d for d in common if drug_counts[d] == 0],
                "min_patients_per_drug_train": int(drug_counts.min()),
                "inner": inner,
            })
        rows.append({
            "fold": fold["fold"],
            "n_test_patients": len(test_patients),
            "n_test_patients_with_at_least_3_common_drugs": int(test_counts.ge(3).sum()),
            "test_drug_count_min": int(test_counts.min()),
            "test_drug_count_max": int(test_counts.max()),
            "fractions": levels,
        })
    return {
        "status": "IDENTITY_AND_LABEL_AVAILABILITY_ONLY_NO_RESPONSE_VALUES_READ",
        "split_sha256": _hash(split_path),
        "target_pairs_sha256": _hash(target_pairs),
        "source_drugs_sha256": _hash(source_drugs),
        "target_drugs_sha256": _hash(target_drugs),
        "n_common_drugs": len(common), "common_drug_ids": common,
        "n_unique_admitted_sample_drug_rows": len(pairs),
        "folds": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--target-samples", type=Path, required=True)
    parser.add_argument("--target-pairs", type=Path, required=True)
    parser.add_argument("--target-drugs", type=Path, required=True)
    parser.add_argument("--source-drugs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.splits, args.target_samples, args.target_pairs,
                   args.target_drugs, args.source_drugs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"n_common_drugs": result["n_common_drugs"],
                      "n_rows": result["n_unique_admitted_sample_drug_rows"],
                      "n_evaluable_test_patients": sum(
                          row["n_test_patients_with_at_least_3_common_drugs"]
                          for row in result["folds"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
