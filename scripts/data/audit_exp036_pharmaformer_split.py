"""Audit frozen target patient folds and rank denominators before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(input_dir: Path, split_path: Path, fractions: tuple[float, ...] = (0.5,)) -> dict:
    pairs = pd.read_parquet(input_dir / "patient_pairs.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    all_patients = set(pairs["patient_id"])
    results = []
    for fraction_value in fractions:
        if fraction_value not in (0.25, 0.5, 1.0):
            raise ValueError(f"unsupported preregistered target label fraction: {fraction_value}")
        rows = []
        for fold in split["target_folds"]:
            number = fold["fold"]
            train_patients, test_patients = set(fold["outer_train_ids"]), set(fold["outer_test_ids"])
            if train_patients & test_patients or train_patients | test_patients != all_patients:
                raise ValueError(f"fold {number} does not partition all admitted patients")
            fraction = next(entry for entry in fold["label_fractions"]
                            if float(entry["fraction"]) == fraction_value)
            selected_patients = set(fraction["train_ids"])
            if not selected_patients <= train_patients:
                raise ValueError(f"fold {number} training fraction escapes outer training patients")
            selected = pairs.loc[pairs["patient_id"].isin(selected_patients)]
            seen = set(selected["drug_id"])
            test_all = pairs.loc[pairs["patient_id"].isin(test_patients)]
            test = test_all.loc[test_all["drug_id"].isin(seen)]
            per_test = test.groupby("patient_id")["drug_id"].nunique().reindex(sorted(test_patients), fill_value=0)
            per_test_all = test_all.groupby("patient_id")["drug_id"].nunique().reindex(sorted(test_patients), fill_value=0)
            inner_receipts = []
            for inner in fraction["inner_folds"]:
                train_ids, valid_ids = set(inner["train_ids"]), set(inner["validation_ids"])
                if train_ids & valid_ids or train_ids | valid_ids != selected_patients:
                    raise ValueError(f"fold {number} inner split does not partition selected patients")
                inner_train = pairs.loc[pairs["patient_id"].isin(train_ids)]
                inner_seen = set(inner_train["drug_id"])
                valid_all = pairs.loc[pairs["patient_id"].isin(valid_ids)]
                valid = valid_all.loc[valid_all["drug_id"].isin(inner_seen)]
                per_valid = valid.groupby("patient_id")["drug_id"].nunique().reindex(sorted(valid_ids), fill_value=0)
                valid_eligible = int(per_valid.ge(3).sum())
                observed_validation_drugs = set(valid_all["drug_id"])
                inner_receipts.append({"inner_fold": inner["inner_fold"],
                                       "n_train_patients": len(train_ids),
                                       "n_validation_patients": len(valid_ids),
                                       "n_validation_patients_at_least_three_seen_drugs": valid_eligible,
                                       "n_validation_pairs_seen_drug": len(valid),
                                       "n_validation_pairs_full_panel": len(valid_all),
                                       "unseen_validation_drugs": sorted(set(valid_all["drug_id"]) - inner_seen),
                                       "validation_drugs_without_observations": sorted(set(inner_seen) - observed_validation_drugs),
                                       "patients_below_three_seen_drugs": per_valid[per_valid.lt(3)].to_dict(),
                                       "n_train_drugs": len(inner_seen)})
            rows.append({"fold": number, "n_outer_train_patients": len(train_patients),
                         "n_selected_train_patients": len(selected_patients),
                         "n_test_patients": len(test_patients),
                         "n_train_pairs": len(selected), "n_test_pairs_seen_drug": len(test),
                         "n_test_pairs_full_panel": len(test_all),
                         "n_test_patients_at_least_three_full_panel_drugs": int(per_test_all.ge(3).sum()),
                         "n_test_patients_at_least_three_seen_drugs": int(per_test.ge(3).sum()),
                         "unseen_test_drugs": sorted(set(test_all["drug_id"]) - seen),
                         "patients_below_three_seen_drugs": per_test[per_test.lt(3)].to_dict(),
                         "n_train_drugs": len(seen), "inner_folds": inner_receipts})
        gate = all(row["n_test_patients_at_least_three_seen_drugs"] == row["n_test_patients"]
                   and all(inner["n_validation_patients_at_least_three_seen_drugs"]
                           == inner["n_validation_patients"] for inner in row["inner_folds"])
                   for row in rows)
        results.append({"fraction": fraction_value,
                        "status": "PATIENT_RANK_GATE_PASS" if gate else "PATIENT_RANK_GATE_FAIL",
                        "scope": "patient-held-out ranking; eligibility requires at least three target-train-seen drugs per patient",
                        "folds": rows})
    all_pass = all(result["status"] == "PATIENT_RANK_GATE_PASS" for result in results)
    return {"experiment_id": "EXP-036", "status": "TARGET_FRACTIONAL_SPLIT_AUDIT_PASS" if all_pass else "TARGET_FRACTIONAL_SPLIT_AUDIT_REVIEW_REQUIRED",
            "scope": "fixed target patient folds at preregistered 25%, 50%, and 100% label fractions; preserve drug coverage gaps separately from patient-rank eligibility",
            "input_manifest_sha256": sha256(input_dir / "manifest.json"),
            "split_sha256": sha256(split_path), "fractions": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, action="append", dest="fractions",
                        choices=(0.25, 0.5, 1.0),
                        help="fraction to audit; may be repeated (default: 0.5)")
    args = parser.parse_args()
    result = audit(args.input_dir, args.split, tuple(args.fractions or (0.5,)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "TARGET_FRACTIONAL_SPLIT_AUDIT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
