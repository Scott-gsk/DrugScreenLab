"""EXP-013 minimum closed loop: B0 vs B1 on an accessible PDO cohort.

This runner does not train neural models and does not use test labels for
feature or hyperparameter selection. Hyperparameters are frozen in the config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.data.root import data_root
from drug_screen.transfer.baselines import fit_b0_drug_means, fit_b1_per_drug_ridge, predict_b0, predict_b1
from drug_screen.transfer.bundles import load_indexed_bundle, sample_index_map
from drug_screen.transfer.contract import TransferObservation, assert_higher_is_sensitive, eligibility_report
from drug_screen.transfer.metrics import evaluate_predictions
from drug_screen.transfer.splits import assign_patient_folds, assert_patient_isolation, split_for_fold


def _inchikey_means(pairs, drugs):
    merged = pairs.merge(drugs[["drug_id", "inchikey"]], on="drug_id", how="left")
    grouped = merged.groupby("inchikey")["response"].mean()
    return grouped.to_dict()


def _fit_affine(source_scores, observed):
    pairs = [
        (float(score), float(y))
        for score, y in zip(source_scores, observed)
        if score is not None
    ]
    if len(pairs) < 8:
        return 0.0, 0.0, 0
    xs = [row[0] for row in pairs]
    ys = [row[1] for row in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x == 0:
        return 0.0, y_mean, len(pairs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / var_x
    intercept = y_mean - slope * x_mean
    return slope, intercept, len(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="mcrc_pdo")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--n-hvg", type=int, default=512)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-drugs-per-patient", type=int, default=5)
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path; default data/artifacts/EXP-013/<cohort>_b0_b1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_indexed_bundle(args.cohort)
    index_map = sample_index_map(bundle)
    pairs = bundle.pairs.merge(
        bundle.samples[["sample_id", "patient_id", "pdo_id"]],
        on="sample_id",
        how="left",
    )
    if pairs["patient_id"].isna().any():
        raise ValueError("pairs contain sample_id values missing from the sample table")
    observations = [
        TransferObservation(
            patient_id=str(row.patient_id),
            model_id=str(row.pdo_id or row.sample_id),
            study_id=args.cohort,
            domain="pdo",
            cancer_type="unknown_pending_target_audit",
            drug_id=str(row.drug_id),
            response=float(row.response),
            response_name=str(row.response_name),
            higher_is_sensitive=bool(row.higher_is_sensitive),
            split_id="unassigned",
        )
        for row in pairs.itertuples(index=False)
    ]
    print(f"loaded {args.cohort} n_pairs={len(pairs)} n_patients={pairs['patient_id'].nunique()}", flush=True)
    source_means = {}
    try:
        source = load_indexed_bundle("gdsc2_cell_line")
        source_means = _inchikey_means(source.pairs, source.drugs)
        print(f"loaded gdsc2 source means n_inchikey={len(source_means)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"source GDSC2 not used for B2: {exc}", flush=True)
    target_inchikey = dict(zip(bundle.drugs["drug_id"].astype(str), bundle.drugs["inchikey"].astype(str)))
    assert_higher_is_sensitive(observations)
    eligibility = eligibility_report(observations, source_drug_ids=list(source_means))
    assignment = assign_patient_folds(
        [row.patient_id for row in observations],
        n_folds=args.n_folds,
        seed=args.seed,
    )

    fold_summaries = []
    pooled_patient: list[str] = []
    pooled_drug: list[str] = []
    pooled_y: list[float] = []
    pooled_b1: list[float] = []
    pooled_b0: list[float] = []
    pooled_b2: list[float] = []

    for fold in range(args.n_folds):
        split_by_patient = {
            patient: split_for_fold(assignment, patient, fold)
            for patient in assignment.patient_to_test_fold
        }
        tagged = [
            TransferObservation(
                patient_id=row.patient_id,
                model_id=row.model_id,
                study_id=row.study_id,
                domain=row.domain,
                cancer_type=row.cancer_type,
                drug_id=row.drug_id,
                response=row.response,
                response_name=row.response_name,
                higher_is_sensitive=row.higher_is_sensitive,
                split_id=split_by_patient[row.patient_id],
            )
            for row in observations
        ]
        assert_patient_isolation(tagged, split_by_patient)

        train_mask = pairs["patient_id"].map(split_by_patient).eq("train")
        test_mask = pairs["patient_id"].map(split_by_patient).eq("test")
        train = pairs.loc[train_mask]
        test = pairs.loc[test_mask]
        print(f"fold {fold} train={int(train['patient_id'].nunique())} test={int(test['patient_id'].nunique())}", flush=True)

        b0 = fit_b0_drug_means(train["drug_id"].tolist(), train["response"].tolist())
        train_indices = [index_map[sample_id] for sample_id in train["sample_id"]]
        test_indices = [index_map[sample_id] for sample_id in test["sample_id"]]
        b1 = fit_b1_per_drug_ridge(
            bundle.expression,
            train_indices,
            train["drug_id"].tolist(),
            train["response"].tolist(),
            n_hvg=args.n_hvg,
            alpha=args.ridge_alpha,
        )
        b0_hat = predict_b0(b0, test["drug_id"].tolist())
        b1_hat = predict_b1(b1, bundle.expression, test_indices, test["drug_id"].tolist())
        train_source = [
            source_means.get(target_inchikey.get(str(drug)))
            for drug in train["drug_id"].astype(str)
        ]
        slope, intercept, n_cal = _fit_affine(train_source, train["response"].tolist())
        b2_hat = []
        for drug, mean_pred in zip(test["drug_id"].astype(str), b0_hat):
            score = source_means.get(target_inchikey.get(str(drug)))
            if score is None or n_cal == 0:
                b2_hat.append(mean_pred)
            else:
                b2_hat.append(intercept + slope * score)
        fold_eval = evaluate_predictions(
            test["patient_id"].astype(str).tolist(),
            test["drug_id"].astype(str).tolist(),
            test["response"].astype(float).tolist(),
            b1_hat,
            b0_hat,
            min_drugs_per_patient=args.min_drugs_per_patient,
            seed=args.seed + fold,
        )
        fold_summaries.append(
            {
                "fold": fold,
                "n_train_patients": int(train["patient_id"].nunique()),
                "n_test_patients": int(test["patient_id"].nunique()),
                "n_b1_drugs": len(b1.models),
                "metrics": {
                    key: fold_eval[key]
                    for key in (
                        "n_scored_patients",
                        "n_scored_drugs",
                        "within_patient_spearman",
                        "within_patient_baseline_spearman",
                        "within_patient_ndcg",
                        "per_drug_spearman",
                        "per_drug_baseline_spearman",
                        "within_patient_delta",
                        "per_drug_delta",
                        "within_patient_delta_ci",
                        "per_drug_delta_ci",
                        "verdict",
                    )
                },
            }
        )
        pooled_patient.extend(test["patient_id"].astype(str).tolist())
        pooled_drug.extend(test["drug_id"].astype(str).tolist())
        pooled_y.extend(test["response"].astype(float).tolist())
        pooled_b1.extend(b1_hat)
        pooled_b0.extend(b0_hat)
        pooled_b2.extend(b2_hat)

    overall = evaluate_predictions(
        pooled_patient,
        pooled_drug,
        pooled_y,
        pooled_b1,
        pooled_b0,
        min_drugs_per_patient=args.min_drugs_per_patient,
        seed=args.seed,
    )
    b2_overall = evaluate_predictions(
        pooled_patient,
        pooled_drug,
        pooled_y,
        pooled_b2,
        pooled_b0,
        min_drugs_per_patient=args.min_drugs_per_patient,
        seed=args.seed + 17,
    )
    payload = {
        "experiment_id": "EXP-013",
        "status": "BASELINE_LOOP_COMPLETE",
        "cohort_id": args.cohort,
        "role": bundle.role,
        "expression_unit": bundle.expression_unit,
        "response_name": bundle.response_name,
        "higher_is_sensitive": bundle.higher_is_sensitive,
        "hyperparameters_frozen": {
            "n_hvg": args.n_hvg,
            "ridge_alpha": args.ridge_alpha,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "inner_cv": False,
            "note": "No inner search; hyperparameters frozen before looking at test labels",
        },
        "eligibility": dict(eligibility),
        "folds": fold_summaries,
        "overall": {
            key: overall[key]
            for key in (
                "n_scored_patients",
                "n_scored_drugs",
                "within_patient_spearman",
                "within_patient_baseline_spearman",
                "within_patient_ndcg",
                "per_drug_spearman",
                "per_drug_baseline_spearman",
                "within_patient_delta",
                "per_drug_delta",
                "within_patient_delta_ci",
                "per_drug_delta_ci",
                "verdict",
            )
        },
        "b2_overall": {
            key: b2_overall[key]
            for key in (
                "n_scored_patients",
                "n_scored_drugs",
                "within_patient_spearman",
                "within_patient_baseline_spearman",
                "within_patient_ndcg",
                "per_drug_spearman",
                "per_drug_baseline_spearman",
                "within_patient_delta",
                "per_drug_delta",
                "within_patient_delta_ci",
                "per_drug_delta_ci",
                "verdict",
            )
        },
        "limitations": [
            "Preferred pancreatic PDO RNA (HRA010277) remains controlled-access; only Supplementary Data 10 AUC labels are local.",
            "Official NCI CTRPv2 text/zip redirected to studycatalog; Zenodo PharmacoSet RDS is registered but not yet table-materialized. B2 used GDSC2 1-AUC means by InChIKey.",
            "GSE114845 is not Tiriac 2018; that accession is an unrelated mouse sleep study and was not registered.",
            "This is not a clinical efficacy claim.",
        ],
    }
    output = (
        Path(args.output)
        if args.output
        else data_root() / "artifacts" / "EXP-013" / f"{args.cohort}_b0_b1.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(output), "verdict": overall["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
