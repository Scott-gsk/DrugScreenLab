"""Independently recalculate EXP-038 measured-versus-predicted reversal rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from .verify_exp039_b5_reversal import (
        _equal_tables,
        negative_cosine,
        reversal_reference,
        sha256,
        top_metrics,
    )
except ImportError:
    from verify_exp039_b5_reversal import (
        _equal_tables,
        negative_cosine,
        reversal_reference,
        sha256,
        top_metrics,
    )


MODELS = ("conditional", "state_ablated")
SEEDS = (20260922, 20260923, 20260924)
BASELINES = ("train_drug_mean", "zero")
DRUG_COLUMNS = (
    "fold", "arm", "seed", "cell_id", "pert_id", "n_conditions",
    "measured_reversal", "predicted_reversal", "measured_negative_cosine",
    "predicted_negative_cosine",
)
CELL_COLUMNS = (
    "fold", "arm", "seed", "cell_id", "spearman", "n_drugs", "reason",
    "negative_cosine_spearman", "n_negative_cosine_drugs", "top_decile_overlap",
    "measured_selection_loss", "top_decile_count",
)


def verify(condition_dir: Path, contract_dir: Path, signature_dir: Path,
           panel_dir: Path, split_path: Path, folds_dir: Path,
           baseline_dir: Path, scoring_dir: Path) -> dict:
    summary_path = scoring_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-038"
            or summary.get("status") != "B2_SCORED_PENDING_INDEPENDENT_REVIEW"
            or summary.get("signature") != signature_dir.name
            or set(summary.get("arms", [])) != set((*MODELS, *BASELINES))
            or summary.get("seeds") != list(SEEDS)):
        raise ValueError("scoring receipt is not a pending EXP-038 result for the fixed arms")
    if (summary["source_sha256"]["panel"] != sha256(panel_dir / "panel_conditions.parquet")
            or summary["source_sha256"]["signature"] != sha256(signature_dir / "manifest.json")
            or summary["source_sha256"]["split"] != sha256(split_path)
            or summary["source_sha256"]["condition_manifest"] != sha256(condition_dir / "manifest.json")
            or summary["source_sha256"]["condition_matrix"] != sha256(condition_dir / "condition_delta.npy")
            or summary["source_sha256"]["gene_axis"] != sha256(contract_dir / "genes.json")
            or summary["source_sha256"]["baseline_summary"] != sha256(baseline_dir / "run_summary.json")):
        raise ValueError("scoring receipt does not match the frozen panel, model inputs, signature, or split")
    for name, digest in summary["files_sha256"].items():
        if sha256(scoring_dir / name) != digest:
            raise ValueError(f"scored artifact changed after its receipt: {name}")

    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    up = set(json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8")))
    down = set(json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8")))
    if (len(genes) != 978 or len(set(genes)) != 978 or len(up) < 20 or len(down) < 20
            or (up | down) - set(genes) or up & down):
        raise ValueError("frozen disease signature does not uniquely map to the 978-gene axis")
    landmark = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    if landmark["pr_gene_id"].duplicated().any():
        raise ValueError("continuous disease signature contains duplicate gene identities")
    effect = landmark.set_index("pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(float)

    panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    baseline_summary = json.loads((baseline_dir / "run_summary.json").read_text(encoding="utf-8"))
    if (baseline_summary.get("experiment_id") != "EXP-037"
            or baseline_summary.get("status") != "BASELINES_COMPLETE_PENDING_REVIEW"
            or baseline_summary["data"]["condition_manifest_sha256"] != sha256(condition_dir / "manifest.json")
            or baseline_summary["data"]["condition_matrix_sha256"] != sha256(condition_dir / "condition_delta.npy")
            or baseline_summary["data"]["split_sha256"] != sha256(split_path)):
        raise ValueError("EXP-037 train-side drug means do not match the frozen inputs")
    if (keys["condition_id"].duplicated().any()
            or delta.shape != (len(keys), len(genes))):
        raise ValueError("measured condition keys and continuous expression matrix are misaligned")
    condition_positions = pd.Index(keys["condition_id"]).get_indexer
    expected_drugs, expected_cells, fold_receipts = [], [], []

    for fold in range(5):
        fold_dir = folds_dir / f"fold_{fold}"
        model_summary_path = fold_dir / "fold_summary.json"
        model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        verification_path = fold_dir / "verification_v2.json"
        if not verification_path.is_file():
            verification_path = fold_dir / "verification_continuation.json"
        if not verification_path.is_file():
            raise ValueError(f"upstream perturbation fold {fold} has no supported verification receipt")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if (model_summary.get("experiment_id") != "EXP-037"
                or model_summary.get("fold") != fold
                or model_summary.get("status") != "FOLD_COMPLETE_PENDING_CROSS_FOLD_REVIEW"
                or model_summary.get("split_sha256") != sha256(split_path)
                or verification.get("status") != "PASS" or verification.get("fold") != fold):
            raise ValueError(f"upstream EXP-037 fold {fold} is incomplete or unverified")
        for name, digest in verification["files_sha256"].items():
            if sha256(fold_dir / name) != digest:
                raise ValueError(f"EXP-037 prediction changed after verification: {name}")
        selected = panel.loc[
            panel["fold"].eq(fold) & panel["panel_status"].eq("ELIGIBLE")
        ].copy().reset_index(drop=True)
        if selected.empty or selected["condition_id"].duplicated().any():
            raise ValueError(f"frozen EXP-038 fold {fold} panel is empty or duplicated")
        observed_positions = condition_positions(selected["condition_id"].to_numpy())
        if (observed_positions < 0).any():
            raise ValueError(f"EXP-038 fold {fold} panel condition is absent from the measured matrix")
        observed = np.asarray(delta[observed_positions], dtype=np.float32)
        groups = selected.groupby(["cell_id", "pert_id"], sort=True).indices
        measured_vectors = np.empty((len(groups), len(genes)), dtype=np.float32)
        group_rows = []
        for index, ((cell, drug), positions) in enumerate(groups.items()):
            positions = np.asarray(positions, dtype=np.int64)
            measured_vectors[index] = observed[positions].mean(axis=0, dtype=np.float64)
            group_rows.append({"cell_id": cell, "pert_id": drug, "n_conditions": len(positions)})
        group_frame = pd.DataFrame(group_rows)
        measured_scores = np.asarray(
            [reversal_reference(row, genes, up, down) for row in measured_vectors], dtype=np.float64)
        measured_cosines = negative_cosine(measured_vectors, effect)

        test_ids = np.load(fold_dir / "test_condition_ids.npy")
        prediction_positions = pd.Index(test_ids).get_indexer(selected["condition_id"])
        if (prediction_positions < 0).any():
            raise ValueError(f"EXP-038 fold {fold} panel condition is absent from EXP-037 predictions")

        drug_ids_path = baseline_dir / f"fold_{fold}" / "drug_ids.json"
        drug_mean_path = baseline_dir / f"fold_{fold}" / "drug_mean.npy"
        drug_ids = json.loads(drug_ids_path.read_text(encoding="utf-8"))
        drug_mean = np.load(drug_mean_path, mmap_mode="r")
        drug_index = {drug: index for index, drug in enumerate(drug_ids)}
        if (len(drug_index) != len(drug_ids) or drug_mean.shape != (len(drug_ids), len(genes))
                or not set(group_frame["pert_id"]) <= set(drug_index)):
            raise ValueError(f"EXP-038 fold {fold} training-side drug mean is missing panel drugs")
        training_cells = set(json.loads(split_path.read_text(encoding="utf-8"))["folds"][fold]["outer_train_ids"])
        if set(selected["cell_id"]) & training_cells:
            raise ValueError(f"EXP-038 fold {fold} contains a training cell in its measured test panel")
        training_index = np.flatnonzero(keys["cell_id"].isin(training_cells).to_numpy())
        train_frame = keys.iloc[training_index].reset_index(drop=True)
        if not np.array_equal(np.asarray(drug_ids), np.sort(train_frame["pert_id"].unique())):
            raise ValueError(f"EXP-038 fold {fold} baseline drug axis differs from outer training cells")
        expected_means = np.empty_like(drug_mean)
        grouped_train = train_frame.groupby("pert_id", sort=True).indices
        train_values = np.asarray(delta[training_index], dtype=np.float32)
        for index, positions in enumerate(grouped_train.values()):
            expected_means[index] = train_values[np.asarray(positions, dtype=np.int64)].mean(
                axis=0, dtype=np.float64)
        baseline_error = float(np.max(np.abs(expected_means.astype(np.float64)
                                             - np.asarray(drug_mean, dtype=np.float64))))
        if baseline_error > 1e-7:
            raise ValueError(f"EXP-038 fold {fold} drug means are not an outer-training fit")
        fold_rows = next(row for row in summary["folds"] if int(row["fold"]) == fold)
        if (fold_rows["upstream_fold_summary_sha256"] != sha256(model_summary_path)
                or fold_rows.get("upstream_verification_file") != verification_path.name
                or fold_rows["upstream_verification_sha256"] != sha256(verification_path)
                or fold_rows["baseline_drug_ids_sha256"] != sha256(drug_ids_path)
                or fold_rows["baseline_drug_mean_sha256"] != sha256(drug_mean_path)):
            raise ValueError(f"EXP-038 scoring receipt references a different EXP-037 fold {fold}")

        for arm in (*MODELS, *BASELINES):
            seeds = SEEDS if arm in MODELS else (0,)
            for seed in seeds:
                if arm in MODELS:
                    prediction = np.load(
                        fold_dir / f"predicted_delta_{arm}_seed{seed}.npy", mmap_mode="r")
                    if prediction.shape != (len(test_ids), len(genes)):
                        raise ValueError(f"EXP-037 {arm} prediction shape is invalid")
                    vectors = np.empty_like(measured_vectors)
                    for index, positions in enumerate(groups.values()):
                        positions = np.asarray(positions, dtype=np.int64)
                        vectors[index] = np.asarray(
                            prediction[prediction_positions[positions]], dtype=np.float32
                        ).mean(axis=0, dtype=np.float64)
                elif arm == "zero":
                    vectors = np.zeros_like(measured_vectors)
                else:
                    vectors = np.stack([drug_mean[drug_index[drug]]
                                        for drug in group_frame["pert_id"]]).astype(np.float32)
                predicted_scores = np.asarray(
                    [reversal_reference(row, genes, up, down) for row in vectors], dtype=np.float64)
                predicted_cosines = negative_cosine(vectors, effect)
                for index, row in group_frame.iterrows():
                    expected_drugs.append({
                        "fold": fold, "arm": arm, "seed": seed,
                        **row.to_dict(), "measured_reversal": measured_scores[index],
                        "predicted_reversal": predicted_scores[index],
                        "measured_negative_cosine": measured_cosines[index],
                        "predicted_negative_cosine": predicted_cosines[index],
                    })
                frame = group_frame.copy()
                frame["measured_reversal"] = measured_scores
                frame["predicted_reversal"] = predicted_scores
                frame["measured_negative_cosine"] = measured_cosines
                frame["predicted_negative_cosine"] = predicted_cosines
                for cell, cell_frame in frame.groupby("cell_id", sort=True):
                    y = cell_frame["measured_reversal"].to_numpy(float)
                    yhat = cell_frame["predicted_reversal"].to_numpy(float)
                    reason = None
                    if len(cell_frame) < 5:
                        reason = "FEWER_THAN_FIVE_DRUGS"
                    elif np.ptp(y) == 0:
                        reason = "CONSTANT_MEASURED_REVERSAL"
                    elif np.ptp(yhat) == 0:
                        reason = "CONSTANT_PREDICTED_REVERSAL"
                    rho = float(spearmanr(y, yhat).statistic) if reason is None else float("nan")
                    if reason is None and not np.isfinite(rho):
                        reason = "UNDEFINED_CORRELATION"
                    cosine_mask = cell_frame[
                        ["measured_negative_cosine", "predicted_negative_cosine"]
                    ].notna().all(axis=1)
                    cosine_frame = cell_frame.loc[cosine_mask]
                    cosine_rho = (
                        float(spearmanr(cosine_frame["measured_negative_cosine"],
                                        cosine_frame["predicted_negative_cosine"]).statistic)
                        if len(cosine_frame) >= 5 else float("nan"))
                    overlap, loss, count = top_metrics(
                        cell_frame["pert_id"].to_numpy(), y, yhat)
                    expected_cells.append({
                        "fold": fold, "arm": arm, "seed": seed,
                        "cell_id": cell, "spearman": rho, "n_drugs": len(cell_frame),
                        "reason": reason, "negative_cosine_spearman": cosine_rho,
                        "n_negative_cosine_drugs": len(cosine_frame),
                        "top_decile_overlap": overlap, "measured_selection_loss": loss,
                        "top_decile_count": count,
                    })
        fold_receipts.append({"fold": fold, "n_conditions": len(selected),
                              "n_cells": int(selected["cell_id"].nunique()),
                              "upstream_fold_summary_sha256": sha256(model_summary_path),
                              "upstream_verification_sha256": sha256(verification_path)})

    actual_drugs = pd.read_parquet(scoring_dir / "drug_scores.parquet")
    actual_cells = pd.read_parquet(scoring_dir / "cell_scores.parquet")
    max_error = max(
        _equal_tables(actual_drugs, pd.DataFrame(expected_drugs), DRUG_COLUMNS,
                      ("fold", "arm", "seed", "cell_id", "pert_id")),
        _equal_tables(actual_cells, pd.DataFrame(expected_cells), CELL_COLUMNS,
                      ("fold", "arm", "seed", "cell_id")),
    )
    return {
        "status": "PASS",
        "experiment_id": "EXP-038",
        "signature": signature_dir.name,
        "run_summary_sha256": sha256(summary_path),
        "verified_drug_score_rows": len(expected_drugs),
        "verified_cell_score_rows": len(expected_cells),
        "max_abs_recalculation_error": max_error,
        "reversal_implementation": "independent prefix-walk implementation with tied deltas grouped",
        "folds": fold_receipts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.condition_dir, args.contract_dir, args.signature_dir,
                    args.panel_dir, args.split, args.folds_dir, args.baseline_dir,
                    args.scoring_dir)
    output = args.scoring_dir / "independent_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
