"""Independently recalculate EXP-039 fixed-panel reversal rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ARMS = ("huber", "b5_rank")
SEEDS = (20260922, 20260923, 20260924)
DRUG_COLUMNS = (
    "fold", "signature", "arm", "seed", "cell_id", "pert_id", "n_conditions",
    "measured_reversal", "predicted_reversal", "measured_negative_cosine",
    "predicted_negative_cosine",
)
CELL_COLUMNS = (
    "fold", "signature", "arm", "seed", "cell_id", "spearman", "n_drugs",
    "reason", "negative_cosine_spearman", "n_negative_cosine_drugs",
    "top_decile_overlap", "measured_selection_loss", "top_decile_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _es(values: np.ndarray, genes: list[str], members: set[str]) -> float:
    membership = np.fromiter((gene in members for gene in genes), dtype=bool)
    hits, n = int(membership.sum()), len(values)
    if not 0 < hits < n:
        raise ValueError("signature direction must be non-empty and smaller than the gene axis")
    order = np.argsort(-values, kind="stable")
    ordered_values, ordered_hits = values[order], membership[order]
    cumulative_hits = np.cumsum(ordered_hits, dtype=np.int64)
    ends = np.r_[np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1, n]
    seen_hits = cumulative_hits[ends - 1]
    walk = seen_hits / hits - (ends - seen_hits) / (n - hits)
    high, low = float(walk.max()), float(walk.min())
    if abs(high + low) <= 1e-12:
        return 0.0
    return high if high > -low else low


def reversal_reference(delta: np.ndarray, genes: list[str],
                       up: set[str], down: set[str]) -> float:
    values = np.asarray(delta, dtype=np.float64)
    if values.shape != (len(genes),) or not np.isfinite(values).all():
        raise ValueError("reversal vector is not finite or aligned to the gene axis")
    if np.ptp(values) == 0:
        return 0.0
    return float((_es(values, genes, down) - _es(values, genes, up)) / 2.0)


def negative_cosine(vectors: np.ndarray, effect: np.ndarray) -> np.ndarray:
    mapped = np.isfinite(effect)
    if mapped.sum() < 20:
        raise ValueError("continuous signature maps fewer than 20 genes")
    matrix = np.asarray(vectors[:, mapped], dtype=np.float64)
    reference = effect[mapped]
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(reference)
    output = np.full(len(matrix), np.nan, dtype=np.float64)
    valid = denominator > 0
    output[valid] = -(matrix[valid] @ reference) / denominator[valid]
    return output


def top_metrics(drugs: np.ndarray, measured: np.ndarray,
                predicted: np.ndarray) -> tuple[float, float, int]:
    count = max(1, int(np.ceil(len(drugs) * 0.1)))
    measured_top = np.lexsort((drugs, -measured))[:count]
    predicted_top = np.lexsort((drugs, -predicted))[:count]
    overlap = len(set(measured_top) & set(predicted_top)) / count
    loss = float(measured[measured_top].mean() - measured[predicted_top].mean())
    return overlap, loss, count


def _equal_tables(actual: pd.DataFrame, expected: pd.DataFrame,
                  columns: tuple[str, ...], key_columns: tuple[str, ...]) -> float:
    if len(actual) != len(expected) or set(actual.columns) != set(columns):
        raise ValueError("scored table has an unexpected schema or row count")
    left = actual.loc[:, columns].sort_values(list(key_columns)).reset_index(drop=True)
    right = expected.loc[:, columns].sort_values(list(key_columns)).reset_index(drop=True)
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(right[column])]
    categorical = [column for column in columns if column not in numeric]
    max_error = 0.0
    for column in categorical:
        a = left[column].fillna("<NA>").astype(str).to_numpy()
        b = right[column].fillna("<NA>").astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"independent score differs on key/label column {column}")
    for column in numeric:
        a = left[column].to_numpy(dtype=np.float64)
        b = right[column].to_numpy(dtype=np.float64)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(f"independent score has a different missingness pattern in {column}")
        finite = np.isfinite(a)
        error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        max_error = max(max_error, error)
        if error > 1e-12:
            raise ValueError(f"independent score differs in {column}: max_abs_error={error}")
    return max_error


def verify(condition_dir: Path, contract_dir: Path, signature_dir: Path,
           panel_dir: Path, split_path: Path, folds_dir: Path,
           scoring_dir: Path) -> dict:
    summary_path = scoring_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-039"
            or summary.get("status") != "B5_REVERSAL_SCORED_PENDING_INDEPENDENT_REVIEW"
            or summary.get("signature") != signature_dir.name):
        raise ValueError("scoring receipt is not a pending EXP-039 result for this signature")
    if (summary["source_sha256"]["panel"] != sha256(panel_dir / "panel_conditions.parquet")
            or summary["source_sha256"]["signature"] != sha256(signature_dir / "manifest.json")
            or summary["source_sha256"]["split"] != sha256(split_path)
            or summary["source_sha256"]["condition_manifest"] != sha256(condition_dir / "manifest.json")
            or summary["source_sha256"]["condition_matrix"] != sha256(condition_dir / "condition_delta.npy")
            or summary["source_sha256"]["gene_axis"] != sha256(contract_dir / "genes.json")):
        raise ValueError("scoring receipt does not match the frozen panel, signature, or split")
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
        raise ValueError("continuous signature contains duplicate mapped gene identities")
    effect = landmark.set_index("pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(float)
    panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    exposure = pd.read_parquet(condition_dir.parent / "continuous_inputs_v2"
                               / "condition_exposure.parquet")
    if (not np.array_equal(keys["condition_id"].to_numpy(), np.arange(len(keys)))
            or delta.shape != (len(keys), len(genes))
            or not exposure["condition_id"].equals(keys["condition_id"])):
        raise ValueError("measured condition keys, exposure, and continuous gene matrix are misaligned")
    measured_by_id = pd.Index(keys["condition_id"]).get_indexer
    fold_receipts = {int(row["fold"]): row for row in summary["folds"]}
    expected_drugs, expected_cells, checked_folds = [], [], []

    for fold in range(5):
        fold_dir = folds_dir / f"fold_{fold}"
        model_receipt = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        verification_path = fold_dir / "verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if (model_receipt.get("experiment_id") != "EXP-039"
                or model_receipt.get("fold") != fold
                or model_receipt.get("status") != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
                or verification.get("status") != "PASS" or verification.get("fold") != fold
                or verification.get("cell_scores_sha256") != sha256(fold_dir / "cell_scores_delta.parquet")):
            raise ValueError(f"EXP-039 model fold {fold} lacks a passing upstream verification")
        fold_panel = panel.loc[panel["fold"].eq(fold)
                               & panel["panel_status"].eq("ELIGIBLE")].copy().reset_index(drop=True)
        if fold_panel.empty or fold_panel["condition_id"].duplicated().any():
            raise ValueError(f"frozen fold {fold} panel is empty or has duplicate conditions")
        fold_keys = set(fold_panel["cell_id"])
        observed_positions = measured_by_id(fold_panel["condition_id"].to_numpy())
        if (observed_positions < 0).any():
            raise ValueError(f"fold {fold} panel condition is absent from the observed delta matrix")
        observed = np.asarray(delta[observed_positions], dtype=np.float32)
        groups = fold_panel.groupby(["cell_id", "pert_id"], sort=True).indices
        measured_vectors = np.empty((len(groups), len(genes)), dtype=np.float32)
        group_rows = []
        for index, ((cell, drug), positions) in enumerate(groups.items()):
            positions = np.asarray(positions, dtype=np.int64)
            measured_vectors[index] = observed[positions].mean(axis=0, dtype=np.float64)
            group_rows.append({"cell_id": cell, "pert_id": drug, "n_conditions": len(positions)})
        group_frame = pd.DataFrame(group_rows)
        measured_score = np.asarray(
            [reversal_reference(row, genes, up, down) for row in measured_vectors], dtype=np.float64)
        measured_cosine = negative_cosine(measured_vectors, effect)

        test_ids = np.load(fold_dir / "test_condition_ids.npy")
        prediction_positions = pd.Index(test_ids).get_indexer(fold_panel["condition_id"])
        if (prediction_positions < 0).any():
            raise ValueError(f"fold {fold} panel condition is absent from verified predictions")
        for arm in ARMS:
            for seed in SEEDS:
                predicted_rows = np.load(
                    fold_dir / f"predicted_delta_{arm}_seed{seed}.npy", mmap_mode="r")
                if predicted_rows.shape != (len(test_ids), len(genes)):
                    raise ValueError(f"fold {fold} {arm} seed {seed} prediction shape is invalid")
                predicted_vectors = np.empty_like(measured_vectors)
                for index, positions in enumerate(groups.values()):
                    positions = np.asarray(positions, dtype=np.int64)
                    predicted_vectors[index] = np.asarray(
                        predicted_rows[prediction_positions[positions]], dtype=np.float32
                    ).mean(axis=0, dtype=np.float64)
                predicted_score = np.asarray(
                    [reversal_reference(row, genes, up, down) for row in predicted_vectors],
                    dtype=np.float64)
                predicted_cosine = negative_cosine(predicted_vectors, effect)
                for index, row in group_frame.iterrows():
                    expected_drugs.append({
                        "fold": fold, "signature": signature_dir.name, "arm": arm, "seed": seed,
                        **row.to_dict(), "measured_reversal": measured_score[index],
                        "predicted_reversal": predicted_score[index],
                        "measured_negative_cosine": measured_cosine[index],
                        "predicted_negative_cosine": predicted_cosine[index],
                    })
                temp = group_frame.copy()
                temp["measured_reversal"] = measured_score
                temp["predicted_reversal"] = predicted_score
                temp["measured_negative_cosine"] = measured_cosine
                temp["predicted_negative_cosine"] = predicted_cosine
                for cell, frame in temp.groupby("cell_id", sort=True):
                    y = frame["measured_reversal"].to_numpy(float)
                    yhat = frame["predicted_reversal"].to_numpy(float)
                    reason = None
                    if len(frame) < 5:
                        reason = "FEWER_THAN_FIVE_DRUGS"
                    elif np.ptp(y) == 0:
                        reason = "CONSTANT_MEASURED_REVERSAL"
                    elif np.ptp(yhat) == 0:
                        reason = "CONSTANT_PREDICTED_REVERSAL"
                    rho = float(spearmanr(y, yhat).statistic) if reason is None else float("nan")
                    if reason is None and not np.isfinite(rho):
                        reason = "UNDEFINED_CORRELATION"
                    cosine_mask = frame[
                        ["measured_negative_cosine", "predicted_negative_cosine"]
                    ].notna().all(axis=1)
                    cosine_frame = frame.loc[cosine_mask]
                    cosine_rho = (
                        float(spearmanr(cosine_frame["measured_negative_cosine"],
                                        cosine_frame["predicted_negative_cosine"]).statistic)
                        if len(cosine_frame) >= 5 else float("nan"))
                    overlap, loss, top_n = top_metrics(
                        frame["pert_id"].to_numpy(), y, yhat)
                    expected_cells.append({
                        "fold": fold, "signature": signature_dir.name, "arm": arm, "seed": seed,
                        "cell_id": cell, "spearman": rho, "n_drugs": len(frame),
                        "reason": reason, "negative_cosine_spearman": cosine_rho,
                        "n_negative_cosine_drugs": len(cosine_frame),
                        "top_decile_overlap": overlap, "measured_selection_loss": loss,
                        "top_decile_count": top_n,
                    })
        if not fold_keys <= set(split_cell_ids(split_path, fold)):
            raise ValueError(f"fold {fold} measured panel includes cells outside the frozen test IDs")
        fold_receipt = fold_receipts[fold]
        if (fold_receipt["fold_summary_sha256"] != sha256(fold_dir / "fold_summary.json")
                or fold_receipt["verification_sha256"] != sha256(verification_path)):
            raise ValueError(f"scoring receipt references a different model fold {fold}")
        checked_folds.append({"fold": fold, "n_panel_conditions": len(fold_panel),
                              "n_cells": len(fold_keys),
                              "fold_summary_sha256": sha256(fold_dir / "fold_summary.json"),
                              "verification_sha256": sha256(verification_path)})

    expected_drug_table = pd.DataFrame(expected_drugs)
    expected_cell_table = pd.DataFrame(expected_cells)
    observed_drug_table = pd.read_parquet(scoring_dir / "drug_scores.parquet")
    observed_cell_table = pd.read_parquet(scoring_dir / "cell_scores.parquet")
    max_error = max(
        _equal_tables(observed_drug_table, expected_drug_table, DRUG_COLUMNS,
                      ("fold", "arm", "seed", "cell_id", "pert_id")),
        _equal_tables(observed_cell_table, expected_cell_table, CELL_COLUMNS,
                      ("fold", "arm", "seed", "cell_id")),
    )
    return {
        "status": "PASS",
        "experiment_id": "EXP-039",
        "signature": signature_dir.name,
        "run_summary_sha256": sha256(summary_path),
        "verified_drug_score_rows": len(expected_drug_table),
        "verified_cell_score_rows": len(expected_cell_table),
        "max_abs_recalculation_error": max_error,
        "reversal_implementation": "independent prefix-walk implementation with tied deltas grouped",
        "folds": checked_folds,
    }


def split_cell_ids(split_path: Path, fold: int) -> list[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return split["folds"][fold]["outer_test_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.condition_dir, args.contract_dir, args.signature_dir,
                    args.panel_dir, args.split, args.folds_dir, args.scoring_dir)
    output = args.scoring_dir / "independent_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
