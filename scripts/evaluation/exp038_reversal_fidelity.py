"""Score frozen measured and held-out predicted perturbations on the CRC signature."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.reversal import reversal_score


ARMS = ("conditional", "state_ablated", "train_drug_mean", "zero")
SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_scores(vectors: np.ndarray, genes: list[str], up: list[str], down: list[str]) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[1] != len(genes):
        raise ValueError("delta gene axis differs from frozen signature axis")
    return np.array([reversal_score(row, genes, up, down) for row in vectors], dtype=np.float64)


def negative_cosine(vectors: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Direction-only secondary score on mapped continuous disease effects."""
    mapped = np.isfinite(effect)
    if mapped.sum() < 20 or vectors.shape[1] != len(effect):
        raise ValueError("continuous disease effect axis is unavailable")
    values = np.asarray(vectors[:, mapped], dtype=np.float64)
    reference = effect[mapped]
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(reference)
    result = np.full(len(vectors), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = -(values[valid] @ reference) / denominator[valid]
    return result


def top_selection(drugs: np.ndarray, measured: np.ndarray,
                  predicted: np.ndarray) -> tuple[float, float, int]:
    """Deterministic top-decile overlap and measured-score selection loss."""
    if len(drugs) < 5 or len(set(drugs)) != len(drugs):
        raise ValueError("top selection requires at least five unique drugs")
    count = max(1, int(np.ceil(len(drugs) * 0.1)))
    measured_top = np.lexsort((drugs, -measured))[:count]
    predicted_top = np.lexsort((drugs, -predicted))[:count]
    overlap = len(set(measured_top) & set(predicted_top)) / count
    loss = float(measured[measured_top].mean() - measured[predicted_top].mean())
    return overlap, loss, count


def aggregate_drugs(frame: pd.DataFrame, values: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) != len(values):
        raise ValueError("condition and delta axes have different lengths")
    groups = frame.groupby(["cell_id", "pert_id"], sort=True).indices
    rows = []
    mean_values = np.empty((len(groups), values.shape[1]), dtype=np.float32)
    for index, ((cell, drug), positions) in enumerate(groups.items()):
        rows.append({"cell_id": cell, "pert_id": drug, "n_conditions": len(positions)})
        mean_values[index] = values[np.asarray(positions)].mean(axis=0, dtype=np.float64)
    return pd.DataFrame(rows), mean_values


def evaluate(condition_dir: Path, contract_dir: Path, signature_dir: Path,
             panel_dir: Path, split_path: Path, folds_dir: Path,
             baseline_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"reversal output already exists: {output_dir}")
    panel_receipt = json.loads((panel_dir / "panel_audit.json").read_text(encoding="utf-8"))
    if (panel_receipt["source_sha256"]["split"] != sha256(split_path)
            or panel_receipt["source_sha256"]["signature_manifest"] != sha256(signature_dir / "manifest.json")
            or panel_receipt["panel_sha256"] != sha256(panel_dir / "panel_conditions.parquet")):
        raise ValueError("panel differs from the frozen split or signature")
    panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    panel = panel.loc[panel["panel_status"].eq("ELIGIBLE")].copy()
    if panel["condition_id"].duplicated().any():
        raise ValueError("fixed panel has duplicate condition IDs")
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if len(genes) != 978 or len(up) < 20 or len(down) < 20:
        raise ValueError("frozen gene axes fail the registered gate")
    landmark = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    if landmark["pr_gene_id"].duplicated().any():
        raise ValueError("continuous disease signature has duplicate LINCS gene IDs")
    effects = landmark.set_index("pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(
        dtype=np.float64)
    if int(np.isfinite(effects).sum()) < 20:
        raise ValueError("continuous disease signature has too few mapped genes")
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    measured = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if measured.shape != (len(keys), len(genes)):
        raise ValueError("measured delta axis is not aligned with condition keys")
    rows = []
    receipts = []
    for fold in range(5):
        fold_dir = folds_dir / f"fold_{fold}"
        fold_receipt = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        verification_path = fold_dir / "verification_v2.json"
        if not verification_path.is_file():
            verification_path = fold_dir / "verification_continuation.json"
        if not verification_path.is_file():
            raise ValueError(f"upstream perturbation fold {fold} has no supported verification receipt")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if (fold_receipt["fold"] != fold or fold_receipt["status"] != "FOLD_COMPLETE_PENDING_CROSS_FOLD_REVIEW"
                or fold_receipt["split_sha256"] != sha256(split_path)
                or verification["status"] != "PASS" or verification["fold"] != fold):
            raise ValueError(f"upstream perturbation fold {fold} is not verified")
        for name, digest in verification["files_sha256"].items():
            if sha256(fold_dir / name) != digest:
                raise ValueError(f"upstream perturbation file changed after verification: {name}")
        selected = panel.loc[panel["fold"].eq(fold)].copy()
        if not set(selected["cell_id"]) <= set(split["folds"][fold]["outer_test_ids"]):
            raise ValueError("panel cell escapes held-out fold")
        test_ids = np.load(fold_dir / "test_condition_ids.npy")
        positions = pd.Index(test_ids).get_indexer(selected["condition_id"])
        if (positions < 0).any():
            raise ValueError("fixed panel condition absent from held-out predictions")
        index = selected["condition_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(keys.iloc[index]["condition_id"].to_numpy(), index):
            raise ValueError("condition ID differs from measured matrix row")
        groups, observed_drug = aggregate_drugs(selected, np.asarray(measured[index], dtype=np.float32))
        observed_score = rank_scores(observed_drug, genes, up, down)
        observed_cosine = negative_cosine(observed_drug, effects)
        if not np.isfinite(observed_score).all():
            raise ValueError("measured reversal scores contain undefined values")
        drug_ids_path = baseline_dir / f"fold_{fold}" / "drug_ids.json"
        drug_mean_path = baseline_dir / f"fold_{fold}" / "drug_mean.npy"
        drug_ids = json.loads(drug_ids_path.read_text(encoding="utf-8"))
        drug_means = np.load(drug_mean_path, mmap_mode="r")
        drug_position = {drug: i for i, drug in enumerate(drug_ids)}
        if (len(drug_position) != len(drug_ids) or drug_means.shape != (len(drug_ids), len(genes))
                or not set(groups["pert_id"]) <= set(drug_position)):
            raise ValueError("training-side drug mean lacks fixed-panel drugs")
        baseline_vectors = np.asarray(drug_means[[drug_position[x] for x in groups["pert_id"]]],
                                      dtype=np.float32)
        baseline_score = rank_scores(baseline_vectors, genes, up, down)
        baseline_cosine = negative_cosine(baseline_vectors, effects)
        for arm in ARMS:
            for seed in (SEEDS if arm in {"conditional", "state_ablated"} else (0,)):
                if arm == "train_drug_mean":
                    predicted_score = baseline_score
                    predicted_cosine = baseline_cosine
                elif arm == "zero":
                    predicted_score = np.zeros(len(groups), dtype=np.float64)
                    predicted_cosine = np.full(len(groups), np.nan, dtype=np.float64)
                else:
                    prediction = np.load(fold_dir / f"predicted_delta_{arm}_seed{seed}.npy", mmap_mode="r")
                    if prediction.shape != (len(test_ids), len(genes)):
                        raise ValueError("saved delta prediction shape differs from held-out axis")
                    predicted = np.asarray(prediction[positions], dtype=np.float32)
                    recovered_groups, predicted_drug = aggregate_drugs(selected, predicted)
                    pd.testing.assert_frame_equal(recovered_groups, groups)
                    predicted_score = rank_scores(predicted_drug, genes, up, down)
                    predicted_cosine = negative_cosine(predicted_drug, effects)
                part = groups.copy()
                part.insert(0, "fold", fold)
                part.insert(1, "arm", arm)
                part.insert(2, "seed", seed)
                part["measured_reversal"] = observed_score
                part["predicted_reversal"] = predicted_score
                part["measured_negative_cosine"] = observed_cosine
                part["predicted_negative_cosine"] = predicted_cosine
                if not np.isfinite(part[["measured_reversal", "predicted_reversal"]].to_numpy()).all():
                    raise ValueError("fixed-panel reversal score is non-finite")
                rows.append(part)
        receipts.append({"fold": fold, "n_conditions": len(selected),
                        "n_cells": int(selected["cell_id"].nunique()), "n_cell_drugs": len(groups),
                        "upstream_verification_file": verification_path.name,
                        "upstream_fold_summary_sha256": sha256(fold_dir / "fold_summary.json"),
                         "upstream_verification_sha256": sha256(verification_path),
                         "baseline_drug_ids_sha256": sha256(drug_ids_path),
                         "baseline_drug_mean_sha256": sha256(drug_mean_path)})
    drug_scores = pd.concat(rows, ignore_index=True)
    cell_rows = []
    for (fold, arm, seed, cell), frame in drug_scores.groupby(["fold", "arm", "seed", "cell_id"], sort=True):
        measured_score = frame["measured_reversal"].to_numpy()
        predicted_score = frame["predicted_reversal"].to_numpy()
        reason = None
        if len(frame) < 5:
            reason = "FEWER_THAN_FIVE_DRUGS"
        elif np.ptp(measured_score) == 0:
            reason = "CONSTANT_MEASURED_REVERSAL"
        elif np.ptp(predicted_score) == 0:
            reason = "CONSTANT_PREDICTED_REVERSAL"
        score = float(spearmanr(measured_score, predicted_score).statistic) if reason is None else float("nan")
        if reason is None and not np.isfinite(score):
            reason = "UNDEFINED_CORRELATION"
        cosine_valid = frame[["measured_negative_cosine", "predicted_negative_cosine"]].notna().all(axis=1)
        cosine_frame = frame.loc[cosine_valid]
        cosine_score = (float(spearmanr(cosine_frame["measured_negative_cosine"],
                                       cosine_frame["predicted_negative_cosine"]).statistic)
                        if len(cosine_frame) >= 5 else float("nan"))
        top_overlap, selection_loss, top_count = top_selection(
            frame["pert_id"].to_numpy(), measured_score, predicted_score)
        cell_rows.append({"fold": fold, "arm": arm, "seed": seed, "cell_id": cell,
                          "spearman": score, "n_drugs": len(frame), "reason": reason,
                          "negative_cosine_spearman": cosine_score,
                          "n_negative_cosine_drugs": len(cosine_frame),
                          "top_decile_overlap": top_overlap,
                          "measured_selection_loss": selection_loss,
                          "top_decile_count": top_count})
    cell_scores = pd.DataFrame(cell_rows)
    output_dir.mkdir(parents=True)
    drug_path = output_dir / "drug_scores.parquet"
    cell_path = output_dir / "cell_scores.parquet"
    drug_scores.to_parquet(drug_path, index=False)
    cell_scores.to_parquet(cell_path, index=False)
    report = {"experiment_id": "EXP-038", "status": "B2_SCORED_PENDING_INDEPENDENT_REVIEW",
              "signature": signature_dir.name,
              "metric": "within-cell Spearman of predicted versus measured drug reversal scores",
              "secondary_metric": "within-cell Spearman of predicted versus measured negative cosine",
              "top_decile_ties": "sort higher score first, then drug ID ascending",
              "selection_loss": "mean measured reversal of measured top decile minus mean measured reversal of predicted top decile",
              "aggregation": "mean deltas within held-out cell/drug, score each mean, then rank drugs within cell",
              "arms": list(ARMS), "seeds": list(SEEDS), "n_cells": int(panel["cell_id"].nunique()),
              "n_conditions": len(panel), "n_cell_scores": len(cell_scores),
              "undefined_cell_scores": {str(key): int(value) for key, value in
                                        cell_scores["reason"].value_counts(dropna=True).items()},
              "folds": receipts,
              "files_sha256": {drug_path.name: sha256(drug_path), cell_path.name: sha256(cell_path)},
              "source_sha256": {"panel": sha256(panel_dir / "panel_conditions.parquet"),
                                "signature": sha256(signature_dir / "manifest.json"),
                                "split": sha256(split_path),
                                "condition_manifest": sha256(condition_dir / "manifest.json"),
                                "condition_matrix": sha256(condition_dir / "condition_delta.npy"),
                                "gene_axis": sha256(contract_dir / "genes.json"),
                                "baseline_summary": sha256(baseline_dir / "run_summary.json")}}
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.condition_dir, args.contract_dir, args.signature_dir,
                      args.panel_dir, args.split, args.folds_dir, args.baseline_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
