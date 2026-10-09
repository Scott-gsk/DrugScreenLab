"""Score EXP-039 held-out predictions against measured reversal rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.reversal import reversal_score


ARMS = ("huber", "b5_rank")
SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scores(vectors: np.ndarray, genes: list[str], up: list[str], down: list[str]) -> np.ndarray:
    return np.asarray([reversal_score(vector, genes, up, down) for vector in vectors], dtype=np.float64)


def negative_cosine(vectors: np.ndarray, effect: np.ndarray) -> np.ndarray:
    mapped = np.isfinite(effect)
    if mapped.sum() < 20:
        raise ValueError("fewer than 20 signature effects map to the LINCS gene axis")
    values = np.asarray(vectors[:, mapped], dtype=np.float64)
    reference = effect[mapped]
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(reference)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = denominator > 0
    result[valid] = -(values[valid] @ reference) / denominator[valid]
    return result


def top_metrics(drugs: np.ndarray, measured: np.ndarray, predicted: np.ndarray) -> tuple[float, float, int]:
    count = max(1, int(np.ceil(len(drugs) * 0.1)))
    measured_top = np.lexsort((drugs, -measured))[:count]
    predicted_top = np.lexsort((drugs, -predicted))[:count]
    overlap = len(set(measured_top) & set(predicted_top)) / count
    loss = float(measured[measured_top].mean() - measured[predicted_top].mean())
    return overlap, loss, count


def score_fold(condition_dir: Path, contract_dir: Path, signature_dir: Path,
               panel_dir: Path, split_path: Path, folds_dir: Path,
               fold: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold_dir = folds_dir / f"fold_{fold}"
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((fold_dir / "verification.json").read_text(encoding="utf-8"))
    if (summary["experiment_id"] != "EXP-039" or summary["fold"] != fold
            or summary["status"] != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or verification["status"] != "PASS" or verification["fold"] != fold
            or verification.get("run_provenance_sha256") != summary.get("run_provenance_sha256")
            or verification["cell_scores_sha256"] != sha256(fold_dir / "cell_scores_delta.parquet")):
        raise ValueError(f"EXP-039 fold {fold} is incomplete or unverified")
    for name, digest in summary["files_sha256"].items():
        if sha256(fold_dir / name) != digest:
            raise ValueError(f"EXP-039 upstream artifact changed: {name}")
    panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    panel = panel.loc[(panel["fold"].eq(fold)) & panel["panel_status"].eq("ELIGIBLE")].copy()
    if panel["condition_id"].duplicated().any() or panel.empty:
        raise ValueError("fixed EXP-038 panel has duplicate or missing held-out conditions")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if not set(panel["cell_id"]) <= set(split["folds"][fold]["outer_test_ids"]):
        raise ValueError("fixed reversal panel contains non-held-out cells")
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if (len(genes) != 978 or len(up) < 20 or len(down) < 20
            or (set(up) | set(down)) - set(genes) or set(up) & set(down)):
        raise ValueError("fixed disease signature does not map uniquely to the 978-gene axis")
    landmark = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    if landmark["pr_gene_id"].duplicated().any():
        raise ValueError("signature gene IDs are not unique")
    effects = landmark.set_index("pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(float)

    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    deltas = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    if deltas.shape != (len(keys), len(genes)):
        raise ValueError("condition delta matrix differs from signature gene axis")
    condition_rows = keys.set_index("condition_id", verify_integrity=True)
    measured_row_index = condition_rows.index.get_indexer(panel["condition_id"])
    if (measured_row_index < 0).any():
        raise ValueError("fixed panel condition is absent from measured delta keys")
    measured = np.asarray(deltas[measured_row_index], dtype=np.float32)
    base_keys = panel[["cell_id", "pert_id", "condition_id"]].reset_index(drop=True)
    groups = base_keys.groupby(["cell_id", "pert_id"], sort=True).indices
    group_rows = []
    observed_vectors = np.empty((len(groups), len(genes)), dtype=np.float32)
    for i, ((cell, drug), positions) in enumerate(groups.items()):
        positions = np.asarray(positions, dtype=np.int64)
        observed_vectors[i] = measured[positions].mean(axis=0, dtype=np.float64)
        group_rows.append({"cell_id": cell, "pert_id": drug, "n_conditions": len(positions)})
    group_frame = pd.DataFrame(group_rows)
    measured_score = scores(observed_vectors, genes, up, down)
    measured_cosine = negative_cosine(observed_vectors, effects)

    test_ids = np.load(fold_dir / "test_condition_ids.npy")
    prediction_positions = pd.Index(test_ids).get_indexer(panel["condition_id"])
    if (prediction_positions < 0).any():
        raise ValueError("fixed panel condition is missing from the verified prediction axis")
    drug_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            prediction = np.load(fold_dir / f"predicted_delta_{arm}_seed{seed}.npy", mmap_mode="r")
            predicted_vectors = np.empty_like(observed_vectors)
            for i, positions in enumerate(groups.values()):
                positions = np.asarray(positions, dtype=np.int64)
                prediction_rows = prediction_positions[positions]
                predicted_vectors[i] = np.asarray(prediction[prediction_rows], dtype=np.float32).mean(
                    axis=0, dtype=np.float64)
            predicted_score = scores(predicted_vectors, genes, up, down)
            predicted_cosine = negative_cosine(predicted_vectors, effects)
            frame = group_frame.copy()
            frame.insert(0, "fold", fold)
            frame.insert(1, "signature", signature_dir.name)
            frame.insert(2, "arm", arm)
            frame.insert(3, "seed", seed)
            frame["measured_reversal"] = measured_score
            frame["predicted_reversal"] = predicted_score
            frame["measured_negative_cosine"] = measured_cosine
            frame["predicted_negative_cosine"] = predicted_cosine
            drug_rows.extend(frame.to_dict("records"))

    drug_scores = pd.DataFrame(drug_rows)
    cell_rows = []
    for (arm, seed, cell), frame in drug_scores.groupby(["arm", "seed", "cell_id"], sort=True):
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
        cosine_mask = frame[["measured_negative_cosine", "predicted_negative_cosine"]].notna().all(axis=1)
        cosine_rows = frame.loc[cosine_mask]
        cosine_rho = (float(spearmanr(cosine_rows["measured_negative_cosine"],
                                      cosine_rows["predicted_negative_cosine"]).statistic)
                      if len(cosine_rows) >= 5 else float("nan"))
        overlap, loss, top_n = top_metrics(frame["pert_id"].to_numpy(), y, yhat)
        cell_rows.append({"fold": fold, "signature": signature_dir.name,
                          "arm": arm, "seed": seed, "cell_id": cell,
                          "spearman": rho, "n_drugs": len(frame), "reason": reason,
                          "negative_cosine_spearman": cosine_rho,
                          "n_negative_cosine_drugs": len(cosine_rows),
                          "top_decile_overlap": overlap,
                          "measured_selection_loss": loss, "top_decile_count": top_n})
    cell_scores = pd.DataFrame(cell_rows)
    receipt = {"fold": fold, "n_conditions": len(panel), "n_cells": int(panel["cell_id"].nunique()),
               "n_cell_drugs": len(group_frame),
               "fold_summary_sha256": sha256(fold_dir / "fold_summary.json"),
               "verification_sha256": sha256(fold_dir / "verification.json"),
               "prediction_sha256": {f"{arm}_{seed}": sha256(
                   fold_dir / f"predicted_delta_{arm}_seed{seed}.npy")
                   for arm in ARMS for seed in SEEDS}}
    return drug_scores, cell_scores, receipt


def run(condition_dir: Path, contract_dir: Path, signature_dir: Path,
        panel_dir: Path, split_path: Path, folds_dir: Path,
        output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"B5 reversal output already exists: {output_dir}")
    panel_audit = json.loads((panel_dir / "panel_audit.json").read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (panel_audit["status"] != "MEASURED_PANEL_QC_PASS_PENDING_PREDICTIONS"
            or panel_audit["source_sha256"]["split"] != sha256(split_path)
            or panel_audit["source_sha256"]["condition_matrix"] != sha256(condition_dir / "condition_delta.npy")
            or panel_audit["source_sha256"]["gene_axis"] != sha256(contract_dir / "genes.json")):
        raise ValueError("frozen EXP-038 measured panel differs from scoring inputs")
    all_drugs, all_cells, receipts = [], [], []
    for fold in range(5):
        drugs, cells, receipt = score_fold(condition_dir, contract_dir, signature_dir,
                                           panel_dir, split_path, folds_dir, fold)
        all_drugs.append(drugs)
        all_cells.append(cells)
        receipts.append(receipt)
    drug_scores = pd.concat(all_drugs, ignore_index=True)
    cell_scores = pd.concat(all_cells, ignore_index=True)
    if cell_scores["cell_id"].nunique() != panel_audit["n_eligible_cells"]:
        raise ValueError("five-fold B5 scoring does not cover every frozen panel cell")
    output_dir.mkdir(parents=True)
    drug_path = output_dir / "drug_scores.parquet"
    cell_path = output_dir / "cell_scores.parquet"
    drug_scores.to_parquet(drug_path, index=False)
    cell_scores.to_parquet(cell_path, index=False)
    result = {
        "experiment_id": "EXP-039", "status": "B5_REVERSAL_SCORED_PENDING_INDEPENDENT_REVIEW",
        "signature": signature_dir.name,
        "primary_metric": "equal-cell mean within-cell Spearman: predicted versus measured reversal scores",
        "secondary_metrics": ["negative-cosine Spearman", "top-decile overlap", "measured selection loss"],
        "arms": list(ARMS), "seeds": list(SEEDS),
        "n_cells": int(cell_scores["cell_id"].nunique()),
        "n_cell_scores": len(cell_scores), "n_drug_scores": len(drug_scores),
        "undefined_cell_scores": {str(key): int(value) for key, value in
                                   cell_scores["reason"].value_counts(dropna=True).items()},
        "folds": receipts,
        "files_sha256": {drug_path.name: sha256(drug_path), cell_path.name: sha256(cell_path)},
        "source_sha256": {"panel": sha256(panel_dir / "panel_conditions.parquet"),
                          "signature": sha256(signature_dir / "manifest.json"),
                          "split": sha256(split_path),
                          "condition_manifest": sha256(condition_dir / "manifest.json"),
                          "condition_matrix": sha256(condition_dir / "condition_delta.npy"),
                          "gene_axis": sha256(contract_dir / "genes.json")},
    }
    (output_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.condition_dir, args.contract_dir, args.signature_dir,
                 args.panel_dir, args.split, args.folds_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
