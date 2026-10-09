"""Independent verification for EXP-038's CPU baseline-only scoring run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EXPECTED = {
    "split": "e474cede3d7474ad87c5685850726e9a8db813eabbd775e552fd09899df753d7",
    "condition_manifest": "0947fdf55d4f016928258d9a372c699efe12bdbdf6cf404d46253bcdd6f3e68e",
    "delta": "29317d50b44933452126ddd213017709d54b37f6c0dcc2f403cf5ac5ff8112fb",
    "genes": "8cc8d6ade494209a9f323a94450e02c3a74d1ea3ae61a75ed4c084a9ab80502c",
    "panel": "43976344daf4a043d6e96c1bc2b363428b4a2478830e7f5cb44408aa704d932f",
    "gse_manifest": "7515dff33c4ac3689a19273a21e6b5d3c184c62d7c8bb67772ac9a7e8c8c3782",
}
SIGNATURES = {
    "GSE74602_v2": Path("data/processed/dual_route/EXP-038/gse74602_crc_signature_v2"),
    "TCGA_COAD_READ_v1": Path("data/processed/dual_route/EXP-038/tcga_crc_signature_v1"),
}
N_GENES = 978
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260925
METRICS = ("spearman", "negative_cosine_spearman", "top_decile_overlap",
           "measured_selection_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _expected_enrichment(delta: np.ndarray, genes: list[str], members: set[str]) -> float:
    """Independent grouped-prefix implementation of the registered ES readout."""
    order = np.argsort(-delta, kind="stable")
    ordered_values = delta[order]
    hit = np.fromiter((genes[i] in members for i in order), dtype=np.int64)
    hit_count = int(hit.sum())
    if not 0 < hit_count < len(genes):
        raise ValueError("signature set is empty or covers the full output axis")
    ends = np.r_[np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1,
                 len(ordered_values)]
    cumulative = np.cumsum(hit)
    hits_at_end = cumulative[ends - 1]
    running = hits_at_end / hit_count - (ends - hits_at_end) / (len(genes) - hit_count)
    high, low = float(np.max(running)), float(np.min(running))
    if abs(high + low) <= 1e-12:
        return 0.0
    return high if high > -low else low


def _expected_reversal(vector: np.ndarray, genes: list[str], up: set[str],
                       down: set[str]) -> float:
    return (_expected_enrichment(vector, genes, down)
            - _expected_enrichment(vector, genes, up)) / 2.0


def _expected_cosine(vectors: np.ndarray, effects: np.ndarray) -> np.ndarray:
    selected = np.isfinite(effects)
    x = np.asarray(vectors[:, selected], dtype=np.float64)
    y = effects[selected]
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    output = np.full(len(x), np.nan, dtype=np.float64)
    valid = denom > 0
    output[valid] = -(x[valid] @ y) / denom[valid]
    return output


def _top_metrics(ids: np.ndarray, measured: np.ndarray,
                 predicted: np.ndarray | None) -> tuple[float, float, int]:
    names = ids.astype(str)
    n_top = max(1, int(np.ceil(len(names) * 0.1)))
    truth_order = np.lexsort((names, -measured))
    pred_order = (np.argsort(names, kind="stable") if predicted is None
                  else np.lexsort((names, -predicted)))
    truth = truth_order[:n_top]
    selected = pred_order[:n_top]
    overlap = len(set(truth).intersection(selected)) / n_top
    loss = float(measured[truth].mean() - measured[selected].mean())
    return overlap, loss, n_top


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _load_signatures(root: Path, genes: list[str]) -> dict[str, tuple[set[str], set[str], np.ndarray]]:
    result = {}
    for name, relative in SIGNATURES.items():
        folder = root / relative
        manifest = read_json(folder / "manifest.json")
        if name == "GSE74602_v2" and sha256(folder / "manifest.json") != EXPECTED["gse_manifest"]:
            raise ValueError("GSE v2 signature manifest no longer matches the preregistered version")
        required_status = ("SIGNATURE_READY_FOR_DEVELOPMENT" if name == "GSE74602_v2"
                           else "SIGNATURE_READY_FOR_CROSS_SIGNATURE_VALIDATION")
        if manifest.get("experiment_id") != "EXP-038" or manifest.get("status") != required_status:
            raise ValueError(f"{name} signature status is not eligible for this baseline comparison")
        file_hashes = manifest.get("files_sha256", {})
        for relative_name, expected in file_hashes.items():
            if sha256(folder / relative_name) != expected:
                raise ValueError(f"{name} signature source changed: {relative_name}")
        up = set(json.loads((folder / "up_genes.json").read_text(encoding="utf-8")))
        down = set(json.loads((folder / "down_genes.json").read_text(encoding="utf-8")))
        if (len(up) < 20 or len(down) < 20 or len(up) > 100 or len(down) > 100
                or up & down or (up | down) - set(genes)):
            raise ValueError(f"{name} direction sets do not map uniquely to the frozen 978-gene axis")
        effects_frame = pd.read_parquet(folder / "landmark_statistics.parquet")
        effects_frame["pr_gene_id"] = effects_frame["pr_gene_id"].astype(str)
        if effects_frame["pr_gene_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate continuous effect gene IDs")
        effects = effects_frame.set_index("pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(float)
        if int(np.isfinite(effects).sum()) < 20:
            raise ValueError(f"{name} continuous effect axis has fewer than 20 mapped genes")
        result[name] = (up, down, effects)
    return result


def _compare_rows(actual: pd.DataFrame, expected: pd.DataFrame,
                  keys: list[str], numeric: list[str], categorical: list[str]) -> float:
    if set(keys + numeric + categorical) - set(actual.columns):
        missing = sorted(set(keys + numeric + categorical) - set(actual.columns))
        raise ValueError(f"scored table is missing required columns: {missing}")
    left = actual[keys + numeric + categorical].sort_values(keys).reset_index(drop=True)
    right = expected[keys + numeric + categorical].sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(f"scoring row count mismatch: actual={len(left)}, expected={len(right)}")
    for col in keys + categorical:
        a = left[col].fillna("<NA>").astype(str).to_numpy()
        b = right[col].fillna("<NA>").astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"scoring identity/status mismatch in {col}")
    maximum = 0.0
    for col in numeric:
        a = pd.to_numeric(left[col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[col], errors="coerce").to_numpy(float)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise ValueError(f"undefined-value pattern mismatch in {col}")
        valid = np.isfinite(a) & np.isfinite(b)
        error = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
        maximum = max(maximum, error)
        if error > 1e-8:
            raise ValueError(f"independent score mismatch in {col}: max_abs_error={error}")
    return maximum


def _ci(values: np.ndarray, seed: int) -> dict[str, Any] | None:
    x = values[np.isfinite(values)]
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        boot[i] = x[rng.integers(0, len(x), len(x))].mean()
    return {"mean": float(x.mean()), "n_cells": len(x),
            "percentile_95_ci": [float(np.quantile(boot, 0.025)),
                                  float(np.quantile(boot, 0.975))],
            "bootstrap_resamples": N_BOOTSTRAP, "unit": "cell",
            "interpretation": "descriptive uncertainty interval; not a hypothesis test or Holm-family result"}


def _make_summary(cells: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (signature, baseline), frame in cells.groupby(["signature", "baseline"], sort=True):
        overall = {metric: _ci(frame[metric].to_numpy(float), BOOTSTRAP_SEED + index)
                   for index, metric in enumerate(METRICS)}
        fold_summaries = []
        for fold, part in frame.groupby("fold", sort=True):
            row: dict[str, Any] = {"fold": int(fold),
                                   "n_cells": int(part["cell_id"].nunique()),
                                   "n_conditions": int(part["n_measured_conditions"].sum()),
                                   "n_cell_drug_groups": int(part["n_drugs"].sum())}
            for metric in METRICS:
                vals = part[metric].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else None
                row[f"{metric}_defined_cells"] = int(len(vals))
            row["undefined_primary_reasons"] = {
                str(k): int(v) for k, v in part["spearman_reason"].dropna().value_counts().items()}
            row["undefined_negative_cosine_reasons"] = {
                str(k): int(v) for k, v in
                part["negative_cosine_reason"].dropna().value_counts().items()}
            fold_summaries.append(row)
        result[f"{signature}/{baseline}"] = {
            "n_cells": int(frame["cell_id"].nunique()),
            "overall_cell_equal_metrics": overall,
            "folds": fold_summaries,
            "primary_spearman_comparable": baseline == "train_drug_mean",
            "notes": (["Zero predictions have no between-drug rank; primary Spearman and negative cosine are undefined.",
                       "Top-decile metrics use ascending pert_id as the tie-break among all tied zero predictions."]
                      if baseline == "zero" else []),
        }
    return result


def _dict_close(actual: Any, expected: Any, path: str = "summary") -> float:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"summary structure mismatch at {path}")
        return max((_dict_close(actual[key], value, f"{path}.{key}")
                    for key, value in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"summary list mismatch at {path}")
        return max((_dict_close(a, b, f"{path}[{i}]")
                    for i, (a, b) in enumerate(zip(actual, expected))), default=0.0)
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or not np.isfinite(float(actual)):
            raise ValueError(f"summary numeric value missing at {path}")
        error = abs(float(actual) - float(expected))
        if error > 1e-10:
            raise ValueError(f"summary mismatch at {path}: max_abs_error={error}")
        return error
    if actual != expected:
        raise ValueError(f"summary value mismatch at {path}")
    return 0.0


def verify(root: Path, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if (not sys.platform.startswith("linux") or "microsoft" not in platform.release().lower()
            or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu"
            or not os.environ.get("CONDA_PREFIX", "").endswith("/envs/drugscreening-gpu")
            or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None}
            or "torch" in sys.modules):
        raise RuntimeError("independent verifier must use the approved WSL2 Conda CPU-only runtime")
    root = root.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    summary_path = run_dir / "result_summary.json"
    receipt_path = run_dir / "run_receipt.json"
    summary, receipt = read_json(summary_path), read_json(receipt_path)
    if (summary.get("experiment_id") != "EXP-038"
            or summary.get("status") != "BASELINE_ONLY_SCORED_PENDING_INDEPENDENT_VERIFICATION"
            or summary.get("run_id") != run_dir.name
            or summary.get("not_b2_main_complete") is not True
            or summary.get("holm_family_closed") is not False
            or receipt.get("device") != "CPU_ONLY"
            or receipt.get("runtime", {}).get("torch_imported") is not False):
        raise ValueError("run summary/receipt is outside the authorized baseline-only scope")

    condition = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"
    split_path = root / "data/splits/lincs_cell_splits_20260922_v2.json"
    axis_path = root / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"
    panel_dir = root / "artifacts/experiments/EXP-038/20260923_fixed_panel_v1"
    baseline_dir = root / "artifacts/experiments/EXP-037/20260922_baselines_v2"
    actual_hashes = {
        "split": sha256(split_path),
        "condition_manifest": sha256(condition / "manifest.json"),
        "delta": sha256(condition / "condition_delta.npy"),
        "genes": sha256(axis_path),
        "panel": sha256(panel_dir / "panel_conditions.parquet"),
    }
    expected_hashes = {"split": EXPECTED["split"],
                       "condition_manifest": EXPECTED["condition_manifest"],
                       "delta": EXPECTED["delta"], "genes": EXPECTED["genes"],
                       "panel": EXPECTED["panel"]}
    if actual_hashes != expected_hashes:
        raise ValueError(f"frozen baseline inputs changed: {actual_hashes}")
    summary_key = {"genes": "gene_axis", "panel": "panel_conditions", "delta": "condition_delta"}
    for key, value in actual_hashes.items():
        if summary["source_sha256"].get(summary_key.get(key, key)) != value:
            raise ValueError(f"result summary does not bind frozen input {key}")
    for name, digest in receipt.get("files_sha256", {}).items():
        if sha256(run_dir / name) != digest:
            raise ValueError(f"scored output changed after run receipt: {name}")

    genes = json.loads(axis_path.read_text(encoding="utf-8"))
    if len(genes) != N_GENES or len(set(genes)) != N_GENES:
        raise ValueError("frozen output axis is not 978 unique genes")
    keys = pd.read_parquet(condition / "conditions.parquet")
    delta = np.load(condition / "condition_delta.npy", mmap_mode="r", allow_pickle=False)
    if (keys["condition_id"].duplicated().any() or delta.shape != (len(keys), N_GENES)
            or delta.dtype != np.dtype("float32")):
        raise ValueError("condition IDs and 978-axis delta matrix are misaligned")
    index = pd.Index(keys["condition_id"])
    panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    if (panel["condition_id"].duplicated().any()
            or len(panel.loc[panel["panel_status"].eq("ELIGIBLE")]) != 40642):
        raise ValueError("frozen eligible panel identity/count is invalid")
    panel = panel.loc[panel["panel_status"].eq("ELIGIBLE")].copy()
    if panel["condition_id"].isin(keys["condition_id"]).sum() != len(panel):
        raise ValueError("fixed panel condition ID is absent from measured rows")
    split = read_json(split_path)
    baseline_summary = read_json(baseline_dir / "run_summary.json")
    existing_verification = read_json(baseline_dir / "independent_verification.json")
    if (baseline_summary.get("status") != "BASELINES_COMPLETE_PENDING_REVIEW"
            or existing_verification.get("status") != "PASS"
            or baseline_summary.get("data", {}).get("condition_manifest_sha256") != actual_hashes["condition_manifest"]
            or baseline_summary.get("data", {}).get("condition_matrix_sha256") != actual_hashes["delta"]
            or baseline_summary.get("data", {}).get("split_sha256") != actual_hashes["split"]):
        raise ValueError("EXP-037 baseline inputs are not independently bound to this evaluation")
    if (receipt.get("baseline_summary_sha256") != sha256(baseline_dir / "run_summary.json")
            or receipt.get("baseline_independent_verification_sha256")
            != sha256(baseline_dir / "independent_verification.json")):
        raise ValueError("run receipt references different upstream baseline summaries")

    condition_to_position = pd.Series(np.arange(len(keys), dtype=np.int64), index=keys["condition_id"])
    signatures = _load_signatures(root, genes)
    expected_drug_rows: list[dict[str, Any]] = []
    expected_cell_rows: list[dict[str, Any]] = []
    full_mean_checks = []
    fold_cell_counts = []
    for fold, fold_split in enumerate(split["folds"]):
        train_cells = set(map(str, fold_split["outer_train_ids"]))
        test_cells = set(map(str, fold_split["outer_test_ids"]))
        fold_panel = panel.loc[panel["fold"].eq(fold)].copy().reset_index(drop=True)
        if (set(fold_panel["cell_id"].astype(str)) - test_cells
                or set(fold_panel["cell_id"].astype(str)) & train_cells):
            raise ValueError(f"fold {fold} measured panel is not held out by cell")
        train_positions = np.flatnonzero(keys["cell_id"].astype(str).isin(train_cells).to_numpy())
        train_rows = keys.iloc[train_positions].reset_index(drop=True)
        expected_drugs = sorted(train_rows["pert_id"].astype(str).unique())
        saved_ids = json.loads((baseline_dir / f"fold_{fold}/drug_ids.json").read_text(encoding="utf-8"))
        saved_mean = np.load(baseline_dir / f"fold_{fold}/drug_mean.npy", mmap_mode="r", allow_pickle=False)
        if saved_ids != expected_drugs or len(saved_ids) != len(set(saved_ids)):
            raise ValueError(f"fold {fold} training drug identity axis mismatch")
        if saved_mean.shape != (len(saved_ids), N_GENES) or saved_mean.dtype != np.dtype("float32"):
            raise ValueError(f"fold {fold} training-side baseline matrix has wrong shape/dtype")
        train_values = np.asarray(delta[train_positions], dtype=np.float32)
        grouped_train = train_rows.groupby(train_rows["pert_id"].astype(str), sort=True).indices
        index_by_drug = {drug: i for i, drug in enumerate(expected_drugs)}
        recomputed = np.empty_like(saved_mean)
        for drug, row_positions in grouped_train.items():
            recomputed[index_by_drug[drug]] = train_values[np.asarray(row_positions, dtype=np.int64)].mean(
                axis=0, dtype=np.float64)
        mean_error = float(np.max(np.abs(recomputed.astype(np.float64)
                                         - np.asarray(saved_mean, dtype=np.float64))))
        if mean_error > 2e-6:
            raise ValueError(f"fold {fold} full train-side drug mean mismatch: {mean_error}")
        full_mean_checks.append({"fold": fold, "n_training_conditions": len(train_positions),
                                 "n_training_drugs": len(saved_ids),
                                 "n_compared_values": int(saved_mean.size),
                                 "max_absolute_error": mean_error,
                                 "drug_ids_sha256": sha256(baseline_dir / f"fold_{fold}/drug_ids.json"),
                                 "drug_mean_sha256": sha256(baseline_dir / f"fold_{fold}/drug_mean.npy")})
        observed_pos = condition_to_position.reindex(fold_panel["condition_id"]).to_numpy()
        if pd.isna(observed_pos).any():
            raise ValueError(f"fold {fold} fixed panel condition key is missing")
        observed_rows = np.asarray(delta[observed_pos.astype(np.int64)], dtype=np.float32)
        group_index = fold_panel.groupby(["cell_id", "pert_id"], sort=True).indices
        group_rows = []
        measured_vectors = np.empty((len(group_index), N_GENES), dtype=np.float32)
        for i, ((cell, drug), locations) in enumerate(group_index.items()):
            locs = np.asarray(locations, dtype=np.int64)
            measured_vectors[i] = observed_rows[locs].mean(axis=0, dtype=np.float64)
            group_rows.append({"cell_id": str(cell), "pert_id": str(drug),
                               "n_conditions": int(len(locs))})
        grouped = pd.DataFrame(group_rows)
        group_drugs = grouped["pert_id"].astype(str).to_numpy()
        mean_positions = {drug: idx for idx, drug in enumerate(saved_ids)}
        if set(group_drugs) - set(mean_positions):
            raise ValueError(f"fold {fold} panel contains drugs absent from training means")
        train_vectors = np.stack([saved_mean[mean_positions[drug]] for drug in group_drugs]).astype(np.float32)

        for signature, (up, down, effects) in signatures.items():
            measured_rank = np.asarray([_expected_reversal(row, genes, up, down)
                                        for row in measured_vectors], dtype=float)
            measured_cos = _expected_cosine(measured_vectors, effects)
            trained_rank = np.asarray([_expected_reversal(row, genes, up, down)
                                       for row in train_vectors], dtype=float)
            trained_cos = _expected_cosine(train_vectors, effects)
            base = grouped.copy()
            base.insert(0, "fold", fold)
            base.insert(0, "signature", signature)
            base["measured_reversal"] = measured_rank
            base["measured_negative_cosine"] = measured_cos
            for arm in ("train_drug_mean", "zero"):
                frame = base.copy()
                frame.insert(1, "baseline", arm)
                if arm == "train_drug_mean":
                    frame["predicted_reversal"] = trained_rank
                    frame["predicted_negative_cosine"] = trained_cos
                    frame["prediction_status"] = None
                else:
                    frame["predicted_reversal"] = np.nan
                    frame["predicted_negative_cosine"] = np.nan
                    frame["prediction_status"] = "ZERO_VECTOR_RANK_AND_COSINE_UNDEFINED"
                expected_drug_rows.extend(frame.to_dict(orient="records"))

                for cell, row_positions in grouped.groupby("cell_id", sort=True).indices.items():
                    row_pos = np.asarray(row_positions, dtype=np.int64)
                    truth_rank = measured_rank[row_pos]
                    truth_cos = measured_cos[row_pos]
                    cell_ids = grouped.iloc[row_pos]["pert_id"].astype(str).to_numpy()
                    n_drugs = len(row_pos)
                    if arm == "train_drug_mean":
                        prediction = trained_rank[row_pos]
                        pred_cos = trained_cos[row_pos]
                        reason = None
                        if n_drugs < 5:
                            reason, rho = "FEWER_THAN_FIVE_DRUGS", float("nan")
                        elif np.ptp(truth_rank) == 0:
                            reason, rho = "CONSTANT_MEASURED_REVERSAL", float("nan")
                        elif np.ptp(prediction) == 0:
                            reason, rho = "CONSTANT_PREDICTED_REVERSAL", float("nan")
                        else:
                            rho = _rho(truth_rank, prediction)
                            if not np.isfinite(rho):
                                reason = "UNDEFINED_CORRELATION"
                        valid_cos = np.isfinite(truth_cos) & np.isfinite(pred_cos)
                        if (int(valid_cos.sum()) >= 5
                                and np.ptp(truth_cos[valid_cos]) > 0
                                and np.ptp(pred_cos[valid_cos]) > 0):
                            cos_rho = _rho(truth_cos[valid_cos], pred_cos[valid_cos])
                            cos_reason = None if np.isfinite(cos_rho) else "UNDEFINED_CORRELATION"
                        else:
                            cos_rho = float("nan")
                            cos_reason = "INSUFFICIENT_OR_CONSTANT_NEGATIVE_COSINE"
                        overlap, loss, top_n = _top_metrics(cell_ids, truth_rank, prediction)
                        tie = "score descending then pert_id ascending"
                    else:
                        rho, reason = float("nan"), "ZERO_VECTOR_HAS_NO_DRUG_RANK"
                        cos_rho, cos_reason = float("nan"), "ZERO_VECTOR_COSINE_UNDEFINED"
                        valid_cos = np.zeros(n_drugs, dtype=bool)
                        overlap, loss, top_n = _top_metrics(cell_ids, truth_rank, None)
                        tie = "all zero predictions tied; pert_id ascending"
                    expected_cell_rows.append({
                        "signature": signature, "baseline": arm, "fold": fold,
                        "cell_id": str(cell), "n_drugs": n_drugs,
                        "n_measured_conditions": int(grouped.iloc[row_pos]["n_conditions"].sum()),
                        "spearman": rho, "spearman_reason": reason,
                        "negative_cosine_spearman": cos_rho,
                        "negative_cosine_reason": cos_reason,
                        "n_negative_cosine_drugs": int(valid_cos.sum()),
                        "top_decile_overlap": overlap, "measured_selection_loss": loss,
                        "top_decile_count": top_n, "tie_break": tie,
                    })
        fold_cell_counts.append({"fold": fold, "n_cells": int(fold_panel["cell_id"].nunique()),
                                 "n_conditions": len(fold_panel),
                                 "n_cell_drug_groups": len(grouped)})

    actual_drugs = pd.read_parquet(run_dir / "drug_scores.parquet")
    actual_cells = pd.read_parquet(run_dir / "cell_scores.parquet")
    expected_drugs_frame = pd.DataFrame(expected_drug_rows)
    expected_cells_frame = pd.DataFrame(expected_cell_rows)
    drug_error = _compare_rows(
        actual_drugs, expected_drugs_frame,
        ["signature", "baseline", "fold", "cell_id", "pert_id"],
        ["n_conditions", "measured_reversal", "measured_negative_cosine",
         "predicted_reversal", "predicted_negative_cosine"], ["prediction_status"])
    cell_error = _compare_rows(
        actual_cells, expected_cells_frame,
        ["signature", "baseline", "fold", "cell_id"],
        ["n_drugs", "n_measured_conditions", "spearman", "negative_cosine_spearman",
         "n_negative_cosine_drugs", "top_decile_overlap", "measured_selection_loss",
         "top_decile_count"],
        ["spearman_reason", "negative_cosine_reason", "tie_break"])
    summary_error = _dict_close(summary.get("summary"), _make_summary(expected_cells_frame))
    if len(actual_cells) != 2 * len(SIGNATURES) * 30:
        raise ValueError("expected five-fold, 30-cell-per-signature baseline cell rows")
    for (signature, baseline), frame in actual_cells.groupby(["signature", "baseline"]):
        if baseline == "zero" and (frame["spearman"].notna().any()
                                    or frame["negative_cosine_spearman"].notna().any()):
            raise ValueError(f"zero baseline undefined correlations were replaced for {signature}")

    return {
        "experiment_id": "EXP-038", "run_id": run_dir.name, "status": "PASS",
        "scope": "independent CPU recomputation of preregistered train_drug_mean and zero baselines",
        "not_b2_main_complete": True, "holm_family_closed": False,
        "runtime": {"python": sys.version, "python_executable": sys.executable,
                    "platform": platform.platform(), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "torch_imported": "torch" in sys.modules},
        "run_summary_sha256": sha256(summary_path),
        "run_receipt_sha256": sha256(receipt_path),
        "verified_input_sha256": actual_hashes,
        "verified_drug_score_rows": len(expected_drug_rows),
        "verified_cell_score_rows": len(expected_cell_rows),
        "max_abs_recalculation_error": max(drug_error, cell_error),
        "primary_summary_max_abs_error": summary_error,
        "full_train_side_mean_recalculation": full_mean_checks,
        "fold_panel_counts": fold_cell_counts,
        "method": "independent grouped-prefix reversal ES, direct cosine recomputation, key-aligned cell/drug aggregation, and cell bootstrap",
        "verification_elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir / "independent_verification.json"
    failure = args.run_dir / "verification_failure.json"
    try:
        result = verify(args.root, args.run_dir)
    except Exception as exc:
        if args.run_dir.exists() and not failure.exists():
            failure.write_text(json.dumps({"status": "FAILED_INDEPENDENT_VERIFICATION",
                                           "error_type": type(exc).__name__, "error": str(exc)},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
