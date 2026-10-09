"""Run the separately authorized EXP-038 baseline-only CPU evaluation.

This entrypoint evaluates only the already registered ``train_drug_mean`` and
``zero`` baselines. It does not consume model-fold receipts, train, select a
model, or make a B2/Holm-family claim.
"""

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

from drug_screen.evaluation.reversal import reversal_score


EXPECTED = {
    "split": "e474cede3d7474ad87c5685850726e9a8db813eabbd775e552fd09899df753d7",
    "condition_manifest": "0947fdf55d4f016928258d9a372c699efe12bdbdf6cf404d46253bcdd6f3e68e",
    "condition_delta": "29317d50b44933452126ddd213017709d54b37f6c0dcc2f403cf5ac5ff8112fb",
    "gene_axis": "8cc8d6ade494209a9f323a94450e02c3a74d1ea3ae61a75ed4c084a9ab80502c",
    "panel": "43976344daf4a043d6e96c1bc2b363428b4a2478830e7f5cb44408aa704d932f",
    "gse_signature_manifest": "7515dff33c4ac3689a19273a21e6b5d3c184c62d7c8bb67772ac9a7e8c8c3782",
}
N_GENES = 978
N_FOLDS = 5
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260925
SCORE_METRICS = (
    "spearman",
    "negative_cosine_spearman",
    "top_decile_overlap",
    "measured_selection_loss",
)
SIGNATURES = {
    "GSE74602_v2": Path("data/processed/dual_route/EXP-038/gse74602_crc_signature_v2"),
    "TCGA_COAD_READ_v1": Path("data/processed/dual_route/EXP-038/tcga_crc_signature_v1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_digest(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} digest mismatch: expected={expected}, actual={actual}")
    return actual


def _linux_io_snapshot() -> dict[str, int] | None:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in {"rchar", "read_bytes", "syscr"}:
            values[key] = int(value.strip())
    return values


def _sha_record(path: Path, label: str, hashed: dict[str, int]) -> str:
    actual = sha256(path)
    hashed[label] = path.stat().st_size
    return actual


def _validate_runtime() -> None:
    if not sys.platform.startswith("linux") or "microsoft" not in platform.release().lower():
        raise RuntimeError("formal EXP-038 processing must run in WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("Python must run in the approved drugscreening-gpu Conda environment")
    prefix = os.environ.get("CONDA_PREFIX", "")
    if not prefix.endswith("/envs/drugscreening-gpu"):
        raise RuntimeError(f"unexpected Conda prefix: {prefix!r}")
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only evaluator must not import PyTorch")


def _verify_signature(signature_dir: Path, signature_name: str,
                      genes: list[str]) -> tuple[list[str], list[str], np.ndarray, dict[str, Any]]:
    manifest_path = signature_dir / "manifest.json"
    manifest = read_json(manifest_path)
    expected_status = ("SIGNATURE_READY_FOR_DEVELOPMENT" if signature_name == "GSE74602_v2"
                       else "SIGNATURE_READY_FOR_CROSS_SIGNATURE_VALIDATION")
    if manifest.get("experiment_id") != "EXP-038" or manifest.get("status") != expected_status:
        raise ValueError(f"{signature_name} signature status/version is not accepted")
    file_hashes = manifest.get("files_sha256", {})
    required = ("up_genes.json", "down_genes.json", "landmark_statistics.parquet")
    for name in required:
        expected = file_hashes.get(name)
        if not isinstance(expected, str) or sha256(signature_dir / name) != expected:
            raise ValueError(f"{signature_name} signature file hash mismatch: {name}")
    for name, expected in file_hashes.items():
        if sha256(signature_dir / name) != expected:
            raise ValueError(f"{signature_name} manifest file hash mismatch: {name}")
    up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if (len(up) < 20 or len(down) < 20 or len(up) > 100 or len(down) > 100
            or len(set(up)) != len(up) or len(set(down)) != len(down)
            or set(up) & set(down) or not set(up + down) <= set(genes)):
        raise ValueError(f"{signature_name} signature directions do not match the fixed gene axis")
    landmark = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    if landmark["pr_gene_id"].astype(str).duplicated().any():
        raise ValueError(f"{signature_name} contains duplicate landmark gene IDs")
    effects = landmark.assign(pr_gene_id=landmark["pr_gene_id"].astype(str)).set_index(
        "pr_gene_id")["mean_paired_log2fc"].reindex(genes).to_numpy(dtype=np.float64)
    if int(np.isfinite(effects).sum()) < 20:
        raise ValueError(f"{signature_name} continuous signature has fewer than 20 mapped genes")
    return up, down, effects, {
        "directory": signature_dir.as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "status": manifest["status"],
        "n_up": len(up),
        "n_down": len(down),
        "n_finite_continuous_effects_on_lincs_axis": int(np.isfinite(effects).sum()),
        "file_sha256": {name: sha256(signature_dir / name) for name in file_hashes},
    }


def _negative_cosine(vectors: np.ndarray, effects: np.ndarray) -> np.ndarray:
    valid_effect = np.isfinite(effects)
    if int(valid_effect.sum()) < 20:
        raise ValueError("continuous disease effect has fewer than 20 mapped genes")
    values = np.asarray(vectors[:, valid_effect], dtype=np.float64)
    reference = effects[valid_effect]
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(reference)
    scores = np.full(len(values), np.nan, dtype=np.float64)
    good = denominator > 0
    scores[good] = -(values[good] @ reference) / denominator[good]
    return scores


def _rank_reversal(vectors: np.ndarray, genes: list[str], up: list[str],
                   down: list[str]) -> np.ndarray:
    return np.asarray([reversal_score(row, genes, up, down) for row in vectors], dtype=np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(x, y).statistic
    return float(result) if np.isfinite(result) else float("nan")


def _top_metrics(drugs: np.ndarray, measured: np.ndarray,
                 predicted: np.ndarray | None) -> tuple[float, float, int]:
    if len(drugs) < 5 or len(set(drugs.astype(str))) != len(drugs):
        raise ValueError("top-decile metrics require at least five unique drug IDs")
    count = max(1, int(np.ceil(len(drugs) * 0.1)))
    ids = drugs.astype(str)
    observed_order = np.lexsort((ids, -measured))
    if predicted is None:
        predicted_order = np.argsort(ids, kind="stable")
    else:
        predicted_order = np.lexsort((ids, -predicted))
    measured_top = observed_order[:count]
    predicted_top = predicted_order[:count]
    overlap = len(set(measured_top).intersection(predicted_top)) / count
    loss = float(measured[measured_top].mean() - measured[predicted_top].mean())
    return float(overlap), loss, count


def _aggregate_vectors(panel: pd.DataFrame, delta: np.ndarray,
                       *, expected_genes: int) -> tuple[pd.DataFrame, np.ndarray]:
    condition_ids = panel["condition_id"].to_numpy(dtype=np.int64)
    if len(condition_ids) == 0 or len(np.unique(condition_ids)) != len(condition_ids):
        raise ValueError("scoring panel has no conditions or repeated condition IDs")
    selected = np.asarray(delta[condition_ids], dtype=np.float32)
    if selected.shape != (len(panel), expected_genes) or not np.isfinite(selected).all():
        raise ValueError("scored measured delta is non-finite or misaligned")
    groups = panel.groupby(["cell_id", "pert_id"], sort=True).indices
    key_rows = []
    vectors = np.empty((len(groups), expected_genes), dtype=np.float32)
    for row_index, ((cell, drug), positions) in enumerate(groups.items()):
        positions = np.asarray(positions, dtype=np.int64)
        vectors[row_index] = selected[positions].mean(axis=0, dtype=np.float64)
        key_rows.append({"cell_id": str(cell), "pert_id": str(drug),
                         "n_conditions": int(len(positions))})
    return pd.DataFrame(key_rows), vectors


def _score_signature(signature_name: str, panel: pd.DataFrame, delta: np.ndarray,
                     genes: list[str], up: list[str], down: list[str], effects: np.ndarray,
                     baseline_dir: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    drug_rows: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    fold_mean_summary = {}
    for fold in range(N_FOLDS):
        fold_panel = panel.loc[panel["fold"].eq(fold)].copy().reset_index(drop=True)
        groups, observed_vectors = _aggregate_vectors(fold_panel, delta, expected_genes=len(genes))
        observed_reversal = _rank_reversal(observed_vectors, genes, up, down)
        observed_cosine = _negative_cosine(observed_vectors, effects)
        fold_dir = baseline_dir / f"fold_{fold}"
        ids = json.loads((fold_dir / "drug_ids.json").read_text(encoding="utf-8"))
        mean = np.load(fold_dir / "drug_mean.npy", mmap_mode="r", allow_pickle=False)
        positions = {str(drug): index for index, drug in enumerate(ids)}
        if not set(groups["pert_id"]) <= set(positions):
            raise ValueError(f"{signature_name} fold {fold}: panel drug lacks a training-side mean")
        predicted_vectors = np.stack([mean[positions[drug]] for drug in groups["pert_id"]]).astype(np.float32)
        predicted_reversal = _rank_reversal(predicted_vectors, genes, up, down)
        predicted_cosine = _negative_cosine(predicted_vectors, effects)
        base = groups.copy()
        base.insert(0, "fold", fold)
        base.insert(0, "signature", signature_name)
        base["measured_reversal"] = observed_reversal
        base["measured_negative_cosine"] = observed_cosine
        for arm in ("train_drug_mean", "zero"):
            current = base.copy()
            current.insert(1, "baseline", arm)
            if arm == "train_drug_mean":
                current["predicted_reversal"] = predicted_reversal
                current["predicted_negative_cosine"] = predicted_cosine
            else:
                current["predicted_reversal"] = np.nan
                current["predicted_negative_cosine"] = np.nan
                current["prediction_status"] = "ZERO_VECTOR_RANK_AND_COSINE_UNDEFINED"
            drug_rows.append(current)
        for cell, row_positions in groups.groupby("cell_id", sort=True).indices.items():
            row_positions = np.asarray(row_positions, dtype=np.int64)
            cell_measured = observed_reversal[row_positions]
            cell_predicted = predicted_reversal[row_positions]
            cell_measured_cos = observed_cosine[row_positions]
            cell_predicted_cos = predicted_cosine[row_positions]
            drug_ids = groups.iloc[row_positions]["pert_id"].astype(str).to_numpy()
            n_drugs = int(len(row_positions))
            reason = None
            if n_drugs < 5:
                reason = "FEWER_THAN_FIVE_DRUGS"
                main_spearman = float("nan")
            elif np.ptp(cell_measured) == 0:
                reason = "CONSTANT_MEASURED_REVERSAL"
                main_spearman = float("nan")
            elif np.ptp(cell_predicted) == 0:
                reason = "CONSTANT_PREDICTED_REVERSAL"
                main_spearman = float("nan")
            else:
                main_spearman = _spearman(cell_measured, cell_predicted)
                if not np.isfinite(main_spearman):
                    reason = "UNDEFINED_CORRELATION"
            cosine_valid = np.isfinite(cell_measured_cos) & np.isfinite(cell_predicted_cos)
            if int(cosine_valid.sum()) >= 5 and np.ptp(cell_measured_cos[cosine_valid]) > 0 \
                    and np.ptp(cell_predicted_cos[cosine_valid]) > 0:
                cosine_spearman = _spearman(cell_measured_cos[cosine_valid],
                                            cell_predicted_cos[cosine_valid])
                cosine_reason = None if np.isfinite(cosine_spearman) else "UNDEFINED_CORRELATION"
            else:
                cosine_spearman = float("nan")
                cosine_reason = "INSUFFICIENT_OR_CONSTANT_NEGATIVE_COSINE"
            overlap, loss, top_count = _top_metrics(drug_ids, cell_measured, cell_predicted)
            cell_rows.append({
                "signature": signature_name, "baseline": "train_drug_mean", "fold": fold,
                "cell_id": str(cell), "n_drugs": n_drugs,
                "n_measured_conditions": int(groups.iloc[row_positions]["n_conditions"].sum()),
                "spearman": main_spearman, "spearman_reason": reason,
                "negative_cosine_spearman": cosine_spearman,
                "negative_cosine_reason": cosine_reason,
                "n_negative_cosine_drugs": int(cosine_valid.sum()),
                "top_decile_overlap": overlap, "measured_selection_loss": loss,
                "top_decile_count": top_count,
                "tie_break": "score descending then pert_id ascending",
            })
            zero_overlap, zero_loss, zero_count = _top_metrics(drug_ids, cell_measured, None)
            cell_rows.append({
                "signature": signature_name, "baseline": "zero", "fold": fold,
                "cell_id": str(cell), "n_drugs": n_drugs,
                "n_measured_conditions": int(groups.iloc[row_positions]["n_conditions"].sum()),
                "spearman": float("nan"),
                "spearman_reason": "ZERO_VECTOR_HAS_NO_DRUG_RANK",
                "negative_cosine_spearman": float("nan"),
                "negative_cosine_reason": "ZERO_VECTOR_COSINE_UNDEFINED",
                "n_negative_cosine_drugs": 0,
                "top_decile_overlap": zero_overlap, "measured_selection_loss": zero_loss,
                "top_decile_count": zero_count,
                "tie_break": "all zero predictions tied; pert_id ascending",
            })
        fold_mean_summary[str(fold)] = {
            "n_conditions": int(len(fold_panel)),
            "n_cells": int(groups["cell_id"].nunique()),
            "n_cell_drug_groups": int(len(groups)),
        }
    return pd.concat(drug_rows, ignore_index=True), pd.DataFrame(cell_rows)


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> dict[str, Any] | None:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        samples[index] = valid[rng.integers(0, len(valid), len(valid))].mean()
    return {
        "mean": float(valid.mean()), "n_cells": int(len(valid)),
        "percentile_95_ci": [float(np.quantile(samples, 0.025)),
                              float(np.quantile(samples, 0.975))],
        "bootstrap_resamples": N_BOOTSTRAP,
        "unit": "cell",
        "interpretation": "descriptive uncertainty interval; not a hypothesis test or Holm-family result",
    }


def _summary_for_scores(cells: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (signature, baseline), frame in cells.groupby(["signature", "baseline"], sort=True):
        overall = {}
        for metric_index, metric in enumerate(SCORE_METRICS):
            overall[metric] = _bootstrap_interval(
                frame[metric].to_numpy(dtype=np.float64),
                seed=BOOTSTRAP_SEED + metric_index,
            )
        fold_rows = []
        for fold, part in frame.groupby("fold", sort=True):
            fold_summary = {"fold": int(fold), "n_cells": int(part["cell_id"].nunique()),
                            "n_conditions": int(part["n_measured_conditions"].sum()),
                            "n_cell_drug_groups": int(part["n_drugs"].sum())}
            for metric in SCORE_METRICS:
                valid = part[metric].to_numpy(dtype=np.float64)
                valid = valid[np.isfinite(valid)]
                fold_summary[f"{metric}_mean"] = float(valid.mean()) if len(valid) else None
                fold_summary[f"{metric}_defined_cells"] = int(len(valid))
            fold_summary["undefined_primary_reasons"] = {
                str(key): int(value) for key, value in
                part["spearman_reason"].dropna().value_counts().items()
            }
            fold_summary["undefined_negative_cosine_reasons"] = {
                str(key): int(value) for key, value in
                part["negative_cosine_reason"].dropna().value_counts().items()
            }
            fold_rows.append(fold_summary)
        result[f"{signature}/{baseline}"] = {
            "n_cells": int(frame["cell_id"].nunique()),
            "overall_cell_equal_metrics": overall,
            "folds": fold_rows,
            "primary_spearman_comparable": baseline == "train_drug_mean",
            "notes": (["Zero predictions have no between-drug rank; primary Spearman and negative cosine are undefined.",
                       "Top-decile metrics use ascending pert_id as the tie-break among all tied zero predictions."]
                      if baseline == "zero" else []),
        }
    return result


def _prepare(root: Path, hashed: dict[str, int]) -> dict[str, Any]:
    condition_dir = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"
    contract_dir = root / "data/processed/lincs/GSE92742/continuous_contract_v2"
    split_path = root / "data/splits/lincs_cell_splits_20260922_v2.json"
    panel_dir = root / "artifacts/experiments/EXP-038/20260923_fixed_panel_v1"
    baseline_dir = root / "artifacts/experiments/EXP-037/20260922_baselines_v2"
    condition_manifest_path = condition_dir / "manifest.json"
    condition_path = condition_dir / "conditions.parquet"
    delta_path = condition_dir / "condition_delta.npy"
    genes_path = contract_dir / "genes.json"
    split_sha = _require_digest(split_path, EXPECTED["split"], "frozen split")
    condition_manifest_sha = _require_digest(condition_manifest_path,
                                             EXPECTED["condition_manifest"], "condition manifest")
    delta_sha = _require_digest(delta_path, EXPECTED["condition_delta"], "condition delta")
    genes_sha = _require_digest(genes_path, EXPECTED["gene_axis"], "978-gene axis")
    panel_sha = _require_digest(panel_dir / "panel_conditions.parquet",
                                EXPECTED["panel"], "frozen fixed panel")
    panel_receipt = read_json(panel_dir / "panel_audit.json")
    if (panel_receipt.get("source_sha256", {}).get("split") != split_sha
            or panel_receipt.get("source_sha256", {}).get("condition_manifest") != condition_manifest_sha
            or panel_receipt.get("source_sha256", {}).get("condition_matrix") != delta_sha
            or panel_receipt.get("source_sha256", {}).get("gene_axis") != genes_sha
            or panel_receipt.get("panel_sha256") != panel_sha):
        raise ValueError("fixed panel receipt does not bind to current split/condition/gene inputs")
    condition_manifest = read_json(condition_manifest_path)
    if condition_manifest.get("status") != "PASS":
        raise ValueError("continuous condition manifest is not PASS")
    if condition_manifest.get("files_sha256", {}).get("conditions.parquet") != sha256(condition_path):
        raise ValueError("condition key parquet differs from its manifest")
    if condition_manifest.get("files_sha256", {}).get("condition_delta.npy") != delta_sha:
        raise ValueError("condition delta differs from its manifest")
    for name, path in (("split", split_path), ("condition_manifest", condition_manifest_path),
                       ("condition_keys", condition_path), ("condition_delta", delta_path),
                       ("gene_axis", genes_path),
                       ("panel_conditions", panel_dir / "panel_conditions.parquet"),
                       ("panel_audit", panel_dir / "panel_audit.json")):
        hashed[name] = path.stat().st_size

    genes = json.loads(genes_path.read_text(encoding="utf-8"))
    if not isinstance(genes, list) or len(genes) != N_GENES or len(set(genes)) != N_GENES:
        raise ValueError("frozen output gene axis must contain 978 unique genes")
    keys = pd.read_parquet(condition_path)
    delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if (len(keys) != int(condition_manifest.get("n_conditions", -1))
            or keys["condition_id"].duplicated().any()
            or not np.array_equal(pd.to_numeric(keys["condition_id"], errors="raise").to_numpy(np.int64),
                                  np.arange(len(keys)))
            or delta.shape != (len(keys), N_GENES)
            or delta.dtype != np.dtype("float32")):
        raise ValueError("condition IDs, float32 delta rows, or 978-gene axis disagree")
    for column in ("condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
                   "pert_time", "pert_time_unit"):
        if column not in keys.columns:
            raise ValueError(f"condition table missing key column {column}")
    if not keys["pert_dose_unit"].astype(str).eq("um").all() \
            or not keys["pert_time_unit"].astype(str).eq("h").all():
        raise ValueError("frozen condition dose/time units differ from panel contract")
    split = read_json(split_path)
    if len(split.get("folds", [])) != N_FOLDS:
        raise ValueError("frozen split does not have exactly five outer folds")
    cell_axis = set(keys["cell_id"].astype(str))
    test_owner: dict[str, int] = {}
    for fold_index, fold in enumerate(split["folds"]):
        train = set(map(str, fold["outer_train_ids"]))
        test = set(map(str, fold["outer_test_ids"]))
        if train & test or train | test != cell_axis:
            raise ValueError(f"frozen fold {fold_index} has overlapping/incomplete cell roles")
        for cell in test:
            if cell in test_owner:
                raise ValueError(f"cell {cell} appears as test in multiple frozen folds")
            test_owner[cell] = fold_index
    if set(test_owner) != cell_axis:
        raise ValueError("frozen outer test folds do not cover all condition cells exactly once")

    dose = pd.to_numeric(keys["pert_dose"], errors="raise").to_numpy(dtype=np.float64)
    time_hours = pd.to_numeric(keys["pert_time"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(dose).all() or not np.isfinite(time_hours).all():
        raise ValueError("condition exposure contains non-finite dose/time")
    fixed = np.isclose(dose, 10.0, rtol=0, atol=1e-6) & (time_hours == 24.0)
    expected_parts = []
    fold_reports = []
    for fold_index, fold in enumerate(split["folds"]):
        train_ids = set(map(str, fold["outer_train_ids"]))
        test_ids = set(map(str, fold["outer_test_ids"]))
        train_mask = keys["cell_id"].astype(str).isin(train_ids).to_numpy()
        seen_drugs = set(keys.loc[train_mask, "pert_id"].astype(str))
        fixed_test_mask = fixed & keys["cell_id"].astype(str).isin(test_ids).to_numpy()
        test = keys.loc[fixed_test_mask].copy()
        before_count = int(len(test))
        test = test.loc[test["pert_id"].astype(str).isin(seen_drugs)].copy()
        if not set(test["cell_id"].astype(str)) <= test_ids:
            raise ValueError(f"fold {fold_index} panel contains a seen cell")
        if not set(test["pert_id"].astype(str)) <= seen_drugs:
            raise ValueError(f"fold {fold_index} panel includes a drug absent from training cells")
        counts = test.groupby(test["cell_id"].astype(str), sort=True)["pert_id"].nunique()
        eligible_cells = set(counts.loc[counts.ge(5)].index)
        status = np.where(test["cell_id"].astype(str).isin(eligible_cells),
                          "ELIGIBLE", "FEWER_THAN_FIVE_DRUGS")
        test.insert(0, "fold", fold_index)
        test["panel_status"] = status
        expected_parts.append(test)
        fold_reports.append({
            "fold": fold_index, "fixed_test_conditions": before_count,
            "seen_drug_conditions": int(len(test)),
            "eligible_conditions": int(np.sum(status == "ELIGIBLE")),
            "eligible_cells": len(eligible_cells),
            "eligible_cell_drug_groups": int(test.loc[test["panel_status"].eq("ELIGIBLE")]
                                              .groupby("cell_id")["pert_id"].nunique().sum()),
        })
    expected_panel = pd.concat(expected_parts, ignore_index=True)
    if expected_panel["condition_id"].duplicated().any():
        raise ValueError("a fixed condition is assigned to multiple held-out folds")
    actual_panel = pd.read_parquet(panel_dir / "panel_conditions.parquet")
    if actual_panel["condition_id"].duplicated().any():
        raise ValueError("frozen panel has duplicate condition IDs")
    compare_columns = ["condition_id", "fold", "cell_id", "pert_id", "pert_dose",
                       "pert_dose_unit", "pert_time", "pert_time_unit", "panel_status"]
    for label, frame in (("reconstructed", expected_panel), ("frozen", actual_panel)):
        missing = [column for column in compare_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} panel missing column {missing[0]}")
    left = expected_panel[compare_columns].sort_values("condition_id").reset_index(drop=True)
    right = actual_panel[compare_columns].sort_values("condition_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    eligible = actual_panel.loc[actual_panel["panel_status"].eq("ELIGIBLE")].copy()
    cell_drug_counts = eligible.groupby("cell_id")["pert_id"].nunique()
    if len(eligible) != 40642 or len(cell_drug_counts) != 30 or not cell_drug_counts.ge(5).all():
        raise ValueError("reconstructed fixed panel fails the frozen coverage totals")
    # Exact set equality above proves every eligible replicate condition is represented once.
    if (panel_receipt.get("n_fixed_conditions") != 40761
            or panel_receipt.get("n_seen_drug_heldout_conditions") != 40642
            or panel_receipt.get("n_eligible_conditions") != len(eligible)):
        raise ValueError("panel audit totals differ from independently reconstructed panel")

    baseline_summary_path = baseline_dir / "run_summary.json"
    baseline_verification_path = baseline_dir / "independent_verification.json"
    baseline_summary = read_json(baseline_summary_path)
    baseline_verification = read_json(baseline_verification_path)
    if (baseline_summary.get("status") != "BASELINES_COMPLETE_PENDING_REVIEW"
            or baseline_summary.get("run_id") != "20260922_baselines_v2"
            or baseline_summary.get("data", {}).get("condition_manifest_sha256") != condition_manifest_sha
            or baseline_summary.get("data", {}).get("condition_matrix_sha256") != delta_sha
            or baseline_summary.get("data", {}).get("split_sha256") != split_sha
            or baseline_verification.get("status") != "PASS"
            or baseline_verification.get("verified_folds") != N_FOLDS):
        raise ValueError("EXP-037 baseline summary or existing verification is not bound to these inputs")
    model_coverage = {(int(row["fold"]), str(row["model"]))
                      for row in baseline_summary.get("fold_results", [])}
    required_coverage = {(fold, arm) for fold in range(N_FOLDS)
                         for arm in ("zero", "train_drug_mean")}
    if not required_coverage <= model_coverage:
        raise ValueError("EXP-037 baseline summary lacks one or more required fold baselines")
    hashed["baseline_summary"] = baseline_summary_path.stat().st_size
    hashed["baseline_existing_verification"] = baseline_verification_path.stat().st_size
    baseline_inputs = {
        "summary_sha256": sha256(baseline_summary_path),
        "existing_verification_sha256": sha256(baseline_verification_path),
        "folds": [],
    }
    mean_reconstruction = []
    for fold_index, fold in enumerate(split["folds"]):
        train_cells = set(map(str, fold["outer_train_ids"]))
        train_positions = np.flatnonzero(keys["cell_id"].astype(str).isin(train_cells).to_numpy())
        train_keys = keys.iloc[train_positions].reset_index(drop=True)
        grouped = train_keys.groupby(train_keys["pert_id"].astype(str), sort=True).indices
        expected_ids = sorted(grouped)
        fold_dir = baseline_dir / f"fold_{fold_index}"
        ids_path = fold_dir / "drug_ids.json"
        mean_path = fold_dir / "drug_mean.npy"
        drug_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        mean = np.load(mean_path, mmap_mode="r", allow_pickle=False)
        if (drug_ids != expected_ids or len(drug_ids) != len(set(drug_ids))
                or mean.shape != (len(drug_ids), N_GENES)
                or mean.dtype != np.dtype("float32") or not np.isfinite(mean).all()):
            raise ValueError(f"fold {fold_index} saved training drug means have invalid IDs/shape/values")
        train_values = np.asarray(delta[train_positions], dtype=np.float32)
        if not np.isfinite(train_values).all():
            raise ValueError(f"fold {fold_index} training-side delta contains non-finite values")
        reconstructed = np.empty_like(mean)
        expected_position = {drug: index for index, drug in enumerate(expected_ids)}
        for drug, positions in grouped.items():
            reconstructed[expected_position[drug]] = train_values[np.asarray(positions, dtype=np.int64)].mean(
                axis=0, dtype=np.float64)
        max_abs_error = float(np.max(np.abs(reconstructed - np.asarray(mean))))
        exact = bool(np.array_equal(reconstructed, np.asarray(mean)))
        if not exact and max_abs_error > 2e-6:
            raise ValueError(f"fold {fold_index} complete train-side mean reconstruction failed: {max_abs_error}")
        ids_sha = sha256(ids_path)
        mean_sha = sha256(mean_path)
        baseline_inputs["folds"].append({
            "fold": fold_index, "drug_ids_sha256": ids_sha,
            "drug_mean_sha256": mean_sha,
            "n_train_conditions": int(len(train_positions)),
            "n_training_drugs": len(drug_ids),
        })
        mean_reconstruction.append({
            "fold": fold_index, "n_training_conditions_recomputed": int(len(train_positions)),
            "n_unique_training_drugs": len(drug_ids), "n_values_compared": int(mean.size),
            "max_absolute_error": max_abs_error, "bitwise_equal": exact,
            "drug_ids_sha256": ids_sha, "saved_drug_mean_sha256": mean_sha,
        })
        hashed[f"baseline_fold_{fold_index}_drug_ids"] = ids_path.stat().st_size
        hashed[f"baseline_fold_{fold_index}_drug_mean"] = mean_path.stat().st_size
    for name, signature_rel in SIGNATURES.items():
        path = root / signature_rel
        manifest_path = path / "manifest.json"
        expected_manifest = EXPECTED["gse_signature_manifest"] if name == "GSE74602_v2" else None
        if expected_manifest:
            _require_digest(manifest_path, expected_manifest, f"{name} signature manifest")
    signatures = {}
    signature_data = {}
    for name, signature_rel in SIGNATURES.items():
        signature_path = root / signature_rel
        signature_data[name] = _verify_signature(signature_path, name, genes)
        signatures[name] = signature_data[name][3]
        hashed[f"{name}_manifest"] = (signature_path / "manifest.json").stat().st_size
        for file_name in signature_data[name][3]["file_sha256"]:
            hashed[f"{name}_{file_name}"] = (signature_path / file_name).stat().st_size
    for name in ("baseline_summary", "baseline_existing_verification"):
        hashed.pop(name, None)
    return {
        "root": root, "condition_dir": condition_dir, "condition_path": condition_path,
        "delta_path": delta_path, "condition_manifest_sha256": condition_manifest_sha,
        "split_path": split_path, "split_sha256": split_sha, "split": split,
        "genes": genes, "gene_axis_sha256": genes_sha, "keys": keys, "delta": delta,
        "eligible_panel": eligible, "panel_sha256": panel_sha,
        "panel_receipt_sha256": sha256(panel_dir / "panel_audit.json"),
        "panel_folds": fold_reports, "baseline_dir": baseline_dir,
        "baseline_inputs": baseline_inputs, "mean_reconstruction": mean_reconstruction,
        "signatures": signatures, "signature_data": signature_data,
        "hashed_bytes": hashed,
        "counts": {"n_conditions": int(len(keys)), "n_genes": N_GENES,
                   "n_fixed_conditions": int(sum(x["fixed_test_conditions"] for x in fold_reports)),
                   "n_seen_drug_conditions": int(sum(x["seen_drug_conditions"] for x in fold_reports)),
                   "n_eligible_conditions": int(len(eligible)),
                   "n_eligible_cells": int(eligible["cell_id"].nunique()),
                   "n_eligible_cell_drug_groups": int(eligible.groupby(["cell_id", "pert_id"]).ngroups)},
    }


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    io_before = _linux_io_snapshot()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _validate_runtime()
    root = root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    allowed = (root / "artifacts/experiments/EXP-038").resolve(strict=True)
    if allowed not in output_dir.parents:
        raise ValueError("output directory must be a new child of artifacts/experiments/EXP-038")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing EXP-038 output: {output_dir}")
    output_dir.mkdir(parents=True)
    phase_times: dict[str, float] = {}
    hashed: dict[str, int] = {}
    try:
        phase = time.perf_counter()
        prepared = _prepare(root, hashed)
        phase_times["version_identity_panel_baseline_reconstruction_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        outputs = []
        for signature_name, data in prepared["signature_data"].items():
            drug, cell = _score_signature(
                signature_name, prepared["eligible_panel"], prepared["delta"],
                prepared["genes"], data[0], data[1], data[2],
                prepared["baseline_dir"], output_dir,
            )
            outputs.append(drug)
            outputs.append(cell)
        drug_scores = pd.concat([frame for frame in outputs if "pert_id" in frame.columns],
                                ignore_index=True)
        cell_scores = pd.concat([frame for frame in outputs if "cell_id" in frame.columns
                                 and "spearman_reason" in frame.columns], ignore_index=True)
        phase_times["signature_scoring_seconds"] = time.perf_counter() - phase
        phase = time.perf_counter()
        drug_path = output_dir / "drug_scores.parquet"
        cell_path = output_dir / "cell_scores.parquet"
        drug_scores.to_parquet(drug_path, index=False)
        cell_scores.to_parquet(cell_path, index=False)
        summary = {
            "experiment_id": "EXP-038",
            "run_id": output_dir.name,
            "status": "BASELINE_ONLY_SCORED_PENDING_INDEPENDENT_VERIFICATION",
            "scope": "CPU evaluation of registered train_drug_mean and zero baselines only",
            "not_a_model_result": True,
            "not_b2_main_complete": True,
            "holm_family_closed": False,
            "test_delta_role": "fixed scoring labels only; not used for training, selection, thresholding, or cell choice",
            "condition_aggregation": "average all fixed panel technical-repeat deltas within held-out cell and drug before scoring",
            "primary_metric": "cell-equal mean of within-cell Spearman between train_drug_mean and measured reversal ranks",
            "secondary_metric": "cell-equal within-cell Spearman of negative cosine scores",
            "top_decile": "ceil(10% of unique drugs), at least one; score descending then pert_id ascending",
            "zero_policy": "primary rank and negative cosine are undefined for zero vectors; retain NaN, use ascending pert_id only to break top-selection ties",
            "uncertainty": "2000-resample percentile bootstrap over cells per descriptive metric; no p values and no Holm-family inference",
            "baselines": ["train_drug_mean", "zero"],
            "panel_counts": prepared["counts"],
            "panel_fold_counts": prepared["panel_folds"],
            "signatures": prepared["signatures"],
            "baseline_input_receipt": prepared["baseline_inputs"],
            "train_side_mean_reconstruction": prepared["mean_reconstruction"],
            "summary": _summary_for_scores(cell_scores),
            "source_sha256": {
                "split": prepared["split_sha256"],
                "condition_manifest": prepared["condition_manifest_sha256"],
                "condition_delta": EXPECTED["condition_delta"],
                "gene_axis": prepared["gene_axis_sha256"],
                "panel_conditions": prepared["panel_sha256"],
                "panel_audit": prepared["panel_receipt_sha256"],
            },
            "files_sha256": {drug_path.name: sha256(drug_path), cell_path.name: sha256(cell_path)},
            "code_sha256": {"scripts/evaluation/exp038_baseline_only.py": sha256(Path(__file__))},
        }
        summary_path = output_dir / "result_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
        phase_times["write_output_seconds"] = time.perf_counter() - phase
        phase_times["total_elapsed_before_receipt_seconds"] = time.perf_counter() - started
        io_after = _linux_io_snapshot()
        receipt = {
            "experiment_id": "EXP-038", "run_id": output_dir.name,
            "status": summary["status"], "device": "CPU_ONLY",
            "runtime": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(),
                        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                        "conda_prefix": os.environ.get("CONDA_PREFIX"),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch_imported": "torch" in sys.modules,
                        "cpu_count": os.cpu_count()},
            "phase_seconds": phase_times,
            "wall_seconds": float(time.perf_counter() - started),
            "linux_process_io_before": io_before, "linux_process_io_after": io_after,
            "linux_process_io_delta": ({key: io_after.get(key, 0) - io_before.get(key, 0)
                                         for key in io_after.keys() & io_before.keys()}
                                        if io_before is not None and io_after is not None else None),
            "hashed_input_bytes": prepared["hashed_bytes"],
            "input_sha256": summary["source_sha256"],
            "baseline_summary_sha256": prepared["baseline_inputs"]["summary_sha256"],
            "baseline_independent_verification_sha256": prepared["baseline_inputs"]["existing_verification_sha256"],
            "baseline_fold_files": prepared["baseline_inputs"]["folds"],
            "reproducer": "CUDA_VISIBLE_DEVICES= PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/exp038_baseline_only.py --root /mnt/d/Code/DrugScreenLab --output-dir artifacts/experiments/EXP-038/20260925_baseline_only_v1",
            "verifier": "CUDA_VISIBLE_DEVICES= PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/verify_exp038_baseline_only.py --root /mnt/d/Code/DrugScreenLab --run-dir artifacts/experiments/EXP-038/20260925_baseline_only_v1",
        }
        receipt["files_sha256"] = {path.name: sha256(path) for path in
                                   (drug_path, cell_path, summary_path)}
        (output_dir / "run_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        return {"status": summary["status"], "run_id": output_dir.name,
                "panel_counts": prepared["counts"], "summary": summary["summary"],
                "wall_seconds": receipt["wall_seconds"]}
    except Exception as exc:
        failure = {
            "experiment_id": "EXP-038", "run_id": output_dir.name,
            "status": "FAILED_VALIDATION_OR_SCORING",
            "error_type": type(exc).__name__, "error": str(exc),
            "device": "CPU_ONLY", "elapsed_seconds": float(time.perf_counter() - started),
            "phase_seconds": phase_times,
        }
        path = output_dir / "failure_receipt.json"
        if not path.exists():
            path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
