#!/usr/bin/env python3
"""Analyze frozen XPert heldout transcription-response prediction quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SOURCE_RUN_ID = "20261001_xpert_independent_object_fit_v1"
SOURCE_ATTEMPT = "attempt_001"
DEFAULT_RUN_ID = "20261001_xpert_independent_response_diagnostics_v2"
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20261001


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result without a schema: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def finite_metric(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return None
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def object_summary(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "pearson_r", "spearman_rho", "predicted_to_observed_abs_delta_ratio",
        "signed_gene_agreement_nonzero_target",
    ]
    by_object_drug = rows.groupby(
        ["source_object_id", "full_inchikey"], sort=True
    )[numeric].mean().reset_index()
    grouped_keys = by_object_drug.groupby("source_object_id", sort=True)
    grouped_conditions = rows.groupby("source_object_id", sort=True)
    result = grouped_keys[numeric].mean().reset_index()
    result["n_conditions"] = grouped_conditions["condition_id"].nunique().to_numpy()
    result["n_object_drug_keys"] = grouped_keys["full_inchikey"].nunique().to_numpy()
    result["n_defined_pearson_object_drug_keys"] = grouped_keys["pearson_r"].count().to_numpy()
    result["n_defined_spearman_object_drug_keys"] = grouped_keys["spearman_rho"].count().to_numpy()
    result["n_defined_amplitude_object_drug_keys"] = grouped_keys[
        "predicted_to_observed_abs_delta_ratio"
    ].count().to_numpy()
    return result


def summarize_objects(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    metrics = [
        "pearson_r", "spearman_rho", "predicted_to_observed_abs_delta_ratio",
        "signed_gene_agreement_nonzero_target",
    ]
    summary: dict[str, Any] = {"n_independent_objects": int(len(frame))}
    for metric in metrics:
        values = frame[metric].dropna().to_numpy(dtype=float)
        summary[metric] = {
            "object_macro_mean": finite_metric(values.mean()) if len(values) else None,
            "object_median": finite_metric(np.median(values)) if len(values) else None,
            "defined_objects": int(len(values)),
            "paired_object_bootstrap_95_percentile_interval": paired_bootstrap(values, rng),
        }
    return summary


def metric_comparison(frame: pd.DataFrame, model_column: str, reference_column: str,
                      rng: np.random.Generator) -> dict[str, Any]:
    paired = frame[["source_object_id", model_column, reference_column]].dropna()
    delta = (paired[model_column] - paired[reference_column]).to_numpy(dtype=float)
    return {
        "metric": "NMSE; lower is better",
        "n_independent_objects": int(len(delta)),
        "model_minus_reference_object_macro_mean": float(delta.mean()) if len(delta) else None,
        "median_object_difference": float(np.median(delta)) if len(delta) else None,
        "objects_model_better": int((delta < 0).sum()),
        "objects_tied": int((delta == 0).sum()),
        "objects_model_worse": int((delta > 0).sum()),
        "paired_object_bootstrap_95_percentile_interval": paired_bootstrap(delta, rng),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    source_dir = root / "artifacts/experiments/EXP-050" / SOURCE_RUN_ID / "attempts" / SOURCE_ATTEMPT
    output_dir = root / "artifacts/experiments/EXP-050" / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing diagnostic run: {output_dir}")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    receipt_path = source_dir / "attempt_receipt.json"
    result_path = source_dir / "fit_result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    fit = json.loads(result_path.read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETED_ONE_FRESH_XPERT_FIT" or fit["status"] != receipt["status"]:
        raise ValueError("Source XPert attempt is not a completed fresh fit")
    if fit["outer_or_locked_eval_data_read"] or fit["pdo_treated_labels_read"]:
        raise ValueError("Fit receipt reports forbidden outer/locked or PDO labels were read")
    if fit["direct_sensitivity_values_read"]:
        raise ValueError("Direct sensitivity values were read by the XPert fit")

    verified_inputs: dict[str, str] = {}
    for rel, expected in receipt["output_sha256"].items():
        file_path = source_dir / rel
        actual = sha256(file_path)
        if actual != expected:
            raise ValueError(f"Attempt output hash mismatch: {rel}")
        verified_inputs[rel] = actual

    keys_path = source_dir / "heldout_source_prediction_keys.tsv"
    prediction_path = source_dir / "heldout_source_prediction_target_order.npy"
    truth_path = source_dir / "heldout_source_true_delta_target_order.npy"
    keys = pd.read_csv(keys_path, sep="\t", dtype={"condition_id": str})
    all_metrics = pd.read_csv(source_dir / "heldout_source_per_condition_metrics.tsv",
                              sep="\t", dtype={"condition_id": str})
    common_metrics = pd.read_csv(source_dir / "heldout_common78_per_condition_metrics.tsv",
                                 sep="\t", dtype={"condition_id": str})
    prediction = np.load(prediction_path, allow_pickle=False)
    truth = np.load(truth_path, allow_pickle=False)

    if keys["condition_id"].duplicated().any() or keys["condition_id"].tolist() != all_metrics["condition_id"].tolist():
        raise ValueError("Prediction keys and saved heldout metric rows do not have exact ordered identity")
    if prediction.shape != truth.shape or prediction.shape != (len(keys), 978):
        raise ValueError(f"Unexpected prediction/target shape: {prediction.shape} vs {truth.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Non-finite prediction or heldout response value")
    if (all_metrics["split_role"].astype(str) != "heldout_source").any():
        raise ValueError("Prediction key table contains a non-heldout role")
    if keys["source_object_id"].nunique() != fit["n_heldout_source_objects"]:
        raise ValueError("Heldout object denominator does not match the frozen fit receipt")

    condition_rows: list[dict[str, Any]] = []
    for i, row in enumerate(keys.itertuples(index=False)):
        pred = prediction[i].astype(np.float64)
        obs = truth[i].astype(np.float64)
        nonzero = obs != 0
        pearson_value = pearsonr(pred, obs).statistic
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_value = spearmanr(pred, obs).statistic
        condition_rows.append({
            "condition_id": str(row.condition_id),
            "source_object_id": str(row.source_object_id),
            "full_inchikey": str(row.full_inchikey),
            "split_role": str(all_metrics.iloc[i]["split_role"]),
            "pearson_r": finite_metric(pearson_value),
            "spearman_rho": finite_metric(spearman_value),
            "observed_mean_abs_delta": float(np.mean(np.abs(obs))),
            "predicted_mean_abs_delta": float(np.mean(np.abs(pred))),
            "predicted_to_observed_abs_delta_ratio": (
                float(np.mean(np.abs(pred)) / np.mean(np.abs(obs)))
                if np.mean(np.abs(obs)) > 0 else None
            ),
            "signed_gene_agreement_nonzero_target": (
                float(np.mean(np.sign(pred[nonzero]) == np.sign(obs[nonzero])))
                if int(nonzero.sum()) else None
            ),
            "n_nonzero_observed_genes": int(nonzero.sum()),
            "raw_gene_rmse": float(np.sqrt(np.mean(np.square(pred - obs)))),
        })
    per_condition = pd.DataFrame(condition_rows)
    common_ids = set(common_metrics["condition_id"].astype(str))
    common_condition = per_condition[per_condition["condition_id"].isin(common_ids)].copy()
    if len(common_condition) != len(common_metrics):
        raise ValueError("Common-key evaluation rows do not map one-to-one to heldout predictions")

    all_object = object_summary(per_condition)
    common_object = object_summary(common_condition)
    if all_object["source_object_id"].tolist() != common_object["source_object_id"].tolist():
        raise ValueError("Full and common-key heldout scopes have different independent objects")
    all_object_metrics = pd.read_csv(source_dir / "heldout_source_per_object_metrics.tsv", sep="\t")
    common_object_metrics = pd.read_csv(source_dir / "heldout_common78_per_object_metrics.tsv", sep="\t")
    if all_object["source_object_id"].tolist() != all_object_metrics["source_object_id"].tolist():
        raise ValueError("Full-scope diagnostics and official object metrics have different object order")
    if common_object["source_object_id"].tolist() != common_object_metrics["source_object_id"].tolist():
        raise ValueError("Common-scope diagnostics and official object metrics have different object order")
    all_object = all_object.merge(all_object_metrics, on="source_object_id", validate="one_to_one")
    common_object = common_object.merge(common_object_metrics, on="source_object_id", validate="one_to_one")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {
        "run_id": args.run_id,
        "source_run": SOURCE_RUN_ID,
        "source_attempt": SOURCE_ATTEMPT,
        "source_attempt_status": receipt["status"],
        "selected_epoch": fit["selected_epoch"],
        "epochs_completed": fit["n_epochs_completed"],
        "gpu_hours_charged_by_source_attempt": receipt["charged_gpu_hours"],
        "analysis_scope": {
            "heldout_conditions_all_supported": len(per_condition),
            "heldout_objects": int(per_condition["source_object_id"].nunique()),
            "heldout_full_inchikeys_all_supported": int(per_condition["full_inchikey"].nunique()),
            "heldout_conditions_on_common78_support": len(common_condition),
            "heldout_full_inchikeys_on_common78_support": int(common_condition["full_inchikey"].nunique()),
            "heldout_objects_on_common78_support": int(common_condition["source_object_id"].nunique()),
            "response_axis_genes": int(prediction.shape[1]),
            "metric_aggregation": "per condition, average duplicate conditions within object x drug key, equal drug-key weight within object, then equal object weight; bootstrap resamples the 10 independent objects",
        },
        "prediction_quality_all_supported": summarize_objects(all_object, rng),
        "prediction_quality_common78_support": summarize_objects(common_object, rng),
        "official_nmse_comparisons": {
            "all_supported_model_vs_zero": metric_comparison(all_object, "model_nmse", "zero_nmse", rng),
            "common78_model_vs_zero": metric_comparison(common_object, "model_nmse", "zero_nmse", rng),
            "common78_model_vs_train_drug_mean": metric_comparison(common_object, "model_nmse", "train_drug_mean_nmse", rng),
        },
        "data_boundary": {
            "heldout_response_read_after_checkpoint_frozen": bool(fit["target_delta_read_after_checkpoint_frozen"]),
            "direct_sensitivity_values_read": False,
            "pdo_treated_labels_read": False,
            "outer_or_locked_eval_data_read": False,
            "claim_limit": "source-object transcription-response prediction only; not PDO performance, direct drug efficacy, clinical value, or evidence that drug sensitivity labels were used",
        },
        "bootstrap": {
            "unit": "independent source object",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "paired object-level percentile interval; descriptive with only 10 heldout objects",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    condition_path = output_dir / "heldout_condition_response_diagnostics.tsv"
    object_path = output_dir / "heldout_object_response_diagnostics.tsv"
    common_object_path = output_dir / "heldout_common78_object_response_diagnostics.tsv"
    summary_path = output_dir / "summary.json"
    write_tsv(condition_path, condition_rows)
    write_tsv(object_path, all_object.to_dict(orient="records"))
    write_tsv(common_object_path, common_object.to_dict(orient="records"))
    summary["inputs"] = {rel: value for rel, value in sorted(verified_inputs.items())}
    summary["outputs"] = {
        path.name: sha256(path) for path in [condition_path, object_path, common_object_path]
    }
    summary["started_at"] = started_at
    summary["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": args.run_id,
        "experiment_id": "EXP-050",
        "operation": "post-freeze CPU-only diagnostics of XPert heldout source transcription-response predictions",
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "command": f"python scripts/evaluation/analyze_exp050_xpert_independent_response_diagnostics.py --repo-root {root} --run-id {args.run_id}",
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "source_run": f"artifacts/experiments/EXP-050/{SOURCE_RUN_ID}/attempts/{SOURCE_ATTEMPT}",
        "output_sha256": {
            path.name: sha256(path) for path in [summary_path, condition_path, object_path, common_object_path]
        },
        "started_at": started_at,
        "finished_at": summary["finished_at"],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "run_id": args.run_id,
        "n_conditions": len(per_condition),
        "n_objects": int(per_condition["source_object_id"].nunique()),
        "all_supported": summary["prediction_quality_all_supported"],
        "common78": summary["prediction_quality_common78_support"],
        "nmse": summary["official_nmse_comparisons"],
        "output_dir": str(output_dir.relative_to(root)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
