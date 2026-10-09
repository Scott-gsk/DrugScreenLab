#!/usr/bin/env python3
"""Root-side independent readback for the frozen EXP-062 source comparison.

This verifier consumes only the immutable metadata freeze and the completed
attempt's keyed NPZ/metrics files. It does not reopen expression matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "zero_change", "per_drug_mean", "linear_state_residual",
    "nonlinear_state_residual", "capacity_matched_drug_only", "shuffled_state",
)
CONTROLS = ("per_drug_mean", "linear_state_residual", "capacity_matched_drug_only", "shuffled_state")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def same_number(actual: Any, expected: Any, *, atol: float = 2e-7, rtol: float = 2e-7) -> bool:
    a, e = finite_number(actual), finite_number(expected)
    if a is None or e is None:
        return a is None and e is None
    return math.isclose(a, e, rel_tol=rtol, abs_tol=atol)


def role_metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                 gene_variance: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usable = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    if not usable.any():
        raise ValueError("TRAIN-only gene variance has no usable genes")
    err = prediction.astype(np.float64) - truth.astype(np.float64)
    condition_nmse = np.square(err[:, usable] / np.sqrt(gene_variance[usable])[None, :]).mean(axis=1)
    condition_mae = np.abs(err).mean(axis=1)
    condition_corr = np.full(len(keys), np.nan, dtype=np.float64)
    for i, (a, b) in enumerate(zip(truth, prediction, strict=True)):
        if np.std(a) > 0 and np.std(b) > 0:
            condition_corr[i] = float(np.corrcoef(a, b)[0, 1])
    condition_direction = (np.sign(truth) == np.sign(prediction)).mean(axis=1)
    object_rows: list[dict[str, Any]] = []
    for obj in sorted(keys.source_object_id.astype(str).unique()):
        idx = np.flatnonzero(keys.source_object_id.astype(str).to_numpy() == obj)
        corr = condition_corr[idx]
        corr = corr[np.isfinite(corr)]
        target_abs = float(np.abs(truth[idx]).mean())
        pred_abs = float(np.abs(prediction[idx]).mean())
        object_rows.append({
            "source_object_id": obj,
            "nmse": float(condition_nmse[idx].mean()),
            "mae": float(condition_mae[idx].mean()),
            "condition_gene_correlation": float(corr.mean()) if len(corr) else None,
            "direction_agreement": float(condition_direction[idx].mean()),
            "amplitude_ratio": pred_abs / target_abs if target_abs > 0 else None,
            "conditions": int(len(idx)),
            "condition_correlations_defined": int(len(corr)),
        })
    summary: dict[str, Any] = {}
    for name in ("nmse", "mae", "condition_gene_correlation", "direction_agreement", "amplitude_ratio"):
        vals = np.asarray([np.nan if row[name] is None else row[name] for row in object_rows], dtype=np.float64)
        summary[f"object_macro_{name}"] = float(vals[np.isfinite(vals)].mean()) if np.isfinite(vals).any() else None
    summary.update({
        "objects": len(object_rows), "conditions": int(len(keys)),
        "condition_gene_correlations_defined": int(np.isfinite(condition_corr).sum()),
        "genes_in_nmse": int(usable.sum()),
        "zero_or_invalid_variance_genes_excluded": int((~usable).sum()),
        "object_scores": {row["source_object_id"]: {k: v for k, v in row.items() if k != "source_object_id"}
                          for row in object_rows},
    })
    return summary, object_rows


def check_summary(recomputed: dict[str, Any], recorded: dict[str, Any], label: str) -> None:
    for key in ("objects", "conditions", "condition_gene_correlations_defined", "genes_in_nmse",
                "zero_or_invalid_variance_genes_excluded"):
        if recomputed.get(key) != recorded.get(key):
            raise ValueError(f"{label} denominator mismatch at {key}: {recomputed.get(key)} != {recorded.get(key)}")
    for key in ("object_macro_nmse", "object_macro_mae", "object_macro_condition_gene_correlation",
                "object_macro_direction_agreement", "object_macro_amplitude_ratio"):
        if not same_number(recomputed.get(key), recorded.get(key)):
            raise ValueError(f"{label} independent metric mismatch at {key}: {recomputed.get(key)} != {recorded.get(key)}")
    # Per-object scores are intentionally persisted in object_metrics.tsv,
    # not nested in metrics_summary.json; compare that full table below.


def parse_bootstrap_spec(cfg: dict[str, Any]) -> tuple[int, int]:
    spec = str(cfg["metrics"]["paired_interval"])
    match = re.fullmatch(r"source_object_bootstrap_(\d+)_seed_(\d+)", spec)
    if match is None:
        raise ValueError(f"unrecognized registered bootstrap recipe: {spec}")
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-062" or cfg.get("run_id") != args.run_id:
        raise ValueError("config/run identity mismatch")
    out = ROOT / cfg["output"]["root"]
    attempt = out / args.attempt_id
    freeze_path, root_meta_path = out / "metadata_freeze_manifest.json", out / "root_metadata_readback.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
    if root_meta.get("status") != "PASS_ROOT_INDEPENDENT_METADATA_READBACK":
        raise PermissionError("Stage M root readback is not PASS")
    if root_meta.get("metadata_freeze_sha256") != sha256_file(freeze_path):
        raise ValueError("Stage M root receipt does not bind this exact metadata freeze")
    manifest_path = attempt / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED_FIXED_SOURCE_DEVELOPMENT_COMPARISON":
        raise ValueError(f"attempt is not a completed full comparison: {manifest.get('status')}")
    if manifest.get("experiment_id") != "EXP-062" or manifest.get("run_id") != args.run_id:
        raise ValueError("attempt manifest identity mismatch")
    if manifest.get("device") != "cpu" or float(manifest.get("gpu_hours", -1)) != 0.0:
        raise ValueError("attempt resource receipt conflicts with CPU-only authorization")
    if manifest.get("forbidden_numeric_inputs_opened"):
        raise PermissionError("manifest reports forbidden numeric inputs")

    keys = pd.read_csv(out / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
    if len(keys) != int(freeze["selection"]["selected_conditions"]):
        raise ValueError("frozen condition count mismatch")
    if keys.full_condition_key.duplicated().any() or keys.source_row.duplicated().any():
        raise ValueError("frozen keys/rows are not unique")
    train = keys.loc[keys.effective_split_role.eq("train")].reset_index(drop=True)
    dev = keys.loc[keys.effective_split_role.eq("inner_dev")].reset_index(drop=True)
    if not set(keys.effective_split_role).issubset({"train", "inner_dev"}):
        raise ValueError("frozen data contain an unauthorized role")
    train_objects, dev_objects = set(train.source_object_id), set(dev.source_object_id)
    if train_objects & dev_objects:
        raise ValueError("TRAIN and inner-dev source-object overlap")

    pred_path = attempt / "keyed_predictions.npz"
    # This first-party NPZ was produced by the authorized runner and its bytes
    # are bound by two attempt manifests. Verify both hashes before NumPy is
    # allowed to deserialize the runner's object-dtype key arrays. Never use
    # this path for an external or otherwise untrusted pickle/NPZ input.
    root_input_manifest_path = attempt / "root_readback_input_manifest.json"
    root_input_manifest = json.loads(root_input_manifest_path.read_text(encoding="utf-8"))
    expected_prediction_hash = manifest.get("outputs_sha256", {}).get("keyed_predictions.npz")
    if (root_input_manifest.get("experiment_id") != "EXP-062"
            or root_input_manifest.get("run_id") != args.run_id
            or root_input_manifest.get("attempt_id") != args.attempt_id
            or root_input_manifest.get("status") != "COMPLETED_FIXED_SOURCE_DEVELOPMENT_COMPARISON"
            or not expected_prediction_hash
            or root_input_manifest.get("prediction_sha256") != expected_prediction_hash
            or root_input_manifest.get("output_sha256", {}).get("keyed_predictions.npz") != expected_prediction_hash
            or sha256_file(pred_path) != expected_prediction_hash):
        raise ValueError("keyed prediction archive hash/identity is not bound by both attempt manifests")
    archive = np.load(pred_path, allow_pickle=True)
    with archive as z:
        train_ids = z["train_full_condition_key"].astype(str)
        dev_ids = z["inner_dev_full_condition_key"].astype(str)
        if not np.array_equal(train_ids, train.full_condition_key.astype(str).to_numpy()):
            raise ValueError("TRAIN keyed prediction rows are not in exact frozen key order")
        if not np.array_equal(dev_ids, dev.full_condition_key.astype(str).to_numpy()):
            raise ValueError("inner-dev keyed prediction rows are not in exact frozen key order")
        for role_keys, prefix in ((train, "train"), (dev, "inner_dev")):
            for suffix, column in (("source_object_id", "source_object_id"), ("full_inchikey", "full_inchikey")):
                if not np.array_equal(z[f"{prefix}_{suffix}"].astype(str), role_keys[column].astype(str).to_numpy()):
                    raise ValueError(f"{prefix} prediction {column} axis differs from frozen keys")
        genes = z["gene_axis"].astype(str)
        if len(genes) != 978 or len(set(genes)) != 978:
            raise ValueError("prediction gene axis is not unique length 978")
        axis_sha = hashlib.sha256("\n".join(genes.tolist()).encode("utf-8")).hexdigest()
        if axis_sha != freeze.get("gene_axis_sha256"):
            raise ValueError("prediction gene axis hash differs from frozen official 978-axis")
        y_train = z["train_target_delta"].astype(np.float64)
        y_dev = z["inner_dev_target_delta"].astype(np.float64)
        train_pred = {m: z[f"{m}_train"].astype(np.float64) for m in METHODS}
        dev_pred = {m: z[f"{m}_inner_dev"].astype(np.float64) for m in METHODS}
    if y_train.shape != (len(train), 978) or y_dev.shape != (len(dev), 978):
        raise ValueError("keyed target array shape differs from frozen condition/gene axes")
    for role_name, y, preds in (("train", y_train, train_pred), ("inner_dev", y_dev, dev_pred)):
        if not np.isfinite(y).all():
            raise FloatingPointError(f"keyed {role_name} target has non-finite values")
        for method, prediction in preds.items():
            if prediction.shape != y.shape or not np.isfinite(prediction).all():
                raise FloatingPointError(f"keyed {role_name} prediction invalid for {method}")

    # Independent metric calculation from the frozen keyed arrays; no runner metric function is imported.
    gene_variance = np.var(y_train, axis=0, ddof=0)
    recomputed: dict[str, dict[str, Any]] = {}
    object_rows: list[dict[str, Any]] = []
    role_data = {"train_in_sample": (train, y_train, train_pred), "inner_dev": (dev, y_dev, dev_pred)}
    for method in METHODS:
        recomputed[method] = {}
        for role_name, (role_keys, truth, predictions) in role_data.items():
            summary, rows = role_metrics(role_keys, truth, predictions[method], gene_variance)
            recomputed[method][role_name] = summary
            object_rows.extend({"method": method, "role": role_name, **row} for row in rows)

    recorded = json.loads((attempt / "metrics_summary.json").read_text(encoding="utf-8"))
    if recorded.get("status") != "COMPLETED_FIXED_SOURCE_DEVELOPMENT_COMPARISON":
        raise ValueError("metrics summary is not a completed comparison")
    for method in METHODS:
        for role_name in role_data:
            check_summary(recomputed[method][role_name], recorded["metrics"][method][role_name],
                          f"{method}/{role_name}")

    object_frame = pd.DataFrame(object_rows)
    persisted_objects = pd.read_csv(attempt / "object_metrics.tsv", sep="\t")
    expected_cols = ["method", "role", "source_object_id", "nmse", "mae", "condition_gene_correlation",
                     "direction_agreement", "amplitude_ratio", "conditions", "condition_correlations_defined"]
    if set(expected_cols) - set(persisted_objects.columns):
        raise ValueError("object metrics table is missing required fields")
    if len(persisted_objects) != len(object_frame):
        raise ValueError("object metrics row count mismatch")
    join = object_frame.merge(persisted_objects[expected_cols], on=["method", "role", "source_object_id"],
                              suffixes=("_root", "_saved"), validate="one_to_one")
    for metric in ("nmse", "mae", "condition_gene_correlation", "direction_agreement", "amplitude_ratio"):
        left, right = f"{metric}_root", f"{metric}_saved"
        for a, b in zip(join[left], join[right], strict=True):
            if not same_number(a, b):
                raise ValueError(f"saved object table differs from independent {metric}")
    for metric in ("conditions", "condition_correlations_defined"):
        if not np.array_equal(join[f"{metric}_root"].to_numpy(), join[f"{metric}_saved"].to_numpy()):
            raise ValueError(f"saved object denominator mismatch: {metric}")

    candidate = recomputed["nonlinear_state_residual"]["inner_dev"]
    control_nmses = {c: recomputed[c]["inner_dev"]["object_macro_nmse"] for c in CONTROLS}
    strongest = min(CONTROLS, key=lambda c: float(control_nmses[c]))
    reference = recomputed[strongest]["inner_dev"]
    cand_scores = candidate["object_scores"]
    ref_scores = reference["object_scores"]
    paired_ids = sorted(set(cand_scores) & set(ref_scores))
    diff = np.asarray([ref_scores[o]["nmse"] - cand_scores[o]["nmse"] for o in paired_ids], dtype=np.float64)
    reps, seed = parse_bootstrap_spec(cfg)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(paired_ids), size=(reps, len(paired_ids)))
    ci = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975]).astype(float).tolist()
    fraction = float(np.mean(diff > 0)) if len(diff) else None
    paired_controls: dict[str, Any] = {}
    for control in CONTROLS:
        scores = recomputed[control]["inner_dev"]["object_scores"]
        common = sorted(set(cand_scores) & set(scores))
        d = np.asarray([scores[o]["nmse"] - cand_scores[o]["nmse"] for o in common], dtype=np.float64)
        boot_draws = rng.integers(0, len(common), size=(reps, len(common)))
        paired_controls[control] = {
            "paired_objects": len(common), "control_minus_candidate_nmse": float(d.mean()),
            "bootstrap_95pct_ci": np.quantile(d[boot_draws].mean(axis=1), [0.025, 0.975]).astype(float).tolist(),
            "objects_improved": int((d > 0).sum()),
            "fraction_objects_improved": float(np.mean(d > 0)) if len(d) else None,
        }
    relative_gain = ((float(reference["object_macro_nmse"]) - float(candidate["object_macro_nmse"]))
                     / float(reference["object_macro_nmse"]))
    amplitude = candidate["object_macro_amplitude_ratio"]
    beats_all = all(float(candidate["object_macro_nmse"]) < float(control_nmses[c]) for c in CONTROLS)
    gate_pass = bool(relative_gain >= 0.05 and ci[0] > 0 and fraction is not None and fraction >= 0.60
                     and amplitude is not None and 0.5 <= float(amplitude) <= 2.0 and beats_all)
    independent_decision = {
        "strongest_fixed_control": strongest,
        "candidate_inner_dev_object_macro_nmse": candidate["object_macro_nmse"],
        "strongest_control_inner_dev_object_macro_nmse": reference["object_macro_nmse"],
        "relative_nmse_gain": relative_gain,
        "paired_object_bootstrap_95pct_ci_absolute_gain": ci,
        "objects_improved": int((diff > 0).sum()), "paired_objects": len(paired_ids),
        "fraction_objects_improved": fraction,
        "candidate_amplitude_ratio": amplitude,
        "paired_object_bootstrap_vs_each_fixed_control": paired_controls,
        "beats_every_fixed_control": beats_all, "expansion_gate_pass": gate_pass,
        "candidate_disposition": "QUALIFIES_FOR_AUTHORIZED_CONFIRMATION" if gate_pass else "CLOSED_NO_SEARCH_EXPANSION",
        "claim_boundary": "fixed XPert source-development split only; no unseen drug, independent test, PDO efficacy, or clinical claim",
    }
    saved_decision = json.loads((attempt / "decision.json").read_text(encoding="utf-8"))
    for key in ("strongest_fixed_control", "objects_improved", "paired_objects", "beats_every_fixed_control",
                "expansion_gate_pass", "candidate_disposition"):
        if independent_decision[key] != saved_decision.get(key):
            raise ValueError(f"decision mismatch at {key}")
    for key in ("candidate_inner_dev_object_macro_nmse", "strongest_control_inner_dev_object_macro_nmse",
                "relative_nmse_gain", "fraction_objects_improved", "candidate_amplitude_ratio"):
        if not same_number(independent_decision[key], saved_decision.get(key)):
            raise ValueError(f"decision numeric mismatch at {key}")
    if not np.allclose(independent_decision["paired_object_bootstrap_95pct_ci_absolute_gain"],
                       saved_decision["paired_object_bootstrap_95pct_ci_absolute_gain"], rtol=0, atol=2e-7):
        raise ValueError("strongest-control paired bootstrap interval mismatch")
    for control in CONTROLS:
        root_control, saved_control = paired_controls[control], saved_decision["paired_object_bootstrap_vs_each_fixed_control"][control]
        if root_control["paired_objects"] != saved_control.get("paired_objects"):
            raise ValueError(f"paired object denominator mismatch for {control}")
        for key in ("control_minus_candidate_nmse", "fraction_objects_improved"):
            if not same_number(root_control[key], saved_control.get(key)):
                raise ValueError(f"paired comparison mismatch for {control}/{key}")
        if not np.allclose(root_control["bootstrap_95pct_ci"], saved_control["bootstrap_95pct_ci"], rtol=0, atol=2e-7):
            raise ValueError(f"paired bootstrap interval mismatch for {control}")

    curve = pd.read_csv(attempt / "training_curves.tsv", sep="\t")
    if len(curve) != int(recorded["training_curves"]) or not set(curve.arm).issubset(set(METHODS)):
        raise ValueError("training curve count/arm identity mismatch")
    trainable_arms = {"nonlinear_state_residual", "capacity_matched_drug_only", "shuffled_state"}
    if set(curve.arm) != trainable_arms or set(curve.loc[curve.epoch.eq(1), "arm"]) != trainable_arms:
        raise ValueError("first-epoch training curves do not cover all three fitted arms")
    if not np.isfinite(curve["train_object_balanced_normalized_residual_mse"].to_numpy(float)).all():
        raise FloatingPointError("training loss curves contain non-finite values")

    run_file_hashes = manifest.get("outputs_sha256", {})
    for name, digest in run_file_hashes.items():
        # run_manifest is finalized after this payload hash map is assembled,
        # so a self-hash would necessarily refer to its earlier draft bytes.
        if name == "run_manifest.json":
            continue
        if sha256_file(attempt / name) != digest:
            raise ValueError(f"attempt output hash mismatch: {name}")
    root_input_manifest = json.loads((attempt / "root_readback_input_manifest.json").read_text(encoding="utf-8"))
    if root_input_manifest.get("summary_sha256") != sha256_file(attempt / "metrics_summary.json"):
        raise ValueError("attempt summary hash differs from root readback input manifest")
    if root_input_manifest.get("prediction_sha256") != sha256_file(pred_path):
        raise ValueError("attempt prediction hash differs from root readback input manifest")
    if int(recorded["denominators"]["train_conditions"]) != len(train) or int(recorded["denominators"]["inner_dev_conditions"]) != len(dev):
        raise ValueError("summary row denominators differ from frozen split")
    if int(recorded["denominators"]["train_objects"]) != len(train_objects) or int(recorded["denominators"]["inner_dev_objects"]) != len(dev_objects):
        raise ValueError("summary object denominators differ from frozen split")
    if int(recorded["denominators"]["object_overlap"]) != 0 or int(recorded["denominators"]["full_inchikeys"]) != keys.full_inchikey.nunique() or int(recorded["denominators"]["genes"]) != 978:
        raise ValueError("summary identity/gene denominator mismatch")
    if float(recorded["cpu_wall_seconds"]) > int(cfg["resources"]["cpu_hard_cap_seconds"]):
        raise RuntimeError("EXP-062 package CPU cap exceeded")
    if float(recorded["EXP055_cumulative_cpu_seconds"]) > int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"]):
        raise RuntimeError("inherited EXP-055 CPU cap exceeded")

    prior_readback_failure_seconds = 0.0
    for failure_path in sorted(out.glob(f"root_independent_readback_{args.attempt_id}_v*/failure_close_receipt.json")):
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("status") == "FAILED_CLOSED" and failure.get("immutable") is True:
            prior_readback_failure_seconds += float(failure.get("resource", {}).get("cpu_wall_seconds_conservative", 0.0))
    receipt_dir = out / f"root_independent_readback_{args.attempt_id}_v5"
    receipt_dir.mkdir(parents=True, exist_ok=False)
    metrics_table = []
    for method in METHODS:
        for role_name in role_data:
            metrics_table.append({"method": method, "role": role_name,
                                  **{k: v for k, v in recomputed[method][role_name].items()
                                     if k != "object_scores"}})
    pd.DataFrame(metrics_table).to_csv(receipt_dir / "root_recomputed_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(object_rows).to_csv(receipt_dir / "root_recomputed_object_metrics.tsv", sep="\t", index=False)
    cpu_seconds = time.monotonic() - started
    exp062_package_total = float(recorded["cpu_wall_seconds"]) + prior_readback_failure_seconds + cpu_seconds
    exp055_total = float(recorded["EXP055_cumulative_cpu_seconds"]) + prior_readback_failure_seconds + cpu_seconds
    if exp062_package_total > int(cfg["resources"]["cpu_hard_cap_seconds"]):
        raise RuntimeError("root verification accounting exceeds the EXP-062 CPU cap")
    if exp055_total > int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"]):
        raise RuntimeError("root verification accounting exceeds inherited EXP-055 CPU cap")
    receipt = {
        "experiment_id": "EXP-062", "run_id": args.run_id, "attempt_id": args.attempt_id,
        "status": "PASS_ROOT_INDEPENDENT_NUMERIC_READBACK",
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "metadata_freeze_sha256": sha256_file(freeze_path),
        "root_metadata_readback_sha256": sha256_file(root_meta_path),
        "attempt_manifest_sha256": sha256_file(manifest_path),
        "attempt_metrics_sha256": sha256_file(attempt / "metrics_summary.json"),
        "attempt_decision_sha256": sha256_file(attempt / "decision.json"),
        "attempt_keyed_predictions_sha256": sha256_file(pred_path),
        "source_values_independently_reopened": False,
        "source_value_provenance": "Stage V input_receipt binds the only authorized frozen source rows; root independently re-keyed and recomputed every saved metric/decision from the immutable keyed target/prediction archive. No new source X/X_ctl read.",
        "forbidden_numeric_values_read": [],
        "denominators": {"train_conditions": len(train), "train_objects": len(train_objects),
            "inner_dev_conditions": len(dev), "inner_dev_objects": len(dev_objects),
            "train_inner_dev_object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()),
            "genes": len(genes)},
        "root_recomputed_metrics": {m: {r: {k: v for k, v in recomputed[m][r].items() if k != "object_scores"}
                                         for r in role_data} for m in METHODS},
        "independent_decision": independent_decision,
        "metric_summary_max_abs_diff": "see paired root_recomputed_metrics.tsv and immutable metrics_summary.json",
        "metric_output_directory": str(receipt_dir.relative_to(ROOT)),
        "cpu_wall_seconds": cpu_seconds,
        "prior_failed_root_readback_cpu_seconds": prior_readback_failure_seconds,
        "EXP062_cumulative_cpu_seconds_including_root_readback": exp062_package_total,
        "EXP055_cumulative_cpu_seconds_including_root_readback": exp055_total,
        "gpu_hours": 0,
    }
    (receipt_dir / "root_independent_readback.json").write_text(
        json.dumps(receipt, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt["root_receipt_sha256"] = sha256_file(receipt_dir / "root_independent_readback.json")
    # Keep a separate sidecar digest so the immutable receipt itself remains non-self-referential.
    (receipt_dir / "root_independent_readback_sha256.txt").write_text(
        receipt["root_receipt_sha256"] + "  root_independent_readback.json\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "root_receipt_sha256": receipt["root_receipt_sha256"],
                      "decision": independent_decision, "denominators": receipt["denominators"],
                      "cpu_wall_seconds": cpu_seconds, "gpu_hours": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
