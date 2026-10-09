#!/usr/bin/env python3
"""Independent CPU-only keyed prediction and metric readback for EXP-060."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA = "fa75ce08d6013c42ee52f503230cfc0f1bbe4a30036b11c4e705f7d2981d92f2"
TRAIN_BASELINE_SHA = "57605394f80d78e37fb8ebd48d5d35b2554db6976a45ac2698054cf64bd0f0ac"
SEEDS = tuple(range(20261006, 20261022))
FAILED_READBACK_DEBIT_SECONDS = 19.3
CURRENT_READBACK_WRAPPER_ALLOWANCE_SECONDS = 8.0

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def close(a, b, *, tol=2e-7, label="value"):
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            raise ValueError(f"mapping mismatch at {label}")
        for key in b:
            close(a[key], b[key], tol=tol, label=f"{label}.{key}")
    elif isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            raise ValueError(f"list mismatch at {label}")
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, tol=tol, label=f"{label}[{i}]")
    elif isinstance(b, (int, float)) and not isinstance(b, bool):
        if a is None or not math.isfinite(float(a)) or abs(float(a)-float(b)) > tol*max(1.0, abs(float(b))):
            raise ValueError(f"numeric mismatch at {label}: {a} vs {b}")
    elif a != b:
        raise ValueError(f"mismatch at {label}: {a!r} vs {b!r}")

def archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def safe(v):
    v = float(v)
    return v if math.isfinite(v) else None

def metric(keys: pd.DataFrame, y: np.ndarray, p: np.ndarray, var: np.ndarray):
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 978:
        raise ValueError("metric axes must be the complete keyed 978 genes")
    use = np.isfinite(var) & (var > 1e-12)
    if int(use.sum()) != 978:
        raise ValueError("TRAIN normalization denominator changed")
    yf, pf = y.astype(np.float64), p.astype(np.float64)
    err = pf-yf
    row_nmse = np.mean(err[:, use]**2/var[use][None, :], axis=1)
    row_mae = np.mean(np.abs(err), axis=1)
    corr, direction = [], []
    for yt, yp in zip(yf, pf):
        corr.append(None if np.ptp(yt) <= 0 or np.ptp(yp) <= 0 else safe(np.corrcoef(yt, yp)[0, 1]))
        nz = np.abs(yt) > 1e-12
        direction.append(float(np.mean(np.sign(yt[nz]) == np.sign(yp[nz]))) if nz.any() else None)
    rows = keys[["source_object_id", "full_inchikey"]].reset_index(drop=True).copy()
    rows["nmse"], rows["mae"], rows["corr"], rows["direction"] = row_nmse, row_mae, corr, direction
    obj = rows.groupby("source_object_id", sort=True).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"), corr=("corr", "mean"), direction=("direction", "mean"))
    amps = []
    for oid, pos in rows.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(pos, dtype=np.int64)
        obs, pred = float(np.abs(yf[ix]).mean()), float(np.abs(pf[ix]).mean())
        if obs > 0:
            amps.append(pred/obs)
    result = {
        "n_conditions": int(len(keys)), "n_source_objects": int(keys.source_object_id.nunique()),
        "n_full_inchikeys": int(keys.full_inchikey.nunique()), "n_genes_in_nmse_denominator": int(use.sum()),
        "n_zero_variance_genes_excluded": int((~use).sum()),
        "object_macro_nmse": safe(obj["nmse"].mean()), "object_macro_mae": safe(obj["mae"].mean()),
        "object_macro_gene_correlation": safe(obj["corr"].mean()),
        "object_macro_direction_agreement": safe(obj["direction"].mean()),
        "object_macro_amplitude_ratio": safe(np.mean(amps)) if amps else None,
        "n_amplitude_objects_with_nonzero_observed_mean_abs": len(amps),
        "aggregation": "conditions averaged within source cell-line object, then objects equally weighted",
        "normalizer": "per-gene variance fitted on TRAIN delta rows only; zero-variance genes excluded and counted",
    }
    return result, {str(oid): float(v) for oid, v in obj["nmse"].items()}

def validate_summary_table(path: Path, expected_rows: list[dict], key_columns: list[str], numeric_columns: list[str]):
    observed = pd.read_csv(path, sep="\t", dtype={name: str for name in key_columns}, keep_default_na=True)
    if len(observed) != len(expected_rows):
        raise ValueError(f"summary-table row count differs for {path.name}: {len(observed)} vs {len(expected_rows)}")
    observed = observed.sort_values(key_columns).reset_index(drop=True)
    expected = pd.DataFrame(expected_rows).sort_values(key_columns).reset_index(drop=True)
    for col in key_columns:
        if observed[col].astype(str).tolist() != expected[col].astype(str).tolist():
            raise ValueError(f"summary-table identity differs for {path.name}/{col}")
    for col in numeric_columns:
        for i, value in enumerate(expected[col].tolist()):
            actual = observed.loc[i, col]
            if value is None:
                if not pd.isna(actual):
                    raise ValueError(f"expected undefined value in {path.name}/{col}/{i}")
            elif pd.isna(actual) or abs(float(actual)-float(value)) > 2e-7*max(1.0, abs(float(value))):
                raise ValueError(f"numeric mismatch in {path.name}/{col}/{i}")

def run(run_id: str, config_path: Path):
    w0, c0 = time.perf_counter(), time.process_time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for the root readback")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-060" or cfg.get("authorization", {}).get("user_scope") != "EXP060-FROZEN-INFERENCE-0103":
        raise ValueError("config is not bound to the authorized EXP-060 package")
    run_root = ROOT / "artifacts/experiments/EXP-060" / run_id
    attempt = run_root / cfg["execution"]["attempt_id"]
    out = attempt / "root_independent_readback.json"
    if out.exists():
        raise FileExistsError("root readback is immutable; refusing overwrite")
    report_path = attempt / "comparison_result.json"
    if not report_path.exists():
        report_path = attempt / "train_comparison_result.json"
    report = read_json(report_path)
    close_receipt = read_json(attempt / "attempt_close_receipt.json")
    manifest = read_json(attempt / "run_manifest.json")
    inp = read_json(attempt / "input_receipt.json")
    selection = read_json(attempt / "train_selection_receipt.json")
    checks = {}
    checks["attempt_and_config_binding"] = (
        manifest.get("run_id") == run_id
        and manifest.get("attempt_id") == cfg["execution"]["attempt_id"]
        and manifest.get("config_sha256") == digest(config_path)
        and close_receipt.get("run_id") == run_id and close_receipt.get("gpu_hours") == 0.0
        and manifest.get("runner_sha256") == digest(ROOT / "scripts/evaluation/evaluate_transigen_frozen_inference.py")
        and report.get("config_sha256") == digest(config_path)
        and report.get("device", "cpu") == "cpu" and report.get("gpu_hours", 0.0) == 0.0
        and report.get("optimizer_loaded", False) is False and report.get("optimizer_updates", 0) == 0
        and report.get("batchnorm_updates", 0) == 0)
    if not checks["attempt_and_config_binding"]:
        raise ValueError("attempt/config/close receipt binding failed")
    ckpt = ROOT / cfg["data"]["checkpoint"]["path"]
    checks["frozen_checkpoint_sha"] = digest(ckpt) == CHECKPOINT_SHA == manifest.get("checkpoint_sha256_before")
    if not checks["frozen_checkpoint_sha"]:
        raise ValueError("frozen checkpoint SHA changed")
    train_baseline = ROOT / "artifacts/experiments/EXP-059/20261008_exp059_transigen_lr1e4_bindingfix_v2/fit_attempt_001/train_baseline_metrics.json"
    if digest(train_baseline) != TRAIN_BASELINE_SHA or manifest.get("parent_train_baseline_sha256") != TRAIN_BASELINE_SHA:
        raise ValueError("previously root-bound TRAIN baseline artifact SHA changed")
    if inp.get("inner_dev_numeric_rows_loaded") is not False:
        raise ValueError("input receipt indicates premature inner-dev access")
    for key in ("pdo_outer_locked_numeric_values_opened", "panacea_numeric_values_opened"):
        if inp.get(key) is not False:
            raise ValueError(f"forbidden data access indicated by {key}")
    selected = selection.get("selected_candidate")
    if selected not in (None, "mc16_mean", "mu_both_levels"):
        raise ValueError("unregistered candidate in selection receipt")
    if selection.get("inner_dev_numeric_rows_loaded") is not False or selection.get("selection_used_inner_dev") is not False:
        raise ValueError("selection receipt fails the TRAIN-only freeze")
    checks["train_selection_frozen_before_dev"] = True
    partial = report.get("valid_full_comparison") is False or close_receipt.get("fixed_comparison_valid") is False

    if report.get("cost_gate", {}).get("status") == "FAIL_CLOSE_AFTER_FIRST_FORMAL_INFERENCE":
        seed_files = list(attempt.glob("random_seed_*_train_keyed_predictions.npz"))
        if len(seed_files) != 1 or selected is not None or (attempt / "inner_dev_result.json").exists():
            raise ValueError("cost-gate closeout has unexpected predictions/DEV access")
        one = archive(seed_files[0])
        if one["predicted_delta"].shape != (419, 978) or not np.isfinite(one["predicted_delta"]).all():
            raise ValueError("retained first formal prediction is incomplete or non-finite")
        if (report.get("first_seed_prediction_sha256") != digest(seed_files[0])
                or hashlib.sha256(one["target_delta"].astype(np.float32).tobytes()).hexdigest()
                != inp.get("target_delta_sha256")):
            raise ValueError("cost-gated first prediction hash or TRAIN target receipt differs")
        if not (attempt / "model_state_after_inference.json").exists():
            raise ValueError("cost-gated attempt lacks post-inference model state receipt")
        status = "PASS_ROOT_READBACK_COST_GATE_STOP_NO_FULL_COMPARISON"
        summary = {"retained_first_formal_seed": seed_files[0].name,
                   "full_16_seed_comparison": False, "inner_dev_numeric_rows_loaded": False}
        checks["cost_gate_retained_only_first_formal_prediction"] = True
    elif partial:
        if not report.get("fixed_random_seed_failure") or selected is not None or (attempt / "inner_dev_result.json").exists():
            raise ValueError("partial stochastic failure package has inconsistent selection or DEV artifact")
        failure = report["fixed_random_seed_failure"]
        seed_records = report.get("fixed_inference_seed_records", [])
        if len(seed_records) != int(failure.get("seed_index", 0)):
            raise ValueError("partial seed records do not match the first failed seed index")
        status = "PASS_ROOT_READBACK_PARTIAL_RANDOM_ARM_FAILED_NO_DEV"
        summary = {"completed_fixed_seed_predictions": len(seed_records),
                   "first_failed_seed": failure.get("failure", {}).get("seed"),
                   "full_16_seed_comparison": False, "inner_dev_numeric_rows_loaded": False}
        checks["partial_stochastic_failure_closed_without_dev"] = True
    else:
        if selected != report.get("selected_candidate"):
            raise ValueError("final report differs from the frozen selection receipt")
        train_path = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/response_blind_conditions.tsv"
        all_keys = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=False)
        train = all_keys.loc[all_keys.split_role.eq("train")].reset_index(drop=True)
        if len(train) != 419 or train.source_object_id.nunique() != 110 or train.full_inchikey.nunique() != 5:
            raise ValueError("frozen TRAIN identity table no longer matches 419/110/5")
        expected_keys = train.full_condition_key.astype(str).to_numpy()
        expected_objects = train.source_object_id.astype(str).to_numpy()
        expected_drugs = train.full_inchikey.astype(str).to_numpy()
        files = [attempt / f"random_seed_{seed}_train_keyed_predictions.npz" for seed in SEEDS]
        records = report.get("fixed_inference_seed_records", [])
        if len(records) != 16 or any(not p.is_file() for p in files):
            raise ValueError("not all fixed 16 stochastic predictions were retained")
        first = archive(files[0])
        fields = set(first)
        if fields != {"condition_key", "source_object_id", "full_inchikey", "genes", "target_delta",
                      "predicted_delta", "role", "method", "inference_seed"}:
            raise ValueError("keyed archive field set changed")
        y = first["target_delta"].astype(np.float32)
        genes = first["genes"].astype(str)
        if (y.shape != (419, 978) or not np.isfinite(y).all() or len(np.unique(genes)) != 978
                or not np.array_equal(first["condition_key"].astype(str), expected_keys)
                or not np.array_equal(first["source_object_id"].astype(str), expected_objects)
                or not np.array_equal(first["full_inchikey"].astype(str), expected_drugs)):
            raise ValueError("TRAIN keys, identities or gene axis fail independent readback")
        if hashlib.sha256(y.tobytes()).hexdigest() != inp.get("target_delta_sha256"):
            raise ValueError("TRAIN target matrix hash differs from the runner input receipt")
        var = np.var(y.astype(np.float64), axis=0, ddof=0)
        key_frame = train[["source_object_id", "full_inchikey"]].copy()
        preds, seed_nmse, seed_objects = [], [], []
        objects = sorted(train.source_object_id.astype(str).unique().tolist())
        for seed, path, record in zip(SEEDS, files, records):
            if record.get("seed") != seed or record.get("prediction_sha256") != digest(path):
                raise ValueError(f"seed receipt/hash mismatch: {seed}")
            data = archive(path)
            if set(data) != fields or int(data["inference_seed"][0]) != seed:
                raise ValueError(f"seed archive contract mismatch: {seed}")
            for name, expected in (("condition_key", expected_keys), ("source_object_id", expected_objects),
                                   ("full_inchikey", expected_drugs), ("genes", genes)):
                if not np.array_equal(data[name].astype(str), expected):
                    raise ValueError(f"key/axis mismatch for seed {seed}/{name}")
            p = data["predicted_delta"]
            if p.shape != (419, 978) or not np.isfinite(p).all() or not np.array_equal(data["target_delta"], y):
                raise ValueError(f"non-finite prediction or different target in seed {seed}")
            m, obj = metric(key_frame, y, p, var)
            close(m, record["metric"], label=f"seed_{seed}_metric")
            preds.append(p.astype(np.float32))
            seed_nmse.append(float(m["object_macro_nmse"]))
            seed_objects.append([obj[o] for o in objects])
        random_ref = float(np.mean(seed_nmse))
        if abs(random_ref - float(selection["random_reference"]["object_macro_nmse"])) > 2e-7:
            raise ValueError("primary random reference does not independently reproduce")
        drug_predictions = np.stack([
            y[expected_drugs == drug].mean(axis=0, dtype=np.float64).astype(np.float32)
            for drug in expected_drugs])
        zero = np.zeros_like(y)
        base = {"per_drug_mean_delta": metric(key_frame, y, drug_predictions, var)[0],
                "zero_change": metric(key_frame, y, zero, var)[0]}
        close(base, report["baseline_train_metrics_reused_and_recomputed"], label="train_baselines")
        candidate_predictions = {"mc16_mean": np.mean(np.stack(preds).astype(np.float64), axis=0).astype(np.float32)}
        if "mu_both_levels" in report.get("candidate_train_metrics", {}):
            mu = archive(attempt / "mu_both_levels_train_keyed_predictions.npz")
            candidate_predictions["mu_both_levels"] = mu["predicted_delta"].astype(np.float32)
        passing = []
        candidate_results = {}
        for name, p in candidate_predictions.items():
            path = attempt / f"{name}_train_keyed_predictions.npz"
            data = archive(path)
            if not np.isfinite(p).all() or data["predicted_delta"].shape != (419, 978):
                raise ValueError(f"candidate archive is invalid: {name}")
            if not np.array_equal(data["predicted_delta"], p) and name == "mc16_mean":
                raise ValueError("mc16 prediction mean differs from the 16 individual draws")
            if (not np.array_equal(data["target_delta"], y)
                    or not np.array_equal(data["condition_key"].astype(str), expected_keys)
                    or not np.array_equal(data["source_object_id"].astype(str), expected_objects)
                    or not np.array_equal(data["full_inchikey"].astype(str), expected_drugs)
                    or not np.array_equal(data["genes"].astype(str), genes)):
                raise ValueError(f"candidate target differs from fixed TRAIN: {name}")
            m, obj = metric(key_frame, y, data["predicted_delta"], var)
            selection_metric = selection["train_only_candidate_metrics"][name]
            report_metric = report["candidate_train_metrics"][name]
            close(m, {k: selection_metric[k] for k in m}, label=f"candidate_{name}")
            close(m, {k: report_metric[k] for k in m}, label=f"final_candidate_{name}")
            rand_gain = (random_ref - m["object_macro_nmse"]) / random_ref if random_ref > 0 else None
            drug_nmse = base["per_drug_mean_delta"]["object_macro_nmse"]
            drug_gain = (drug_nmse - m["object_macro_nmse"]) / drug_nmse
            amp = m["object_macro_amplitude_ratio"]
            passed = (rand_gain is not None and rand_gain >= 0.10 and drug_gain >= 0.10
                      and amp is not None and 0.5 <= amp <= 2.0)
            candidate_results[name] = {"metrics": m, "passes_train_gates": passed}
            if passed:
                passing.append(name)
        expected_selected = min(passing, key=lambda n: (
            candidate_results[n]["metrics"]["object_macro_nmse"], 0 if n == "mu_both_levels" else 1)) if passing else None
        if expected_selected != selected:
            raise ValueError("independently recomputed TRAIN selection differs from frozen receipt")
        method_predictions = {f"random_seed_{seed}": pred for seed, pred in zip(SEEDS, preds)}
        method_predictions.update(candidate_predictions)
        method_predictions["per_drug_mean_delta"] = drug_predictions
        method_predictions["zero_change"] = zero
        per_drug_expected, object_expected = [], []
        for method_name, prediction in method_predictions.items():
            _, obj_values = metric(key_frame, y, prediction, var)
            object_expected.extend({"method": method_name, "source_object_id": oid,
                                    "object_macro_nmse": value} for oid, value in obj_values.items())
            # The runner's per-drug table covers stochastic seeds and the two
            # candidates. Baseline per-drug metrics are recorded in the bound
            # baseline JSON and object table, not repeated in this table.
            if method_name not in ("per_drug_mean_delta", "zero_change"):
                for drug in sorted(set(expected_drugs.tolist())):
                    pos = np.flatnonzero(expected_drugs == drug)
                    submetric, _ = metric(key_frame.iloc[pos].reset_index(drop=True), y[pos], prediction[pos], var)
                    per_drug_expected.append({"method": method_name, "full_inchikey": drug, **submetric})
        validate_summary_table(attempt / "train_per_drug_metrics.tsv", per_drug_expected,
            ["method", "full_inchikey"], ["n_conditions", "n_source_objects", "n_full_inchikeys",
            "n_genes_in_nmse_denominator", "n_zero_variance_genes_excluded", "object_macro_nmse",
            "object_macro_mae", "object_macro_gene_correlation", "object_macro_direction_agreement",
            "object_macro_amplitude_ratio", "n_amplitude_objects_with_nonzero_observed_mean_abs"])
        validate_summary_table(attempt / "train_object_metrics.tsv", object_expected,
            ["method", "source_object_id"], ["object_macro_nmse"])
        rng = np.random.default_rng(2060)
        random_obj_mean = np.mean(np.asarray(seed_objects, dtype=np.float64), axis=0)
        drug_obj = metric(key_frame, y, drug_predictions, var)[1]
        zero_obj = metric(key_frame, y, zero, var)[1]
        bootstrap_rows = []
        for candidate_name, prediction in candidate_predictions.items():
            cand_obj_map = metric(key_frame, y, prediction, var)[1]
            cand_obj = np.asarray([cand_obj_map[oid] for oid in objects], dtype=np.float64)
            for reference_name, reference in (("mean_of_16_single_random_object_losses", random_obj_mean),
                                               ("TRAIN_per_full_inchikey_mean", np.asarray([drug_obj[o] for o in objects])),
                                               ("zero_change", np.asarray([zero_obj[o] for o in objects]))):
                difference = cand_obj - reference
                draws = rng.integers(0, len(difference), size=(2000, len(difference)))
                sampled_mean = difference[draws].mean(axis=1)
                bootstrap_rows.append({"role": "TRAIN", "candidate": candidate_name,
                    "reference": reference_name, "n_objects": len(difference), "n_resamples": 2000,
                    "difference_definition": "candidate minus reference object-macro NMSE; negative favors candidate",
                    "observed_mean_difference": float(difference.mean()),
                    "ci_95_low": float(np.quantile(sampled_mean, 0.025)),
                    "ci_95_high": float(np.quantile(sampled_mean, 0.975)),
                    "selection_use": False, "rng_seed": 2060})
        validate_summary_table(attempt / "paired_object_bootstrap.tsv", bootstrap_rows,
            ["role", "candidate", "reference"], ["n_objects", "n_resamples", "observed_mean_difference",
            "ci_95_low", "ci_95_high", "selection_use", "rng_seed"])
        draws = np.stack(preds, axis=0).astype(np.float64)
        standardized = np.var(draws, axis=0, ddof=0)[:, var > 1e-12] / var[var > 1e-12][None, :]
        variance_receipt = read_json(attempt / "stochastic_prediction_variance.json")
        close({"mean_predictive_variance_over_train_gene_variance": float(standardized.mean()),
               "median_predictive_variance_over_train_gene_variance": float(np.median(standardized)),
               "mean_standardized_predictive_sd": float(np.sqrt(standardized).mean()),
               "condition_gene_denominator": int(standardized.size),
               "random_single_reference_object_macro_nmse": random_ref,
               "mc16_mean_prediction_object_macro_nmse": candidate_results["mc16_mean"]["metrics"]["object_macro_nmse"],
               "single_reference_minus_mc16_mean_nmse": random_ref-candidate_results["mc16_mean"]["metrics"]["object_macro_nmse"]},
              {k: variance_receipt[k] for k in (
                  "mean_predictive_variance_over_train_gene_variance",
                  "median_predictive_variance_over_train_gene_variance",
                  "mean_standardized_predictive_sd", "condition_gene_denominator",
                  "random_single_reference_object_macro_nmse",
                  "mc16_mean_prediction_object_macro_nmse", "single_reference_minus_mc16_mean_nmse")},
              label="stochastic_variance")
        checks["per_drug_and_per_object_metric_tables_recomputed"] = True
        checks["paired_object_bootstrap_recomputed"] = True
        checks["stochastic_prediction_variance_recomputed"] = True
        checks["all_16_keyed_predictions_and_metrics_recomputed"] = True
        checks["train_selection_gates_recomputed"] = True
        dev_loaded = False
        if selected is None:
            if (attempt / "inner_dev_result.json").exists() or list(attempt.glob("*_inner_dev_keyed_predictions.npz")):
                raise ValueError("no candidate passed but inner-dev output exists")
        else:
            dev_path = attempt / f"{selected}_inner_dev_keyed_predictions.npz"
            dev = archive(dev_path)
            dv = all_keys.loc[all_keys.split_role.eq("inner_dev")].reset_index(drop=True)
            if len(dv) != 93 or dv.source_object_id.nunique() != 24 or set(dv.source_object_id) & set(train.source_object_id):
                raise ValueError("inner-dev metadata identity or nonoverlap failed")
            if (dev["target_delta"].shape != (93, 978) or not np.isfinite(dev["target_delta"]).all()
                    or not np.isfinite(dev["predicted_delta"]).all()
                    or not np.array_equal(dev["condition_key"].astype(str), dv.full_condition_key.astype(str).to_numpy())
                    or not np.array_equal(dev["source_object_id"].astype(str), dv.source_object_id.astype(str).to_numpy())
                    or not np.array_equal(dev["full_inchikey"].astype(str), dv.full_inchikey.astype(str).to_numpy())
                    or not np.array_equal(dev["genes"].astype(str), genes)):
                raise ValueError("selected candidate DEV identity/axis/finite readback failed")
            dev_metric = metric(dv, dev["target_delta"], dev["predicted_delta"], var)[0]
            dev_result = read_json(attempt / "inner_dev_result.json")
            if dev_result.get("selected_candidate") != selected or dev_result.get("inner_dev_numeric_rows_loaded") is not True:
                raise ValueError("inner-dev score is not bound to the single TRAIN-selected candidate")
            close(dev_metric, dev_result["metrics"]["candidate"], label="selected_inner_dev_metric")
            if report.get("data_access", {}).get("inner_dev_loaded_only_after_selection_freeze") is not True:
                raise ValueError("report does not prove post-selection inner-dev access")
            dev_loaded = True
        checks["inner_dev_access_matches_frozen_train_selection"] = True
        checks["valid_fixed_inference_comparison"] = True
        status = "PASS_ROOT_INDEPENDENT_READBACK"
        summary = {"fixed_seeds": 16, "random_reference_object_macro_nmse": random_ref,
                   "recomputed_train_baselines": base, "train_candidates": candidate_results,
                   "selected_candidate": selected, "inner_dev_numeric_rows_loaded": dev_loaded}

    before = read_json(attempt / "model_state_before_inference.json")
    after_path = attempt / "model_state_after_inference.json"
    if not after_path.exists():
        raise FileNotFoundError("post-inference model/BN state receipt missing")
    after = read_json(after_path)
    checks["model_state_and_bn_exactly_unchanged"] = (
        after.get("exact") is True and before.get("state_tree_sha256") == after.get("state_tree_sha256")
        and before.get("bn_buffer_sha256") == after.get("bn_buffer_sha256"))
    if not checks["model_state_and_bn_exactly_unchanged"]:
        raise ValueError("model or BatchNorm state changed during read-only inference")
    if report.get("pdo_outer_locked_numeric_values_opened", False) or report.get("panacea_numeric_values_opened", False):
        raise ValueError("final report indicates forbidden numeric data access")
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    own_charge = wall + cpu
    total = (float(close_receipt.get("conservative_charge_seconds", 0.0)) + own_charge
             + FAILED_READBACK_DEBIT_SECONDS + CURRENT_READBACK_WRAPPER_ALLOWANCE_SECONDS)
    new_cap = float(cfg["authorization"]["new_package_cpu_cap_seconds"])
    inherited = float(cfg["authorization"]["inherited_exp059_cpu_remaining_seconds"])
    if total > new_cap or total > inherited:
        raise TimeoutError("runner plus root readback exceeds a frozen CPU cap")
    receipt = {
        "status": status, "experiment_id": "EXP-060", "run_id": run_id,
        "attempt_id": attempt.name, "authorization": cfg["authorization"]["user_scope"],
        "checks": checks, "summary": summary,
        "runner_close_receipt_sha256": digest(attempt / "attempt_close_receipt.json"),
        "comparison_result_sha256": digest(report_path),
        "train_selection_receipt_sha256": digest(attempt / "train_selection_receipt.json"),
        "checkpoint_sha256": CHECKPOINT_SHA, "gpu_hours": 0.0,
        "inner_dev_numeric_rows_loaded_only_after_train_selection_freeze": selected is not None,
        "pdo_outer_locked_numeric_values_opened": False, "panacea_numeric_values_opened": False,
        "root_readback_wall_seconds": wall, "root_readback_process_seconds": cpu,
        "root_readback_cpu_charge_seconds": own_charge,
        "failed_readback_attempts_debit_seconds": FAILED_READBACK_DEBIT_SECONDS,
        "current_wrapper_startup_allowance_seconds": CURRENT_READBACK_WRAPPER_ALLOWANCE_SECONDS,
        "runner_plus_all_root_readback_cpu_charge_seconds": total,
        "new_package_cpu_cap_seconds": new_cap,
        "inherited_exp059_cpu_remaining_cap_seconds": inherited,
    }
    write_json(out, receipt)
    return receipt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default="configs/experiments/EXP-060-transigen-frozen-inference.yaml")
    args = ap.parse_args()
    result = run(args.run_id, Path(args.config).resolve())
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "runner_plus_root_cpu_seconds": result["runner_plus_all_root_readback_cpu_charge_seconds"],
                      "gpu_hours": 0.0}, ensure_ascii=False), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
