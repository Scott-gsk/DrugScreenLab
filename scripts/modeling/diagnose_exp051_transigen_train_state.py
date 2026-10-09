"""Bounded TRAIN-only diagnosis of the EXP-051 TranSiGen evaluation-loss issue.

The selected model is the upstream TranSiGen architecture. This diagnostic
reuses only the frozen TRAIN rows from the response-blind EXP-051 source run;
it never evaluates inner-dev, outer/locked, or PDO labels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAX_GPU_SECONDS = 1800.0
RESERVATION_GPU_HOURS = 0.60
LOSS_NAMES = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert")


def _array_summary(value: Any) -> dict[str, Any]:
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {"shape": list(array.shape), "finite": bool(finite.all()),
            "finite_count": int(finite.sum()), "total_count": int(array.size),
            "min": float(finite_values.min()) if finite_values.size else None,
            "max": float(finite_values.max()) if finite_values.size else None,
            "mean": float(finite_values.mean(dtype=np.float64)) if finite_values.size else None,
            "std": float(finite_values.std(dtype=np.float64)) if finite_values.size else None,
            "max_abs": float(np.max(np.abs(finite_values))) if finite_values.size else None}


def _state_summary(torch: Any, model: Any) -> dict[str, Any]:
    parameters = list(model.named_parameters())
    finite_parameters = sum(int(torch.isfinite(value).all()) for _, value in parameters)
    flat_norm = math.sqrt(sum(float(torch.sum(value.detach().double() ** 2).cpu()) for _, value in parameters))
    nonfinite_parameters = [name for name, value in parameters if not bool(torch.isfinite(value).all())]
    buffers = {}
    for name, value in model.named_buffers():
        if any(token in name for token in ("running_mean", "running_var", "num_batches_tracked")):
            flat = value.detach().float().cpu().numpy().astype("<f4", copy=False)
            buffers[name] = {**_array_summary(flat), "sha256": hashlib.sha256(flat.tobytes()).hexdigest()}
    return {"parameter_tensors": len(parameters), "finite_parameter_tensors": finite_parameters,
            "nonfinite_parameter_names": nonfinite_parameters,
            "parameter_l2_norm": flat_norm,
            "batchnorm_buffers": buffers}


def _optimizer_finite(torch: Any, optimizer: Any) -> dict[str, Any]:
    nonfinite = []
    tensor_count = 0
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                if not bool(torch.isfinite(value).all()):
                    nonfinite.append(str(name))
    return {"state_tensor_count": tensor_count, "all_finite": not nonfinite,
            "nonfinite_state_names": sorted(set(nonfinite))}


def _bn_state(torch: Any, model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if any(token in name for token in ("running_mean", "running_var", "num_batches_tracked"))}


def _bn_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return before.keys() == after.keys() and all(np.array_equal(before[key].numpy(), after[key].numpy())
                                                 for key in before)


def run_diagnostic(fit: Any, cfg: dict[str, Any], config_sha: str, attempt_id: str) -> dict[str, Any]:
    """Called from the frozen EXP-051 entry point with that runner's helpers."""
    fit.configure_source_run(fit.SOURCE_RUN_ID)
    fit.configure_run(fit.RUN_ID)
    limits = fit.resource_limits(cfg)
    accounting = fit.resource_accounting(
        fit.ROOT / "artifacts/experiments/EXP-051", fit.RUN_ID,
        limits["package_cap_gpu_hours"], limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"])
    remaining = float(accounting["remaining_authorized_gpu_hours"])
    reservation = min(RESERVATION_GPU_HOURS, remaining)
    if reservation <= 0:
        raise RuntimeError("no settled-ledger EXP-051 balance remains for bounded TranSiGen diagnosis")
    if remaining + 1e-12 < RESERVATION_GPU_HOURS:
        raise RuntimeError("shared package balance cannot cover the frozen 0.60 GPUh diagnostic reservation")
    attempt = fit.new_attempt_path(fit.RUN_DIR, attempt_id)
    fit.RUN_DIR.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(parents=False, exist_ok=False)
    process_id = os.getpid()
    preflight = {"run_id": fit.RUN_ID, "attempt": attempt_id, "selected_models": ["transigen"],
                 "status": "CPU_PREFLIGHT_STARTED", "started_at_utc": fit.utc_now(),
                 "wsl_pid": process_id, "runner_sha256": fit.sha256_file(Path(fit.__file__).resolve()),
                 "diagnostic_runner_sha256": fit.sha256_file(Path(__file__).resolve()),
                 "source_run_id": fit.SOURCE_RUN_ID, "config_sha256": config_sha,
                 "diagnostic_gpu_wall_cap_seconds": MAX_GPU_SECONDS,
                 "reserved_gpu_hours": reservation}
    fit.atomic_json(attempt / "pre_gpu_preflight_started.json", preflight)
    try:
        receipt_paths = {"freeze": fit.SOURCE_RUN_DIR / "metadata_freeze_receipt.json",
                         "axis": fit.SOURCE_RUN_DIR / "gene_axis_receipt.json",
                         "values": fit.SOURCE_RUN_DIR / "value_contract_receipt.json",
                         "root_values": fit.SOURCE_RUN_DIR / "root_values_readback_receipt.json",
                         "feature": fit.SOURCE_RUN_DIR / "transigen_ecfp4_feature_audit_receipt.json"}
        for name, path in receipt_paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"required frozen {name} receipt is missing: {path}")
        freeze = json.loads(receipt_paths["freeze"].read_text(encoding="utf-8"))
        axis = json.loads(receipt_paths["axis"].read_text(encoding="utf-8"))
        value_receipt = json.loads(receipt_paths["values"].read_text(encoding="utf-8"))
        root_values = json.loads(receipt_paths["root_values"].read_text(encoding="utf-8"))
        if (freeze.get("run_id") != fit.SOURCE_RUN_ID or axis.get("status") != "PASS_EXACT_978_AXIS"
                or root_values.get("checks", {}).get("result") != "PASS"):
            raise ValueError("source run is not bound to the frozen exact-978 and root-readback receipts")
        keys_path = fit.SOURCE_RUN_DIR / "response_blind_conditions.tsv"
        values_path = fit.SOURCE_RUN_DIR / "frozen_train_inner_values.npz"
        if fit.sha256_file(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
            raise ValueError("frozen TRAIN/inner value file checksum does not match the receipt")
        keys = pd.read_csv(keys_path, sep="\t", dtype=str)
        train_mask = keys.split_role.eq("train").to_numpy()
        if int(train_mask.sum()) != 419 or keys.loc[train_mask, "source_object_id"].nunique() != 110:
            raise ValueError("response-blind TRAIN mask differs from 419 conditions/110 objects")
        # The source NPZ is an internally generated, receipt-bound archive. Only
        # TRAIN-indexed slices are retained or passed to any model/metric code.
        with np.load(values_path, allow_pickle=True) as frozen:
            if not np.array_equal(frozen["split_role"].astype(str), keys.split_role.astype(str).to_numpy()):
                raise ValueError("frozen key-table role differs from the value archive")
            x_ctl = frozen["X_ctl"][train_mask].astype(np.float32, copy=True)
            x_treated = frozen["X"][train_mask].astype(np.float32, copy=True)
            delta = frozen["delta"][train_mask].astype(np.float32, copy=True)
            genes = frozen["genes"].astype(str)
            full_keys = frozen["full_inchikey"][train_mask].astype(str)
        train_keys = keys.loc[train_mask].reset_index(drop=True)
        if (x_ctl.shape != (419, 978) or x_treated.shape != (419, 978) or delta.shape != (419, 978)
                or len(genes) != 978 or len(set(genes.tolist())) != 978
                or not np.isfinite(x_ctl).all() or not np.isfinite(x_treated).all()
                or not np.isfinite(delta).all() or not np.array_equal(full_keys, train_keys.full_inchikey.astype(str).to_numpy())):
            raise ValueError("TRAIN-only values, axis, identity, or finite-value gate failed")
        # The frozen expression arrays are float32. Reconstructing a treated
        # value after loading/casting can differ by one float32 rounding step,
        # so require a strict absolute reconstruction tolerance and record the
        # measured residual instead of demanding bitwise equality.
        decomposition_max_abs_error = float(np.max(np.abs(
            x_treated.astype(np.float64) - x_ctl.astype(np.float64) - delta.astype(np.float64))))
        if decomposition_max_abs_error > 1e-6:
            raise ValueError(
                "TRAIN treated values do not reconstruct from paired controls plus delta "
                f"within 1e-6; measured max_abs_error={decomposition_max_abs_error:.9g}")
        feature_by_key, feature_receipt = fit._load_audited_ecfp4(fit.SOURCE_RUN_DIR)
        feature_rows = fit._full_key_features(train_keys, np.arange(len(train_keys)), feature_by_key)
        if not np.isfinite(feature_rows).all() or feature_rows.shape != (419, 2048):
            raise ValueError("audited full-key ECFP4 TRAIN feature rows are invalid")
        source_hashes = fit._source_hashes()
        snapshot = fit._device_snapshot()
        if (snapshot["selected_gpu"]["memory_used_mib"] > 1024
                or snapshot["selected_gpu"]["utilization_percent"] > 20):
            raise RuntimeError("approved A5000 is not idle enough for the TranSiGen diagnostic")
        receipt_set = {name: {"path": str(path), "sha256": fit.sha256_file(path)}
                       for name, path in receipt_paths.items()}
        manifest = {"experiment_id": "EXP-051", "run_id": fit.RUN_ID, "fit_attempt": attempt_id,
                    "selected_models": ["transigen"], "wsl_pid": process_id,
                    "current_runner_sha256": preflight["runner_sha256"],
                    "diagnostic_runner_sha256": preflight["diagnostic_runner_sha256"],
                    "config_sha256": config_sha, "source_run_id": fit.SOURCE_RUN_ID,
                    "resource_accounting_before_attempt": accounting,
                    "reservation_gpu_hours": reservation,
                    "resource_limits": limits,
                    "model_contract": "upstream original TranSiGen scratch; 419 TRAIN rows only; normal model.train() and ordinary BatchNorm updates; no eval-time BN calibration",
                    "diagnostic_question": "Locate the first abnormal TRAIN objective/gradient/parameter/optimizer or eval-only loss term; independently score the frozen TRAIN delta prediction if finite.",
                    "scope": {"train_conditions": 419, "train_objects": 110,
                              "full_inchikeys": int(train_keys.full_inchikey.nunique()), "genes": len(genes),
                              "train_treated_equals_control_plus_delta_max_abs_error": decomposition_max_abs_error,
                              "decomposition_absolute_tolerance": 1e-6,
                              "inner_dev_evaluated": False, "outer_or_locked_opened": False,
                              "PDO_sensitivity_opened": False, "external_expression_opened": False},
                    "bounded_limits": {"max_gpu_wall_seconds": MAX_GPU_SECONDS,
                                       "reservation_gpu_hours": reservation,
                                       "stop_at_first_nonfinite_training_state": True,
                                       "stop_at_first_auxiliary_eval_loss_nonfinite_with_finite_prediction": True},
                    "source_receipts": receipt_set,
                    "source_inputs": {"keys_sha256": fit.sha256_file(keys_path),
                                      "values_sha256": fit.sha256_file(values_path),
                                      "feature_npz_sha256": feature_receipt["feature_npz_sha256"]},
                    "source_code_sha256": source_hashes,
                    "device_uuid": fit.GPU_UUID,
                    "evidence_boundary": "TRAIN-side optimization/state diagnosis only; no inner-dev selection, no transfer or PDO efficacy inference."}
        fit.atomic_json(attempt / "fit_smoke_manifest.json", manifest)
        ledger = attempt / "resource_ledger.jsonl"
        fit.append_jsonl(ledger, {"event": "GPU_SMOKE_STARTED", "at_utc": fit.utc_now(),
                                  "wsl_pid": process_id, "device_uuid": fit.GPU_UUID,
                                  "selected_models": ["transigen"], "reserved_gpu_hours": reservation,
                                  "diagnostic_cap_gpu_hours": MAX_GPU_SECONDS / 3600,
                                  "prior_package_settled_gpu_hours": accounting["settled_gpu_hours"],
                                  "pre_smoke_gpu_snapshot": snapshot["selected_gpu"],
                                  "runner_sha256": preflight["runner_sha256"],
                                  "diagnostic_runner_sha256": preflight["diagnostic_runner_sha256"],
                                  "config_sha256": config_sha})

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the WSL2 Conda environment")
        torch.cuda.set_device(torch.device("cuda:0"))
        device = torch.device("cuda:0")
        start_gpu = time.perf_counter()
        features = torch.from_numpy(feature_rows)
        dataset = TensorDataset(torch.from_numpy(x_ctl), torch.from_numpy(x_treated), features)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
                            num_workers=0, generator=torch.Generator().manual_seed(fit.SEED))
        model, module = fit._load_transigen(torch, device, fit.SEED, features_dim=2048)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        gene_variance = np.var(delta.astype(np.float64), axis=0, ddof=0)
        per_drug_baseline = np.stack([
            delta[train_keys.full_inchikey.astype(str).to_numpy() == key].mean(axis=0, dtype=np.float64).astype(np.float32)
            for key in train_keys.full_inchikey.astype(str)
        ], axis=0)
        baseline_metrics = fit.keyed_metrics(train_keys, delta, per_drug_baseline, gene_variance)
        progress_path = attempt / "attempt_progress.json"
        trajectory = attempt / "transigen_train_diagnostic.jsonl"
        last_prediction: np.ndarray | None = None
        last_eval_epoch: int | None = None
        first_abnormal: dict[str, Any] | None = None
        epoch = 0
        fit.atomic_json(progress_path, {"phase": "TRAIN_ONLY_TRANSIGEN_INITIALIZED",
                                       "updated_at_utc": fit.utc_now(), "wsl_pid": process_id,
                                       "gpu_hours": 0.0})
        fit.append_jsonl(trajectory, {"event": "INITIAL_STATE", "at_utc": fit.utc_now(),
                                      "model_state": _state_summary(torch, model),
                                      "train_per_drug_mean_baseline": baseline_metrics})
        while time.perf_counter() - start_gpu < MAX_GPU_SECONDS:
            epoch += 1
            model.train()
            loss_sum = np.zeros(len(LOSS_NAMES), dtype=np.float64)
            update_gradients: list[float] = []
            update_count = 0
            sample_count = 0
            for batch_index, batch in enumerate(loader):
                if time.perf_counter() - start_gpu >= MAX_GPU_SECONDS:
                    break
                batch = tuple(tensor.to(device) for tensor in batch)
                sample_count += int(batch[0].shape[0])
                optimizer.zero_grad(set_to_none=True)
                try:
                    x1, x2, predicted_treated, terms = fit._transigen_forward_batch(torch, model, batch, device)
                except BaseException as error:
                    first_abnormal = {"stage": "TRAIN_FORWARD_OR_LOSS_RAISED", "epoch": epoch,
                                      "batch_index": batch_index, "exception_type": type(error).__name__,
                                      "exception_message": str(error)}
                    fit.append_jsonl(trajectory, {"event": "FIRST_ABNORMAL", "at_utc": fit.utc_now(),
                                                  "diagnostic": first_abnormal,
                                                  "model_state": _state_summary(torch, model)})
                    fit.atomic_json(attempt / "train_diagnostic_partial_result.json", {
                        "status": "FAILED_CLOSED_TRAIN_FORWARD_ABNORMAL", "first_abnormal": first_abnormal,
                        "predictions_finite": False, "forbidden_roles_opened": []})
                    raise FloatingPointError(f"TranSiGen {first_abnormal['stage']}: {error}") from error
                term_values = [_array_summary(term) for term in terms]
                term_finite = [bool(torch.isfinite(term).all()) for term in terms]
                if not all(term_finite):
                    first_abnormal = {"stage": "TRAIN_LOSS_NONFINITE", "epoch": epoch,
                                      "batch_index": batch_index, "loss_names": list(LOSS_NAMES),
                                      "loss_finite": term_finite, "loss_summary": term_values,
                                      "model_state": _state_summary(torch, model)}
                    fit.append_jsonl(trajectory, {"event": "FIRST_ABNORMAL", "at_utc": fit.utc_now(),
                                                  "diagnostic": first_abnormal})
                    fit.atomic_json(attempt / "train_diagnostic_partial_result.json", {
                        "status": "FAILED_CLOSED_TRAIN_LOSS_NONFINITE", "first_abnormal": first_abnormal,
                        "predictions_finite": bool(torch.isfinite(predicted_treated).all()),
                        "forbidden_roles_opened": []})
                    raise FloatingPointError(f"non-finite TRAIN loss at epoch={epoch}, batch={batch_index}")
                terms[0].backward()
                gradient_norm = fit._grad_global_norm(torch, model)
                gradients_finite = math.isfinite(gradient_norm) and all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters())
                if not gradients_finite:
                    first_abnormal = {"stage": "TRAIN_GRADIENT_NONFINITE", "epoch": epoch,
                                      "batch_index": batch_index, "gradient_norm": gradient_norm,
                                      "loss_summary": term_values, "model_state": _state_summary(torch, model)}
                    fit.atomic_json(attempt / "train_diagnostic_partial_result.json", {
                        "status": "FAILED_CLOSED_TRAIN_GRADIENT_NONFINITE", "first_abnormal": first_abnormal,
                        "forbidden_roles_opened": []})
                    raise FloatingPointError(f"non-finite TRAIN gradients at epoch={epoch}, batch={batch_index}")
                update_gradients.append(gradient_norm)
                loss_sum += np.asarray([float(term.detach().cpu()) for term in terms], dtype=np.float64)
                optimizer.step()
                optimizer_finite = _optimizer_finite(torch, optimizer)
                model_summary = _state_summary(torch, model)
                update_count += 1
                fit.append_jsonl(trajectory, {"event": "TRAIN_UPDATE", "at_utc": fit.utc_now(),
                                              "epoch": epoch, "batch_index": batch_index,
                                              "batch_conditions": int(batch[0].shape[0]),
                                              "loss_names": list(LOSS_NAMES), "loss": term_values,
                                              "gradient_norm": gradient_norm,
                                              "post_step_parameters_and_batchnorm": model_summary,
                                              "optimizer_state": optimizer_finite})
                if (not optimizer_finite["all_finite"] or model_summary["nonfinite_parameter_names"]):
                    first_abnormal = {"stage": "POST_STEP_STATE_NONFINITE", "epoch": epoch,
                                      "batch_index": batch_index, "optimizer_state": optimizer_finite,
                                      "model_state": model_summary}
                    fit.atomic_json(attempt / "train_diagnostic_partial_result.json", {
                        "status": "FAILED_CLOSED_POST_STEP_STATE_NONFINITE", "first_abnormal": first_abnormal,
                        "forbidden_roles_opened": []})
                    raise FloatingPointError("TranSiGen parameters or optimizer state became non-finite after a TRAIN update")
            if not update_count:
                break
            if sample_count != len(train_keys):
                if time.perf_counter() - start_gpu >= MAX_GPU_SECONDS:
                    break
                raise RuntimeError("an ordinary TRAIN pass did not process exactly 419 conditions")
            if time.perf_counter() - start_gpu + 15.0 >= MAX_GPU_SECONDS:
                break

            # One eval-mode diagnostic over TRAIN only. Snapshot and compare
            # BatchNorm buffers to confirm eval did not recalibrate them.
            model.train(False)
            bn_before = _bn_state(torch, model)
            torch.manual_seed(fit.SEED + 10000 + epoch)
            torch.cuda.manual_seed_all(fit.SEED + 10000 + epoch)
            pred_chunks: list[np.ndarray] = []
            component_sums = np.zeros(len(LOSS_NAMES), dtype=np.float64)
            component_finite = np.ones(len(LOSS_NAMES), dtype=bool)
            observations = 0
            output_summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in
                ("x1_reconstruction", "x2_reconstruction", "treated_prediction", "mu1", "logvar1",
                 "mu2", "logvar2", "mu_pred", "logvar_pred", "z2_pred")}
            with torch.no_grad():
                eval_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0)
                for batch_index, batch in enumerate(eval_loader):
                    batch = tuple(tensor.to(device) for tensor in batch)
                    x1, x2, features_batch = batch
                    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = model.forward(x1, features_batch)
                    z2, mu2, logvar2 = model.encode_x2(x2)
                    x2_rec = model.decode_x2(z2)
                    terms = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec,
                                       mu2, logvar2, x2_pred, mu_pred, logvar_pred)
                    for j, term in enumerate(terms):
                        finite = bool(torch.isfinite(term).all())
                        component_finite[j] &= finite
                        if finite:
                            component_sums[j] += float(term.detach().cpu())
                    for name, value in (("x1_reconstruction", x1_rec), ("x2_reconstruction", x2_rec),
                                        ("treated_prediction", x2_pred), ("mu1", mu1), ("logvar1", logvar1),
                                        ("mu2", mu2), ("logvar2", logvar2), ("mu_pred", mu_pred),
                                        ("logvar_pred", logvar_pred), ("z2_pred", z2_pred)):
                        output_summaries[name].append(_array_summary(value))
                    pred_chunks.append((x2_pred - x1).detach().cpu().numpy().astype(np.float32))
                    observations += int(x1.shape[0])
            bn_after = _bn_state(torch, model)
            if observations != len(train_keys):
                raise RuntimeError("TRAIN eval pass did not read exactly 419 conditions")
            prediction = np.concatenate(pred_chunks, axis=0)
            last_prediction = prediction
            last_eval_epoch = epoch
            pred_finite = bool(np.isfinite(prediction).all())
            train_metric = fit.keyed_metrics(train_keys, delta, prediction, gene_variance) if pred_finite else None
            eval_losses = {name: (float(component_sums[i] / observations) if component_finite[i] else None)
                           for i, name in enumerate(LOSS_NAMES)}
            eval_record = {"event": "TRAIN_EVAL", "at_utc": fit.utc_now(), "epoch": epoch,
                           "training_raw_loss_per_condition": (loss_sum / sample_count).tolist(),
                           "train_prediction_metrics": train_metric,
                           "eval_loss_per_condition": eval_losses,
                           "eval_loss_component_finite": {name: bool(component_finite[i]) for i, name in enumerate(LOSS_NAMES)},
                           "prediction_finite": pred_finite, "prediction_summary": _array_summary(prediction),
                           # ``output_summaries`` already contains JSON-safe
                           # shape/range summaries; summarizing those dicts a
                           # second time coerces them to object arrays.
                           "eval_outputs": {name: pieces for name, pieces in output_summaries.items()},
                           "batchnorm_unchanged_during_eval": _bn_unchanged(bn_before, bn_after),
                           "batchnorm_before_eval": _state_summary(torch, model)["batchnorm_buffers"],
                           "gradient_norm_mean": float(np.mean(update_gradients)),
                           "gradient_norm_max": float(np.max(update_gradients)),
                           "post_eval_optimizer_state": _optimizer_finite(torch, optimizer),
                           "post_eval_model_state": _state_summary(torch, model)}
            fit.append_jsonl(trajectory, eval_record)
            first_bad_terms = [name for i, name in enumerate(LOSS_NAMES) if not component_finite[i]]
            if first_bad_terms:
                aux_bad_only = pred_finite and all(name != "raw_total_loss" for name in first_bad_terms)
                first_abnormal = {"stage": ("EVAL_AUX_LOSS_NONFINITE_TRAIN_PREDICTION_FINITE" if aux_bad_only
                                               else "EVAL_LOSS_NONFINITE_TRAIN_PREDICTION_FINITE" if pred_finite
                                               else "EVAL_LOSS_AND_TRAIN_PREDICTION_NONFINITE"),
                                  "epoch": epoch, "loss_terms_nonfinite": first_bad_terms,
                                  "all_eval_loss_terms": eval_losses, "prediction_finite": pred_finite,
                                  "train_prediction_metrics": train_metric}
                fit.atomic_json(attempt / "train_diagnostic_partial_result.json", {
                    "status": "EVAL_AUX_LOSS_NONFINITE_PREDICTION_FINITE" if pred_finite else "EVAL_MAIN_PREDICTION_NONFINITE",
                    "first_abnormal": first_abnormal, "forbidden_roles_opened": []})
                if not pred_finite:
                    if not first_bad_terms:
                        first_abnormal["stage"] = "EVAL_MAIN_PREDICTION_NONFINITE"
                    raise FloatingPointError("TranSiGen TRAIN main delta prediction is non-finite")
                break
            progress = {"phase": "TRAIN_ONLY_TRANSIGEN_DIAGNOSTIC_ACTIVE", "updated_at_utc": fit.utc_now(),
                        "wsl_pid": process_id, "completed_train_epochs": epoch,
                        "actual_gpu_hours_upper_bound": (time.perf_counter() - start_gpu) / 3600,
                        "last_train_nmse": train_metric["object_macro_nmse"] if train_metric else None,
                        "first_abnormal": first_abnormal}
            fit.atomic_json(progress_path, progress)
            if epoch == 1 or epoch % 5 == 0:
                print(json.dumps({"stage": progress["phase"], "epoch": epoch,
                                  "train_nmse": progress["last_train_nmse"],
                                  "elapsed_gpu_hours": progress["actual_gpu_hours_upper_bound"]}, ensure_ascii=False), flush=True)
            if time.perf_counter() - start_gpu >= MAX_GPU_SECONDS:
                break

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_gpu
        if elapsed > MAX_GPU_SECONDS + 1.0:
            raise TimeoutError("TranSiGen TRAIN-only diagnosis crossed its 30-minute hard wall limit")
        if last_prediction is None or last_eval_epoch is None:
            raise RuntimeError("diagnostic ended without at least one eval-mode TRAIN prediction")
        if np.isfinite(last_prediction).all():
            np.savez_compressed(attempt / "transigen_train_predictions.npz",
                                 train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
                                 train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
                                 train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
                                 genes=genes, train_target_delta=delta, train_prediction_delta=last_prediction)
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "model": "upstream TranSiGen", "diagnostic_epoch": last_eval_epoch,
                        "seed": fit.SEED}, attempt / "transigen_train_diagnostic_state.pt")
        status = ("EVAL_AUX_LOSS_NONFINITE_PREDICTION_FINITE" if first_abnormal and
                  first_abnormal["stage"] == "EVAL_AUX_LOSS_NONFINITE_TRAIN_PREDICTION_FINITE" else
                  "TRAIN_ONLY_DIAGNOSTIC_COMPLETED" if elapsed >= MAX_GPU_SECONDS - 10 else
                  "TRAIN_ONLY_DIAGNOSTIC_STOPPED_AT_FIRST_ABNORMAL" if first_abnormal else
                  "TRAIN_ONLY_DIAGNOSTIC_COMPLETED_WITHOUT_ABNORMALITY")
        result = {"status": status, "run_id": fit.RUN_ID, "attempt_id": attempt_id,
                  "diagnostic_runner_sha256": fit.sha256_file(Path(__file__).resolve()),
                  "runner_sha256": fit.sha256_file(Path(fit.__file__).resolve()),
                  "resource": {"actual_gpu_hours": elapsed / 3600, "hard_cap_gpu_seconds": MAX_GPU_SECONDS,
                               "reserved_gpu_hours": reservation, "gpu_uuid": fit.GPU_UUID},
                  "denominator": {"train_conditions": 419, "train_objects": 110,
                                  "full_inchikeys": int(train_keys.full_inchikey.nunique()), "genes": len(genes),
                                  "inner_dev_evaluated": False, "outer_locked_or_pdo_opened": False},
                  "first_abnormal": first_abnormal,
                  "last_train_prediction_metrics": fit.keyed_metrics(train_keys, delta, last_prediction, gene_variance),
                  "per_drug_train_mean_baseline_metrics": baseline_metrics,
                  "prediction_file": "transigen_train_predictions.npz" if (attempt / "transigen_train_predictions.npz").is_file() else None,
                  "prediction_sha256": fit.sha256_file(attempt / "transigen_train_predictions.npz")
                      if (attempt / "transigen_train_predictions.npz").is_file() else None,
                  "bn_behavior": "Only ordinary TRAIN updates changed running buffers; eval-mode diagnostic snapshots were required to remain exactly unchanged.",
                  "forbidden_roles_opened": [], "claim_boundary": "TRAIN-only model-state and fit diagnostic; not inner-dev, transfer, or PDO validation."}
        fit.atomic_json(attempt / "transigen_train_diagnostic_result.json", result)
        final_gpu = fit._device_snapshot()["selected_gpu"]
        close = {"status": "COMPLETED_TRAIN_ONLY_DIAGNOSTIC", "run_id": fit.RUN_ID,
                 "attempt": attempt_id, "wsl_pid": process_id, "device_uuid": fit.GPU_UUID,
                 "actual_gpu_hours": elapsed / 3600, "reserved_gpu_hours": reservation,
                 "unused_reservation_released_gpu_hours": reservation - elapsed / 3600,
                 "cumulative_conservative_settled_gpu_hours": accounting["settled_gpu_hours"] + elapsed / 3600,
                 "runner_sha256": preflight["runner_sha256"],
                 "diagnostic_runner_sha256": preflight["diagnostic_runner_sha256"],
                 "diagnostic_result_sha256": fit.sha256_file(attempt / "transigen_train_diagnostic_result.json"),
                 "first_abnormal": first_abnormal, "device_snapshot_at_close": final_gpu}
        fit.atomic_json(attempt / "attempt_close_receipt.json", close)
        fit.append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": fit.utc_now(),
                                  "wsl_pid": process_id, "device_uuid": fit.GPU_UUID,
                                  "selected_models": ["transigen"], "actual_gpu_hours": elapsed / 3600,
                                  "reserved_gpu_hours": reservation,
                                  "unused_reservation_released_gpu_hours": reservation - elapsed / 3600,
                                  "cumulative_conservative_charge_including_prior_attempts_gpu_hours": accounting["settled_gpu_hours"] + elapsed / 3600,
                                  "close_receipt": "attempt_close_receipt.json"})
        fit.atomic_json(progress_path, {"phase": "CLOSED_TRAIN_ONLY_DIAGNOSTIC", "updated_at_utc": fit.utc_now(),
                                        "wsl_pid": process_id, "actual_gpu_hours": elapsed / 3600,
                                        "first_abnormal": first_abnormal})
        print(json.dumps({"stage": status, "attempt": attempt_id,
                          "gpu_hours": elapsed / 3600, "first_abnormal": first_abnormal,
                          "train_metrics": result["last_train_prediction_metrics"]}, ensure_ascii=False), flush=True)
        return result
    except BaseException:
        raise

