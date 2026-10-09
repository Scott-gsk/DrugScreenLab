#!/usr/bin/env python3
"""Read-only root verification of an EXP-051 TRAIN-only TranSiGen diagnostic.

Purpose: cross-check an immutable diagnostic's manifests, terminal resource receipt,
JSONL trajectory, TRAIN denominator, and first non-finite event without loading model
weights or any validation/PDO/external labels. Reuse the runner's JSON receipts rather
than duplicating its training logic; this verifies the recorded trajectory, not the
scientific correctness of the upstream model implementation.
Retention: reusable CPU-only verifier; writes a new receipt beside the attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def max_bn_running_variance(record: dict[str, Any]) -> float | None:
    candidates: list[dict[str, Any]] = []
    for key in ("post_eval_model_state", "post_step_parameters_and_batchnorm"):
        state = record.get(key)
        if isinstance(state, dict):
            buffers = state.get("batchnorm_buffers", {})
            if isinstance(buffers, dict):
                candidates.extend(
                    value for name, value in buffers.items()
                    if name.endswith("running_var") and isinstance(value, dict)
                )
    values = [float(item["max_abs"]) for item in candidates if item.get("finite") is True and item.get("max_abs") is not None]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    args = parser.parse_args()
    attempt_dir = args.attempt_dir.resolve()
    required = {
        name: attempt_dir / name for name in (
            "fit_smoke_manifest.json", "attempt_progress.json",
            "train_diagnostic_partial_result.json", "failure_close_receipt.json",
            "resource_ledger.jsonl", "transigen_train_diagnostic.jsonl",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing immutable evidence: " + ", ".join(missing))

    manifest = read_json(required["fit_smoke_manifest.json"])
    progress = read_json(required["attempt_progress.json"])
    partial = read_json(required["train_diagnostic_partial_result.json"])
    failure = read_json(required["failure_close_receipt.json"])
    ledger = [json.loads(line) for line in required["resource_ledger.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    events: list[dict[str, Any]] = []
    with required["transigen_train_diagnostic.jsonl"].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Non-object JSONL event at line {line_number}")
            events.append(item)

    scope = manifest.get("scope", {})
    first_abnormal = partial.get("first_abnormal", {})
    close_event = next((item for item in reversed(ledger) if item.get("event") == "GPU_ATTEMPT_FAILED_CLOSED"), None)
    eval_events = [item for item in events if item.get("event") == "TRAIN_EVAL"]
    update_events = [item for item in events if item.get("event") == "TRAIN_UPDATE"]
    train_metrics = [item.get("train_prediction_metrics", {}) for item in eval_events if isinstance(item.get("train_prediction_metrics"), dict)]
    nmse_values = [float(item["object_macro_nmse"]) for item in train_metrics if item.get("object_macro_nmse") is not None]
    finite_flags = [item.get("prediction_finite") for item in eval_events if "prediction_finite" in item]
    nonfinite_eval = next((item for item in eval_events if item.get("prediction_finite") is False), None)
    max_gradients = [float(item[key]) for item in events for key in ("gradient_norm", "gradient_norm_max") if item.get(key) is not None]
    bn_variances = [value for item in events if (value := max_bn_running_variance(item)) is not None]

    checks = {
        "attempt_id_matches": failure.get("attempt") == attempt_dir.name,
        "run_id_matches_manifest": failure.get("run_id") == manifest.get("run_id"),
        "closed_failed": failure.get("status") == "FAILED_CLOSED" and close_event is not None,
        "source_runner_hash_matches": sha256(args.runner.resolve()) == manifest.get("current_runner_sha256"),
        "train_only_scope": (
            scope.get("train_conditions") == 419 and scope.get("train_objects") == 110
            and scope.get("genes") == 978 and scope.get("inner_dev_evaluated") is False
            and scope.get("outer_or_locked_opened") is False and scope.get("PDO_sensitivity_opened") is False
        ),
        "no_forbidden_roles_opened": partial.get("forbidden_roles_opened") == [],
        "progress_matches_failure_receipt": progress.get("completed_train_epochs") == failure.get("last_progress", {}).get("completed_train_epochs"),
        "abnormal_epoch_matches_first_nonfinite_eval": (
            first_abnormal.get("epoch") == (nonfinite_eval or {}).get("epoch")
            and first_abnormal.get("prediction_finite") is False
        ),
        "all_prior_eval_predictions_finite": all(flag is True for flag in finite_flags[:-1]) if finite_flags else False,
        "train_evaluations_have_frozen_denominator": all(
            item.get("n_conditions") == 419 and item.get("n_source_objects") == 110
            and item.get("n_full_inchikeys") == 5 and item.get("n_genes_in_nmse_denominator") == 978
            for item in train_metrics
        ) and bool(train_metrics),
    }
    receipt = {
        "status": "ROOT_INDEPENDENT_TRAIN_DIAGNOSTIC_READBACK_PASS" if all(checks.values()) else "ROOT_INDEPENDENT_TRAIN_DIAGNOSTIC_READBACK_FAIL",
        "run_id": failure.get("run_id"),
        "attempt": failure.get("attempt"),
        "input_hashes": {key: sha256(path) for key, path in required.items()},
        "runner_sha256": sha256(args.runner.resolve()),
        "denominator": {key: scope.get(key) for key in (
            "train_conditions", "train_objects", "full_inchikeys", "genes",
            "inner_dev_evaluated", "outer_or_locked_opened", "PDO_sensitivity_opened",
        )},
        "trajectory": {
            "train_update_events": len(update_events),
            "train_eval_events": len(eval_events),
            "train_nmse_first": nmse_values[0] if nmse_values else None,
            "train_nmse_min_recorded": min(nmse_values) if nmse_values else None,
            "train_nmse_last_finite": nmse_values[-1] if nmse_values else None,
            "largest_recorded_gradient_norm": max(max_gradients) if max_gradients else None,
            "largest_finite_batchnorm_running_variance": max(bn_variances) if bn_variances else None,
            "first_abnormal_epoch": first_abnormal.get("epoch"),
            "first_abnormal_terms": first_abnormal.get("loss_terms_nonfinite"),
            "train_prediction_finite_at_failure": first_abnormal.get("prediction_finite"),
        },
        "resource": {
            "charged_gpu_hours_conservative_upper_bound": failure.get("charged_gpu_hours_conservative_upper_bound"),
            "released_unused_reservation_gpu_hours": failure.get("unused_reservation_released_gpu_hours"),
            "close_event_gpu_hours": close_event.get("charged_gpu_hours_conservative_upper_bound") if close_event else None,
        },
        "checks": checks,
        "interpretation_boundary": "This verifies the recorded TRAIN-only diagnostic and its numerical failure. It does not estimate inner-development or PDO transfer performance, and does not establish absence of biological signal.",
    }
    output = attempt_dir / "root_independent_train_diagnostic_readback.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
