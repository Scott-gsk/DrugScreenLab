"""Label-blind CPU structure, input, parameter, initialization, and gradient gate for EXP-048."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from drug_screen.models.exp048_fusion import (
    ADDED_OUTPUT_INIT_STD,
    ARMS,
    EXP048FusionArm,
    check_capacity_tolerance,
    finite_positive_gradient,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/EXP-048.yaml"
INPUT_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/receipt.json"
BINDING_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_target_feature_binding_v1/receipt.json"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
TARGET_INPUTS = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
SOURCE_VECTOR = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/source_reference_perturbation.npy"
SOURCE_CHECKPOINT = ROOT / "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
OFFICIAL_SOURCE = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
OUTPUT = ROOT / "artifacts/experiments/EXP-048/20260926_five_arm_cpu_preflight_v1"
EXPECTED_INPUT_RECEIPT_SHA256 = "7f70be0fdf1106038463455dd4d2bf25a01b61b140f2b78e9046c60eb24283e2"
EXPECTED_BINDING_RECEIPT_SHA256 = "4877ba04d628258033f42770728bce52c56bcf94a2c0335a20464bb80fae6caf"
SEED = 20260922


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_exp048", path)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer source could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: {actual} != {expected}")


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(value.float()))).detach().cpu())


def main() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None}:
        raise RuntimeError("EXP-048 preflight must be CPU-only")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(SEED)

    require_sha(INPUT_RECEIPT, EXPECTED_INPUT_RECEIPT_SHA256, "source/key receipt")
    require_sha(BINDING_RECEIPT, EXPECTED_BINDING_RECEIPT_SHA256, "target-feature binding receipt")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source_receipt = json.loads(INPUT_RECEIPT.read_text(encoding="utf-8"))
    binding_receipt = json.loads(BINDING_RECEIPT.read_text(encoding="utf-8"))
    target_manifest_path = TARGET_INPUTS / "manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    if binding_receipt.get("status") != "TARGET_FEATURE_IDENTITY_AND_SHARED_KEY_MASK_PASS":
        raise ValueError("previous EXP-048 target identity binding is not PASS")
    if sha256(target_manifest_path) != source_receipt["target_key_coverage"]["target_input_manifest_sha256"]:
        raise ValueError("reused target feature manifest changed")
    for filename, expected in target_manifest["files_sha256"].items():
        if sha256(TARGET_INPUTS / filename) != expected:
            raise ValueError(f"reused target feature checksum changed: {filename}")

    panel_indexes = [int(value) for value in config["drug_panel"]["target_indices"]]
    panel_keys = [str(value) for value in config["drug_panel"]["target_inchikeys"]]
    panel_pert_ids = [str(value) for value in config["pert_ids"]]
    if not (len(panel_indexes) == len(panel_keys) == len(panel_pert_ids) == 9):
        raise ValueError("frozen EXP-048 identity axis must contain exactly nine drugs")
    source_id_rows = binding_receipt["drug_identity_binding"]
    if ([int(row["drug_index"]) for row in source_id_rows] != panel_indexes
            or [str(row["target_inchikey"]) for row in source_id_rows] != panel_keys
            or [str(row["lincs_pert_id"]) for row in source_id_rows] != panel_pert_ids):
        raise ValueError("source OOF vector order is not bound to the frozen target/pert_id identity order")
    patients = [str(value) for value in json.loads((TARGET_INPUTS / "patient_axis.json").read_text(encoding="utf-8"))]
    expression = np.load(TARGET_INPUTS / "patient_expression.npy", mmap_mode="r", allow_pickle=False)
    ordinal = np.load(TARGET_INPUTS / "drug_ordinal128.npy", mmap_mode="r", allow_pickle=False)
    source_vector = np.load(SOURCE_VECTOR, mmap_mode="r", allow_pickle=False)
    if expression.shape != (51, 19091) or not np.isfinite(expression).all():
        raise ValueError("verified target expression input no longer matches the 51x19091 axis")
    if ordinal.shape != (14, 128) or not np.issubdtype(ordinal.dtype, np.number):
        raise ValueError("verified target drug feature axis changed")
    if source_vector.shape != (9, 978) or not np.isfinite(source_vector).all():
        raise ValueError("verified source OOF vector no longer matches the 9x978 axis")
    if sha256(SOURCE_VECTOR) != source_receipt["source_oof"]["vector_sha256"]:
        raise ValueError("source OOF vector is not bound to the accepted source receipt")

    drug_axis = pd.read_parquet(TARGET_INPUTS / "drug_axis.parquet", columns=["drug_index", "drug_id"])
    drug_id_by_index = {int(row.drug_index): str(row.drug_id) for row in drug_axis.itertuples(index=False)}
    if [drug_id_by_index.get(index) for index in panel_indexes] != panel_keys:
        raise ValueError("frozen target drug indices and InChIKeys no longer agree")
    if len(patients) != 51 or len(set(patients)) != 51:
        raise ValueError("target patient axis must contain 51 unique patients")

    mask_path = ROOT / binding_receipt["target_key_mask"]["path"]
    if sha256(mask_path) != binding_receipt["target_key_mask"]["sha256"]:
        raise ValueError("shared patient-key mask changed after independent identity binding")
    key_mask = pd.read_csv(mask_path)
    if len(key_mask) != 459 or not key_mask.groupby("patient_id")["target_key_present"].sum().eq(7).all():
        raise ValueError("common observed-key mask must retain 459 keys and seven per patient")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    fold0 = split["target_folds"][0]
    fraction = next(row for row in fold0["label_fractions"] if float(row["fraction"]) == 0.5)
    train_patients = [str(value) for value in fraction["train_ids"]]
    if not train_patients or any(value not in set(patients) for value in train_patients):
        raise ValueError("frozen outer-fold-0 training patient axis is invalid")
    chosen_patient = train_patients[0]
    patient_index = patients.index(chosen_patient)
    patient_mask = key_mask.loc[
        key_mask["patient_id"].astype(str).eq(chosen_patient)
        & key_mask["target_key_present"].astype(bool)
    ].copy()
    if len(patient_mask) != 7:
        raise ValueError("preflight patient must have exactly seven admitted observed keys")
    panel_position = {key: index for index, key in enumerate(panel_keys)}
    positions = [panel_position[str(key)] for key in patient_mask["drug_id"]]
    if len(set(positions)) != 7:
        raise ValueError("preflight observed drug keys are not unique")
    drug_row_by_key = {key: index for key, index in zip(panel_keys, panel_indexes)}
    selected_keys = [str(key) for key in patient_mask["drug_id"]]
    selected_drug_rows = [drug_row_by_key[key] for key in selected_keys]
    selected_positions = [panel_position[key] for key in selected_keys]
    smiles = torch.from_numpy(np.asarray(ordinal[selected_drug_rows], dtype=np.float32).copy())
    expression_batch = torch.from_numpy(
        np.repeat(np.asarray(expression[patient_index:patient_index + 1], dtype=np.float32), len(selected_keys), axis=0)
    )
    perturbation = torch.from_numpy(np.asarray(source_vector[selected_positions], dtype=np.float32).copy())
    shuffled_positions = [(position + 1) % 9 for position in selected_positions]
    shuffled_perturbation = torch.from_numpy(np.asarray(source_vector[shuffled_positions], dtype=np.float32).copy())
    if (len(set(shuffled_positions)) != len(shuffled_positions)
            or any(source == destination for source, destination in zip(selected_positions, shuffled_positions))):
        raise ValueError("fixed cyclic perturbation derangement is not one-to-one and nonidentity")

    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel":
        raise ValueError("registered source checkpoint is incompatible")
    official = load_official(OFFICIAL_SOURCE)
    model_config = checkpoint["model_config"]
    reference_predictions: np.ndarray | None = None
    arm_reports: dict[str, dict] = {}
    maximum_new_module_rms_ratio = 0.0
    capacity = None

    for arm in ARMS:
        projected = expression_latent = delta = residual = direct = None
        torch.manual_seed(SEED)
        backbone = official.CombinedModel(
            gene_input_size=model_config["gene_input_size"],
            gene_hidden_size=model_config["gene_hidden_size"],
            drug_hidden_size=model_config["drug_hidden_size"],
            feature_dim=model_config["feature_dim"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
        )
        backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = EXP048FusionArm(backbone, arm, seed=SEED)
        model.eval()
        expression_input = None if arm == "F-B" else expression_batch
        arm_perturbation = (
            perturbation if arm in {"F-B", "F-F"}
            else shuffled_perturbation if arm == "F-S"
            else None
        )

        if arm == "F-B":
            with torch.no_grad():
                projected = model.perturbation_delta(perturbation)
                input_norm = float(torch.linalg.vector_norm(perturbation).cpu())
                output_norm = float(torch.linalg.vector_norm(projected).cpu())
            init_ratio = output_norm / input_norm if input_norm > 0.0 else float("inf")
            if init_ratio > 0.01:
                raise ValueError("F-B perturbation projection exceeds the frozen 1% initial norm gate")
            latent_ratio = init_ratio
        else:
            with torch.no_grad():
                expression_latent = model.encode_expression(expression_batch)
                if arm in {"F-F", "F-S"}:
                    delta = model.perturbation_delta(arm_perturbation)
                    latent_ratio = rms(delta) / max(rms(expression_latent), 1e-12)
                elif arm == "F-C":
                    residual = model.direct_residual(expression_latent)
                    latent_ratio = rms(residual) / max(rms(expression_latent), 1e-12)
                else:
                    latent_ratio = 0.0
            if latent_ratio > 0.01:
                raise ValueError(f"{arm} added-module RMS ratio exceeds the frozen 1% initialization gate")

        with torch.no_grad():
            predictions = model(expression_input, smiles, arm_perturbation).flatten()
        if predictions.shape != (7,) or not bool(torch.isfinite(predictions).all()):
            raise ValueError(f"{arm} failed the seven-key finite CPU forward path")
        if arm == "F-A":
            with torch.no_grad():
                direct = backbone(expression_batch, smiles).flatten()
            if not torch.allclose(predictions, direct, rtol=0.0, atol=1e-7):
                raise ValueError("F-A wrapper differs from the official direct forward path")
            reference_predictions = predictions.detach().cpu().numpy().copy()
        assert reference_predictions is not None or arm == "F-A"
        if arm in {"F-F", "F-C", "F-S"} and reference_predictions is not None:
            output_change = float(np.sqrt(np.mean(np.square(predictions.detach().cpu().numpy() - reference_predictions))))
        else:
            output_change = 0.0
        if arm == "F-B":
            output_change = None  # It is a separate input arm, not a paired direct delta.

        if arm in {"F-F", "F-C", "F-S"}:
            added_params = model.added_module_parameter_count()
            if arm == "F-C":
                capacity = check_capacity_tolerance(model.gene_hidden_size)
                if added_params != int(capacity["F-C_residual_adapter"]):
                    raise ValueError("F-C implementation parameter count differs from frozen formula")
            if arm == "F-F" and added_params != int(check_capacity_tolerance(model.gene_hidden_size)["F-F_projection"]):
                raise ValueError("F-F implementation parameter count differs from frozen formula")
        else:
            added_params = model.added_module_parameter_count()

        model.train()
        model.zero_grad(set_to_none=True)
        pseudo_target = torch.linspace(-0.5, 0.5, 7, dtype=torch.float32)
        pseudo_loss = F.smooth_l1_loss(model(expression_input, smiles, arm_perturbation).flatten(), pseudo_target)
        pseudo_loss.backward()
        new_gradients = finite_positive_gradient(model.added_module_parameters())
        if arm != "F-A" and new_gradients["nonzero_gradient_tensors"] != new_gradients["parameter_tensors"]:
            raise ValueError(f"{arm} did not deliver finite nonzero gradients to all added parameter tensors")
        if arm == "F-A":
            backbone_gradient = any(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                and float(torch.linalg.vector_norm(parameter.grad).detach().cpu()) > 0.0
                for parameter in model.parameters() if parameter.requires_grad
            )
            if not backbone_gradient:
                raise ValueError("F-A did not deliver a finite gradient to its direct model")

        trainable_total = model.trainable_parameter_count()
        if arm == "F-B" and any(parameter.requires_grad for parameter in backbone.feature_extractor.gene_fc1.parameters()):
            raise ValueError("F-B expression encoder parameters must be excluded from optimization")
        arm_reports[arm] = {
            "status": "CPU_FORWARD_AND_GRADIENT_PASS",
            "expression_passed_to_model": arm != "F-B",
            "expression_encoder_executed": arm != "F-B",
            "finite_prediction_count": int(predictions.numel()),
            "source_perturbation_rows": 0 if arm == "F-A" or arm == "F-C" else 7,
            "added_module_parameters": added_params,
            "total_trainable_parameters": trainable_total,
            "new_module_gradient": new_gradients,
            "initial_added_latent_rms_ratio": float(latent_ratio),
            "initial_output_rmse_vs_F_A_not_a_performance_metric": output_change,
        }
        if arm != "F-A":
            maximum_new_module_rms_ratio = max(maximum_new_module_rms_ratio, float(latent_ratio))
        del model, backbone, predictions, pseudo_loss, expression_input, arm_perturbation
        del projected, expression_latent, delta, residual, direct
        gc.collect()

    if capacity is None:
        raise ValueError("F-C/F-F parameter-capacity gate did not execute")
    if maximum_new_module_rms_ratio > 0.01:
        raise ValueError("new module initialization exceeded the frozen global 1% latent RMS gate")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    report = {
        "schema_version": 1,
        "experiment_id": "EXP-048",
        "artifact_id": OUTPUT.name,
        "status": "FIVE_ARM_REAL_INPUT_CPU_STRUCTURE_INIT_MASK_AND_GRADIENT_PASS",
        "created_at_utc": now,
        "runtime": {"environment": "WSL2 Conda drugscreening-gpu", "device": "cpu", "cuda_visible_devices": "", "gpu_hours": 0},
        "model_source": {
            "checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "official_source": str(OFFICIAL_SOURCE.relative_to(ROOT)),
            "official_source_sha256": sha256(OFFICIAL_SOURCE),
            "checkpoint_variant": checkpoint["variant"],
            "model_config": model_config,
            "sequence_length": int((model_config["gene_hidden_size"] + model_config["drug_hidden_size"]) // model_config["feature_dim"]),
            "expression_slots": int(model_config["gene_hidden_size"] // model_config["feature_dim"]),
            "drug_slots": int(model_config["drug_hidden_size"] // model_config["feature_dim"]),
        },
        "input_binding": {
            "source_receipt_sha256": sha256(INPUT_RECEIPT),
            "target_binding_receipt_sha256": sha256(BINDING_RECEIPT),
            "target_feature_manifest_sha256": sha256(target_manifest_path),
            "source_oof_vector_sha256": sha256(SOURCE_VECTOR),
            "target_patient_count_in_contract": 51,
            "patient_key_mask_rows_in_contract": int(len(key_mask)),
            "preflight_patient_count_forwarded": 1,
            "same_observed_drug_keys_all_arms": 7,
            "target_response_values_read": False,
            "target_predictions_or_outer_scores_read": False,
            "outer_test_labels_used": False,
        },
        "frozen_protocol": {
            "arms": list(ARMS),
            "candidate_grid": "EXP036_STANDARD_CANDIDATES_exactly_18_per_arm",
            "inner_folds": 3,
            "max_epochs": 100,
            "patience": 10,
            "final_seeds": [20260922, 20260923, 20260924],
            "new_output_initialization_std": ADDED_OUTPUT_INIT_STD,
            "new_output_bias": 0.0,
            "shuffled_panel_rule": "one-position-cyclic-shift_over_frozen_panel_order",
        },
        "capacity_match": capacity,
        "arms": arm_reports,
        "scientific_result": False,
        "notes": [
            "One outer-fold-0 training-side patient and its seven fixed observed keys were used for a CPU wiring check only.",
            "The pseudo-target gradient probe is an engineering connectivity check; no optimizer step or target label was used.",
            "F-B receives expression=None and its expression encoder is not executed or optimized.",
            "Initial output differences are implementation diagnostics, not predictive performance comparisons.",
        ],
        "next_gate": "EXP-036/041_core_completion_then_full_five_arm_CPU_cost_profile_against_the_existing_shared_40_GPU_hour_ledger; no_GPU_profile_or_formal_training_authorized_here",
    }
    OUTPUT.mkdir(parents=True)
    receipt_path = OUTPUT / "receipt.json"
    receipt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report["receipt_sha256"] = sha256(receipt_path)
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
