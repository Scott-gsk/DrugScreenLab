"""Run the frozen EXP-050 source-domain five-arm development comparison.

Only the nine CTRPv2 TRAIN PatientID source objects are used. Existing XPert
predictions come from a model trained on source identities disjoint from these
objects. All arm fits, scalers and fusion weights are restricted to the outer
training objects; held-out labels are scored after each fold is frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_exp050_source_five_arm_development_v3"
CONFIG_REL = Path("configs/experiments/EXP-050-source-five-arm-development-v3.yaml")
OUT_REL = Path("artifacts/experiments/EXP-050") / RUN_ID
V2_RUNNER_SNAPSHOT_REL = OUT_REL / "previous_v2_runner_snapshot.py"
V2_SMOKE_FAILURE_REL = Path("artifacts/experiments/EXP-050/20261006_exp050_source_five_arm_development_v2/smoke_failure_attempt_001.json")
V3_CORRECTION_RECEIPT_REL = OUT_REL / "protocol_correction_receipt.md"
SYNTHETIC_GATE_CASE_REL = Path("tests/evaluation/test_exp050_source_five_arm_candidate_domain_gate.py")
SYNTHETIC_GATE_RECEIPT_REL = OUT_REL / "synthetic_candidate_domain_gate_receipt.json"
SOURCE_34_CODE_REL = Path("scripts/evaluation/analyze_exp050_independent_object_response_drug_effect.py")
XPERT_FIT_CODE_REL = Path("scripts/modeling/exp050_xpert_independent_response_fit.py")
PAIRED_REL = Path("artifacts/experiments/EXP-050/20261001_independent_object_response_drug_effect_v1/paired_object_drug_values.tsv")
ROLE_REL = Path("artifacts/experiments/EXP-050/20261001_independent_object_response_drug_effect_v1/object_roles_key_support.tsv")
SUPPORT_REL = Path("artifacts/experiments/EXP-050/20261001_independent_object_response_drug_effect_v1/train_val_exact_object_drug_key_support.tsv")
XPERT_REL = Path("artifacts/experiments/EXP-050/20261001_xpert_independent_object_fit_v1/attempts/attempt_001")
ASSET_AUDIT_REL = Path("artifacts/experiments/EXP-050/20261006_exp050_source_five_arm_development_v1/read_only_asset_audit")
PRED_KEYS_REL = XPERT_REL / "heldout_source_prediction_keys.tsv"
PRED_NPY_REL = XPERT_REL / "heldout_source_prediction_target_order.npy"
PRED_RECEIPT_REL = XPERT_REL / "attempt_receipt.json"
FIT_MAP_REL = XPERT_REL / "fit_condition_identity_key_map.tsv"
FIT_PROTOCOL_REL = XPERT_REL / "training_protocol_snapshot.json"
ID_SPLIT_REL = Path("artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json")
EXPR_REL = Path("data/processed/depmap/24q2_rnaseq_exact978/ccle_24q2_exact978_log2tpm.npy")
EXPR_MAP_REL = Path("data/processed/depmap/24q2_rnaseq_exact978/ccle_24q2_exact978_mapping.json")
EXPR_AUDIT_REL = Path("data/processed/depmap/24q2_rnaseq_exact978/intake_audit.json")
MODEL_REL = Path("data/raw/depmap/24q2/Model.csv")
GENES_REL = Path("data/processed/lincs/GSE92742/continuous_contract_v2/genes.json")
DRUG_REL = Path("data/processed/lincs/GSE92742/continuous_inputs_v2/drugs.parquet")
DRUG_MORGAN_REL = Path("data/processed/lincs/GSE92742/continuous_inputs_v2/drug_morgan2048.npy")
EXPECTED_TRAIN_OBJECTS = 9
EXPECTED_GENES = 978
MORGAN_DIM = 2048
EXPECTED_NONUNIQUE_MORGAN_KEYS = {
    "BNRNXUUZRGQAQC-UHFFFAOYSA-N",
    "GCIKSSRWRFVXBI-UHFFFAOYSA-N",
}
EXPECTED_VALID_CANDIDATE_KEYS = 121
RIDGE_ALPHA = 1.0
CPU_CAP_MINUTES = 240.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def input_paths() -> dict[str, Path]:
    rels = {
        "config": CONFIG_REL, "source_3_4_pipeline_code": SOURCE_34_CODE_REL,
        "xpert_fit_source_code": XPERT_FIT_CODE_REL,
        "v1_readonly_asset_audit_code": ASSET_AUDIT_REL / "verify_exp050_source_asset_contract.py",
        "v1_readonly_asset_audit_receipt": ASSET_AUDIT_REL / "audit_receipt.json",
        "v1_readonly_asset_audit_pair_details": ASSET_AUDIT_REL / "per_object_key_exposure_and_fit_support.tsv",
        "v1_readonly_asset_audit_morgan_details": ASSET_AUDIT_REL / "exact_key_morgan_asset_audit.tsv",
        "paired_source_asset": PAIRED_REL, "source_role_support": ROLE_REL,
        "source_object_drug_support": SUPPORT_REL, "xpert_prediction_keys": PRED_KEYS_REL,
        "xpert_prediction": PRED_NPY_REL,
        "xpert_attempt_receipt": PRED_RECEIPT_REL, "xpert_identity_key_map": FIT_MAP_REL,
        "xpert_training_protocol": FIT_PROTOCOL_REL, "frozen_identity_split": ID_SPLIT_REL,
        "depmap_expression": EXPR_REL, "depmap_row_identity": EXPR_MAP_REL,
        "depmap_gene_axis_audit": EXPR_AUDIT_REL, "depmap_model_patient": MODEL_REL,
        "lincs_gene_axis": GENES_REL,
        "lincs_drug_registry": DRUG_REL, "lincs_drug_morgan2048": DRUG_MORGAN_REL,
        "previous_v2_smoke_failure_receipt": V2_SMOKE_FAILURE_REL,
        "previous_v2_runner_snapshot": V2_RUNNER_SNAPSHOT_REL,
        "v3_protocol_correction_receipt": V3_CORRECTION_RECEIPT_REL,
        "synthetic_candidate_domain_gate_case": SYNTHETIC_GATE_CASE_REL,
        "synthetic_candidate_domain_gate_receipt": SYNTHETIC_GATE_RECEIPT_REL,
        "previous_v1_cpu_accounting": Path("artifacts/experiments/EXP-050/20261006_exp050_source_five_arm_development_v1/cpu_accounting.json"),
        "previous_v2_cpu_accounting": Path("artifacts/experiments/EXP-050/20261006_exp050_source_five_arm_development_v2/cpu_accounting.json"),
        "cpu_budget_carryover_receipt": OUT_REL / "cpu_budget_carryover.json",
    }
    return {name: ROOT / rel for name, rel in rels.items()}


def read_train_labels(train_objects: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project only CTRPv2 TRAIN row values; other endpoint/role values are skipped."""
    rows: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {"paired_table_rows": 0, "rows_by_endpoint_role": {},
                                  "unique_object_key_pairs_by_endpoint_role": {}}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    with (ROOT / PAIRED_REL).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        needed = {"endpoint", "split_role", "source_object_id", "full_inchikey", "sensitivity"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"paired 3.4 asset lacks columns: {sorted(needed - set(reader.fieldnames or []))}")
        for raw in reader:
            row_counts["paired_table_rows"] += 1
            endpoint = raw["endpoint"]
            role = raw["split_role"]
            role_key = f"{endpoint}|{role}"
            row_counts["rows_by_endpoint_role"][role_key] = row_counts["rows_by_endpoint_role"].get(role_key, 0) + 1
            metadata_obj = str(raw["source_object_id"])
            metadata_key = str(raw["full_inchikey"]).strip().upper()
            pair_sets.setdefault(role_key, set()).add((metadata_obj, metadata_key))
            if endpoint != "CTRPv2" or role != "train":
                continue
            obj = metadata_obj
            if obj not in train_objects:
                raise ValueError(f"CTRPv2 TRAIN label row has an unregistered TRAIN object: {obj}")
            key = str(raw["full_inchikey"]).strip().upper()
            if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
                raise ValueError(f"invalid exact full InChIKey on allowed row: {obj}/{key}")
            y = float(raw["sensitivity"])
            if not np.isfinite(y):
                raise ValueError(f"non-finite allowed CTRPv2 label: {obj}/{key}")
            rows.append({"source_object_id": obj, "full_inchikey": key, "sensitivity": y})
            row_counts["CTRPv2_TRAIN_numeric_rows"] = row_counts.get("CTRPv2_TRAIN_numeric_rows", 0) + 1
    row_counts["unique_object_key_pairs_by_endpoint_role"] = {
        key: len(value) for key, value in sorted(pair_sets.items())
    }
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["source_object_id", "full_inchikey"]).any():
        raise ValueError("allowed CTRPv2 TRAIN source-object/full-InChIKey labels are empty or duplicated")
    if set(frame.source_object_id.astype(str)) != train_objects:
        raise ValueError("the paired source asset does not contain all nine registered CTRPv2 TRAIN objects")
    return frame.sort_values(["source_object_id", "full_inchikey"]).reset_index(drop=True), row_counts


def verified_xpert_exposure(train_objects: set[str], targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt = json.loads((ROOT / PRED_RECEIPT_REL).read_text(encoding="utf-8"))
    for rel in (PRED_KEYS_REL, PRED_NPY_REL, FIT_MAP_REL):
        expected = receipt.get("output_sha256", {}).get(rel.name)
        if not expected or sha256(ROOT / rel) != expected:
            raise ValueError(f"XPert frozen attempt output hash mismatch: {rel}")
    fit_map = pd.read_csv(ROOT / FIT_MAP_REL, sep="\t",
                          usecols=["condition_id", "source_row", "split_role", "source_object_id", "full_inchikey", "pert_dose", "pert_time"],
                          dtype={"split_role": "string", "source_object_id": "string", "full_inchikey": "string"})
    fit_map["condition_id"] = pd.to_numeric(fit_map.condition_id, errors="raise").astype(int)
    fit_map["source_row"] = pd.to_numeric(fit_map.source_row, errors="raise").astype(int)
    fit_map["full_inchikey"] = fit_map.full_inchikey.astype(str).str.strip().str.upper()
    fit_map["pert_dose"] = pd.to_numeric(fit_map.pert_dose, errors="raise").astype(float)
    fit_map["pert_time"] = pd.to_numeric(fit_map.pert_time, errors="raise").astype(float)
    xpert_train = set(fit_map.loc[fit_map.split_role.eq("train"), "source_object_id"].dropna().astype(str))
    xpert_dev = set(fit_map.loc[fit_map.split_role.isin(["inner_dev", "validation", "dev"]), "source_object_id"].dropna().astype(str))
    overlap = sorted(train_objects & xpert_train)
    if overlap:
        raise ValueError(f"XPert fit-trained source identities overlap CTRPv2 TRAIN objects: {overlap}")
    target_train = targets.loc[targets.source_object_id.astype(str).isin(train_objects)].copy()
    target_objects = set(target_train.source_object_id.astype(str))
    if target_objects != train_objects:
        raise ValueError(f"XPert frozen prediction asset missing TRAIN objects: {sorted(train_objects - target_objects)}")
    if not target_train.full_inchikey.astype(str).str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").all():
        raise ValueError("XPert target keys include invalid complete InChIKeys")
    target_train["condition_id"] = pd.to_numeric(target_train.condition_id, errors="raise").astype(int)
    target_train["source_row"] = pd.to_numeric(target_train.source_row, errors="raise").astype(int)
    target_map = fit_map.loc[fit_map.split_role.eq("heldout_source") & fit_map.source_object_id.astype(str).isin(train_objects),
                             ["condition_id", "source_row", "source_object_id", "full_inchikey", "pert_dose", "pert_time"]].copy()
    if target_map.condition_id.duplicated().any():
        raise ValueError("hash-frozen XPert attempt map has duplicate allowed heldout condition_id")
    target_train = target_train.merge(target_map,
        on=["condition_id", "source_row", "source_object_id", "full_inchikey"], how="left",
        validate="one_to_one", indicator=True)
    if len(target_train) != int(target_train._merge.eq("both").sum()):
        raise ValueError("one or more allowed XPert target rows differ from the hash-frozen attempt identity/exposure map")
    target_train.drop(columns=["_merge"], inplace=True)
    if not np.isfinite(target_train[["pert_dose", "pert_time"]].to_numpy(dtype=float)).all():
        raise ValueError("allowed XPert source condition dose/time metadata is non-finite")
    if not np.allclose(target_train.pert_dose, 10.0, rtol=0.0, atol=1e-12) or not np.allclose(target_train.pert_time, 24.0, rtol=0.0, atol=1e-12):
        raise ValueError("the frozen EXP-050 3.5 input contract requires exact LINCS 10 uM x 24 h conditions")
    if "prediction_row" not in target_train.columns:
        raise ValueError("the frozen prediction row-order key is missing")
    positions = target_train.prediction_row.astype(int).to_numpy()
    pred = np.load(ROOT / PRED_NPY_REL, mmap_mode="r", allow_pickle=False)
    if pred.shape != (len(targets), EXPECTED_GENES):
        raise ValueError(f"frozen XPert prediction axis mismatch: {pred.shape}")
    # Index only the nine authorized TRAIN objects. No measured perturbation array is read.
    selected_pred = np.asarray(pred[positions], dtype=np.float32)
    if not np.isfinite(selected_pred).all():
        raise ValueError("allowed TRAIN source predicted perturbation array contains non-finite values")
    result_rows: list[dict[str, Any]] = []
    for (obj, key), positions_local in target_train.groupby(["source_object_id", "full_inchikey"], sort=True).indices.items():
        idx = np.asarray(positions_local, dtype=np.int64)
        result_rows.append({
            "source_object_id": str(obj), "full_inchikey": str(key),
            "predicted_delta": np.median(selected_pred[idx], axis=0).astype(np.float32),
            "n_conditions": int(len(idx)),
            "predicted_perturbation_strength": float(np.abs(np.median(selected_pred[idx], axis=0)).mean()),
        })
    features = pd.DataFrame(result_rows)
    if features.duplicated(["source_object_id", "full_inchikey"]).any():
        raise ValueError("frozen XPert object/full-InChIKey response aggregation is not unique")
    audit = {
        "attempt_id": receipt.get("attempt_id"), "frozen_model": "XPertNet attempt_001",
        "xpert_training_source_object_count": len(xpert_train),
        "xpert_training_source_object_ids": sorted(xpert_train),
        "xpert_inner_dev_source_object_count": len(xpert_dev),
        "ctrpv2_train_source_object_count": len(train_objects),
        "ctrpv2_train_source_object_ids": sorted(train_objects),
        "fit_train_ctrpv2_train_identity_overlap": overlap,
        "identity_overlap_status": "PASS_ZERO_OVERLAP",
        "target_train_condition_rows": int(len(target_train)),
        "target_train_object_drug_pairs": int(len(features)),
        "target_condition_dose_uM": sorted(set(target_train.pert_dose.astype(float).tolist())),
        "target_condition_time_h": sorted(set(target_train.pert_time.astype(float).tolist())),
        "target_dose_time_condition_counts": {f"{float(dose):g}uM_x_{float(hours):g}h": int(count)
                                               for (dose, hours), count in target_train.groupby(["pert_dose", "pert_time"], sort=True).size().items()},
        "condition_map_source": "attempt_001 fit_condition_identity_key_map.tsv, matched by condition_id/source_row/source_object_id/full_inchikey and verified against its attempt_receipt output SHA; continuous_inputs_v2 exposure condition IDs are not joined",
        "prediction_array_row_binding": "prediction array positions are row order of heldout_source_prediction_keys.tsv; source_row is a global LINCS matrix index, not a prediction output position",
        "aggregation": "median of signed predicted 978-gene deltas per source_object_id x exact full InChIKey, across all frozen matched 10 uM x 24 h target conditions",
        "prediction_array_axis": "row order follows heldout_source_prediction_keys.tsv; source_row identifies a row in the global LINCS matrix and is not a prediction-array index",
        "val_prediction_values_accessed": False,
        "outer_or_locked_values_accessed": False,
        "pdo_values_accessed": False,
    }
    return features, audit


def load_expression(train_objects: set[str], genes: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    audit = json.loads((ROOT / EXPR_AUDIT_REL).read_text(encoding="utf-8"))
    actual_axis = hashlib.sha256(("\n".join(genes) + "\n").encode("utf-8")).hexdigest()
    expected_axis = audit["gene_harmonization"]["ordered_gene_ids_sha256"]
    if actual_axis != expected_axis:
        raise ValueError("DepMap exact978 ordered gene axis differs from frozen LINCS target gene axis")
    map_rows = pd.DataFrame(json.loads((ROOT / EXPR_MAP_REL).read_text(encoding="utf-8")))
    need = {"depmap_id", "exact978_row"}
    if not need.issubset(map_rows.columns) or map_rows.depmap_id.astype(str).duplicated().any():
        raise ValueError("DepMap exact978 row identity map is incomplete or duplicated")
    model = pd.read_csv(ROOT / MODEL_REL, usecols=["ModelID", "PatientID"], dtype="string")
    if model.ModelID.astype(str).duplicated().any():
        raise ValueError("DepMap ModelID to PatientID mapping is not unique")
    joined = map_rows.merge(model, left_on="depmap_id", right_on="ModelID", how="inner", validate="one_to_one")
    allowed = joined.loc[joined.PatientID.astype(str).isin(train_objects)].copy()
    if set(allowed.PatientID.astype(str)) != train_objects:
        raise ValueError(f"DepMap baseline expression is missing TRAIN source objects: {sorted(train_objects - set(allowed.PatientID.astype(str)))}")
    if allowed.exact978_row.duplicated().any():
        raise ValueError("DepMap source object row projection contains duplicate expression rows")
    matrix = np.load(ROOT / EXPR_REL, mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_GENES:
        raise ValueError(f"DepMap expression matrix is not the frozen 978-gene axis: {matrix.shape}")
    rows: dict[str, np.ndarray] = {}
    id_counts: dict[str, int] = {}
    for obj, group in allowed.groupby("PatientID", sort=True):
        idx = group.exact978_row.astype(int).to_numpy()
        if (idx < 0).any() or (idx >= matrix.shape[0]).any():
            raise ValueError(f"DepMap expression row index out of range for {obj}")
        values = np.asarray(matrix[idx], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite source baseline expression for allowed TRAIN object {obj}")
        rows[str(obj)] = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        id_counts[str(obj)] = int(len(idx))
    return rows, {
        "matrix_shape": list(matrix.shape), "selected_patient_objects": len(rows),
        "selected_depmap_models_by_object": id_counts,
        "expression_aggregation": "mean across DepMap Public 24Q2 ModelID rows sharing the frozen TRAIN PatientID",
        "gene_count": EXPECTED_GENES, "ordered_gene_axis_sha256": actual_axis,
        "target_val_or_locked_expression_rows_read": False,
    }


def load_drug_features(keys: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    registry = pd.read_parquet(ROOT / DRUG_REL, columns=["drug_index", "pert_id", "canonical_smiles", "inchi_key"])
    registry["inchi_key"] = registry.inchi_key.astype(str).str.strip().str.upper()
    morgan = np.load(ROOT / DRUG_MORGAN_REL, mmap_mode="r", allow_pickle=False)
    if morgan.ndim != 2 or morgan.shape[1] != MORGAN_DIM or len(registry) != morgan.shape[0]:
        raise ValueError(f"registered Morgan feature rows/shape mismatch: {morgan.shape}, registry={len(registry)}")
    registry["drug_index"] = pd.to_numeric(registry.drug_index, errors="raise").astype(int)
    if registry.drug_index.duplicated().any() or not np.array_equal(registry.drug_index.to_numpy(), np.arange(len(registry))):
        raise ValueError("drug_index does not uniquely match the registered Morgan matrix row positions")
    result: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    nonunique: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(keys):
        rows = registry.loc[registry.inchi_key.eq(key)]
        indices = rows.drug_index.to_numpy(dtype=int)
        if len(indices) == 0:
            continue
        values = np.asarray(morgan[indices], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite Morgan feature for exact InChIKey {key}")
        if not np.all(values == values[0]):
            nonunique[key] = [
                {"drug_index": int(row.drug_index), "pert_id": str(row.pert_id),
                 "canonical_smiles": str(row.canonical_smiles),
                 "morgan_popcount": int(values[i].sum()),
                 "morgan_vector_sha256": hashlib.sha256(np.ascontiguousarray(values[i]).tobytes()).hexdigest()}
                for i, row in enumerate(rows.itertuples(index=False))
            ]
            continue
        result[key] = values[0]
        counts[key] = int(len(indices))
    if set(nonunique) != EXPECTED_NONUNIQUE_MORGAN_KEYS:
        raise ValueError(f"the frozen two-key Morgan ambiguity list changed: {sorted(nonunique)}")
    return result, {"requested_exact_keys": len(keys), "mapped_unique_exact_keys": len(result),
                    "unique_registry_rows_by_key": counts,
                    "excluded_nonunique_exact_morgan_keys": nonunique,
                    "nonunique_key_resolution": "exclude both ambiguous full InChIKeys from the shared candidate and every fold/arm denominator; no vector is selected or averaged",
                    "fingerprint": "precomputed EXP-037 registered RDKit Morgan bit fingerprint, radius 2, 2048 bits; verified drug_index equals matrix row position"}


def choose_block_derangement(objects: list[str], key_sets: dict[str, set[str]]) -> dict[str, str]:
    """Maximum exact-key support bijective derangement; no labels or scores used."""
    n = len(objects)
    if n < 2:
        raise ValueError("block shuffle requires at least two outer-training objects")
    best: tuple[int, int, tuple[str, ...]] | None = None
    best_map: dict[str, str] | None = None
    for perm in itertools.permutations(objects):
        if any(objects[i] == perm[i] for i in range(n)):
            continue
        overlap = [len(key_sets[objects[i]] & key_sets[perm[i]]) for i in range(n)]
        candidate = (min(overlap), sum(overlap), tuple(perm))
        if best is None or candidate > best:
            best = candidate
            best_map = {objects[i]: perm[i] for i in range(n)}
    assert best_map is not None
    return best_map


def freeze_support(labels: pd.DataFrame, features: pd.DataFrame, train_objects: list[str], drug_features: dict[str, np.ndarray], out: Path) -> dict[str, Any]:
    label_keys = {obj: set(labels.loc[labels.source_object_id.eq(obj), "full_inchikey"].astype(str)) for obj in train_objects}
    feature_keys = {obj: set(features.loc[features.source_object_id.eq(obj), "full_inchikey"].astype(str)) & set(drug_features)
                    for obj in train_objects}
    candidate_domain = set().union(*(label_keys[obj] & feature_keys[obj] & set(drug_features) for obj in train_objects))
    if not candidate_domain:
        raise ValueError("the exact full-InChIKey public candidate domain is empty")
    if len(candidate_domain) != EXPECTED_VALID_CANDIDATE_KEYS:
        raise ValueError(f"expected 121 common exact keys after the two frozen Morgan ambiguity exclusions; found {len(candidate_domain)}")
    fold_maps: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for test_obj in train_objects:
        fit_objects = [obj for obj in train_objects if obj != test_obj]
        mapping = choose_block_derangement(fit_objects, feature_keys)
        fold_maps[test_obj] = mapping
        training_rows: list[tuple[str, str]] = []
        for obj in fit_objects:
            donor = mapping[obj]
            common = label_keys[obj] & feature_keys[obj] & feature_keys[donor] & set(drug_features)
            training_rows.extend((obj, key) for key in sorted(common))
        counts: dict[str, int] = {}
        for obj, key in training_rows:
            counts[key] = counts.get(key, 0) + 1
        donor = sorted(fit_objects, key=lambda d: (-len(label_keys[test_obj] & feature_keys[test_obj] & feature_keys[d] & set(drug_features)), d))[0]
        eligible = sorted(
            (label_keys[test_obj] & feature_keys[test_obj] & feature_keys[donor] & set(drug_features))
            & {key for key, count in counts.items() if count >= 2}
        )
        all_test_label_keys = sorted(label_keys[test_obj])
        for key in all_test_label_keys:
            train_label_objects = sorted(obj for obj in fit_objects if key in label_keys[obj])
            rows.append({
                "outer_test_object": test_obj, "source_role": "TRAIN_only_outer_LOO",
                "source_object_id": test_obj, "full_inchikey": key,
                "in_candidate_domain": key in candidate_domain,
                "test_label_supported": True,
                "test_predicted_delta_supported": key in feature_keys[test_obj],
                "exact_drug_feature_supported": key in drug_features,
                "test_shuffle_donor": donor,
                "test_donor_delta_supported": key in feature_keys[donor],
                "outer_train_label_support_objects": len(train_label_objects),
                "outer_train_label_support_ids": ";".join(train_label_objects),
                "same_denominator_all_arms": key in eligible,
                "exclusion_reason": "" if key in eligible else ";".join(
                    reason for cond, reason in [
                        (key in feature_keys[test_obj], "no_test_predicted_delta"),
                        (key in drug_features, "no_exact_Morgan_key"),
                        (key in feature_keys[donor], "no_whole_block_donor_key"),
                        (counts.get(key, 0) >= 2, "fewer_than_two_outer_train_object_labels_after_common_shuffle_support"),
                    ] if not cond
                ),
            })
        fold_summaries.append({
            "test_object": test_obj, "outer_train_objects": fit_objects,
            "outer_train_block_donor_map": mapping, "test_block_donor": donor,
            "outer_train_model_rows_after_common_shuffle_support": len(training_rows),
            "outer_train_objects_with_rows": sorted({obj for obj, _ in training_rows}),
            "candidate_domain_drug_keys": len(candidate_domain),
            "test_label_pairs": len(label_keys[test_obj]),
            "test_exact_feature_pairs": len(label_keys[test_obj] & feature_keys[test_obj] & set(drug_features)),
            "test_common_all_arm_pairs": len(eligible),
            "test_common_full_inchikeys": eligible,
            "test_key_list_sha256": canonical_sha256(eligible),
            "test_train_block_donor_common_pairs": len(label_keys[test_obj] & feature_keys[test_obj] & feature_keys[donor] & set(drug_features)),
        })
    fold_summaries.sort(key=lambda r: r["test_object"])
    if any(row["test_common_all_arm_pairs"] < 2 for row in fold_summaries):
        raise ValueError("one or more outer folds have fewer than two exact common all-arm test keys")
    # Select the smoke fold using support only; no endpoint values or metrics are consulted.
    smoke_fold = sorted(fold_summaries, key=lambda r: (-r["test_common_all_arm_pairs"], r["test_object"]))[0]
    support = pd.DataFrame(rows).sort_values(["outer_test_object", "full_inchikey"]).reset_index(drop=True)
    support_path = out / "frozen_fold_support.tsv"
    support.to_csv(support_path, sep="\t", index=False, na_rep="")
    mapping_rows = []
    for test_obj, mapping in sorted(fold_maps.items()):
        for target, donor in sorted(mapping.items()):
            mapping_rows.append({"outer_test_object": test_obj, "target_train_object": target,
                                 "donor_source_object": donor, "mapping_type": "whole_978_gene_x_123_key_object_block_derangement"})
        selected_donor = next(x["test_block_donor"] for x in fold_summaries if x["test_object"] == test_obj)
        mapping_rows.append({"outer_test_object": test_obj, "target_train_object": test_obj,
                             "donor_source_object": selected_donor, "mapping_type": "heldout_block_from_outer_training_object"})
    pd.DataFrame(mapping_rows).to_csv(out / "frozen_block_permutations.tsv", sep="\t", index=False)
    summary = {
        "run_id": RUN_ID, "status": "SUPPORT_AND_SPLIT_FROZEN",
        "label_support": {"endpoint": "CTRPv2", "role": "TRAIN", "objects": len(train_objects),
                          "object_drug_pairs": int(len(labels)), "unique_exact_full_inchikeys": int(labels.full_inchikey.nunique()),
                          "pairs_by_object": labels.groupby("source_object_id").size().astype(int).to_dict()},
        "pre_feature_gate_label_key_union_size": int(labels.full_inchikey.nunique()),
        "excluded_nonunique_morgan_keys": sorted(EXPECTED_NONUNIQUE_MORGAN_KEYS),
        "prediction_support": {"object_drug_pairs": int(len(features)),
                               "object_pairs_by_object": features.groupby("source_object_id").size().astype(int).to_dict()},
        "candidate_domain": sorted(candidate_domain), "candidate_domain_size": len(candidate_domain),
        "candidate_domain_sha256": canonical_sha256(sorted(candidate_domain)),
        "fold_support": fold_summaries,
        "smoke_fold": smoke_fold["test_object"],
        "smoke_selection_rule": "maximum same-fold exact all-arm object-drug key count; tie by sorted source_object_id; support only, no label magnitude/score",
        "frozen_fold_support_sha256": sha256(support_path),
        "frozen_block_permutations_sha256": sha256(out / "frozen_block_permutations.tsv"),
        "all_nine_objects_have_valid_fold": True,
    }
    write_json(out / "protocol_snapshot.json", {
        "run_id": RUN_ID, "frozen_at_utc": utc_now(), "config_sha256": sha256(ROOT / CONFIG_REL),
        "candidate_domain_sha256": summary["candidate_domain_sha256"],
        "train_object_ids": train_objects, "train_object_ids_sha256": canonical_sha256(train_objects),
        "fold_support_sha256": summary["frozen_fold_support_sha256"],
        "block_permutation_sha256": summary["frozen_block_permutations_sha256"],
        "model": "one fixed sklearn Ridge(alpha=1.0, solver='auto') family; no model/seed search",
        "fold_rule": "9 outer leave-one-TRAIN-source-object-out folds; inner leave-one-object-out predictions for fusion-weight selection",
        "test_denominator_rule": "test-label object-key pair with exact baseline and predicted delta, one unambiguous exact Morgan feature, at least two outer-training label objects, and exact-key support in the held-out object's single whole-block shuffle donor; identical per fold across all arms/references",
        "label_scope": "CTRPv2 TRAIN labels only; VAL/outer/locked/PDO/PRISM numeric values are not loaded by this runner",
        "cost_limit_minutes": CPU_CAP_MINUTES,
        "smoke_fold": smoke_fold["test_object"],
        "support_freeze_sha256": canonical_sha256(summary),
    })
    write_json(out / "support_freeze.json", summary)
    return summary


def fold_frames(labels: pd.DataFrame, features: pd.DataFrame, folds: dict[str, Any], test_obj: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], str]:
    details = next(x for x in folds["fold_support"] if x["test_object"] == test_obj)
    mapping = details["outer_train_block_donor_map"]
    donor_test = details["test_block_donor"]
    train_objects = details["outer_train_objects"]
    candidate_domain = set(str(key) for key in folds["candidate_domain"])
    if not candidate_domain:
        raise ValueError("frozen unique-Morgan candidate domain is empty")
    feat_lookup = features.set_index(["source_object_id", "full_inchikey"])
    label_keys_by_obj = {obj: set(labels.loc[labels.source_object_id.eq(obj), "full_inchikey"].astype(str)) for obj in train_objects + [test_obj]}
    rows: list[dict[str, Any]] = []
    for obj in train_objects:
        donor = mapping[obj]
        # Apply the one shared feature-supported key gate before outer or inner fitting.
        keys = sorted(label_keys_by_obj[obj] & set(feat_lookup.loc[obj].index.astype(str))
                      & set(feat_lookup.loc[donor].index.astype(str)) & candidate_domain)
        for key in keys:
            y = labels.loc[labels.source_object_id.eq(obj) & labels.full_inchikey.eq(key), "sensitivity"].iloc[0]
            rows.append({"source_object_id": obj, "full_inchikey": key, "sensitivity": float(y),
                         "shuffle_donor": donor, "predicted_perturbation_strength": float(feat_lookup.loc[(obj, key), "predicted_perturbation_strength"])})
    training = pd.DataFrame(rows)
    selected_keys = set(details["test_common_full_inchikeys"])
    test_rows: list[dict[str, Any]] = []
    for key in sorted(selected_keys):
        y = labels.loc[labels.source_object_id.eq(test_obj) & labels.full_inchikey.eq(key), "sensitivity"].iloc[0]
        test_rows.append({"source_object_id": test_obj, "full_inchikey": key, "sensitivity": float(y),
                          "shuffle_donor": donor_test, "predicted_perturbation_strength": float(feat_lookup.loc[(test_obj, key), "predicted_perturbation_strength"])})
    testing = pd.DataFrame(test_rows)
    if training.empty or testing.empty:
        raise ValueError(f"empty frozen fold training/test frame for {test_obj}")
    return training, testing, mapping, donor_test


def raw_features(frame: pd.DataFrame, features: pd.DataFrame, profiles: dict[str, np.ndarray], drugs: dict[str, np.ndarray], shuffle: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing_drug_keys = sorted(set(frame.full_inchikey.astype(str)) - set(drugs))
    if missing_drug_keys:
        raise ValueError(f"feature frame contains keys outside the unique Morgan feature domain: {missing_drug_keys[:5]}")
    feat_lookup = features.set_index(["source_object_id", "full_inchikey"])
    expr = np.stack([profiles[str(obj)] for obj in frame.source_object_id])
    pred_rows = []
    for row in frame.itertuples(index=False):
        source = str(row.shuffle_donor) if shuffle else str(row.source_object_id)
        pred_rows.append(feat_lookup.loc[(source, str(row.full_inchikey)), "predicted_delta"])
    pred = np.stack(pred_rows).astype(np.float32)
    chem = np.stack([drugs[str(key)] for key in frame.full_inchikey]).astype(np.float32)
    return expr.astype(np.float32), pred, chem


def transformed_pair(train: pd.DataFrame, valid: pd.DataFrame, features: pd.DataFrame,
                     profiles: dict[str, np.ndarray], drugs: dict[str, np.ndarray],
                     arm: str, shuffle: bool = False) -> tuple[np.ndarray, np.ndarray]:
    ex_tr, pr_tr, ch_tr = raw_features(train, features, profiles, drugs, shuffle=shuffle)
    ex_va, pr_va, ch_va = raw_features(valid, features, profiles, drugs, shuffle=shuffle)
    exp_scaler = StandardScaler().fit(np.stack([profiles[x] for x in sorted(set(train.source_object_id.astype(str)))]))
    pert_scaler = StandardScaler().fit(pr_tr)
    chem_scaler = StandardScaler().fit(np.stack([drugs[k] for k in sorted(set(train.full_inchikey.astype(str)))]))
    ex_tr = exp_scaler.transform(ex_tr).astype(np.float32)
    ex_va = exp_scaler.transform(ex_va).astype(np.float32)
    pr_tr = pert_scaler.transform(pr_tr).astype(np.float32)
    pr_va = pert_scaler.transform(pr_va).astype(np.float32)
    ch_tr = chem_scaler.transform(ch_tr).astype(np.float32)
    ch_va = chem_scaler.transform(ch_va).astype(np.float32)
    if arm == "direct":
        return np.concatenate([ex_tr, ch_tr], axis=1), np.concatenate([ex_va, ch_va], axis=1)
    if arm in {"predicted_perturbation", "shuffled_perturbation"}:
        return np.concatenate([pr_tr, ch_tr], axis=1), np.concatenate([pr_va, ch_va], axis=1)
    if arm == "equal_capacity_direct":
        # A second independently fitted copy of the direct model matches the
        # nominal fitted-parameter count of late fusion.
        return np.concatenate([ex_tr, ch_tr], axis=1), np.concatenate([ex_va, ch_va], axis=1)
    raise ValueError(arm)


def row_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("source_object_id").source_object_id.transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def train_mean(frame: pd.DataFrame) -> dict[str, float]:
    return frame.groupby("full_inchikey").sensitivity.mean().astype(float).to_dict()


def fit_residual_model(train: pd.DataFrame, valid: pd.DataFrame, features: pd.DataFrame,
                       profiles: dict[str, np.ndarray], drugs: dict[str, np.ndarray], arm: str,
                       shuffle: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    means = train_mean(train)
    y = train.sensitivity.to_numpy(dtype=float) - train.full_inchikey.map(means).to_numpy(dtype=float)
    x_train, x_valid = transformed_pair(train, valid, features, profiles, drugs, arm, shuffle=shuffle)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train, y, sample_weight=row_weights(train))
    pred = valid.full_inchikey.map(means).to_numpy(dtype=float) + model.predict(x_valid)
    if not np.isfinite(pred).all():
        raise ValueError(f"non-finite prediction from {arm}")
    return pred, {"arm": arm, "ridge_alpha": RIDGE_ALPHA, "n_train_pairs": len(train),
                  "n_train_objects": int(train.source_object_id.nunique()), "n_test_pairs": len(valid),
                  "n_features": int(x_train.shape[1]), "fit_train_object_ids": sorted(train.source_object_id.astype(str).unique()),
                  "fit_test_object_id": str(valid.source_object_id.iloc[0]), "scalers_fit_train_only": True}


def inner_oof_for_mix(outer_train: pd.DataFrame, features: pd.DataFrame, profiles: dict[str, np.ndarray],
                      drugs: dict[str, np.ndarray], min_count: int = 2) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions: list[pd.DataFrame] = []
    objects = sorted(outer_train.source_object_id.astype(str).unique())
    for inner_test_obj in objects:
        inner_fit = outer_train.loc[outer_train.source_object_id.astype(str).ne(inner_test_obj)].copy()
        inner_keys = inner_fit.groupby("full_inchikey").source_object_id.nunique()
        eligible = set(inner_keys.loc[inner_keys.ge(min_count)].index.astype(str))
        inner_test = outer_train.loc[(outer_train.source_object_id.astype(str).eq(inner_test_obj)) & outer_train.full_inchikey.astype(str).isin(eligible)].copy()
        if inner_test.empty or inner_fit.empty:
            continue
        pred_direct, _ = fit_residual_model(inner_fit, inner_test, features, profiles, drugs, "direct")
        pred_pert, _ = fit_residual_model(inner_fit, inner_test, features, profiles, drugs, "predicted_perturbation")
        pred_direct_dup, _ = fit_residual_model(inner_fit, inner_test, features, profiles, drugs, "equal_capacity_direct")
        part = inner_test[["source_object_id", "full_inchikey", "sensitivity"]].copy()
        part["pred_direct"] = pred_direct
        part["pred_perturbation"] = pred_pert
        part["pred_direct_dup"] = pred_direct_dup
        predictions.append(part)
    oof = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if oof.empty:
        raise ValueError("inner object-crossfit produced no validation predictions for fusion weight")
    return oof, {"inner_oof_objects": sorted(oof.source_object_id.astype(str).unique()),
                 "inner_oof_object_count": int(oof.source_object_id.nunique()),
                 "inner_oof_pairs": int(len(oof)), "inner_split": "leave one outer-training source object out",
                 "all_inner_preprocessing_fit_on_inner_train_objects": True}


def choose_mix(oof: pd.DataFrame, first: str, second: str) -> tuple[float, dict[str, Any]]:
    best_weight = 0.5
    best_score = float("inf")
    candidates = np.linspace(0.0, 1.0, 101)
    y_by_object = {obj: grp.sensitivity.to_numpy(dtype=float) for obj, grp in oof.groupby("source_object_id")}
    a_by_object = {obj: grp[first].to_numpy(dtype=float) for obj, grp in oof.groupby("source_object_id")}
    b_by_object = {obj: grp[second].to_numpy(dtype=float) for obj, grp in oof.groupby("source_object_id")}
    scores = []
    for w in candidates:
        obj_errors = [float(np.mean(np.square(y_by_object[o] - (w * a_by_object[o] + (1.0 - w) * b_by_object[o])))) for o in y_by_object]
        score = float(np.mean(obj_errors))
        scores.append(score)
        if score < best_score - 1e-12 or (abs(score - best_score) <= 1e-12 and abs(w - 0.5) < abs(best_weight - 0.5)):
            best_score, best_weight = score, float(w)
    return best_weight, {"grid": "0.00 to 1.00 in 0.01 increments", "selected_weight_first_arm": best_weight,
                         "inner_oof_object_macro_mse": best_score, "n_inner_oof_objects": len(y_by_object),
                         "all_candidate_weights_fit_from_outer_train_oof_predictions": True}


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    result: dict[str, Any] = {"n_pairs": int(len(y)), "rmse": float(np.sqrt(np.mean(np.square(y - pred)))),
                              "mae": float(np.mean(np.abs(y - pred))),
                              "spearman": None, "spearman_undefined_reason": None}
    if len(y) < 2:
        result["spearman_undefined_reason"] = "fewer_than_two_object_drug_keys"
    elif np.unique(y).size < 2:
        result["spearman_undefined_reason"] = "constant_observed_AAC"
    elif np.unique(pred).size < 2:
        result["spearman_undefined_reason"] = "constant_model_prediction"
    else:
        val = float(spearmanr(y, pred).statistic)
        result["spearman"] = val if np.isfinite(val) else None
        if result["spearman"] is None:
            result["spearman_undefined_reason"] = "nonfinite_spearman"
    return result


def run_fold(labels: pd.DataFrame, features: pd.DataFrame, profiles: dict[str, np.ndarray], drugs: dict[str, np.ndarray], support: dict[str, Any], test_obj: str) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    train, test, mapping, donor_test = fold_frames(labels, features, support, test_obj)
    # Keep one common drug-mean/residual support: at least two distinct outer-training objects per key.
    counts = train.groupby("full_inchikey").source_object_id.nunique()
    eligible_keys = set(counts.loc[counts.ge(2)].index.astype(str))
    train = train.loc[train.full_inchikey.astype(str).isin(eligible_keys)].copy().reset_index(drop=True)
    test = test.loc[test.full_inchikey.astype(str).isin(eligible_keys)].copy().reset_index(drop=True)
    if len(train) < 8 or train.source_object_id.nunique() < 5 or test.full_inchikey.nunique() < 2:
        raise ValueError(f"insufficient fold support after whole-block common denominator for {test_obj}: train={len(train)}/{train.source_object_id.nunique()} objects, test={len(test)} keys")
    expected_test_keys = set(next(x for x in support["fold_support"] if x["test_object"] == test_obj)["test_common_full_inchikeys"])
    if set(test.full_inchikey.astype(str)) != expected_test_keys:
        raise ValueError("fold test denominator differs from frozen support list")
    oof, oof_info = inner_oof_for_mix(train, features, profiles, drugs)
    fusion_weight, fusion_info = choose_mix(oof, "pred_direct", "pred_perturbation")
    eq_weight, eq_info = choose_mix(oof, "pred_direct", "pred_direct_dup")
    pred_direct, direct_log = fit_residual_model(train, test, features, profiles, drugs, "direct")
    pred_pert, pert_log = fit_residual_model(train, test, features, profiles, drugs, "predicted_perturbation")
    pred_shuffle, shuffle_log = fit_residual_model(train, test, features, profiles, drugs, "shuffled_perturbation", shuffle=True)
    pred_eq1, eq_log1 = fit_residual_model(train, test, features, profiles, drugs, "direct")
    pred_eq2, eq_log2 = fit_residual_model(train, test, features, profiles, drugs, "equal_capacity_direct")
    if not np.allclose(pred_eq1, pred_eq2, rtol=1e-7, atol=1e-8):
        raise ValueError("equal-capacity duplicated direct fit no longer matches its preregistered duplicate representation")
    pred_fusion = fusion_weight * pred_direct + (1.0 - fusion_weight) * pred_pert
    pred_eqcap = eq_weight * pred_eq1 + (1.0 - eq_weight) * pred_eq2
    train_means = train_mean(train)
    pred_mean = test.full_inchikey.map(train_means).to_numpy(dtype=float)
    strength_train = train.predicted_perturbation_strength.to_numpy(dtype=float)
    strength_mean = float(np.average(strength_train, weights=row_weights(train)))
    strength_sd = float(np.sqrt(np.average(np.square(strength_train - strength_mean), weights=row_weights(train))))
    strength_sd = strength_sd if strength_sd > 0 else 1.0
    x_strength = ((strength_train - strength_mean) / strength_sd).reshape(-1, 1)
    x_strength_test = ((test.predicted_perturbation_strength.to_numpy(dtype=float) - strength_mean) / strength_sd).reshape(-1, 1)
    y_resid = train.sensitivity.to_numpy(dtype=float) - train.full_inchikey.map(train_means).to_numpy(dtype=float)
    strength_model = Ridge(alpha=RIDGE_ALPHA).fit(x_strength, y_resid, sample_weight=row_weights(train))
    pred_strength = test.full_inchikey.map(train_means).to_numpy(dtype=float) + strength_model.predict(x_strength_test)
    predictions = test[["source_object_id", "full_inchikey", "sensitivity", "shuffle_donor", "predicted_perturbation_strength"]].copy()
    predictions = predictions.rename(columns={"sensitivity": "observed_aac"})
    predictions["pred_direct"] = pred_direct
    predictions["pred_predicted_perturbation"] = pred_pert
    predictions["pred_fusion"] = pred_fusion
    predictions["pred_equal_capacity_direct"] = pred_eqcap
    predictions["pred_shuffled_perturbation"] = pred_shuffle
    predictions["pred_train_drug_mean"] = pred_mean
    predictions["pred_overall_perturbation_strength"] = pred_strength
    arm_cols = {
        "direct": "pred_direct", "predicted_perturbation": "pred_predicted_perturbation",
        "fusion": "pred_fusion", "equal_capacity_direct": "pred_equal_capacity_direct",
        "shuffled_perturbation": "pred_shuffled_perturbation",
        "training_drug_mean": "pred_train_drug_mean",
        "overall_perturbation_strength": "pred_overall_perturbation_strength",
    }
    metrics = []
    ytrue = predictions.observed_aac.to_numpy(dtype=float)
    for arm, col in arm_cols.items():
        metrics.append({"source_object_id": test_obj, "arm": arm,
                        "n_train_objects": int(train.source_object_id.nunique()),
                        "n_train_pairs": len(train), "n_test_drug_keys": int(len(test)),
                        "n_test_unique_drugs": int(test.full_inchikey.nunique()), **metric(ytrue, predictions[col].to_numpy(dtype=float))})
    model_log = {
        "test_object": test_obj, "outer_train_objects": sorted(train.source_object_id.astype(str).unique()),
        "outer_train_rows": int(len(train)), "outer_train_drug_keys": int(train.full_inchikey.nunique()),
        "test_drug_keys": int(len(test)), "test_drug_key_list": sorted(test.full_inchikey.astype(str).tolist()),
        "test_key_list_sha256": canonical_sha256(sorted(test.full_inchikey.astype(str).tolist())),
        "train_block_permutation": mapping, "test_block_donor": donor_test,
        "fusion_weight_direct": fusion_weight, "fusion_weight_info": fusion_info,
        "equal_capacity_direct_weight": eq_weight, "equal_capacity_direct_weight_info": eq_info,
        "inner_oof_info": oof_info,
        "base_models": [direct_log, pert_log, shuffle_log, eq_log1, eq_log2],
        "equal_capacity_direct_parameterization": {"direct_fit_count": 2, "ridge_slopes_per_fit": 3026,
            "ridge_intercepts_per_fit": 1, "mix_parameter": 1, "total_nominal_trainable_parameters": 6055,
            "fusion_total_nominal_parameters": 6055,
            "duplicate_direct_feature_rank_deficiency": True},
        "fold_training_only_drug_means_and_preprocessing": True,
        "test_label_used_for_any_fit_or_weight": False,
    }
    return predictions, model_log, pd.DataFrame(metrics)


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    contrasts = [("predicted_perturbation", "direct"), ("fusion", "direct"),
                 ("fusion", "predicted_perturbation"), ("fusion", "equal_capacity_direct"),
                 ("shuffled_perturbation", "predicted_perturbation"),
                 ("fusion", "training_drug_mean"), ("fusion", "overall_perturbation_strength")]
    rows = []
    for a, b in contrasts:
        left = metrics.loc[metrics.arm.eq(a), ["source_object_id", "spearman", "rmse", "mae"]].rename(columns={"spearman": "spearman_a", "rmse": "rmse_a", "mae": "mae_a"})
        right = metrics.loc[metrics.arm.eq(b), ["source_object_id", "spearman", "rmse", "mae"]].rename(columns={"spearman": "spearman_b", "rmse": "rmse_b", "mae": "mae_b"})
        joined = left.merge(right, on="source_object_id", how="inner", validate="one_to_one")
        for measure, lo_key, hi_key, favorable in [("spearman", "spearman_a", "spearman_b", "larger_is_better"),
                                                    ("rmse", "rmse_a", "rmse_b", "lower_is_better"),
                                                    ("mae", "mae_a", "mae_b", "lower_is_better")]:
            valid = joined.dropna(subset=[lo_key, hi_key])
            diffs = (valid[lo_key] - valid[hi_key]).to_numpy(dtype=float)
            if not len(diffs):
                rows.append({"contrast": f"{a}_minus_{b}", "metric": measure, "difference_definition": f"{a} - {b}",
                             "source_object_id": "", "object_difference": None,
                             "n_paired_objects_valid_for_metric": 0, "equal_object_mean_difference": None,
                             "favorable_direction": favorable, "status": "undefined_no_valid_object_pairs"})
                continue
            macro = float(np.mean(diffs))
            for (_, item), diff in zip(valid.iterrows(), diffs):
                rows.append({"contrast": f"{a}_minus_{b}", "metric": measure, "difference_definition": f"{a} - {b}",
                             "source_object_id": str(item.source_object_id), "object_difference": float(diff),
                             "n_paired_objects_valid_for_metric": int(len(diffs)), "equal_object_mean_difference": macro,
                             "favorable_direction": favorable, "status": "descriptive_paired_source_object_difference_only"})
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for arm, group in metrics.groupby("arm", sort=False):
        vals = group.spearman.dropna().to_numpy(dtype=float)
        rows.append({"arm": arm, "objects_scored": int(group.source_object_id.nunique()),
                     "valid_spearman_objects": int(len(vals)), "undefined_spearman_objects": int(group.spearman.isna().sum()),
                     "object_macro_spearman": float(vals.mean()) if len(vals) else None,
                     "object_macro_rmse_aac": float(group.rmse.mean()), "object_macro_mae_aac": float(group.mae.mean()),
                     "total_object_drug_pairs_scored": int(group.n_pairs.sum()),
                     "per_object_denominators": {str(r.source_object_id): int(r.n_pairs) for r in group.itertuples(index=False)}})
    return {"primary_metric": "equal-object macro Spearman between out-of-fold predicted and observed CTRPv2 AAC; larger is better",
            "secondary_metrics": "equal-object macro RMSE and MAE in AAC units; lower is better",
            "arms": rows, "no_drug_or_gene_count_as_independent_n": True}


def record_run(args: argparse.Namespace, started: str, elapsed: float, process_cpu_elapsed: float,
               input_hashes: dict[str, str], output_files: list[Path], support: dict[str, Any],
               status: str, extra: dict[str, Any] | None = None) -> None:
    out = ROOT / OUT_REL
    accounting_path = out / "cpu_accounting.json"
    prior = json.loads(accounting_path.read_text(encoding="utf-8")) if accounting_path.exists() else {"attempts": []}
    accumulated = sum(float(x["elapsed_minutes"]) for x in prior["attempts"])
    current = elapsed / 60.0
    if accumulated + current > CPU_CAP_MINUTES:
        raise RuntimeError(f"240-minute CPU budget exceeded: prior={accumulated:.3f}, current={current:.3f}")
    attempt = {"phase": args.phase, "started_at_utc": started, "finished_at_utc": utc_now(),
               "elapsed_wall_minutes": current, "process_cpu_minutes": process_cpu_elapsed / 60.0,
               "elapsed_minutes": current, "cpu_cap_minutes": CPU_CAP_MINUTES,
               "gpu_hours": 0.0, "status": status}
    prior["attempts"].append(attempt)
    prior["cumulative_elapsed_minutes"] = accumulated + current
    prior["cpu_hard_cap_minutes"] = CPU_CAP_MINUTES
    write_json(accounting_path, prior)
    outputs = {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
               if p.is_file() and p.name != "run_manifest.json"}
    manifest = {
        "experiment_id": "EXP-050", "run_id": RUN_ID, "status": status,
        "execution_environment": "WSL2 Conda drugscreening-gpu",
        "python_executable": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "command": " ".join(sys.argv), "phases": prior["attempts"],
        "actual_compute": {"wall_elapsed_minutes": prior["cumulative_elapsed_minutes"],
                           "process_cpu_minutes": sum(float(x.get("process_cpu_minutes", 0.0)) for x in prior["attempts"]), "gpu_hours": 0.0,
                           "gpu_used": False, "cpu_limit_minutes": CPU_CAP_MINUTES},
        "model": "sklearn Ridge(alpha=1.0); one fixed candidate family",
        "outer_folds": 9, "support_freeze_sha256": canonical_sha256(support),
        "runner_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(ROOT / CONFIG_REL),
        "inputs_sha256": input_hashes, "outputs_sha256": outputs,
        "previous_ctrvp2_label_exposure": "EXP-050 §3.4 source exploratory analysis read CTRPv2 values; all OOF results are exploratory/development and not independent validation",
        "heldout_outer_object_labels_used_for_fit_selection": False,
        "val_outer_locked_pdo_label_values_loaded": False,
    }
    if extra:
        manifest.update(extra)
    write_json(out / "run_manifest.json", manifest)


def freeze_protocol_preflight(args: argparse.Namespace) -> None:
    """Freeze object/key support and fold assignments without reading numeric labels or prediction arrays."""
    started_wall = time.monotonic(); started_cpu = time.process_time(); started_utc = utc_now()
    out = ROOT / OUT_REL
    paths = input_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    audit_receipt = json.loads((ROOT / ASSET_AUDIT_REL / "audit_receipt.json").read_text(encoding="utf-8"))
    facts = audit_receipt["facts"]
    audit_input = audit_receipt["input_sha256"]
    required_hash_matches = {
        "xpert_prediction_keys": audit_input["xpert_prediction_keys"] == input_hashes["xpert_prediction_keys"],
        "xpert_prediction_array": audit_input["xpert_prediction_array"] == input_hashes["xpert_prediction"],
        "xpert_fit_identity_key_map": audit_input["xpert_fit_identity_key_map"] == input_hashes["xpert_identity_key_map"],
    }
    if not all(required_hash_matches.values()):
        raise ValueError(f"read-only asset audit is stale against frozen XPert attempt outputs: {required_hash_matches}")
    if (facts["attempt_map_target_dose_time_is_uniform_10uM_24h"] is not True
            or facts["attempt_map_target_row_key_mismatches"] != 0
            or facts["xpert_exact_join_pairs"] != 404
            or facts["xpert_train_pairs_without_prediction"] != 0
            or facts["xpert_unmatched_prediction_pairs"] != 0
            or facts["xpert_fit_train_identity_overlap_with_nine_targets"] != []):
        raise ValueError("read-only audit failed an exact identity, exposure, or XPert training identity gate")

    identity = json.loads((ROOT / ID_SPLIT_REL).read_text(encoding="utf-8"))
    if identity.get("response_blind") is not True:
        raise ValueError("frozen source identity split is not response-blind")
    roles = pd.read_csv(ROOT / ROLE_REL, sep="\t", usecols=["source_object_id", "split_role"], dtype="string")
    train_objects = sorted(set(roles.loc[roles.split_role.eq("train"), "source_object_id"].dropna().astype(str)))
    if len(train_objects) != EXPECTED_TRAIN_OBJECTS:
        raise ValueError(f"expected nine frozen TRAIN objects, found {len(train_objects)}")
    train_set = set(train_objects)

    # Metadata-only projection: the sensitivity column is not loaded in this phase.
    paired = pd.read_csv(ROOT / PAIRED_REL, sep="\t",
                         usecols=["endpoint", "split_role", "source_object_id", "full_inchikey"], dtype="string")
    train_role_rows = int(paired.split_role.eq("train").sum())
    label_meta = paired.loc[paired.endpoint.eq("CTRPv2") & paired.split_role.eq("train"),
                            ["source_object_id", "full_inchikey"]].copy()
    label_meta["source_object_id"] = label_meta.source_object_id.astype(str)
    label_meta["full_inchikey"] = label_meta.full_inchikey.astype(str).str.strip().str.upper()
    label_meta = label_meta.drop_duplicates().reset_index(drop=True)
    if (len(paired) != 535 or train_role_rows != 466 or len(label_meta) != 404
            or int(label_meta.full_inchikey.nunique()) != 123
            or set(label_meta.source_object_id.astype(str)) != train_set):
        raise ValueError("metadata-only 3.4 CTRPv2 TRAIN role/key support differs from the frozen 535/466/404/123 counts")

    targets = pd.read_csv(ROOT / PRED_KEYS_REL, sep="\t",
                          usecols=["condition_id", "source_row", "source_object_id", "full_inchikey"],
                          dtype={"source_object_id": "string", "full_inchikey": "string"}).reset_index(drop=True)
    if len(targets) != 467 or targets.condition_id.duplicated().any() or targets.source_row.duplicated().any():
        raise ValueError("XPert prediction target key rows are not the frozen 467 unique conditions")
    targets["source_object_id"] = targets.source_object_id.astype(str)
    targets["full_inchikey"] = targets.full_inchikey.astype(str).str.strip().str.upper()
    targets["prediction_row"] = np.arange(len(targets), dtype=np.int64)
    prediction_meta = targets.loc[targets.source_object_id.isin(train_set), ["source_object_id", "full_inchikey"]].drop_duplicates()
    exact_join = label_meta.merge(prediction_meta, on=["source_object_id", "full_inchikey"], how="outer", indicator=True)
    if len(prediction_meta) != 404 or not exact_join._merge.eq("both").all():
        raise ValueError("metadata-only source label/prediction full-key join is not an exact 404/404 match")

    all_label_keys = set(label_meta.full_inchikey.astype(str))
    drug_features, drug_audit = load_drug_features(all_label_keys)
    if set(drug_audit["excluded_nonunique_exact_morgan_keys"]) != EXPECTED_NONUNIQUE_MORGAN_KEYS:
        raise ValueError("registered Morgan ambiguity keys differ from the frozen read-only audit")
    if len(set(drug_features) & all_label_keys) != EXPECTED_VALID_CANDIDATE_KEYS:
        raise ValueError("the exact-key chemical feature-supported domain is not 121 keys")

    features_meta = prediction_meta.copy()
    support = freeze_support(label_meta, features_meta, train_objects, drug_features, out)
    if support["candidate_domain_size"] != EXPECTED_VALID_CANDIDATE_KEYS:
        raise ValueError("frozen candidate domain size differs from the approved 121-key Morgan-supported domain")

    frozen = pd.read_csv(out / "frozen_fold_support.tsv", sep="\t", dtype={"outer_test_object": "string", "full_inchikey": "string"})
    fold_checks = []
    for item in support["fold_support"]:
        test_obj = str(item["test_object"])
        rows = frozen.loc[frozen.outer_test_object.astype(str).eq(test_obj)]
        frozen_keys = sorted(rows.loc[rows.same_denominator_all_arms.astype(bool), "full_inchikey"].astype(str).tolist())
        expected_keys = sorted(item["test_common_full_inchikeys"])
        fit_objects = sorted(item["outer_train_objects"])
        block_map = item["outer_train_block_donor_map"]
        bijective_derangement = (set(block_map) == set(fit_objects) and set(block_map.values()) == set(fit_objects)
                                 and all(target != donor for target, donor in block_map.items()))
        exact_equal = frozen_keys == expected_keys
        if not exact_equal or not bijective_derangement or test_obj in fit_objects:
            raise ValueError(f"frozen fold support check failed for {test_obj}: exact_equal={exact_equal}, block_map={bijective_derangement}")
        fold_checks.append({
            "test_object": test_obj, "outer_train_objects": fit_objects,
            "test_key_count_common_to_all_arms_and_references": len(expected_keys),
            "test_key_list_sha256": item["test_key_list_sha256"],
            "frozen_tsv_key_list_exactly_equal_to_protocol_list": exact_equal,
            "training_block_permutation_is_bijective_derangement": bijective_derangement,
            "test_block_donor": item["test_block_donor"],
            "test_block_donor_is_outer_train_object": item["test_block_donor"] in fit_objects,
        })
    freeze_manifest = {
        "experiment_id": "EXP-050", "run_id": RUN_ID,
        "status": "PROTOCOL_SUPPORT_FROZEN_AWAITING_MANAGER_REVIEW",
        "model_fit_or_scoring_started": False, "numeric_response_label_values_read_in_freeze_phase": False,
        "measured_xpert_delta_array_read": False, "prediction_array_numeric_values_read_in_freeze_phase": False,
        "execution_environment": "WSL2 Conda drugscreening-gpu",
        "input_sha256": input_hashes, "runner_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(ROOT / CONFIG_REL),
        "train_objects": train_objects, "train_object_ids_sha256": canonical_sha256(train_objects),
        "paired_metadata_counts": {"all_rows": int(len(paired)), "TRAIN_role_rows_all_endpoints": train_role_rows,
                                   "CTRPv2_TRAIN_unique_object_key_pairs": int(len(label_meta)),
                                   "CTRPv2_TRAIN_unique_full_keys": int(label_meta.full_inchikey.nunique())},
        "xpert_contract": {"attempt_receipt_sha256": input_hashes["xpert_attempt_receipt"],
                           "identity_key_map_sha256": input_hashes["xpert_identity_key_map"],
                           "condition_map_source": "hash-verified attempt_001 fit_condition_identity_key_map.tsv",
                           "allowed_target_condition_rows": facts["hash_verified_attempt_map_rows_for_nine_train_objects"],
                           "all_target_conditions_are_10uM_24h": facts["attempt_map_target_dose_time_is_uniform_10uM_24h"],
                           "fit_training_identities": facts["xpert_fit_training_identity_count"],
                           "fit_training_overlap": facts["xpert_fit_train_identity_overlap_with_nine_targets"],
                           "prediction_array_row_order": "heldout_source_prediction_keys.tsv row order; source_row is not target-array offset",
                           "per_object_key_aggregation": "gene-wise median across all frozen mapped target conditions; no condition selection"},
        "candidate_domain": {"label_union_keys": len(all_label_keys), "excluded_ambiguous_morgan_keys": sorted(EXPECTED_NONUNIQUE_MORGAN_KEYS),
                             "supported_unique_morgan_keys": support["candidate_domain_size"],
                             "effective_candidate_key_list": support["candidate_domain"],
                             "effective_candidate_key_list_sha256": support["candidate_domain_sha256"]},
        "support_freeze_sha256": canonical_sha256(support),
        "fold_support_tsv_sha256": support["frozen_fold_support_sha256"],
        "block_permutation_tsv_sha256": support["frozen_block_permutations_sha256"],
        "per_fold_exact_support_checks": fold_checks,
        "smoke_object_selected_by_support_only": support["smoke_fold"],
        "fixed_cpu_budget_minutes": CPU_CAP_MINUTES, "gpu_hours": 0.0,
        "next_action": "await manager review of the frozen object list, 121-key candidate domain, per-fold common support lists, and hash-verified asset contract before running the single-object five-arm smoke",
    }
    freeze_manifest_path = out / "protocol_preflight_manifest.json"
    write_json(freeze_manifest_path, freeze_manifest)
    outputs = list(out.rglob("*"))
    record_run(args, started_utc, time.monotonic() - started_wall, time.process_time() - started_cpu,
               input_hashes, [p for p in outputs if p.is_file()], support, "PROTOCOL_SUPPORT_FROZEN_AWAITING_MANAGER_REVIEW")


def execute(args: argparse.Namespace) -> None:
    if platform.system() != "Linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("all EXP-050 Python execution must run in WSL2 Conda drugscreening-gpu")
    if args.run_id != RUN_ID:
        raise ValueError(f"run_id must be {RUN_ID}")
    config_path = ROOT / CONFIG_REL
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["run_id"] != RUN_ID:
        raise ValueError("run_id mismatch in frozen config")
    out = ROOT / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    attempt_dir = out / args.phase
    attempt_dir.mkdir(parents=True, exist_ok=True)
    for p in attempt_dir.iterdir():
        if p.is_file():
            raise FileExistsError(f"refusing to overwrite existing {args.phase} output: {p}")
    if args.phase == "freeze":
        freeze_protocol_preflight(args)
        return
    started_wall = time.monotonic(); started_process_cpu = time.process_time(); started_utc = utc_now()
    paths = input_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    identity = json.loads((ROOT / ID_SPLIT_REL).read_text(encoding="utf-8"))
    if identity.get("response_blind") is not True:
        raise ValueError("EXP-007 frozen identity table is not response-blind")
    role_frame = pd.read_csv(ROOT / ROLE_REL, sep="\t", usecols=["source_object_id", "split_role"], dtype="string")
    train_objects = sorted(set(role_frame.loc[role_frame.split_role.eq("train"), "source_object_id"].dropna().astype(str)))
    if len(train_objects) != EXPECTED_TRAIN_OBJECTS:
        raise ValueError(f"expected exactly nine existing TRAIN source objects, found {len(train_objects)}")
    train_set = set(train_objects)
    labels, label_counts = read_train_labels(train_set)
    train_role_metadata_rows = sum(count for role, count in label_counts["rows_by_endpoint_role"].items() if role.endswith("|train"))
    if label_counts["paired_table_rows"] != 535 or train_role_metadata_rows != 466 or len(labels) != 404:
        raise ValueError(f"paired asset support changed: expected 535 total rows, 466 TRAIN-role metadata rows, and 404 CTRPv2 TRAIN object-key pairs, got {label_counts['paired_table_rows']}, {train_role_metadata_rows}, and {len(labels)}")
    target_keys = pd.read_csv(ROOT / PRED_KEYS_REL, sep="\t", usecols=["condition_id", "source_row", "source_object_id", "full_inchikey"], dtype={"source_object_id": "string", "full_inchikey": "string"})
    target_keys["prediction_row"] = np.arange(len(target_keys), dtype=np.int64)
    if target_keys.condition_id.duplicated().any() or target_keys.source_row.duplicated().any():
        raise ValueError("frozen XPert source target keys are not unique by condition/source row")
    if len(target_keys) != 467:
        raise ValueError(f"frozen XPert target condition count changed: {len(target_keys)}")
    features, exposure_audit = verified_xpert_exposure(train_set, target_keys)
    genes = [str(x).strip() for x in json.loads((ROOT / GENES_REL).read_text(encoding="utf-8"))]
    if len(genes) != EXPECTED_GENES or len(set(genes)) != EXPECTED_GENES:
        raise ValueError("frozen LINCS target gene axis is not 978 unique genes")
    profiles, expression_audit = load_expression(train_set, genes)
    all_label_keys = set(labels.full_inchikey.astype(str))
    drug_features, drug_audit = load_drug_features(all_label_keys)
    label_pairs = set(zip(labels.source_object_id.astype(str), labels.full_inchikey.astype(str)))
    feature_pairs = set(zip(features.source_object_id.astype(str), features.full_inchikey.astype(str)))
    joined_pairs = label_pairs & feature_pairs
    missing_prediction_pairs = sorted(label_pairs - feature_pairs)
    if missing_prediction_pairs:
        raise ValueError(f"exact CTRPv2 TRAIN labels lack frozen prediction feature support: {len(missing_prediction_pairs)} pairs")
    missing_drug_keys = sorted(all_label_keys - set(drug_features))
    ambiguous_morgan_keys = sorted(drug_audit["excluded_nonunique_exact_morgan_keys"])
    if set(missing_drug_keys) != set(ambiguous_morgan_keys) or set(ambiguous_morgan_keys) != EXPECTED_NONUNIQUE_MORGAN_KEYS:
        raise ValueError(f"exact-key Morgan support differs from the frozen two-key ambiguity audit: missing={missing_drug_keys}, ambiguous={ambiguous_morgan_keys}")
    if len(joined_pairs) != 404 or len(all_label_keys) != 123 or len(set(drug_features) & all_label_keys) != EXPECTED_VALID_CANDIDATE_KEYS:
        raise ValueError(f"frozen exact identity domain changed: label-prediction pairs={len(joined_pairs)}, label keys={len(all_label_keys)}, unique-Morgan keys={len(set(drug_features) & all_label_keys)}")
    support = freeze_support(labels, features, train_objects, drug_features, out)
    write_json(out / "leakage_audit.json", {
        "status": "PASS_TRAIN_ROLE_AND_XPERT_IDENTITY_AUDIT",
        "used_numeric_endpoint": "CTRPv2 published AAC only",
        "numeric_label_objects": train_objects, "numeric_label_object_count": len(train_objects),
        "numeric_label_pairs": int(len(labels)), "numeric_label_exact_keys": int(labels.full_inchikey.nunique()),
        "paired_table_other_endpoints_and_roles_numeric_values_parsed": False,
        "xpert_fit_train_source_identity_audit": exposure_audit,
        "label_prediction_support": {"allowed_label_pairs": len(label_pairs), "exact_xpert_prediction_pairs": len(feature_pairs),
                                      "exact_join_pairs": len(joined_pairs), "label_pairs_without_prediction": missing_prediction_pairs,
                                      "label_key_union": len(all_label_keys), "candidate_keys_with_unique_morgan": len(set(drug_features) & all_label_keys),
                                      "excluded_nonunique_morgan_keys": ambiguous_morgan_keys},
        "features": {"baseline_expression": expression_audit, "chemical": drug_audit,
                     "predicted_delta": exposure_audit["aggregation"]},
        "forbidden_numeric_access": {"PRISM": False, "VAL": False, "outer": False, "locked": False, "PDO": False},
        "previous_label_exposure": "CTRPv2 labels were read in the existing EXP-050 §3.4 exploratory analysis; this is not an independent confirmation.",
    })
    write_json(out / "input_audit.json", {
        "paired_source_rows_read": {"all_metadata_rows": label_counts["paired_table_rows"],
                                    "rows_by_endpoint_role": label_counts["rows_by_endpoint_role"],
                                    "unique_object_key_pairs_by_endpoint_role": label_counts["unique_object_key_pairs_by_endpoint_role"],
                                    "train_role_rows_all_endpoints_metadata_only": train_role_metadata_rows,
                                    "CTRPv2_TRAIN_numeric_rows": label_counts.get("CTRPv2_TRAIN_numeric_rows", 0),
                                    "CTRPv2_TRAIN_object_drug_pairs": len(labels),
                                    "CTRPv2_TRAIN_unique_drugs": int(labels.full_inchikey.nunique()),
                                    "XPert_predicted_object_drug_pairs_for_9_TRAIN_objects": len(features),
                                    "label_prediction_exact_pairs": len(joined_pairs),
                                    "whole_label_supported_candidate_drug_keys": len(all_label_keys)},
        "role_denominator_table": support["label_support"], "prediction_support": support["prediction_support"],
        "distinct_denominators_note": "the 123-key exact label union is the frozen starting domain; two keys have nonunique Morgan vectors under the same full InChIKey and are excluded from every arm, leaving 121 keys with unique Morgan support; each fold then uses its explicitly hashed train-supported test-key list. No missing key is imputed or borrowed from VAL/heldout.",
        "other_endpoint_or_role_numeric_labels_parsed": False,
    })
    freeze_outputs = [out / "protocol_snapshot.json", out / "support_freeze.json", out / "frozen_fold_support.tsv",
                      out / "frozen_block_permutations.tsv", out / "leakage_audit.json", out / "input_audit.json"]
    if args.phase == "smoke":
        test_obj = support["smoke_fold"]
        predictions, model_log, metrics = run_fold(labels, features, profiles, drug_features, support, test_obj)
        pd.DataFrame([model_log]).to_json(attempt_dir / "model_fit_and_fold_log.json", orient="records", indent=2)
        predictions.to_csv(attempt_dir / "paired_oof_predictions.tsv", sep="\t", index=False, na_rep="")
        metrics.to_csv(attempt_dir / "per_arm_metrics.tsv", sep="\t", index=False, na_rep="")
        # Smoke checks are real-input end-to-end contract checks; scores are not a development selection result.
        checks = {
            "exact_full_inchikey_predictions_and_labels": bool(predictions.full_inchikey.astype(str).str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").all()),
            "all_five_arms_present": set(metrics.arm) >= {"direct", "predicted_perturbation", "fusion", "equal_capacity_direct", "shuffled_perturbation"},
            "same_key_denominator_across_all_arms": metrics.n_pairs.nunique() == 1 and len(predictions.full_inchikey.unique()) == int(metrics.n_pairs.iloc[0]),
            "finite_predictions_and_labels": bool(np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy()).all()),
            "fusion_weight_from_training_oof": bool(model_log["fusion_weight_info"]["all_candidate_weights_fit_from_outer_train_oof_predictions"]),
            "scalers_fit_on_current_fold_train_only": all(x["scalers_fit_train_only"] for x in model_log["base_models"]),
            "test_labels_not_used_for_fit": model_log["test_label_used_for_any_fit_or_weight"] is False,
            "xpert_fit_training_identity_disjoint": exposure_audit["identity_overlap_status"] == "PASS_ZERO_OVERLAP",
            "readback_keys_equal_saved_keys": True,
        }
        if not all(checks.values()):
            raise ValueError(f"single-object smoke contract failed: {checks}")
        reread = pd.read_csv(attempt_dir / "paired_oof_predictions.tsv", sep="\t", dtype={"full_inchikey": "string"})
        checks["readback_keys_equal_saved_keys"] = reread.full_inchikey.astype(str).tolist() == predictions.full_inchikey.astype(str).tolist()
        if not checks["readback_keys_equal_saved_keys"]:
            raise ValueError("smoke OOF predictions failed exact-key readback")
        write_json(attempt_dir / "smoke_receipt.json", {"status": "PASS_SINGLE_OBJECT_FIVE_ARM_END_TO_END_SMOKE",
                    "run_id": RUN_ID, "test_object": test_obj, "n_object_drug_keys": int(len(predictions)),
                    "all_arm_metrics": metrics.to_dict(orient="records"), "checks": checks,
                    "model_log_sha256": sha256(attempt_dir / "model_fit_and_fold_log.json"),
                    "paired_predictions_sha256": sha256(attempt_dir / "paired_oof_predictions.tsv"),
                    "cpu_only": True, "gpu_hours": 0.0, "scores_used_for_selection": False})
        outputs = freeze_outputs + list(attempt_dir.iterdir())
        record_run(args, started_utc, time.monotonic() - started_wall, time.process_time() - started_process_cpu,
                   input_hashes, outputs, support, "SMOKE_PASSED_SINGLE_OBJECT_FIVE_ARM")
        return
    if args.phase == "full":
        out_full = attempt_dir
        metrics_list: list[pd.DataFrame] = []
        prediction_list: list[pd.DataFrame] = []
        model_logs: list[dict[str, Any]] = []
        for fold in support["fold_support"]:
            test_obj = fold["test_object"]
            preds, log, met = run_fold(labels, features, profiles, drug_features, support, test_obj)
            prediction_list.append(preds); metrics_list.append(met); model_logs.append(log)
        all_predictions = pd.concat(prediction_list, ignore_index=True)
        all_metrics = pd.concat(metrics_list, ignore_index=True)
        if all_predictions.duplicated(["source_object_id", "full_inchikey"]).any():
            raise ValueError("OOF source object/full-InChIKey rows are duplicated")
        arm_cols = ["pred_direct", "pred_predicted_perturbation", "pred_fusion", "pred_equal_capacity_direct", "pred_shuffled_perturbation"]
        if not np.isfinite(all_predictions[["observed_aac", *arm_cols]].to_numpy()).all():
            raise ValueError("full OOF arm predictions or labels contain non-finite values")
        all_predictions.to_csv(out_full / "oof_predictions.tsv", sep="\t", index=False, na_rep="")
        all_metrics.to_csv(out_full / "per_object_arm_metrics.tsv", sep="\t", index=False, na_rep="")
        contrasts = paired_differences(all_metrics)
        contrasts.to_csv(out_full / "paired_arm_differences.tsv", sep="\t", index=False, na_rep="")
        write_json(out_full / "summary.json", {
            "status": "COMPLETED_EXPLORATORY_DEVELOPMENT_NINE_OBJECT_OOF",
            "run_id": RUN_ID, "endpoint": "CTRPv2 published AAC; higher is more sensitive",
            "analysis_unit": "source PatientID object; test keys nested within object, never treated as biological n",
            "n_outer_objects": int(all_predictions.source_object_id.nunique()),
            "n_oof_object_drug_pairs": int(len(all_predictions)),
            "candidate_domain_size": support["candidate_domain_size"],
            "arm_summary": summarize(all_metrics),
            "pairwise_contrasts": "per-object paired differences and equal-object mean differences; descriptive only, no inferential intervals or significance tests",
            "fixed_references": ["outer-fold train-object per-InChIKey AAC mean", "one-feature ridge on predicted full-axis mean absolute response strength"],
            "fusion": "single convex coefficient selected on current outer-training objects' inner leave-one-object-out predictions",
            "equal_capacity_control": "same nominal fitted-parameter count (6,055, including two Ridge intercepts and the blend scalar) as two-fit late fusion; duplicate direct copies are rank-deficient and control parameter count, not feature rank",
            "generalization_boundary": "source-domain exploratory OOF only; no independent validation, PDO, clinical, or outer/locked claim",
            "input_asset_boundary": "3.4 frozen identity, paired source assets, XPert attempt_001 predicted source deltas; CTRPv2 TRAIN rows only",
        })
        pd.DataFrame(model_logs).to_json(out_full / "fold_model_logs.json", orient="records", indent=2)
        # output readback and integrity checks before manifest close
        readback = pd.read_csv(out_full / "oof_predictions.tsv", sep="\t", dtype={"source_object_id": "string", "full_inchikey": "string"})
        if len(readback) != len(all_predictions) or readback.duplicated(["source_object_id", "full_inchikey"]).any():
            raise ValueError("full OOF keyed prediction table failed readback")
        if not readback.full_inchikey.astype(str).isin(support["candidate_domain"]).all():
            raise ValueError("full OOF output includes a key outside the frozen public domain")
        receipt = {"status": "PASS_NINE_OBJECT_FIVE_ARM_OOF", "n_objects": int(readback.source_object_id.nunique()),
                   "n_object_drug_pairs": int(len(readback)), "n_arm_predictions": len(arm_cols),
                   "per_object_denominators": readback.groupby("source_object_id").full_inchikey.nunique().astype(int).to_dict(),
                   "all_arms_same_key_rows": True, "test_labels_used_for_selection": False,
                   "cpu_only": True, "gpu_hours": 0.0, "summary_sha256": sha256(out_full / "summary.json"),
                   "oof_predictions_sha256": sha256(out_full / "oof_predictions.tsv"),
                   "paired_differences_sha256": sha256(out_full / "paired_arm_differences.tsv")}
        write_json(out_full / "completion_receipt.json", receipt)
        outputs = freeze_outputs + list(out_full.iterdir())
        record_run(args, started_utc, time.monotonic() - started_wall, time.process_time() - started_process_cpu,
                   input_hashes, outputs, support, "COMPLETED_EXPLORATORY_NINE_OBJECT_OOF")
        return
    raise ValueError(f"unsupported phase {args.phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["freeze", "smoke", "full"], required=True)
    args = parser.parse_args()
    started_wall = time.monotonic(); started_process_cpu = time.process_time(); started_utc = utc_now()
    try:
        execute(args)
    except Exception as exc:
        out = ROOT / OUT_REL
        out.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - started_wall
        process_cpu_elapsed = time.process_time() - started_process_cpu
        failure_index = len(list(out.glob(f"{args.phase}_failure*.json"))) + 1
        failure = out / f"{args.phase}_failure_attempt_{failure_index:03d}.json"
        write_json(failure, {"run_id": args.run_id, "phase": args.phase, "status": "FAILED",
                             "error_type": type(exc).__name__, "error": str(exc), "failed_at_utc": utc_now(),
                             "started_at_utc": started_utc, "elapsed_wall_minutes": elapsed / 60.0,
                             "process_cpu_minutes": process_cpu_elapsed / 60.0,
                             "cpu_cap_minutes": CPU_CAP_MINUTES, "gpu_hours": 0.0})
        accounting_path = out / "cpu_accounting.json"
        prior = json.loads(accounting_path.read_text(encoding="utf-8")) if accounting_path.exists() else {"attempts": []}
        current = elapsed / 60.0
        accumulated = sum(float(x["elapsed_minutes"]) for x in prior["attempts"])
        prior["attempts"].append({"phase": args.phase, "started_at_utc": started_utc, "finished_at_utc": utc_now(),
                                  "elapsed_wall_minutes": current, "process_cpu_minutes": process_cpu_elapsed / 60.0,
                                  "elapsed_minutes": current, "cpu_cap_minutes": CPU_CAP_MINUTES,
                                  "gpu_hours": 0.0, "status": "FAILED", "error_type": type(exc).__name__})
        prior["cumulative_elapsed_minutes"] = accumulated + current
        prior["cpu_hard_cap_minutes"] = CPU_CAP_MINUTES
        write_json(accounting_path, prior)
        raise


if __name__ == "__main__":
    main()
