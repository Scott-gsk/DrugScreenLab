"""Bind EXP-048's frozen drug identities and shared patient-key mask to EXP-036 features.

This CPU-only check reads target feature arrays and patient_id/drug_id keys;
it never reads target response values, predictions, or outer-fold scores.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/EXP-048.yaml"
PANEL_RECEIPT = ROOT / "artifacts/experiments/EXP-047/20260926_alt_common_panel_protocol_v1/receipt.json"
INPUT_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/receipt.json"
TARGET_INPUTS = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
SOURCE_VECTOR_DIR = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1"
OUTPUT = ROOT / "artifacts/experiments/EXP-048/20260926_target_feature_binding_v1"

EXPECTED_INPUT_RECEIPT_SHA256 = "7f70be0fdf1106038463455dd4d2bf25a01b61b140f2b78e9046c60eb24283e2"
EXPECTED_PANEL_RECEIPT_SHA256 = "2626952099789c1497cad792cbad63e883c594b76e140051156efba0fbcb5981"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", None}:
        raise RuntimeError("this target-input binding step must run CPU-only")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite target binding: {OUTPUT}")

    input_receipt_sha = sha256(INPUT_RECEIPT)
    if input_receipt_sha != EXPECTED_INPUT_RECEIPT_SHA256:
        raise ValueError("the admitted EXP-048 source/key receipt changed")
    panel_receipt_sha = sha256(PANEL_RECEIPT)
    if panel_receipt_sha != EXPECTED_PANEL_RECEIPT_SHA256:
        raise ValueError("the reviewed nine-drug identity receipt changed")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected_indexes = [int(value) for value in config["drug_panel"]["target_indices"]]
    expected_inchikeys = [str(value) for value in config["drug_panel"]["target_inchikeys"]]
    expected_pert_ids = [str(value) for value in config["pert_ids"]]
    if not (len(expected_indexes) == len(expected_inchikeys) == len(expected_pert_ids) == 9):
        raise ValueError("the frozen EXP-048 identity axes must each contain nine drugs")
    if len(set(expected_indexes)) != 9 or len(set(expected_inchikeys)) != 9 or len(set(expected_pert_ids)) != 9:
        raise ValueError("the frozen EXP-048 drug identity axes contain duplicates")

    panel_receipt = json.loads(PANEL_RECEIPT.read_text(encoding="utf-8"))
    unique_mappings = {
        int(row["drug_index"]): str(row["pert_id"])
        for row in panel_receipt["identity_bridge"]["unique_mappings"]
    }
    if [unique_mappings.get(index) for index in expected_indexes] != expected_pert_ids:
        raise ValueError("the EXP-048 config pert_id order differs from the reviewed panel receipt")

    input_receipt = json.loads(INPUT_RECEIPT.read_text(encoding="utf-8"))
    manifest_path = TARGET_INPUTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification_path = TARGET_INPUTS / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise ValueError("the reusable EXP-036 target feature bundle is not independently verified")
    if sha256(manifest_path) != input_receipt["target_key_coverage"]["target_input_manifest_sha256"]:
        raise ValueError("the EXP-036 feature manifest differs from the admitted input receipt")
    if sha256(verification_path) != input_receipt["target_key_coverage"]["target_input_verification_sha256"]:
        raise ValueError("the EXP-036 input verification differs from the admitted receipt")

    target_file_hashes = manifest["files_sha256"]
    for filename, expected_sha in target_file_hashes.items():
        path = TARGET_INPUTS / filename
        if not path.is_file() or sha256(path) != expected_sha:
            raise ValueError(f"target input file hash mismatch: {filename}")

    patients = [str(value) for value in json.loads((TARGET_INPUTS / "patient_axis.json").read_text(encoding="utf-8"))]
    if len(patients) != 51 or len(set(patients)) != 51:
        raise ValueError("the verified EXP-036 patient axis must contain 51 unique patients")
    expression_path = TARGET_INPUTS / "patient_expression.npy"
    expression = np.load(expression_path, mmap_mode="r", allow_pickle=False)
    if expression.shape != (len(patients), 19091) or not np.isfinite(expression).all():
        raise ValueError("the direct-route expression feature axis is incomplete or non-finite")

    drug_axis_path = TARGET_INPUTS / "drug_axis.parquet"
    drug_axis = pd.read_parquet(drug_axis_path, columns=["drug_index", "drug_id"])
    if drug_axis["drug_index"].duplicated().any() or drug_axis["drug_id"].duplicated().any():
        raise ValueError("the verified target drug axis contains duplicate indexes or identities")
    drug_axis = drug_axis.sort_values("drug_index", kind="stable").reset_index(drop=True)
    if drug_axis["drug_index"].astype(int).tolist() != list(range(len(drug_axis))):
        raise ValueError("drug_index does not match the materialized ordinal-array row axis")
    target_id_by_index = {int(row.drug_index): str(row.drug_id) for row in drug_axis.itertuples(index=False)}
    if [target_id_by_index.get(index) for index in expected_indexes] != expected_inchikeys:
        raise ValueError("the direct-response drug axis does not match the frozen nine InChIKeys")

    ordinal_path = TARGET_INPUTS / "drug_ordinal128.npy"
    ordinal = np.load(ordinal_path, mmap_mode="r", allow_pickle=False)
    if ordinal.shape != (len(drug_axis), 128) or not np.issubdtype(ordinal.dtype, np.number):
        raise ValueError("the registered PharmaFormer drug encoding axis is incomplete or changed")
    if (not np.isfinite(ordinal).all() or np.any(ordinal < 0)
            or not np.equal(ordinal, np.floor(ordinal)).all()):
        raise ValueError("the direct-route drug ordinal codes must be finite non-negative integers")

    pairs_path = TARGET_INPUTS / "patient_pairs.parquet"
    pairs = pd.read_parquet(pairs_path, columns=["patient_id", "drug_id"])
    pairs["patient_id"] = pairs["patient_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    if pairs.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("the target patient/drug key projection contains duplicate keys")
    if set(pairs["patient_id"]) != set(patients):
        raise ValueError("target patient/drug keys differ from the verified patient axis")

    selected = pairs.loc[pairs["drug_id"].isin(expected_inchikeys)]
    present = set(zip(selected["patient_id"], selected["drug_id"]))
    rows = []
    for patient_index, patient_id in enumerate(patients):
        for drug_index, inchikey, pert_id in zip(expected_indexes, expected_inchikeys, expected_pert_ids):
            rows.append({
                "patient_index": patient_index,
                "patient_id": patient_id,
                "drug_index": drug_index,
                "drug_id": inchikey,
                "pert_id": pert_id,
                "target_key_present": (patient_id, inchikey) in present,
            })
    mask = pd.DataFrame(rows)
    coverage = mask.groupby("patient_id", sort=False)["target_key_present"].sum()
    if len(coverage) != 51 or not coverage.eq(7).all():
        raise ValueError("the frozen nine-drug mask changed: expected seven observed keys for each of 51 patients")
    admitted_histogram = input_receipt["target_key_coverage"]["patient_drug_count_histogram"]
    if admitted_histogram != {"7": 51}:
        raise ValueError("the existing label-blind key receipt no longer records 51 patients at seven keys")

    vector_path = SOURCE_VECTOR_DIR / "source_reference_perturbation.npy"
    vector = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    vector_receipt = input_receipt["source_oof"]
    if (vector.shape != (9, 978) or not np.isfinite(vector).all()
            or sha256(vector_path) != vector_receipt["vector_sha256"]):
        raise ValueError("the admitted source OOF vector is not bound to the fixed nine-by-978 axis")

    output_dir = OUTPUT
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{OUTPUT.name}.tmp-", dir=OUTPUT.parent) as temp_name:
        staging = Path(temp_name)
        mask_path = staging / "target_patient_drug_key_mask.csv"
        mask.to_csv(mask_path, index=False)
        receipt = {
            "schema_version": 1,
            "experiment_id": "EXP-048",
            "status": "TARGET_FEATURE_IDENTITY_AND_SHARED_KEY_MASK_PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "script_sha256": sha256(Path(__file__).resolve()),
            "input_receipts": {
                "exp048_source_and_key_receipt": str(INPUT_RECEIPT.relative_to(ROOT)),
                "exp048_source_and_key_receipt_sha256": input_receipt_sha,
                "exp047_identity_receipt": str(PANEL_RECEIPT.relative_to(ROOT)),
                "exp047_identity_receipt_sha256": panel_receipt_sha,
                "exp036_target_manifest_sha256": sha256(manifest_path),
                "exp036_target_verification_sha256": sha256(verification_path),
                "exp036_file_hashes": target_file_hashes,
            },
            "drug_identity_binding": [
                {"drug_index": i, "target_inchikey": k, "lincs_pert_id": p}
                for i, k, p in zip(expected_indexes, expected_inchikeys, expected_pert_ids)
            ],
            "target_feature_axes": {
                "patient_count": len(patients),
                "patient_expression_shape": list(expression.shape),
                "patient_expression_all_finite": True,
                "drug_count": len(drug_axis),
                "drug_ordinal_shape": list(ordinal.shape),
                "drug_ordinal_dtype": str(ordinal.dtype),
                "source_vector_shape": list(vector.shape),
                "source_vector_sha256": sha256(vector_path),
            },
            "target_key_mask": {
                "path": str((OUTPUT / mask_path.name).relative_to(ROOT)),
                "sha256": sha256(mask_path),
                "row_count": len(mask),
                "observed_drugs_per_patient_histogram": {"7": 51},
                "shared_across_arms_and_seeds_by_identity_keys": True,
            },
            "data_boundary": {
                "target_pair_columns_read": ["patient_id", "drug_id"],
                "target_response_values_read": False,
                "target_predictions_or_outer_scores_read": False,
                "outer_test_labels_used_for_selection": False,
                "cuda_visible_devices": "",
                "gpu_hours": 0,
                "feature_arrays_copied": False,
            },
            "next_gate": "Freeze five-arm implementations, equal-capacity construction, deterministic perturbation identity shuffle, matched candidate/update budget, and complete CPU feature-to-score chain; then measure/admit costs only against the remaining shared 40 GPU-hour ledger after EXP-036/041 core."
        }
        receipt_path = staging / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
    return receipt


if __name__ == "__main__":
    result = main()
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "feature_axes": result["target_feature_axes"],
        "drug_identities_bound": len(result["drug_identity_binding"]),
        "patient_key_rows": result["target_key_mask"]["row_count"],
        "gpu_hours": result["data_boundary"]["gpu_hours"],
    }, ensure_ascii=False))
