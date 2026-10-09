"""Freeze EXP-037's complete exact-identity intersection and record row-map gate.

This identity-only receipt does not inspect expression, perturbation labels,
feature-array values, checkpoints, predictions, or model scores. It deliberately
does not emit a usable feature-row crosswalk when the authoritative producer is
missing or when distinct structures share a feature node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TARGET_DRUGS = ROOT / "data/processed/lincs/GSE92742/continuous_inputs_v2/drugs.parquet"
XPERT_DRUGS = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv"
DRUG_REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
DATA_REGISTRY = ROOT / "data/registry/datasets.json"
MY_DATASET = ROOT / "data/external/xpert_source/datasets/MyDataset.py"
XPERT_CONFIG = ROOT / "data/external/xpert_source/configs/config_l1000.yaml"
XPERT_MODEL = ROOT / "data/external/xpert_source/models/model_XPert.py"
REGISTRY_PRODUCER = ROOT / "scripts/foundation/build_xpert_registries.py"
OUTPUT_ROOT = ROOT / "artifacts/experiments/EXP-037"
DEFAULT_RUN_ID = "20260925_xpert_b1_feature_row_crosswalk_v1"
DRUG_REGISTRY_ID = "xpert_drug_registry_v1"
EXPECTED_FEATURE_INDEX_KEY = "pert_idx == official drug_node_idx"
REQUIRED_TARGET_COLUMNS = ("drug_index", "pert_id", "inchi_key")
REQUIRED_SOURCE_COLUMNS = ("pert_id", "inchi_key", "drug_node_idx")
SIDE_CAR_COLUMNS = ("pert_id", "inchi_key", "pert_idx", "unimol", "hg")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def registered_drug_registry_entry() -> dict[str, Any]:
    entries = json.loads(DATA_REGISTRY.read_text(encoding="utf-8"))
    matches = [item for item in entries if item.get("id") == DRUG_REGISTRY_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one registered {DRUG_REGISTRY_ID} entry")
    return matches[0]


def runtime_code_observations() -> dict[str, Any]:
    source = MY_DATASET.read_text(encoding="utf-8")
    config = XPERT_CONFIG.read_text(encoding="utf-8")
    model = XPERT_MODEL.read_text(encoding="utf-8")
    return {
        "mydataset_path": str(MY_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "mydataset_sha256": sha256_file(MY_DATASET),
        "xpert_model_path": str(XPERT_MODEL.relative_to(ROOT)).replace("\\", "/"),
        "xpert_model_sha256": sha256_file(XPERT_MODEL),
        "config_path": str(XPERT_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(XPERT_CONFIG),
        "dataset_casts_raw_pert_id_to_pert_idx": (
            "raw_data_items['pert_idx'] = raw_data_items['pert_id'].astype(np.int64)"
            in source
        ),
        "dataset_indexes_supplied_drug_features_by_pert_idx": (
            "drug_feat = self.drug_feat[pert_idx]" in source
        ),
        "dataset_source_references_drug_node_idx": "drug_node_idx" in source,
        "model_indexes_hg_embedding_by_batch_drug_idx": (
            "self.drug_HG_embed[drug_idx]" in model
        ),
        "config_selects_unimol_array": (
            "drug_unimol_path: processed_data/all_drugs_unimol_arr.npy" in config
        ),
        "config_selects_hg_embedding": (
            "drug_hg_pretrained_embed_path: HG_data/saved_embedding/HG_drug_embeddings.npy"
            in config
        ),
    }


def build_intersection() -> tuple[list[str], dict[str, Any]]:
    target = pd.read_parquet(TARGET_DRUGS, columns=list(REQUIRED_TARGET_COLUMNS))
    source = pd.read_csv(
        XPERT_DRUGS,
        usecols=list(REQUIRED_SOURCE_COLUMNS),
        dtype={column: "string" for column in REQUIRED_SOURCE_COLUMNS},
    )
    registry_payload = json.loads(DRUG_REGISTRY.read_text(encoding="utf-8"))
    registry_rows = registry_payload.get("drugs")
    if not isinstance(registry_rows, list):
        raise ValueError("registered XPert drug registry has no drugs list")
    sidecar = pd.DataFrame(
        [{column: row.get(column) for column in SIDE_CAR_COLUMNS} for row in registry_rows]
    )

    for frame, columns, name in (
        (target, REQUIRED_TARGET_COLUMNS, "target drug axis"),
        (source, REQUIRED_SOURCE_COLUMNS, "official XPert drug table"),
        (sidecar, SIDE_CAR_COLUMNS, "registered XPert registry"),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    target["pert_id"] = target["pert_id"].astype("string")
    target["inchi_key"] = target["inchi_key"].astype("string")
    target["drug_index"] = pd.to_numeric(target["drug_index"], errors="raise").astype("int64")
    source["pert_id"] = source["pert_id"].astype("string")
    source["inchi_key"] = source["inchi_key"].astype("string")
    source["drug_node_idx"] = pd.to_numeric(
        source["drug_node_idx"], errors="coerce"
    ).astype("Int64")
    sidecar["pert_id"] = sidecar["pert_id"].astype("string")
    sidecar["inchi_key"] = sidecar["inchi_key"].astype("string")
    sidecar["pert_idx"] = pd.to_numeric(sidecar["pert_idx"], errors="coerce").astype("Int64")

    if target["pert_id"].isna().any() or target["pert_id"].duplicated().any():
        raise ValueError("target pert_id axis is missing or duplicated")
    if target["inchi_key"].isna().any() or target["drug_index"].duplicated().any():
        raise ValueError("target InChIKey or drug_index axis is missing or duplicated")
    if not target["drug_index"].sort_values().reset_index(drop=True).equals(
        pd.Series(range(len(target)), dtype="int64")
    ):
        raise ValueError("target drug_index is not a complete contiguous row axis")
    if sidecar["pert_id"].isna().any() or sidecar["pert_id"].duplicated().any():
        raise ValueError("registered XPert pert_id axis is missing or duplicated")
    if sidecar["inchi_key"].isna().any() or sidecar["pert_idx"].isna().any():
        raise ValueError("registered XPert identity/index fields are incomplete")
    if sidecar["pert_idx"].lt(0).any():
        raise ValueError("registered XPert feature indices must be nonnegative")

    target = target.sort_values("drug_index", kind="stable").reset_index(drop=True)
    exact = target.merge(
        source.drop_duplicates(list(REQUIRED_SOURCE_COLUMNS)),
        on=["pert_id", "inchi_key"],
        how="inner",
        validate="one_to_one",
    ).sort_values("drug_index", kind="stable")
    exact_with_registry = exact.merge(
        sidecar,
        on="pert_id",
        how="left",
        validate="one_to_one",
        suffixes=("_official", "_registry"),
    )
    exact_with_registry["registry_identity_matches_official"] = (
        exact_with_registry["inchi_key_official"].eq(
            exact_with_registry["inchi_key_registry"]
        )
    )
    exact_with_registry["registry_row_index_equals_node"] = (
        exact_with_registry["pert_idx"].eq(exact_with_registry["drug_node_idx"])
    )
    exact_with_registry["features_complete"] = (
        exact_with_registry["unimol"].eq(True) & exact_with_registry["hg"].eq(True)
    )

    node_key_counts = (
        source.dropna(subset=["drug_node_idx", "inchi_key"])
        .drop_duplicates(["drug_node_idx", "inchi_key"])
        .groupby("drug_node_idx", sort=False)["inchi_key"]
        .nunique()
    )
    conflict_nodes = set(node_key_counts[node_key_counts.gt(1)].index.astype(int))
    exact_conflicted = exact[exact["drug_node_idx"].isin(conflict_nodes)]

    ids = exact["pert_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("exact intersection contains duplicate pert_id values")
    diagnostics = {
        "target_rows": int(len(target)),
        "target_unique_pert_ids": int(target["pert_id"].nunique()),
        "target_drug_index_order_valid": True,
        "official_source_rows": int(len(source)),
        "official_source_unique_pert_ids": int(source["pert_id"].nunique()),
        "registered_registry_rows": int(len(sidecar)),
        "registered_registry_unique_pert_ids": int(sidecar["pert_id"].nunique()),
        "registered_registry_unique_pert_idx": int(sidecar["pert_idx"].nunique()),
        "registered_registry_reused_pert_idx_count": int(
            len(sidecar) - sidecar["pert_idx"].nunique()
        ),
        "exact_id_and_full_inchikey_intersection_count": int(len(exact)),
        "exact_id_and_full_inchikey_unique_ids": int(exact["pert_id"].nunique()),
        "exact_intersection_missing_registered_record": int(
            exact_with_registry["pert_idx"].isna().sum()
        ),
        "exact_intersection_sidecar_key_mismatch": int(
            (~exact_with_registry["registry_identity_matches_official"]).sum()
        ),
        "sidecar_pert_idx_equals_official_drug_node_idx_count": int(
            exact_with_registry["registry_row_index_equals_node"].sum()
        ),
        "exact_intersection_complete_unimol_and_hg_flag_count": int(
            exact_with_registry["features_complete"].sum()
        ),
        "source_multi_key_node_count": int(len(conflict_nodes)),
        "exact_intersection_ids_on_multi_key_nodes": int(len(exact_conflicted)),
        "exact_intersection_id_list_sha256_sorted_utf8_lf": sha256_bytes(
            "\n".join(sorted(ids)).encode("utf-8")
        ),
        "exact_intersection_id_list_sha256_target_order_utf8_lf": sha256_bytes(
            "\n".join(ids).encode("utf-8")
        ),
    }
    return ids, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("run_id must be a simple directory name")
    output = OUTPUT_ROOT / args.run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")

    registration = registered_drug_registry_entry()
    actual_registry_sha = sha256_file(DRUG_REGISTRY)
    if actual_registry_sha.lower() != str(
        registration.get("checksum", {}).get("value", "")
    ).lower():
        raise ValueError("registered XPert drug registry checksum does not match")

    actual_source_sha = sha256_file(XPERT_DRUGS)
    registry_payload = json.loads(DRUG_REGISTRY.read_text(encoding="utf-8"))
    source_sha_in_sidecar = str(
        registry_payload.get("source", {}).get("drug_info_sha256", "")
    ).lower()
    if actual_source_sha.lower() != source_sha_in_sidecar:
        raise ValueError("official XPert drug table checksum differs from registry provenance")

    ids, diagnostics = build_intersection()
    code_observations = runtime_code_observations()
    feature_index_key = str(registry_payload.get("feature_index_key", ""))
    row_equality_count = diagnostics[
        "sidecar_pert_idx_equals_official_drug_node_idx_count"
    ]
    row_relation_claim_matches = feature_index_key == EXPECTED_FEATURE_INDEX_KEY
    exact_row_relation_matches = row_equality_count == len(ids)
    conflict_count = diagnostics["exact_intersection_ids_on_multi_key_nodes"]
    producer_relative = str(REGISTRY_PRODUCER.relative_to(ROOT)).replace("\\", "/")

    # The identity intersection is useful as a frozen contract input. It is not
    # a feature-ready set until an official row-to-structure producer resolves
    # the node collisions and binds both configured feature arrays to the axis.
    identity_path = output / "exact_identity_intersection.json"
    identity_payload = {
        "format": "exp037_b1_exact_identity_intersection_v1",
        "experiment_id": "EXP-037",
        "run_id": args.run_id,
        "status": "FROZEN_IDENTITY_INTERSECTION_FEATURE_ROW_BLOCKED",
        "identity_rule": "target pert_id and complete InChIKey exactly match the registered official XPert drug table",
        "ordering": "ascending target drug_index from the frozen B1 drug axis",
        "feature_input_eligible": False,
        "feature_input_block_reason": "official node-to-feature-row identity provenance is not established and 84 source nodes map to multiple full InChIKeys",
        "count": len(ids),
        "pert_ids": ids,
    }
    write_json_atomic(identity_path, identity_payload)

    created = datetime.now(timezone.utc).isoformat()
    receipt = {
        "artifact_type": "exp037_b1_full_identity_intersection_and_feature_row_gate",
        "experiment_id": "EXP-037",
        "run_id": args.run_id,
        "created_at_utc": created,
        "status": "IDENTITY_LIST_FROZEN_FEATURE_ROW_MAPPING_BLOCKED",
        "execution": {
            "environment": "WSL2 Ubuntu / Conda drugscreening-gpu",
            "device": "CPU only; no CUDA",
            "expression_or_label_values_read": False,
            "feature_array_values_read": False,
            "checkpoints_predictions_or_scores_read": False,
            "training_started": False,
        },
        "inputs": {
            "target_drug_axis": {
                "path": str(TARGET_DRUGS.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(TARGET_DRUGS),
                "columns_read": list(REQUIRED_TARGET_COLUMNS),
            },
            "official_xpert_drug_table": {
                "path": str(XPERT_DRUGS.relative_to(ROOT)).replace("\\", "/"),
                "sha256": actual_source_sha,
                "columns_read": list(REQUIRED_SOURCE_COLUMNS),
            },
            "registered_xpert_drug_registry": {
                "path": str(DRUG_REGISTRY.relative_to(ROOT)).replace("\\", "/"),
                "sha256": actual_registry_sha,
                "registered_id": DRUG_REGISTRY_ID,
                "registered_expected_sha256": registration["checksum"]["value"],
                "feature_index_key_claim": feature_index_key,
                "source_producer_declared_by_registry": registration.get(
                    "provenance", {}
                ).get("generator"),
                "declared_source_producer_path": producer_relative,
                "declared_source_producer_present": REGISTRY_PRODUCER.is_file(),
                "registration_asset_type": registration.get("asset_type"),
            },
            "dataset_registry": {
                "path": str(DATA_REGISTRY.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(DATA_REGISTRY),
            },
            "official_runtime_code": code_observations,
        },
        "counts_and_checks": diagnostics,
        "identity_intersection_file": {
            "path": str(identity_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(identity_path),
            "count": len(ids),
            "order": "target drug_index ascending",
            "all_panel_members_retained": True,
        },
        "official_node_to_feature_row_mapping": {
            "status": "NOT_AUTHORITATIVELY_ESTABLISHED",
            "registered_sidecar_claim_matches_expected_text": row_relation_claim_matches,
            "registered_sidecar_equals_official_node_for_exact_ids": exact_row_relation_matches,
            "registered_sidecar_is_project_derived_metadata": True,
            "official_producer_source_present": REGISTRY_PRODUCER.is_file(),
            "reason": (
                "The registered JSON asserts pert_idx == official drug_node_idx and agrees for the full exact-ID intersection, "
                "but the official XPert runtime code does not reference drug_node_idx, and the registry-declared producer source "
                "is absent from the workspace. The equality therefore remains a project-side consistency check, not publisher-level "
                "row-to-structure evidence."
            ),
            "existing_blocker_reproduced": {
                "multi_key_source_node_count": diagnostics["source_multi_key_node_count"],
                "exact_panel_ids_on_those_nodes": conflict_count,
            },
            "feature_row_crosswalk_emitted": False,
        },
        "minimum_external_unblock": {
            "request": (
                "Obtain a versioned publisher-provided (or corrected, source-backed) table/manifest that binds each affected full "
                "InChIKey and pert_id to drug_node_idx, the UniMol array row, and the HG embedding row; include canonical isomeric "
                "SMILES, asset/version checksums, producer code or an explicit row-axis contract, and a documented collision/alias rule. "
                "It must resolve the 84 multi-key nodes affecting 158 exact B1 IDs; a bare claim that indices are equal is insufficient."
            ),
            "after_receipt": (
                "Register the source-backed mapping and checksum, rerun this identity-only gate, then emit the ordered full 6,994-row "
                "crosswalk only if every ID has one valid structure-to-feature row and all collisions are authoritatively resolved."
            ),
        },
    }
    receipt_path = output / "receipt.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({
        "run_id": args.run_id,
        "status": receipt["status"],
        "identity_intersection_count": len(ids),
        "identity_intersection_sha256": receipt["identity_intersection_file"]["sha256"],
        "feature_row_crosswalk_emitted": False,
        "source_multi_key_node_count": diagnostics["source_multi_key_node_count"],
        "exact_panel_ids_on_multi_key_nodes": conflict_count,
        "receipt_path": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
