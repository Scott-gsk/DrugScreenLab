#!/usr/bin/env python3
"""Response-blind identity and native-feature-axis gate for EXP-063.

This entrypoint reads EXP-062 frozen metadata and the XPert H5AD obs/var
metadata only. It never accesses H5AD X, X_ctl, layers, or any prediction
array values. NumPy files are inspected through their headers and hashed as
bytes; no feature values are decoded.
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

import anndata as ad
import numpy as np
import pandas as pd
import yaml


REPO = Path("/mnt/d/Code/DrugScreenLab")
XPERT = Path("/mnt/d/Code/XPert")
SOURCE_RUN = REPO / "artifacts/experiments/EXP-062/20261008_exp062_coverage20_fixed_v3"
EXP051_RUN = REPO / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
PRIOR_CPU_SECONDS = 305.0  # conservative package charge through v5, failed fit_attempt_001/002, and preflight probes


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def npy_header(path: Path) -> dict[str, Any]:
    """Read only the .npy header; never load the numerical payload."""
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
    return {"shape": list(shape), "fortran_order": bool(fortran_order), "dtype": str(dtype)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/EXP-063-xpert-coverage20.yaml")
    parser.add_argument("--gate-id", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = REPO / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = REPO / config["output"]["root"]
    output_dir = run_root / "preflight" / args.gate_id
    if output_dir.exists():
        raise FileExistsError(f"immutable gate directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "xpert_native_axis_metadata_gate.json"

    checks: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    try:
        if config["experiment_id"] != "EXP-063":
            raise ValueError("wrong experiment in config")
        if config["data"]["rows"] != 1982 or config["data"]["genes"] != 978:
            raise ValueError("frozen row/gene domain differs from EXP-063 contract")

        metadata = load_json(SOURCE_RUN / "metadata_freeze_manifest.json")
        metadata_root = load_json(SOURCE_RUN / "root_metadata_readback.json")
        value_manifest = load_json(SOURCE_RUN / "fit_attempt_002/run_manifest.json")
        numeric_root = load_json(SOURCE_RUN / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json")
        expected_metadata_sha = config["data"].get("metadata_freeze_sha256")
        actual_metadata_sha = sha256_file(SOURCE_RUN / "metadata_freeze_manifest.json")
        checks["metadata_root_pass"] = metadata_root.get("status") == "PASS_ROOT_INDEPENDENT_METADATA_READBACK"
        checks["numeric_root_pass"] = numeric_root.get("status") == "PASS_ROOT_INDEPENDENT_NUMERIC_READBACK"
        checks["metadata_source_values_unopened"] = not metadata["matrix_access"]["h5ad_X_read"] and not metadata["matrix_access"]["h5ad_X_ctl_read"] and not metadata["matrix_access"]["h5ad_layers_read"]
        checks["numeric_root_source_values_not_reopened"] = numeric_root.get("source_values_independently_reopened") is False
        checks["frozen_counts_match"] = (
            metadata["selection"]["selected_conditions"] == 1982
            and metadata["selection"]["selected_drug_count"] == 20
            and metadata["selection"]["selected_train_conditions"] == 1291
            and metadata["selection"]["selected_inner_dev_conditions"] == 691
            and metadata["selection"]["train_inner_dev_object_overlap"] == 0
            and metadata["gene_axis_sha256"] == "8d28d261f078bebbb1914597640b0902aa58568d6b5a81e18e0ec95baf117043"
        )
        if expected_metadata_sha and expected_metadata_sha != actual_metadata_sha:
            raise ValueError("configured metadata freeze SHA256 does not match frozen manifest")
        expected_receipt_shas = {
            "metadata_root_readback_sha256": config["data"].get("metadata_root_readback_sha256"),
            "numeric_root_readback_sha256": config["data"].get("numeric_root_readback_sha256"),
        }
        actual_receipt_shas = {
            "metadata_root_readback_sha256": sha256_file(SOURCE_RUN / "root_metadata_readback.json"),
            "numeric_root_readback_sha256": sha256_file(SOURCE_RUN / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json"),
        }
        if any(expected_receipt_shas[name] != actual_receipt_shas[name] for name in expected_receipt_shas):
            raise ValueError(f"configured EXP-062 root receipt SHA mismatch: {actual_receipt_shas}")
        if not all(checks[k] for k in ("metadata_root_pass", "numeric_root_pass", "metadata_source_values_unopened", "numeric_root_source_values_not_reopened", "frozen_counts_match")):
            raise ValueError(f"EXP-062 source authorization/readback gate failed: {checks}")

        frozen_files = metadata["frozen_files_sha256"]
        for name, expected in frozen_files.items():
            path = SOURCE_RUN / name
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"EXP-062 frozen metadata file hash mismatch: {name}")
            hashes[f"exp062:{name}"] = actual

        npz_path = SOURCE_RUN / "fit_attempt_002/keyed_predictions.npz"
        npz_sha = sha256_file(npz_path)
        checks["baseline_prediction_archive_hash_bound"] = (
            npz_sha == numeric_root["attempt_keyed_predictions_sha256"]
            and npz_sha == value_manifest["outputs_sha256"]["keyed_predictions.npz"]
        )
        if not checks["baseline_prediction_archive_hash_bound"]:
            raise ValueError("EXP-062 baseline archive is not bound to both run and root receipts")
        hashes["exp062:keyed_predictions.npz"] = npz_sha
        hashes["exp062:metadata_freeze_manifest.json"] = actual_metadata_sha
        hashes["exp062:root_metadata_readback.json"] = sha256_file(SOURCE_RUN / "root_metadata_readback.json")
        hashes["exp062:root_numeric_readback.json"] = sha256_file(SOURCE_RUN / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json")

        keys_path = SOURCE_RUN / "response_blind_conditions.tsv"
        keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
        if len(keys) != 1982 or keys["full_inchikey"].nunique() != 20:
            raise ValueError("frozen key table count mismatch")
        keys["source_row_int"] = pd.to_numeric(keys["source_row"], errors="raise").astype(np.int64)
        keys["pert_idx_int"] = pd.to_numeric(keys["pert_idx"], errors="raise").astype(np.int64)
        if keys["source_row_int"].duplicated().any():
            raise ValueError("frozen source rows are not unique")

        h5ad_path = XPERT / "processed_data/l1000_sdst_78453.h5ad"
        h5ad_receipt_sha = "953110b4e25491aa8985a38f6d90e801ddc1dc32bbdfd71de5623c7bc4ade153"
        if metadata["source_h5ad_sha256"] != h5ad_receipt_sha or metadata_root["source_h5ad_sha256"] != h5ad_receipt_sha:
            raise ValueError("EXP-062 source H5AD provenance is inconsistent")

        # Backed read accesses H5AD obs/var only. No X/X_ctl/layers property is touched.
        h5 = ad.read_h5ad(h5ad_path, backed="r")
        try:
            checks["source_h5ad_shape"] = list(h5.shape) == [78453, 978]
            obs = h5.obs
            var = h5.var
            required = ["sample_id", "pert_id", "pert_idx", "cell_iname", "cell_idx", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "project_code", "build_name"]
            missing = [col for col in required if col not in obs.columns]
            if missing:
                raise ValueError(f"H5AD obs metadata missing columns: {missing}")
            rows = keys["source_row_int"].to_numpy()
            selected_obs = obs.iloc[rows]
            comparisons = {
                "obs_name": np.asarray(h5.obs_names[rows]).astype(str),
                "sample_id": selected_obs["sample_id"].astype(str).to_numpy(),
                "pert_id": selected_obs["pert_id"].astype(str).to_numpy(),
                "pert_idx": pd.to_numeric(selected_obs["pert_idx"], errors="raise").to_numpy(dtype=np.int64).astype(str),
                "source_object_id": selected_obs["cell_iname"].astype(str).to_numpy(),
                "pert_dose": selected_obs["pert_dose"].astype(str).to_numpy(),
                "pert_dose_unit": selected_obs["pert_dose_unit"].astype(str).to_numpy(),
                "pert_time": selected_obs["pert_time"].astype(str).to_numpy(),
                "pert_time_unit": selected_obs["pert_time_unit"].astype(str).to_numpy(),
                "project_code": selected_obs["project_code"].astype(str).to_numpy(),
                "build_name": selected_obs["build_name"].astype(str).to_numpy(),
                "source_row": keys["source_row_int"].astype(str).to_numpy(),
                "effective_split_role": keys["effective_split_role"].to_numpy(dtype=str),
            }
            for col, actual in comparisons.items():
                expected_col = col
                if col == "obs_name":
                    expected = keys["obs_name"].to_numpy(dtype=str)
                elif col == "pert_idx":
                    expected = keys["pert_idx_int"].astype(str).to_numpy()
                elif col == "source_row":
                    expected = keys["source_row_int"].astype(str).to_numpy()
                elif col == "source_object_id":
                    expected = keys["source_object_id"].to_numpy(dtype=str)
                elif col == "effective_split_role":
                    expected = keys["effective_split_role"].to_numpy(dtype=str)
                else:
                    expected = keys[expected_col].to_numpy(dtype=str)
                checks[f"selected_h5ad_obs_{col}_exact"] = bool(np.array_equal(actual, expected))

            mapping_path = EXP051_RUN / "gene_axis_mapping.tsv"
            mapping = pd.read_csv(mapping_path, sep="\t", dtype=str, keep_default_na=False)
            checks["gene_mapping_hash_matches_EXP062"] = sha256_file(mapping_path) == metadata["source_asset_sha256"]["exp051_gene_axis_mapping"]
            checks["h5ad_var_axis_matches_EXP051"] = bool(np.array_equal(np.asarray(h5.var_names).astype(str), mapping["h5ad_var_name"].to_numpy(dtype=str)))
            if "gene_id" not in var.columns:
                raise ValueError("H5AD var metadata lacks gene_id axis field")
            checks["h5ad_gene_id_axis_matches_EXP051"] = bool(np.array_equal(var["gene_id"].astype(str).to_numpy(), mapping["h5ad_var_annotation"].to_numpy(dtype=str)))
            checks["gene_axis_length_978"] = len(mapping) == 978 == h5.shape[1]
            if not all(value is True for key, value in checks.items() if key.startswith("selected_h5ad_obs_") or key in {"source_h5ad_shape", "gene_mapping_hash_matches_EXP062", "h5ad_var_axis_matches_EXP051", "h5ad_gene_id_axis_matches_EXP051", "gene_axis_length_978"}):
                raise ValueError(f"selected row or gene axis check failed: {checks}")
        finally:
            h5.file.close()

        drug_table_path = XPERT / "processed_data/l1000_sdst_drugs_info_8432.csv"
        drug_table = pd.read_csv(drug_table_path, usecols=["pert_id", "inchi_key", "drug_node_idx"], dtype=str, keep_default_na=False)
        selected_pairs = keys[["pert_id", "full_inchikey", "pert_idx_int"]].drop_duplicates()
        drug_table["pert_id"] = drug_table["pert_id"].str.strip()
        drug_table["inchi_key"] = drug_table["inchi_key"].str.strip().str.upper()
        drug_table["drug_node_idx_num"] = pd.to_numeric(drug_table["drug_node_idx"], errors="coerce")
        alias_audit = []
        key_to_node: dict[str, int] = {}
        for full_key, group in selected_pairs.groupby("full_inchikey", sort=True):
            aliases = drug_table.loc[drug_table["inchi_key"].eq(full_key)].copy()
            aliases = aliases.loc[aliases["drug_node_idx_num"].notna()]
            nodes = sorted(aliases["drug_node_idx_num"].astype(int).unique().tolist())
            mapped_aliases = []
            for row in group.itertuples(index=False):
                match = drug_table.loc[(drug_table["pert_id"] == row.pert_id) & (drug_table["inchi_key"] == full_key)]
                match_nodes = sorted(match["drug_node_idx_num"].dropna().astype(int).unique().tolist())
                expected_node = int(row.pert_idx_int)
                if match_nodes != [expected_node]:
                    raise ValueError(f"selected pert_id mapping mismatch: {row.pert_id}/{full_key}/{expected_node}/{match_nodes}")
                mapped_aliases.append({"pert_id": row.pert_id, "pert_idx": expected_node})
            if len(nodes) != 1 or not mapped_aliases:
                raise ValueError(f"full InChIKey does not bind to exactly one XPert node: {full_key}/{nodes}")
            if any(int(item["pert_idx"]) != nodes[0] for item in mapped_aliases):
                raise ValueError(f"selected aliases do not bind to one node for {full_key}")
            key_to_node[str(full_key)] = nodes[0]
            alias_audit.append({"full_inchikey": str(full_key), "node": nodes[0], "author_alias_count": int(aliases["pert_id"].nunique()), "selected_pert_ids": mapped_aliases})
        checks["drug_identity_nodes_exact_20"] = len(key_to_node) == 20 and len(set(key_to_node.values())) == 20
        if not checks["drug_identity_nodes_exact_20"]:
            raise ValueError("not all 20 full keys map one-to-one onto 20 XPert nodes")

        asset_paths = {
            "unimol": XPERT / "processed_data/all_drugs_unimol_arr.npy",
            "hg_embeddings": XPERT / "HG_data/saved_embedding/HG_drug_embeddings.npy",
            "ppi_gene_vectors": XPERT / "processed_data/PPI_gene_vector_128d.npy",
        }
        asset_headers = {name: npy_header(path) for name, path in asset_paths.items()}
        node_ids = sorted(key_to_node.values())
        um = asset_headers["unimol"]["shape"]
        hg = asset_headers["hg_embeddings"]["shape"]
        ppi = asset_headers["ppi_gene_vectors"]["shape"]
        checks["unimol_node_axis_contains_all_selected"] = len(um) == 3 and um[1:] == [122, 514] and max(node_ids) < um[0]
        checks["hg_node_axis_contains_all_selected"] = len(hg) >= 2 and max(node_ids) < hg[0]
        checks["ppi_gene_axis_is_978"] = len(ppi) == 2 and ppi[0] == 978
        if not all(checks[k] for k in ("unimol_node_axis_contains_all_selected", "hg_node_axis_contains_all_selected", "ppi_gene_axis_is_978")):
            raise ValueError(f"native XPert feature-axis check failed: {asset_headers}")

        official_code_paths = {
            "xpert_model": XPERT / "models/model_XPert.py",
            "xpert_train": XPERT / "train_xpert.py",
            "xpert_dataset": XPERT / "datasets/MyDataset.py",
            "xpert_utils": XPERT / "utils.py",
            "xpert_config": XPERT / "configs/config_l1000.yaml",
            "xpert_drug_table": drug_table_path,
            "exp051_gene_axis_mapping": EXP051_RUN / "gene_axis_mapping.tsv",
        }
        for name, path in official_code_paths.items():
            hashes[name] = sha256_file(path)
        for name, path in asset_paths.items():
            hashes[f"feature_asset:{name}"] = sha256_file(path)

        upstream_cfg = yaml.safe_load((XPERT / "configs/config_l1000.yaml").read_text(encoding="utf-8"))
        checks["upstream_schedule_values_match_contract"] = (
            int(upstream_cfg["train"]["batch_size"]) == 128
            and int(upstream_cfg["train"]["num_epochs"]) == 2500
            and int(upstream_cfg["train"]["init_epoch"]) == 70
            and int(upstream_cfg["train"]["patience"]) == 50
            and abs(float(upstream_cfg["train"]["train_lr"]) - 0.004) < 1e-12
            and abs(float(upstream_cfg["train"]["weight_decay"]) - 1e-5) < 1e-12
            and list(map(float, upstream_cfg["train"]["loss_weight"])) == [0.2, 0.003, 0.2, 1.0]
        )
        # Some XPert config versions put AdamW and loss weights under differently named sections;
        # those keys are verified by the fitter before it starts updating weights.
        if not checks["upstream_schedule_values_match_contract"]:
            raise ValueError("official XPert schedule changed from the frozen EXP-063 contract")
        hashes["exp063_config"] = sha256_file(config_path)
        hashes["exp063_gate_script"] = sha256_file(Path(__file__).resolve())
        runner_path = REPO / "scripts/modeling/run_exp063_xpert_coverage20.py"
        adapter_path = REPO / "scripts/modeling/exp051_contract_fit_smoke.py"
        hashes["exp063_runner_script"] = sha256_file(runner_path)
        hashes["exp051_adapter_script"] = sha256_file(adapter_path)

        elapsed = time.perf_counter() - started
        receipt = {
            "experiment_id": "EXP-063",
            "gate_id": args.gate_id,
            "status": "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE",
            "checks": checks,
            "domain": {"rows": 1982, "full_inchikeys": 20, "genes": 978, "train_conditions": 1291, "train_objects": 93, "inner_dev_conditions": 691, "inner_dev_objects": 48, "object_overlap": 0},
            "drug_node_mapping": alias_audit,
            "feature_asset_headers_only": asset_headers,
            "asset_sha256": hashes,
            "predecessor_gates": [
                {"gate_id": "metadata_axis_v1", "status": "FAILED_CLOSED_RESPONSE_BLIND_XPERT_AXIS_GATE",
                 "elapsed_cpu_wall_seconds": 5.0},
                {"gate_id": "metadata_axis_v2", "status": "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE",
                 "elapsed_cpu_wall_seconds": 37.14551124710124},
                {"gate_id": "runner_syntax_validation", "status": "PASS_NO_EXECUTION_OR_MODEL_UPDATES",
                 "elapsed_cpu_wall_seconds": 4.6},
                {"gate_id": "metadata_axis_v3", "status": "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE",
                 "elapsed_cpu_wall_seconds": 37.26813974301331},
                {"gate_id": "runner_syntax_validation_after_cost_review", "status": "PASS_NO_EXECUTION_OR_MODEL_UPDATES",
                 "elapsed_cpu_wall_seconds": 4.6},
                {"gate_id": "metadata_axis_v4", "status": "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE",
                 "elapsed_cpu_wall_seconds": 36.82332340802532},
                {"gate_id": "fit_attempt_001_preflight_hash_binding_failure", "status": "FAILED_CLOSED_BEFORE_NUMERIC_ACCESS_OR_GPU_UPDATE",
                 "elapsed_cpu_seconds_conservatively_charged": 15.0},
                {"gate_id": "runner_syntax_validation_after_attempt_001", "status": "PASS_NO_EXECUTION_OR_MODEL_UPDATES",
                 "elapsed_cpu_wall_seconds": 4.6},
                {"gate_id": "metadata_axis_v5", "status": "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE",
                 "elapsed_cpu_wall_seconds": 36.656473799957894},
                {"gate_id": "fit_attempt_002_baseline_field_mapping_failure", "status": "FAILED_CLOSED_AFTER_ALLOWED_SELECTED_ROW_ACCESS_BEFORE_GPU_UPDATE",
                 "elapsed_cpu_seconds_conservatively_charged": 50.0},
                {"gate_id": "post_failure_schema_probes", "status": "COMPLETED_HEADER_AND_KEYED_ARRAY_CHECKS_ONLY",
                 "elapsed_cpu_seconds_conservatively_charged": 13.0},
                {"gate_id": "runner_syntax_validation_after_attempt_002", "status": "PASS_NO_EXECUTION_OR_MODEL_UPDATES",
                 "elapsed_cpu_wall_seconds": 4.6},
            ],
            "data_access": {"h5ad_obs_var_metadata_only": True, "h5ad_X_values_read": False, "h5ad_X_ctl_values_read": False, "h5ad_layers_values_read": False, "exp062_prediction_values_read": False, "pdo_outer_locked_panacea_cdsdb_numeric_values_read": False},
            "resources": {"elapsed_cpu_wall_seconds": elapsed, "prior_package_seconds_conservatively_charged": PRIOR_CPU_SECONDS, "cumulative_cpu_seconds": elapsed + PRIOR_CPU_SECONDS, "cpu_hard_cap_seconds": 900, "gpu_hours": 0, "model_updates": 0},
            "provenance": {"exp062_metadata_manifest_sha256": actual_metadata_sha, "exp062_root_metadata_readback_status": metadata_root["status"], "exp062_root_numeric_readback_status": numeric_root["status"], "source_h5ad_sha256_from_exp062_frozen_chain": h5ad_receipt_sha},
            "platform": {"python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
        }
        if receipt["resources"]["cumulative_cpu_seconds"] > 900:
            receipt["status"] = "FAIL_CLOSED_CPU_CAP_EXCEEDED"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "elapsed": elapsed, "cumulative_cpu": receipt["resources"]["cumulative_cpu_seconds"], "checks": checks}, indent=2))
        return 0 if receipt["status"] == "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE" else 2
    except Exception as exc:
        elapsed = time.perf_counter() - started
        failure = {
            "experiment_id": "EXP-063", "gate_id": args.gate_id,
            "status": "FAILED_CLOSED_RESPONSE_BLIND_XPERT_AXIS_GATE",
            "error_type": type(exc).__name__, "error": str(exc), "checks_observed": checks,
            "asset_sha256": hashes,
            "data_access": {"h5ad_X_values_read": False, "h5ad_X_ctl_values_read": False, "h5ad_layers_values_read": False, "exp062_prediction_values_read": False},
            "resources": {"elapsed_cpu_wall_seconds": elapsed, "prior_package_seconds_conservatively_charged": PRIOR_CPU_SECONDS, "cumulative_cpu_seconds": elapsed + PRIOR_CPU_SECONDS, "cpu_hard_cap_seconds": 900, "gpu_hours": 0, "model_updates": 0},
        }
        receipt_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
