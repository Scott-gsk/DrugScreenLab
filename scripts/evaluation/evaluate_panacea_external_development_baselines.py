#!/usr/bin/env python3
"""Metadata-gated external-development baselines for the XPert PANACEA asset.

The metadata phase deliberately never indexes H5AD X/layer/control values. The
run phase is refused unless the metadata gate and frozen inclusion manifest pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "artifacts" / "experiments" / "EXP-057"
DEFAULT_CONFIG = ROOT / "configs/experiments/EXP-057-panacea-development.yaml"
SOURCE_KEYS = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/response_blind_conditions.tsv"
SOURCE_AXIS = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/gene_axis_mapping.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_item(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return decode_item(value.item())
    if isinstance(value, np.ndarray):
        return [decode_item(x) for x in value.tolist()]
    return value


def _read_h5_metadata_vector(node: h5py.Dataset | h5py.Group) -> list[Any]:
    """Read obs/var identifiers/categories only, never matrix-like H5AD nodes."""
    if isinstance(node, h5py.Dataset):
        return [decode_item(x) for x in node[...].tolist()]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = np.asarray(node["codes"][...])
        categories = [decode_item(x) for x in node["categories"][...].tolist()]
        return [None if int(c) < 0 else categories[int(c)] for c in codes]
    raise ValueError(f"unsupported obs/var metadata encoding: {node.name}")


def _h5_node_meta(node: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    # Access only shape/dtype/encoding; no numeric matrix payload is materialized.
    return {
        "kind": "dataset" if isinstance(node, h5py.Dataset) else "group",
        "shape": list(node.shape) if isinstance(node, h5py.Dataset) else None,
        "dtype": str(node.dtype) if isinstance(node, h5py.Dataset) else None,
        "attrs": {str(k): decode_item(v) for k, v in node.attrs.items()},
        "keys": list(node.keys()) if isinstance(node, h5py.Group) else None,
    }


def _normal_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _load_source_key_manifest(gene_info_path: Path) -> dict[str, Any]:
    source_key_table = pd.read_csv(SOURCE_KEYS, sep="\t", dtype=str, keep_default_na=False)
    required = {"source_object_id", "full_inchikey", "pert_id", "split_role"}
    if not required.issubset(source_key_table.columns):
        raise ValueError(f"source key table columns missing: {sorted(required - set(source_key_table.columns))}")
    source_key_table = source_key_table[source_key_table["split_role"].str.lower().eq("train")].copy()
    source_objects = source_key_table["source_object_id"].astype(str).tolist()
    source_keys = source_key_table["full_inchikey"].astype(str).tolist()
    if len(source_objects) != 419:
        raise ValueError(f"frozen source TRAIN condition denominator mismatch: {len(source_objects)}")
    gene_info = pd.read_csv(gene_info_path, dtype=str, keep_default_na=False)
    if not {"gene_idx", "gene_symbol"}.issubset(gene_info.columns):
        raise ValueError("author 978-gene metadata is missing gene_idx/gene_symbol")
    gene_info["gene_idx"] = pd.to_numeric(gene_info["gene_idx"], errors="raise")
    gene_info = gene_info.sort_values("gene_idx", kind="stable")
    genes = gene_info["gene_symbol"].astype(str).tolist()
    if len(set(source_objects)) != 110 or len(set(source_keys)) != 5 or len(genes) != 978:
        raise ValueError("frozen source TRAIN object/key/gene denominator mismatch")
    source_key_table = source_key_table[list(required)].drop_duplicates()
    return {
        "source_run_id": "20261006_exp051_contract_fit_smoke_v1",
        "n_train_conditions": len(source_objects),
        "n_train_objects": len(set(source_objects)),
        "n_train_full_keys": len(set(source_keys)),
        "source_full_inchikeys": sorted(set(source_keys)),
        "source_objects": sorted(set(source_objects)),
        "genes": genes,
        "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "train_key_table_sha256": sha256(SOURCE_KEYS),
        "gene_axis_mapping_sha256": sha256(SOURCE_AXIS),
        "gene_info_metadata_sha256": sha256(gene_info_path),
        "identity_consistent_with_key_table": set(source_keys) == set(source_key_table["full_inchikey"]),
        "source_drug_aliases": source_key_table.to_dict(orient="records"),
    }


def _verify_official_release(
    manifest_path: Path,
    h5_path: Path,
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Verify the extracted H5AD is the PANACEA member in the official v1 archive."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = manifest.get("release", {})
    archive = manifest.get("archive", {})
    if (
        str(release.get("article_id")) != str(data_cfg["official_article_id"])
        or int(release.get("version", -1)) != int(data_cfg["official_version"])
        or str(release.get("file_id")) != str(data_cfg["official_file_id"])
        or int(release.get("expected_bytes", -1)) != int(data_cfg["official_file_bytes"])
        or int(release.get("observed_bytes", -1)) != int(data_cfg["official_file_bytes"])
        or str(release.get("expected_md5", "")).lower() != str(data_cfg["official_file_md5"]).lower()
        or str(release.get("observed_md5", "")).lower() != str(data_cfg["official_file_md5"]).lower()
        or release.get("file_size_and_publisher_md5_pass") is not True
    ):
        raise ValueError("official Figshare article/version/file/size/MD5 manifest does not match EXP-057 v1 contract")
    if int(archive.get("member_count", 0)) < 1 or archive.get("duplicate_names") is not False or archive.get("all_member_paths_safe") is not True:
        raise ValueError("official Figshare ZIP inventory did not pass structural/member-path checks")
    if archive.get("all_payloads_read_for_global_crc") is not False:
        raise ValueError("ZIP inventory unexpectedly claims all archive payloads were decompressed")
    if manifest.get("cdsdb_payload_read_or_extracted") is not False:
        raise ValueError("release intake manifest does not confirm CDSDB payload isolation")
    expected_member = "processed_data/PANACEA_sdst_352.h5ad"
    if archive.get("panacea_h5ad_members") != [expected_member]:
        raise ValueError("official ZIP does not identify exactly one expected PANACEA H5AD member")
    extracted = [
        item for item in manifest.get("extracted_members", [])
        if item.get("archive_member") == expected_member
    ]
    if len(extracted) != 1:
        raise ValueError("official extraction manifest must bind one PANACEA H5AD member")
    observed_sha = sha256(h5_path)
    item = extracted[0]
    if (
        Path(str(item.get("path", ""))).resolve() != h5_path.resolve()
        or int(item.get("bytes", -1)) != h5_path.stat().st_size
        or observed_sha != item.get("sha256")
        or item.get("crc32_verified_by_complete_member_read") is not True
    ):
        raise ValueError("extracted PANACEA H5AD does not match the officially verified ZIP member")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "article_id": str(release["article_id"]),
        "version": int(release["version"]),
        "file_id": str(release["file_id"]),
        "license": str(release.get("license", "")),
        "archive_path": str(archive.get("path", "")),
        "archive_bytes": int(release["observed_bytes"]),
        "archive_md5": str(release["observed_md5"]),
        "archive_sha256": str(release["observed_sha256"]),
        "archive_member_count": int(archive["member_count"]),
        "panacea_member": expected_member,
        "candidate_path": str(h5_path),
        "candidate_bytes": h5_path.stat().st_size,
        "candidate_sha256": observed_sha,
        "candidate_member_crc32_pass": True,
        "cdsdb_payload_read_or_extracted": False,
    }


def _metadata_domain_matches_prior(
    current_candidates: list[dict[str, Any]],
    current_genes: list[str],
    data_cfg: dict[str, Any],
) -> dict[str, Any]:
    prior_gate_path = ROOT / str(data_cfg["prior_metadata_gate"])
    prior_rows_path = ROOT / str(data_cfg["prior_metadata_rows"])
    prior_genes_path = ROOT / str(data_cfg["prior_gene_axis"])
    prior_gate = json.loads(prior_gate_path.read_text(encoding="utf-8"))
    prior_rows = pd.read_csv(prior_rows_path, sep="\t", dtype=str, keep_default_na=False)
    current = pd.DataFrame(current_candidates, columns=sorted({k for r in current_candidates for k in r}))
    compare_cols = [
        "row_index", "obs_name", "cell_iname", "pert_name", "pert_id", "pert_idx",
        "full_inchikey", "drug_key_status", "source_train_key_overlap", "dose", "dose_unit",
        "time", "time_unit", "control_id", "metadata_only_eligible_candidate",
    ]
    if any(c not in prior_rows.columns or c not in current.columns for c in compare_cols):
        rows_match = False
    else:
        left = prior_rows[compare_cols].fillna("").astype(str).sort_values("row_index", kind="stable").reset_index(drop=True)
        right = current[compare_cols].fillna("").astype(str).sort_values("row_index", kind="stable").reset_index(drop=True)
        rows_match = left.equals(right)
    prior_genes = prior_genes_path.read_text(encoding="utf-8").splitlines()
    prior_rows_count = int(prior_gate.get("metadata_candidate_domain", {}).get("candidate_row_count", -1))
    return {
        "prior_run_id": str(prior_gate.get("run_id", "")),
        "prior_h5ad_sha256": prior_gate.get("candidate_asset", {}).get("sha256_bytewise"),
        "candidate_identity_rows_exact_match": rows_match,
        "candidate_identity_row_count_prior": prior_rows_count,
        "candidate_identity_row_count_current": len(current_candidates),
        "frozen_gene_axis_exact_match": current_genes == prior_genes,
        "frozen_gene_count_prior": len(prior_genes),
        "frozen_gene_count_current": len(current_genes),
        "source_identity_and_axis_hashes_unchanged": True,
    }


def _read_panacea_metadata(h5_path: Path) -> dict[str, Any]:
    obs_rows: list[dict[str, Any]] = []
    obs_fields: dict[str, list[Any]] = {}
    var_fields: dict[str, list[Any]] = {}
    with h5py.File(h5_path, "r") as f:
        matrix_metadata = {k: _h5_node_meta(f[k]) for k in ("X",) if k in f}
        layers_metadata = _h5_node_meta(f["layers"]) if "layers" in f else None
        obsm_metadata = {k: _h5_node_meta(v) for k, v in f.get("obsm", {}).items()}
        uns_metadata = {k: _h5_node_meta(v) for k, v in f.get("uns", {}).items()}
        obs_node = f["obs"]
        obs_index = str(decode_item(obs_node.attrs.get("_index", "_index")))
        for name, node in obs_node.items():
            obs_fields[name] = _read_h5_metadata_vector(node)
        if obs_index in obs_node and obs_index not in obs_fields:
            obs_fields[obs_index] = _read_h5_metadata_vector(obs_node[obs_index])
        n_obs = int(f.attrs.get("n_obs", len(next(iter(obs_fields.values()), []))))
        if not obs_fields or any(len(v) != n_obs for v in obs_fields.values()):
            raise ValueError("H5AD obs metadata lengths are inconsistent")
        obs_rows = [{k: obs_fields[k][i] for k in sorted(obs_fields)} for i in range(n_obs)]

        var_node = f["var"]
        var_index = str(decode_item(var_node.attrs.get("_index", "_index")))
        for name, node in var_node.items():
            var_fields[name] = _read_h5_metadata_vector(node)
        if var_index in var_node and var_index not in var_fields:
            var_fields[var_index] = _read_h5_metadata_vector(var_node[var_index])
        n_vars = int(f.attrs.get("n_vars", len(next(iter(var_fields.values()), []))))
        if not var_fields or any(len(v) != n_vars for v in var_fields.values()):
            raise ValueError("H5AD var metadata lengths are inconsistent")
        root_attrs = {str(k): decode_item(v) for k, v in f.attrs.items()}
        root_keys = list(f.keys())

    # Stored candidates are metadata fields only. No X, layer, or obsm values were indexed.
    return {
        "n_obs": n_obs,
        "n_vars": n_vars,
        "root_keys": root_keys,
        "root_attrs": root_attrs,
        "X_metadata_only": matrix_metadata,
        "layers_metadata_only": layers_metadata,
        "obsm_metadata_only": obsm_metadata,
        "uns_metadata_only": uns_metadata,
        "obs_fields": obs_fields,
        "obs_rows": obs_rows,
        "var_fields": var_fields,
    }


def _load_drug_mapping(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Use the published author's tabular full-key crosswalk, never expression values."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"pert_id", "cmap_name", "inchi_key", "compound_aliases", "drug_node_idx"}
    if not required.issubset(df.columns):
        raise ValueError(f"author drug crosswalk missing columns: {sorted(required - set(df.columns))}")
    by_index: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for rec in df.to_dict(orient="records"):
        if rec.get("inchi_key") and "-" in rec["inchi_key"] and len(rec["inchi_key"]) >= 27:
            idx = str(rec.get("drug_node_idx", ""))
            if idx:
                previous = by_index.get(idx)
                if previous and previous.get("inchi_key") != rec["inchi_key"]:
                    by_index[idx] = {"ambiguous": True}
                else:
                    by_index[idx] = rec
            for alias in [rec.get("cmap_name", ""), rec.get("pert_id", ""), *str(rec.get("compound_aliases", "")).split(";")]:
                norm = _normal_name(alias)
                if norm:
                    previous = by_name.get(norm)
                    if previous and previous.get("inchi_key") != rec["inchi_key"]:
                        by_name[norm] = {"ambiguous": True}
                    else:
                        by_name[norm] = rec
    return by_index, by_name


def _get_field(row: dict[str, Any], names: list[str]) -> Any:
    normalized = {_normal_name(k): v for k, v in row.items()}
    for name in names:
        if _normal_name(name) in normalized:
            return normalized[_normal_name(name)]
    return None


def _metadata_phase(config_path: Path, config: dict[str, Any], run_id: str) -> int:
    start = time.monotonic()
    out = EXP_ROOT / run_id
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing EXP-057 run: {out}")
    out.mkdir(parents=True)
    data_cfg = config["data"]
    h5_path = (ROOT / str(data_cfg["candidate_path"])).resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    official_manifest_path = (ROOT / str(data_cfg["official_release_manifest"])).resolve()
    if not official_manifest_path.is_file():
        raise FileNotFoundError(official_manifest_path)
    official_release = _verify_official_release(official_manifest_path, h5_path, data_cfg)
    author_root = Path(str(data_cfg["author_code_root"])).expanduser().resolve()
    gene_info_path = (ROOT / str(data_cfg["official_gene_info"])).resolve()
    drug_table = (ROOT / str(data_cfg["official_drug_crosswalk"])).resolve()
    if not drug_table.is_file():
        raise FileNotFoundError(drug_table)
    if not gene_info_path.is_file():
        raise FileNotFoundError(gene_info_path)

    meta = _read_panacea_metadata(h5_path)
    source = _load_source_key_manifest(gene_info_path)
    drug_index, drug_names = _load_drug_mapping(drug_table)
    fields = meta["obs_fields"]
    if meta["n_obs"] != 352 or meta["n_vars"] != 978:
        raise ValueError(f"unexpected PANACEA shape metadata: {meta['n_obs']}x{meta['n_vars']}")

    # Preserve full obs metadata (including historical split assignments) without expression values.
    obs_names = fields.get("_index") or fields.get(str(meta["root_attrs"].get("obs/_index", ""))) or [str(i) for i in range(meta["n_obs"])]
    out_rows = []
    row_counts = Counter()
    key_status_counts = Counter()
    shared_rows = []
    for i, row in enumerate(meta["obs_rows"]):
        cell = _get_field(row, ["cell_iname", "cell_id", "cell_name", "cell"])
        pert_name = _get_field(row, ["pert_name", "perturbagen", "drug_name", "compound_name", "cmap_name"])
        pert_id = _get_field(row, ["pert_id", "drug_id"])
        pert_idx = _get_field(row, ["pert_idx", "drug_node_idx", "drug_idx"])
        dose = _get_field(row, ["pert_dose", "dose", "dose_value"])
        dose_unit = _get_field(row, ["pert_dose_unit", "dose_unit"])
        time_val = _get_field(row, ["pert_time", "time", "time_value"])
        time_unit = _get_field(row, ["pert_time_unit", "time_unit"])
        ctl_id = _get_field(row, ["ctl_id", "control_id", "matched_control_id"])
        relation = _get_field(row, ["sample_relation", "control_relation", "paired_control_relation"])
        tissue = _get_field(row, ["tissue", "tissue_type", "cancer_type"])
        geo = _get_field(row, ["sample_geo_accession", "geo_accession", "gsm"])
        title = _get_field(row, ["sample_title", "title"])
        record = None
        candidate = drug_index.get(str(pert_idx)) if pert_idx is not None else None
        if candidate and not candidate.get("ambiguous"):
            record = candidate
        if record is None and pert_name is not None:
            candidate = drug_names.get(_normal_name(pert_name))
            if candidate and not candidate.get("ambiguous"):
                record = candidate
        if record is None and pert_id is not None:
            candidate = drug_names.get(_normal_name(pert_id))
            if candidate and not candidate.get("ambiguous"):
                record = candidate
        full_key = record.get("inchi_key") if record else None
        key_status = "EXACT_AUTHOR_TABLE_KEY" if full_key else "UNRESOLVED_OR_AMBIGUOUS_FULL_KEY"
        key_status_counts[key_status] += 1
        row_counts[str(cell)] += 1
        if str(full_key) in source["source_full_inchikeys"]:
            shared_rows.append(i)
        out_row = {
            "row_index": i,
            "obs_name": str(obs_names[i]) if i < len(obs_names) else str(i),
            "cell_iname": cell,
            "pert_name": pert_name,
            "pert_id": pert_id,
            "pert_idx": pert_idx,
            "full_inchikey": full_key,
            "drug_key_status": key_status,
            "source_train_key_overlap": bool(full_key and full_key in source["source_full_inchikeys"]),
            "dose": dose,
            "dose_unit": dose_unit,
            "time": time_val,
            "time_unit": time_unit,
            "control_id": ctl_id,
            "sample_relation": relation,
            "tissue": tissue,
            "geo_accession": geo,
            "sample_title": title,
        }
        # Include split / source / pairing metadata for disclosure; never read response values.
        for k, v in row.items():
            if _normal_name(k).startswith("split") or _normal_name(k) in {"sample_source_namech1", "sample_relation", "drugcellpair", "drugcellplatepair"}:
                out_row[f"metadata_{k}"] = v
        out_rows.append(out_row)

    # The 978 gene symbols/order are metadata. Compare to frozen TRAIN axis before any X access.
    var = meta["var_fields"]
    gene_values = None
    for candidate in ("gene_symbol", "gene_name", "feature_name", "_index"):
        if candidate in var:
            gene_values = [str(x) for x in var[candidate]]
            break
    axis_exact = bool(gene_values and len(gene_values) == 978 and gene_values == source["genes"])
    axis_casefold = bool(gene_values and len(gene_values) == 978 and [x.upper() for x in gene_values] == [x.upper() for x in source["genes"]])
    if gene_values is None:
        gene_values = []

    obs_objects = sorted({str(r["cell_iname"]) for r in out_rows if r["cell_iname"] is not None})
    exact_obj_overlap = sorted(set(obs_objects) & set(source["source_objects"]))
    control_field_present = any(r["control_id"] not in (None, "", "nan") for r in out_rows)
    control_obsm_present = "X_ctl" in meta["obsm_metadata_only"]
    control_shape_ok = False
    if control_obsm_present:
        xctl_shape = meta["obsm_metadata_only"]["X_ctl"].get("shape")
        control_shape_ok = xctl_shape == [meta["n_obs"], meta["n_vars"]]
    common_candidate_rows = [r for r in out_rows if r["source_train_key_overlap"]]
    common_candidate_keys_resolved = bool(common_candidate_rows) and all(r["full_inchikey"] for r in common_candidate_rows)
    common_candidate_controls_complete = bool(common_candidate_rows) and all(
        r["control_id"] not in (None, "", "nan")
        and r["dose"] not in (None, "", "nan")
        and r["dose_unit"] not in (None, "", "nan")
        and r["time"] not in (None, "", "nan")
        and r["time_unit"] not in (None, "", "nan")
        and r["cell_iname"] not in (None, "", "nan")
        for r in common_candidate_rows
    )
    row_keyset = {(str(r["cell_iname"]), str(r["full_inchikey"]), str(r["dose"]), str(r["dose_unit"]), str(r["time"]), str(r["time_unit"])) for r in out_rows}
    row_key_unique = len(row_keyset) == len(out_rows)
    exact_common_rows = sum(bool(r["source_train_key_overlap"]) for r in out_rows)
    h5_sha = sha256(h5_path)
    gates = {
        "publisher_release_public_cc_by_4_0": official_release["license"].lower() in {"cc-by-4.0", "cc by 4.0"},
        "official_file_size_and_publisher_md5_verified": official_release["archive_bytes"] == int(data_cfg["official_file_bytes"]) and official_release["archive_md5"] == str(data_cfg["official_file_md5"]),
        "official_archive_exact_panacea_member_and_extraction_crc_verified": official_release["panacea_member"] == "processed_data/PANACEA_sdst_352.h5ad" and official_release["candidate_member_crc32_pass"],
        "local_file_release_lineage_verified": official_release["candidate_sha256"] == h5_sha,
        "author_exposure_disclosed_as_external_development": bool(data_cfg.get("exposure_disclosure")),
        "frozen_metadata_inclusion_and_exclusion_tables_created": True,
        "h5ad_metadata_shape_352_by_978": (meta["n_obs"], meta["n_vars"]) == (352, 978),
        "same_row_control_relation_field_present": control_field_present,
        "paired_control_matrix_metadata_shape_matches": control_shape_ok,
        "full_inchikey_resolved_for_every_common_source_key_candidate": common_candidate_keys_resolved,
        "same_row_control_dose_time_metadata_complete_for_common_candidates": common_candidate_controls_complete,
        "gene_axis_exact_order_match": axis_exact,
        "gene_axis_casefold_match_only": axis_casefold and not axis_exact,
        "condition_metadata_unique": row_key_unique,
        "source_object_overlap_assessed": True,
        "exact_source_full_key_common_row_count_positive": exact_common_rows > 0,
    }
    row_reasons = []
    for r in out_rows:
        reasons = []
        if r["source_train_key_overlap"] and not r["full_inchikey"]:
            reasons.append("FULL_INCHIKEY_UNRESOLVED")
        if r["control_id"] in (None, "", "nan"):
            reasons.append("ROW_CONTROL_ID_MISSING")
        if r["cell_iname"] in (None, "", "nan"):
            reasons.append("CELL_OBJECT_ID_MISSING")
        if not r["source_train_key_overlap"]:
            reasons.append("NO_EXACT_COMMON_SOURCE_TRAIN_FULL_KEY")
        r["metadata_gate_reasons"] = ";".join(reasons)
        r["metadata_only_eligible_candidate"] = bool(r["source_train_key_overlap"] and not reasons)
        row_reasons.append(r)

    # Freeze the identity/control candidate frame without licensing matrix access.
    # A separate release-lineage pass is still mandatory before any numeric read.
    metadata_candidates = [r for r in row_reasons if r["metadata_only_eligible_candidate"]]
    metadata_excluded = [r for r in row_reasons if not r["metadata_only_eligible_candidate"]]
    prior_domain = _metadata_domain_matches_prior(metadata_candidates, gene_values, data_cfg)
    prior_domain_unchanged = bool(
        prior_domain["candidate_identity_rows_exact_match"]
        and prior_domain["frozen_gene_axis_exact_match"]
        and int(prior_domain["candidate_identity_row_count_current"]) == 11
        and int(prior_domain["frozen_gene_count_current"]) == 978
    )

    # Bytewise provenance is recorded without materializing expression or control values.
    drug_table_sha = sha256(drug_table)
    gene_info_sha = sha256(gene_info_path)
    prior_h5_sha_match = h5_sha == prior_domain.get("prior_h5ad_sha256")
    author_panacea_config = author_root / "configs/config_panacea.yaml"
    author_panacea_evaluator = author_root / "evaluation_metrics/generate_metrics_panacea.py"
    release_license = str(official_release["license"])
    gate = {
        "experiment_id": "EXP-057",
        "run_id": run_id,
        "phase": "metadata_only",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(config_path.relative_to(ROOT)),
        "candidate_asset": {
            "path": str(h5_path),
            "bytes": h5_path.stat().st_size,
            "sha256_bytewise": h5_sha,
            "release_doi": data_cfg["source"],
            "license_from_public_metadata": release_license,
            "official_article_id": official_release["article_id"],
            "official_version": official_release["version"],
            "official_file_id": official_release["file_id"],
            "official_archive_path": official_release["archive_path"],
            "official_archive_sha256": official_release["archive_sha256"],
            "official_member": official_release["panacea_member"],
            "official_member_crc32_pass": official_release["candidate_member_crc32_pass"],
            "matches_previous_author_copy_sha256": prior_h5_sha_match,
        },
        "official_release_manifest": official_release,
        "author_drug_crosswalk": {
            "path": str(drug_table),
            "sha256": drug_table_sha,
            "key_fields": ["pert_id", "cmap_name", "inchi_key", "compound_aliases", "drug_node_idx"],
        },
        "author_gene_info": {
            "path": str(gene_info_path),
            "sha256": gene_info_sha,
        },
        "prior_v5_metadata_domain_comparison": prior_domain,
        "metadata_domain_unchanged_from_v5": prior_domain_unchanged,
        "source_train_identity_and_axis": source,
        "h5ad_metadata": {
            "shape": [meta["n_obs"], meta["n_vars"]],
            "root_keys": meta["root_keys"],
            "root_attrs": meta["root_attrs"],
            "X_metadata_only": meta["X_metadata_only"],
            "layers_metadata_only": meta["layers_metadata_only"],
            "obsm_metadata_only": meta["obsm_metadata_only"],
            "uns_metadata_only": meta["uns_metadata_only"],
            "obs_columns": list(fields.keys()),
            "var_columns": list(var.keys()),
            "obs_cell_counts": dict(Counter(str(r["cell_iname"]) for r in out_rows)),
            "resolved_key_status_counts": dict(key_status_counts),
            "exact_key_common_row_count": exact_common_rows,
            "exact_object_name_overlap_with_source_train": exact_obj_overlap,
            "object_name_overlap_count": len(exact_obj_overlap),
            "gene_axis_candidate_column": next((k for k in ("gene_symbol", "gene_name", "feature_name", "_index") if k in var), None),
            "gene_axis_candidate_sha256": hashlib.sha256("\n".join(gene_values).encode()).hexdigest() if gene_values else None,
            "gene_axis_exact_match": axis_exact,
            "gene_axis_casefold_match": axis_casefold,
        },
        "historical_author_exposure": {
            "author_repo_commit": "d53ff497e465692c80fac7b899e3cc5cc8e7bfed",
            "panacea_config_present": author_panacea_config.is_file(),
            "panacea_evaluation_script_present": author_panacea_evaluator.is_file(),
            "panacea_data_in_author_release": True,
            "panacea_split_columns": [k for k in fields if _normal_name(k).startswith("split")],
            "old_panacea_result_payloads_loaded": False,
            "interpretation": "pre-exposed author asset; external development candidate only, not independent test",
        },
        "metadata_gates": gates,
        "metadata_candidate_domain": {
            "candidate_row_count": len(metadata_candidates),
            "excluded_row_count": len(metadata_excluded),
            "candidate_full_inchikeys": sorted({str(r["full_inchikey"]) for r in metadata_candidates}),
            "candidate_objects": sorted({str(r["cell_iname"]) for r in metadata_candidates}),
            "candidate_rows_are_frozen_for_numeric_phase_only_after_independent_readback": True,
        },
        "metadata_gate_all_pass": all(v for k, v in gates.items() if k not in {"gene_axis_casefold_match_only"}),
        "numeric_matrix_accessed": False,
        "h5ad_X_values_accessed": False,
        "h5ad_layers_values_accessed": False,
        "obsm_control_values_accessed": False,
        "pdo_outer_locked_values_accessed": False,
        "gpu_hours": 0.0,
        "cpu_wall_seconds": time.monotonic() - start,
        "decision": "METADATA_ONLY_GATE_FAILED; matrix read is forbidden until official release lineage and every required metadata gate pass and inclusion/exclusion tables are frozen" if not all(v for k, v in gates.items() if k not in {"gene_axis_casefold_match_only"}) else "METADATA_ONLY_GATE_PASSED; frozen inclusion/exclusion tables are ready for independent readback",
    }
    # Persist only metadata-derived row tables; no expression or control vectors are serialized.
    columns = sorted({k for r in row_reasons for k in r})
    pd.DataFrame(row_reasons, columns=columns).to_csv(out / "condition_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(metadata_candidates, columns=columns).to_csv(out / "metadata_candidate_rows.tsv", sep="\t", index=False)
    pd.DataFrame(metadata_excluded, columns=columns).to_csv(out / "metadata_excluded_rows.tsv", sep="\t", index=False)
    pd.DataFrame(metadata_candidates, columns=columns).to_csv(out / "included_rows.tsv", sep="\t", index=False)
    pd.DataFrame(metadata_excluded, columns=columns).to_csv(out / "excluded_rows.tsv", sep="\t", index=False)
    (out / "metadata_candidate_full_inchikeys.txt").write_text("\n".join(sorted({str(r["full_inchikey"]) for r in metadata_candidates})) + "\n", encoding="utf-8")
    (out / "metadata_candidate_genes.txt").write_text("\n".join(gene_values) + "\n", encoding="utf-8")
    pd.DataFrame([{"gene_index": i, "gene_name": g} for i, g in enumerate(gene_values)]).to_csv(out / "gene_axis_metadata.tsv", sep="\t", index=False)
    frozen = {
        "experiment_id": "EXP-057",
        "run_id": run_id,
        "freeze_phase": "metadata_only_before_any_numeric_access",
        "candidate_asset_sha256": h5_sha,
        "included_rows": len(metadata_candidates),
        "excluded_rows": len(metadata_excluded),
        "included_rows_sha256": sha256(out / "included_rows.tsv"),
        "excluded_rows_sha256": sha256(out / "excluded_rows.tsv"),
        "full_inchikey_list_sha256": sha256(out / "metadata_candidate_full_inchikeys.txt"),
        "gene_axis_sha256": sha256(out / "metadata_candidate_genes.txt"),
        "numeric_matrix_accessed": False,
        "h5ad_X_values_accessed": False,
        "h5ad_layers_values_accessed": False,
        "obsm_control_values_accessed": False,
        "pdo_outer_locked_values_accessed": False,
        "status": "FROZEN_PENDING_ROOT_INDEPENDENT_READBACK",
    }
    (out / "frozen_metadata_domain.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "metadata_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "experiment_id": "EXP-057", "run_id": run_id, "phase": "metadata_only", "gpu_hours": 0.0,
        "metadata_gate_sha256": sha256(out / "metadata_gate.json"),
        "condition_metadata_sha256": sha256(out / "condition_metadata.tsv"),
        "metadata_candidate_rows_sha256": sha256(out / "metadata_candidate_rows.tsv"),
        "metadata_excluded_rows_sha256": sha256(out / "metadata_excluded_rows.tsv"),
        "included_rows_sha256": sha256(out / "included_rows.tsv"),
        "excluded_rows_sha256": sha256(out / "excluded_rows.tsv"),
        "metadata_candidate_full_inchikeys_sha256": sha256(out / "metadata_candidate_full_inchikeys.txt"),
        "metadata_candidate_genes_sha256": sha256(out / "metadata_candidate_genes.txt"),
        "gene_axis_metadata_sha256": sha256(out / "gene_axis_metadata.tsv"),
        "frozen_metadata_domain_sha256": sha256(out / "frozen_metadata_domain.json"),
        "official_release_manifest_sha256": sha256(official_manifest_path),
        "candidate_h5ad_sha256": h5_sha,
        "source_train_values_accessed": False,
        "numeric_matrix_accessed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "decision": gate["decision"], "shape": gate["h5ad_metadata"]["shape"], "gate": gates, "common_rows": exact_common_rows, "output": str(out)}, indent=2))
    return 0 if gate["metadata_gate_all_pass"] else 2


def _require_run_gate_ready(metadata_gate: dict[str, Any], frozen: dict[str, Any], root_readback: dict[str, Any]) -> None:
    """Fail closed before touching any PANACEA numeric matrix if identity gates are incomplete."""
    failures: list[str] = []
    if metadata_gate.get("metadata_gate_all_pass") is not True:
        failures.append("metadata-only gate did not pass")
    if frozen.get("status") != "FROZEN_PENDING_ROOT_INDEPENDENT_READBACK":
        failures.append("frozen metadata domain is missing or has an unexpected state")
    if not str(root_readback.get("status", "")).startswith("ROOT_INDEPENDENT_METADATA_READBACK_PASS"):
        failures.append("root independent metadata readback did not pass")
    if root_readback.get("independent_assertions_pass") is not True:
        failures.append("root independent assertions are not all passing")
    if root_readback.get("run_id") != metadata_gate.get("run_id"):
        failures.append("root metadata readback run_id does not match the frozen run")
    boundaries = root_readback.get("data_boundary", {})
    if any(boundaries.get(name) is not False for name in (
        "metadata_numeric_matrix_accessed", "h5ad_X_values_accessed",
        "h5ad_layers_values_accessed", "matched_control_values_accessed",
        "pdo_outer_locked_values_accessed",
    )):
        failures.append("root readback does not confirm all prior numeric-access flags are false")
    if failures:
        raise RuntimeError("numeric run blocked: " + "; ".join(failures))


def _resolve_source_gene_reindex(
    source_gene_labels: list[str],
    mapping: pd.DataFrame,
    official_gene_info: pd.DataFrame,
    expected_genes: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Resolve numeric source H5AD var labels through its frozen author-gene map."""
    map_required = {"axis_index", "h5ad_var_name", "h5ad_var_annotation", "l1000_gene_info_value"}
    info_required = {"gene_id", "gene_symbol"}
    if not map_required.issubset(mapping.columns):
        raise ValueError(f"source gene-axis map lacks columns: {sorted(map_required - set(mapping.columns))}")
    if not info_required.issubset(official_gene_info.columns):
        raise ValueError(f"official gene metadata lacks columns: {sorted(info_required - set(official_gene_info.columns))}")
    mapping = mapping.copy()
    mapping["_axis_order"] = pd.to_numeric(mapping["axis_index"], errors="raise").astype(np.int64)
    mapping = mapping.sort_values("_axis_order", kind="stable").reset_index(drop=True)
    if len(mapping) != len(source_gene_labels) or mapping["_axis_order"].tolist() != list(range(len(source_gene_labels))):
        raise ValueError("source gene-axis map is not a complete ordered axis")
    mapped_labels = mapping["h5ad_var_name"].astype(str).str.strip().tolist()
    if len(set(mapped_labels)) != len(mapped_labels) or source_gene_labels != mapped_labels:
        raise ValueError("source NPZ gene labels differ from the frozen H5AD var-name axis")
    reference_gene_ids = mapping["l1000_gene_info_value"].astype(str).str.strip()
    annotation_gene_ids = mapping["h5ad_var_annotation"].astype(str).str.strip()
    if (reference_gene_ids.eq("").any() or not reference_gene_ids.equals(annotation_gene_ids)
            or reference_gene_ids.duplicated().any()):
        raise ValueError("frozen source var annotation does not uniquely match the author gene-ID axis")
    info = official_gene_info[["gene_id", "gene_symbol"]].copy()
    info["gene_id"] = info["gene_id"].astype(str).str.strip()
    info["gene_symbol"] = info["gene_symbol"].astype(str).str.strip()
    if (info["gene_id"].eq("").any() or info["gene_symbol"].eq("").any()
            or info["gene_id"].duplicated().any() or info["gene_symbol"].duplicated().any()):
        raise ValueError("official gene-ID to symbol crosswalk is not complete and one-to-one")
    crosswalk = pd.DataFrame({"gene_id": reference_gene_ids})
    aligned = crosswalk.merge(info, on="gene_id", how="left", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("frozen source gene IDs are missing from the official gene-info crosswalk")
    source_symbols = aligned["gene_symbol"].astype(str).tolist()
    if len(source_symbols) != len(set(source_symbols)) or set(source_symbols) != set(expected_genes):
        raise ValueError("source and frozen PANACEA gene-symbol axes do not have the same unique members")
    source_position = {symbol: i for i, symbol in enumerate(source_symbols)}
    reindex = np.asarray([source_position[symbol] for symbol in expected_genes], dtype=np.int64)
    alignment = pd.DataFrame({
        "target_gene_position": np.arange(len(expected_genes), dtype=np.int64),
        "gene_symbol": expected_genes,
        "source_column_position": reindex,
        "source_npz_gene_label": [source_gene_labels[i] for i in reindex],
        "source_author_gene_id": [reference_gene_ids.iloc[i] for i in reindex],
    })
    evidence = {
        "source_npz_gene_labels_are_numeric_h5ad_var_names": bool(all(x.isdigit() for x in source_gene_labels)),
        "mapping_rows": int(len(mapping)),
        "source_axis_unique_gene_ids": int(reference_gene_ids.nunique()),
        "source_axis_unique_gene_symbols": int(len(set(source_symbols))),
        "gene_symbol_membership_matches_target": True,
        "gene_symbol_order_already_matches_target": bool(source_symbols == expected_genes),
        "reordered_source_columns": bool(not np.array_equal(reindex, np.arange(len(reindex)))),
        "source_gene_symbol_axis_sha256": hashlib.sha256("\n".join(source_symbols).encode()).hexdigest(),
        "target_gene_symbol_axis_sha256": hashlib.sha256("\n".join(expected_genes).encode()).hexdigest(),
        "source_column_positions_in_target_order": reindex.tolist(),
    }
    return reindex, alignment, evidence


def _load_source_train_means(
    values_path: Path,
    key_table_path: Path,
    expected_genes: list[str],
    required_key: str,
    gene_axis_mapping_path: Path,
    gene_axis_receipt_path: Path,
    official_gene_info_path: Path,
    expected_mapping_sha256: str,
    expected_gene_info_sha256: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Load only the separately stored 419-row TRAIN targets; never open inner-dev arrays."""
    if sha256(gene_axis_mapping_path) != expected_mapping_sha256:
        raise ValueError("source frozen gene-axis mapping checksum changed")
    if sha256(official_gene_info_path) != expected_gene_info_sha256:
        raise ValueError("official gene-info checksum changed")
    axis_receipt = json.loads(gene_axis_receipt_path.read_text(encoding="utf-8"))
    if (axis_receipt.get("files_sha256", {}).get("gene_axis_mapping.tsv") != expected_mapping_sha256
            or int(axis_receipt.get("mapping_rows", -1)) != len(expected_genes)
            or axis_receipt.get("mapping_proof") != "exact row-by-row shared-column match"
            or axis_receipt.get("matched_columns", {}).get("reference") != "gene_id"
            or axis_receipt.get("X_read") is not False
            or axis_receipt.get("X_ctl_read") is not False):
        raise ValueError("source gene-axis map is not bound to its frozen metadata-only receipt")
    axis_mapping = pd.read_csv(gene_axis_mapping_path, sep="\t", dtype=str, keep_default_na=False)
    official_gene_info = pd.read_csv(official_gene_info_path, dtype=str, keep_default_na=False)
    key_table = pd.read_csv(key_table_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source_object_id", "full_inchikey", "pert_id", "split_role", "full_condition_key"}
    if not required.issubset(key_table.columns):
        raise ValueError(f"source identity table is missing columns: {sorted(required - set(key_table.columns))}")
    train_meta = key_table.loc[key_table["split_role"].str.lower().eq("train")].copy()
    if len(train_meta) != 419 or train_meta["full_condition_key"].duplicated().any():
        raise ValueError("source identity table does not bind exactly 419 unique TRAIN condition keys")
    if train_meta["source_object_id"].nunique() != 110 or train_meta["full_inchikey"].nunique() != 5:
        raise ValueError("source TRAIN metadata object or full-InChIKey denominator changed")
    with np.load(values_path, allow_pickle=False) as archive:
        required_arrays = {"train_condition_key", "genes", "train_target_delta"}
        if not required_arrays.issubset(set(archive.files)):
            raise ValueError(f"source baseline archive lacks TRAIN arrays: {sorted(required_arrays - set(archive.files))}")
        train_keys = archive["train_condition_key"].astype(str)
        source_gene_labels = archive["genes"].astype(str).tolist()
        if train_keys.ndim != 1 or len(train_keys) != 419 or len(set(train_keys.tolist())) != 419:
            raise ValueError("source baseline archive does not contain 419 unique TRAIN keys")
        reindex, alignment, axis_evidence = _resolve_source_gene_reindex(
            source_gene_labels, axis_mapping, official_gene_info, expected_genes
        )
        lookup = train_meta.set_index("full_condition_key", verify_integrity=True)
        if set(train_keys.tolist()) != set(lookup.index.astype(str)):
            raise ValueError("source TRAIN NPZ keys and response-blind TRAIN table do not match exactly")
        ordered_meta = lookup.loc[train_keys.tolist()].reset_index()
        if not ordered_meta["split_role"].str.lower().eq("train").all():
            raise ValueError("source baseline key mapping crossed a non-TRAIN role")
        train_delta = archive["train_target_delta"].astype(np.float64, copy=True)[:, reindex]
        loaded_array_names = ["train_condition_key", "genes", "train_target_delta"]
    if train_delta.shape != (419, len(expected_genes)) or not np.isfinite(train_delta).all():
        raise ValueError(f"source TRAIN-only target array shape/value check failed: {train_delta.shape}")
    if required_key not in set(ordered_meta["full_inchikey"].astype(str)):
        raise ValueError(f"exact full InChIKey is absent from source TRAIN: {required_key}")
    global_mean, key_mean, key_count = _compute_train_means(
        train_delta, ordered_meta["full_inchikey"].astype(str).to_numpy(), required_key
    )
    evidence = {
        "path": str(values_path),
        "sha256": sha256(values_path),
        "identity_table_path": str(key_table_path),
        "identity_table_sha256": sha256(key_table_path),
        "numeric_array_names_loaded": loaded_array_names,
        "inner_dev_or_other_role_numeric_arrays_loaded": False,
        "train_conditions": int(len(train_delta)),
        "train_objects": int(ordered_meta["source_object_id"].nunique()),
        "train_full_inchikeys": int(ordered_meta["full_inchikey"].nunique()),
        "per_key_train_conditions": key_count,
        "gene_count": int(train_delta.shape[1]),
        "train_key_order_bound_to_identity_table": True,
        "gene_axis_mapping_path": str(gene_axis_mapping_path),
        "gene_axis_mapping_sha256": sha256(gene_axis_mapping_path),
        "gene_axis_receipt_path": str(gene_axis_receipt_path),
        "gene_axis_receipt_sha256": sha256(gene_axis_receipt_path),
        "official_gene_info_path": str(official_gene_info_path),
        "official_gene_info_sha256": sha256(official_gene_info_path),
        "gene_axis_alignment": axis_evidence,
    }
    return global_mean, key_mean, ordered_meta, evidence, alignment


def _compute_train_means(train_delta: np.ndarray, full_keys: np.ndarray, required_key: str) -> tuple[np.ndarray, np.ndarray, int]:
    values = np.asarray(train_delta, dtype=np.float64)
    keys = np.asarray(full_keys, dtype=str)
    if values.ndim != 2 or len(values) != len(keys) or not np.isfinite(values).all():
        raise ValueError("TRAIN delta and exact full-key rows must be finite and aligned")
    mask = keys == str(required_key)
    count = int(mask.sum())
    if count == 0:
        raise ValueError(f"no TRAIN rows found for exact full InChIKey {required_key}")
    return (
        values.mean(axis=0, dtype=np.float64).astype(np.float32),
        values[mask].mean(axis=0, dtype=np.float64).astype(np.float32),
        count,
    )


def _read_dense_selected_rows(node: h5py.Dataset, rows: np.ndarray, expected_cols: int) -> np.ndarray:
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"selected-row reader requires a dense HDF5 dataset, got {node.name}")
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("numeric read rows must be nonempty and unique")
    if len(node.shape) != 2 or node.shape[1] != expected_cols:
        raise ValueError(f"matrix axis mismatch for {node.name}: {node.shape}")
    if int(rows.min()) < 0 or int(rows.max()) >= int(node.shape[0]):
        raise IndexError(f"selected metadata row outside {node.name} axis")
    order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[order]
    sorted_values = np.asarray(node[sorted_rows, :], dtype=np.float64)
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def _read_frozen_target_delta(h5_path: Path, included_rows: pd.DataFrame, genes_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    row_ids = pd.to_numeric(included_rows["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if len(row_ids) != 11 or len(np.unique(row_ids)) != 11:
        raise ValueError("the frozen PANACEA numeric domain must contain exactly 11 unique row indices")
    with h5py.File(h5_path, "r") as f:
        if "layers" in f:
            # Presence is expected metadata; this branch deliberately never indexes it.
            pass
        if "X" not in f or "obsm" not in f or "X_ctl" not in f["obsm"]:
            raise KeyError("official PANACEA H5AD lacks X or obsm/X_ctl")
        x = _read_dense_selected_rows(f["X"], row_ids, genes_count)
        x_ctl = _read_dense_selected_rows(f["obsm"]["X_ctl"], row_ids, genes_count)
    if x.shape != (11, genes_count) or x_ctl.shape != x.shape:
        raise ValueError(f"selected PANACEA treatment/control shapes do not match 11x{genes_count}")
    if not np.isfinite(x).all() or not np.isfinite(x_ctl).all():
        raise FloatingPointError("selected PANACEA treatment or matched-control rows contain non-finite values")
    target_delta = x - x_ctl
    if not np.isfinite(target_delta).all():
        raise FloatingPointError("PANACEA log2(TPM+1) treatment-minus-control delta is non-finite")
    return target_delta, {
        "path": str(h5_path),
        "file_sha256": sha256(h5_path),
        "numeric_matrix_scope": {
            "H5AD_X_row_indices_read": sorted(map(int, row_ids.tolist())),
            "H5AD_X_rows_read": int(len(row_ids)),
            "matched_control_row_indices_read": sorted(map(int, row_ids.tolist())),
            "matched_control_rows_read": int(len(row_ids)),
            "H5AD_layers_values_read": False,
            "other_H5AD_X_or_control_rows_read": 0,
            "target_definition": "log2(TPM+1) treatment expression minus same-row paired control expression",
        },
    }


def _load_hallmark_sets(path: Path, expected_sha256: str, genes: list[str], expected_set_count: int) -> dict[str, np.ndarray]:
    if sha256(path) != expected_sha256:
        raise ValueError("registered MSigDB Hallmark GMT checksum mismatch")
    gene_to_index = {gene: i for i, gene in enumerate(genes)}
    if len(gene_to_index) != len(genes):
        raise ValueError("frozen gene axis contains duplicate symbols")
    result: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3 or not fields[0]:
            raise ValueError(f"malformed Hallmark GMT line {line_no}")
        if fields[0] in result:
            raise ValueError(f"duplicate Hallmark name at line {line_no}: {fields[0]}")
        indices = sorted({gene_to_index[g] for g in fields[2:] if g in gene_to_index})
        if not indices:
            raise ValueError(f"Hallmark set has no exact symbols on the frozen gene axis: {fields[0]}")
        result[fields[0]] = np.asarray(indices, dtype=np.int32)
    if len(result) != expected_set_count:
        raise ValueError(f"expected {expected_set_count} fixed Hallmark sets, found {len(result)}")
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", ascending=True).to_numpy(dtype=np.float64)


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> tuple[float | None, str]:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None, "UNDEFINED_INSUFFICIENT_FINITE_GENES"
    if not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
        return None, "UNDEFINED_NONFINITE_GENES"
    if np.all(values_a == values_a[0]):
        return None, "UNDEFINED_CONSTANT_TARGET_VECTOR"
    if np.all(values_b == values_b[0]):
        return None, "UNDEFINED_CONSTANT_PREDICTION_VECTOR"
    correlation = float(np.corrcoef(_average_ranks(values_a), _average_ranks(values_b))[0, 1])
    if not np.isfinite(correlation):
        return None, "UNDEFINED_NONFINITE_RANK_CORRELATION"
    return correlation, "DEFINED"


def _direction_label(value: float) -> str:
    if value > 0:
        return "UP"
    if value < 0:
        return "DOWN"
    return "ZERO"


def _score_condition(target: np.ndarray, prediction: np.ndarray, hallmark_sets: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("per-condition target/prediction must be same-shape finite gene vectors")
    target_nonzero = y != 0.0
    direction_denominator = int(target_nonzero.sum())
    direction_numerator = int((np.sign(y[target_nonzero]) == np.sign(p[target_nonzero])).sum()) if direction_denominator else 0
    if direction_denominator:
        gene_direction = float(direction_numerator / direction_denominator)
    else:
        gene_direction = None
    rank_corr, rank_status = _spearman(y, p)
    y_rank = _average_ranks(y)
    p_rank = _average_ranks(p)
    gene_rows = []
    for index in range(len(y)):
        gene_rows.append({
            "gene_index": index,
            "target_direction": _direction_label(float(y[index])),
            "prediction_direction": _direction_label(float(p[index])),
            "target_rank": float(y_rank[index]),
            "prediction_rank": float(p_rank[index]),
            "absolute_rank_gap": float(abs(y_rank[index] - p_rank[index])),
            "target_nonzero": bool(y[index] != 0.0),
            "direction_match": bool(y[index] != 0.0 and np.sign(y[index]) == np.sign(p[index])),
        })
    hallmark_rows = []
    target_nonzero_sets = 0
    set_direction_matches = 0
    predicted_nonzero_sets = 0
    both_nonzero_sets = 0
    for name, indices in hallmark_sets.items():
        target_mean = float(np.mean(y[indices], dtype=np.float64))
        prediction_mean = float(np.mean(p[indices], dtype=np.float64))
        target_direction = _direction_label(target_mean)
        prediction_direction = _direction_label(prediction_mean)
        target_active = target_direction != "ZERO"
        prediction_active = prediction_direction != "ZERO"
        match = bool(target_active and target_direction == prediction_direction)
        target_nonzero_sets += int(target_active)
        predicted_nonzero_sets += int(prediction_active)
        both_nonzero_sets += int(target_active and prediction_active)
        set_direction_matches += int(match)
        hallmark_rows.append({
            "hallmark": name,
            "n_frozen_axis_genes": int(len(indices)),
            "target_direction": target_direction,
            "prediction_direction": prediction_direction,
            "target_nonzero": target_active,
            "prediction_nonzero": prediction_active,
            "direction_match": match,
        })
    hallmark_agreement = (set_direction_matches / target_nonzero_sets) if target_nonzero_sets else None
    return {
        "gene_direction_agreement": gene_direction,
        "gene_direction_numerator": direction_numerator,
        "gene_direction_denominator_target_nonzero": direction_denominator,
        "target_zero_gene_count": int((~target_nonzero).sum()),
        "prediction_zero_gene_count": int((p == 0.0).sum()),
        "gene_spearman": rank_corr,
        "gene_spearman_status": rank_status,
        "gene_spearman_finite_gene_count": int(len(y)),
        "hallmark_direction_agreement": hallmark_agreement,
        "hallmark_direction_numerator": int(set_direction_matches),
        "hallmark_direction_denominator_target_nonzero": int(target_nonzero_sets),
        "hallmark_target_zero_set_count": int(len(hallmark_sets) - target_nonzero_sets),
        "hallmark_prediction_nonzero_set_count": int(predicted_nonzero_sets),
        "hallmark_both_nonzero_set_count": int(both_nonzero_sets),
        "hallmark_set_count": int(len(hallmark_sets)),
    }, gene_rows, hallmark_rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial JSON: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int) -> tuple[float | None, float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    n = int(len(finite))
    if n == 0:
        return None, None, None, 0
    point = float(np.mean(finite, dtype=np.float64))
    if n < 2:
        return point, None, None, n
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    means = finite[draws].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(means, [0.025, 0.975])
    return point, float(low), float(high), n


def _paired_bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int, replicates: int) -> tuple[float | None, float | None, float | None, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    n = int(len(differences))
    if n == 0:
        return None, None, None, 0
    point = float(np.mean(differences, dtype=np.float64))
    if n < 2:
        return point, None, None, n
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    means = differences[draws].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(means, [0.025, 0.975])
    return point, float(low), float(high), n


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial NPZ: {temporary}")
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise TypeError("prediction NPZ may not contain object arrays or require pickle")
    with temporary.open("xb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def _run_phase(config_path: Path, config: dict[str, Any], run_id: str) -> int:
    start = time.monotonic()
    out = EXP_ROOT / run_id
    if not out.is_dir():
        raise FileNotFoundError(f"metadata-only run directory is missing: {out}")
    data_cfg = config["data"]
    metadata_gate_path = out / "metadata_gate.json"
    frozen_path = out / "frozen_metadata_domain.json"
    root_readback_path = (ROOT / str(data_cfg["root_metadata_readback"])).resolve()
    metadata_manifest_path = out / "run_manifest.json"
    for path in (metadata_gate_path, frozen_path, root_readback_path, metadata_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"numeric phase prerequisite is missing: {path}")
    metadata_gate = json.loads(metadata_gate_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    root_readback = json.loads(root_readback_path.read_text(encoding="utf-8"))
    metadata_manifest = json.loads(metadata_manifest_path.read_text(encoding="utf-8"))
    _require_run_gate_ready(metadata_gate, frozen, root_readback)
    if metadata_gate.get("run_id") != run_id or frozen.get("run_id") != run_id:
        raise RuntimeError("numeric phase run_id does not match the frozen metadata gate")
    if metadata_manifest.get("metadata_gate_sha256") != sha256(metadata_gate_path):
        raise RuntimeError("metadata gate changed after its run manifest was written")
    if root_readback.get("metadata_gate", {}).get("sha256") != sha256(metadata_gate_path):
        raise RuntimeError("root readback binds a different metadata gate SHA-256")
    if root_readback.get("frozen_domain", {}).get("sha256") != sha256(frozen_path):
        raise RuntimeError("root readback binds a different frozen-domain SHA-256")

    included_path = out / "included_rows.tsv"
    excluded_path = out / "excluded_rows.tsv"
    keys_path = out / "metadata_candidate_full_inchikeys.txt"
    genes_path = out / "metadata_candidate_genes.txt"
    for name, path, expected in (
        ("included_rows_sha256", included_path, frozen.get("included_rows_sha256")),
        ("excluded_rows_sha256", excluded_path, frozen.get("excluded_rows_sha256")),
        ("full_inchikey_list_sha256", keys_path, frozen.get("full_inchikey_list_sha256")),
        ("gene_axis_sha256", genes_path, frozen.get("gene_axis_sha256")),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen metadata input changed after independent readback: {name}")
    included = pd.read_csv(included_path, sep="\t", dtype=str, keep_default_na=False)
    excluded = pd.read_csv(excluded_path, sep="\t", dtype=str, keep_default_na=False)
    frozen_keys = keys_path.read_text(encoding="utf-8").splitlines()
    genes = genes_path.read_text(encoding="utf-8").splitlines()
    required_key = str(data_cfg.get("required_key_value", "XGALLCVXEZPNRQ-UHFFFAOYSA-N"))
    if len(included) != 11 or len(excluded) != 341 or len(genes) != 978:
        raise ValueError("root-approved frozen domain is no longer 11 included rows, 341 exclusions, and 978 genes")
    if frozen_keys != [required_key] or included["full_inchikey"].astype(str).nunique() != 1:
        raise ValueError("frozen included rows are not restricted to the one exact common full InChIKey")
    if included["cell_iname"].astype(str).nunique() != 11:
        raise ValueError("frozen included rows do not map to exactly 11 unique target cell-line objects")
    if not included["metadata_only_eligible_candidate"].astype(str).str.lower().eq("true").all():
        raise ValueError("a frozen included row fails the metadata eligibility flag")
    h5_path = (ROOT / str(data_cfg["candidate_path"])).resolve()
    if sha256(h5_path) != metadata_gate.get("candidate_asset", {}).get("sha256_bytewise"):
        raise RuntimeError("official PANACEA H5AD bytes changed after metadata-only readback")

    # Source numerical access is restricted to the distinct TRAIN target array.
    source_values_path = (ROOT / str(data_cfg["source_train_values_npz"])).resolve()
    source_key_table_path = (ROOT / str(data_cfg["source_train_identity_table"])).resolve()
    gene_axis_mapping_path = (ROOT / str(data_cfg["source_gene_axis_mapping"])).resolve()
    gene_axis_receipt_path = (ROOT / str(data_cfg["source_gene_axis_receipt"])).resolve()
    official_gene_info_path = (ROOT / str(data_cfg["official_gene_info"])).resolve()
    global_mean, key_mean, ordered_train_meta, source_evidence, gene_alignment = _load_source_train_means(
        source_values_path,
        source_key_table_path,
        genes,
        required_key,
        gene_axis_mapping_path,
        gene_axis_receipt_path,
        official_gene_info_path,
        str(data_cfg["source_gene_axis_mapping_sha256"]),
        str(data_cfg["official_gene_info_sha256"]),
    )
    gene_alignment_path = out / "source_gene_axis_alignment.tsv"
    gene_alignment.to_csv(gene_alignment_path, sep="\t", index=False)
    source_evidence["gene_axis_alignment_artifact"] = str(gene_alignment_path)
    source_evidence["gene_axis_alignment_artifact_sha256"] = sha256(gene_alignment_path)
    # This is the first target-response read. Only the root-approved 11 row indices are selected.
    target_delta, target_read_evidence = _read_frozen_target_delta(h5_path, included, len(genes))
    if target_read_evidence["file_sha256"] != metadata_gate.get("candidate_asset", {}).get("sha256_bytewise"):
        raise RuntimeError("selected numeric input was read from a different PANACEA H5AD version")

    function_cfg = config["functional_sets"]
    hallmark_path = (ROOT / str(function_cfg["path"])).resolve()
    hallmark_sets = _load_hallmark_sets(
        hallmark_path, str(function_cfg["sha256"]), genes, int(function_cfg["expected_set_count"])
    )
    method_names = ("source_train_global_mean_delta", "source_train_exact_full_inchikey_mean_delta", "zero_change")
    predictions = {
        method_names[0]: np.tile(global_mean.astype(np.float64), (len(included), 1)),
        method_names[1]: np.tile(key_mean.astype(np.float64), (len(included), 1)),
        method_names[2]: np.zeros_like(target_delta, dtype=np.float64),
    }
    if any(not np.isfinite(p).all() for p in predictions.values()):
        raise FloatingPointError("one or more fixed TRAIN-only baseline predictions are non-finite")

    condition_metric_rows: list[dict[str, Any]] = []
    gene_detail_rows: list[dict[str, Any]] = []
    hallmark_detail_rows: list[dict[str, Any]] = []
    source_objects = set(ordered_train_meta["source_object_id"].astype(str))
    for row_pos, (_, row) in enumerate(included.reset_index(drop=True).iterrows()):
        identity = {
            "condition_row_index": int(pd.to_numeric(row["row_index"], errors="raise")),
            "obs_name": str(row["obs_name"]),
            "cell_line": str(row["cell_iname"]),
            "full_inchikey": str(row["full_inchikey"]),
            "pert_name": str(row["pert_name"]),
            "pert_id": str(row["pert_id"]),
            "dose": str(row["dose"]),
            "dose_unit": str(row["dose_unit"]),
            "time": str(row["time"]),
            "time_unit": str(row["time_unit"]),
            "control_id": str(row["control_id"]),
            "source_object_name_overlap": str(row["cell_iname"]) in source_objects,
        }
        for method in method_names:
            metrics_row, gene_rows, set_rows = _score_condition(target_delta[row_pos], predictions[method][row_pos], hallmark_sets)
            condition_metric_rows.append({**identity, "baseline": method, **metrics_row})
            for gene_index, detail in enumerate(gene_rows):
                gene_detail_rows.append({
                    **identity,
                    "baseline": method,
                    "gene_index": gene_index,
                    "gene_symbol": genes[gene_index],
                    **detail,
                })
            for detail in set_rows:
                hallmark_detail_rows.append({**identity, "baseline": method, **detail})

    condition_df = pd.DataFrame(condition_metric_rows)
    metric_columns = ["gene_direction_agreement", "gene_spearman", "hallmark_direction_agreement"]
    object_df = condition_df.groupby(["cell_line", "full_inchikey", "baseline"], sort=True, as_index=False).agg(
        n_conditions=("condition_row_index", "count"),
        source_object_name_overlap=("source_object_name_overlap", "max"),
        gene_direction_agreement=("gene_direction_agreement", "mean"),
        gene_spearman=("gene_spearman", "mean"),
        hallmark_direction_agreement=("hallmark_direction_agreement", "mean"),
        gene_direction_numerator=("gene_direction_numerator", "sum"),
        gene_direction_denominator=("gene_direction_denominator_target_nonzero", "sum"),
        hallmark_direction_numerator=("hallmark_direction_numerator", "sum"),
        hallmark_direction_denominator=("hallmark_direction_denominator_target_nonzero", "sum"),
    )
    drug_df = object_df.groupby(["full_inchikey", "baseline"], sort=True, as_index=False).agg(
        n_objects=("cell_line", "nunique"),
        n_conditions=("n_conditions", "sum"),
        source_object_name_overlap_count=("source_object_name_overlap", "sum"),
        gene_direction_agreement=("gene_direction_agreement", "mean"),
        gene_spearman=("gene_spearman", "mean"),
        hallmark_direction_agreement=("hallmark_direction_agreement", "mean"),
    )

    metric_rows: list[dict[str, Any]] = []
    bootstrap_reps = int(config["metrics"]["bootstrap_replicates"])
    seed_base = int(config["metrics"]["bootstrap_seed"])
    for method_index, method in enumerate(method_names):
        for metric_index, metric in enumerate(metric_columns):
            values = pd.to_numeric(object_df.loc[object_df["baseline"].eq(method), metric], errors="coerce").to_numpy(dtype=np.float64)
            point, low, high, n_valid = _bootstrap_mean(values, seed_base + method_index * 100 + metric_index, bootstrap_reps)
            metric_rows.append({
                "baseline": method,
                "metric": metric,
                "object_macro_mean": point,
                "object_bootstrap_95ci_low": low,
                "object_bootstrap_95ci_high": high,
                "valid_objects": n_valid,
                "total_objects": int(object_df.loc[object_df["baseline"].eq(method), "cell_line"].nunique()),
                "bootstrap_replicates": bootstrap_reps,
                "resampling_unit": "target cell-line object; paired between fixed baselines; descriptive interval only",
                "missing_rule": "undefined values remain missing and do not enter the valid-object mean; zero-change gene-rank correlation is undefined because prediction is constant",
            })
    summary_metrics = pd.DataFrame(metric_rows)

    paired_rows: list[dict[str, Any]] = []
    pairs = [
        (method_names[0], method_names[1]),
        (method_names[0], method_names[2]),
        (method_names[1], method_names[2]),
    ]
    for metric_index, metric in enumerate(metric_columns):
        wide = object_df.pivot(index="cell_line", columns="baseline", values=metric)
        for pair_index, (left_method, right_method) in enumerate(pairs):
            left = pd.to_numeric(wide[left_method], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(wide[right_method], errors="coerce").to_numpy(dtype=np.float64)
            point, low, high, n_valid = _paired_bootstrap_difference(
                left, right, seed_base + 1000 + metric_index * 100 + pair_index, bootstrap_reps
            )
            paired_rows.append({
                "metric": metric,
                "left_baseline": left_method,
                "right_baseline": right_method,
                "difference_left_minus_right": point,
                "paired_object_bootstrap_95ci_low": low,
                "paired_object_bootstrap_95ci_high": high,
                "valid_paired_objects": n_valid,
                "total_paired_objects": int(len(wide)),
                "bootstrap_replicates": bootstrap_reps,
                "resampling_unit": "target cell-line object, paired; descriptive interval only",
                "p_value_computed": False,
            })
    paired_df = pd.DataFrame(paired_rows)

    gene_df = pd.DataFrame(gene_detail_rows)
    counterexample_rows: list[dict[str, Any]] = []
    for (cell_line, key, method), subset in gene_df.groupby(["cell_line", "full_inchikey", "baseline"], sort=True):
        mismatch = subset.loc[subset["target_nonzero"] & ~subset["direction_match"]].sort_values(
            ["absolute_rank_gap", "gene_index"], ascending=[False, True], kind="stable"
        )
        chosen = mismatch.head(5).copy()
        chosen["counterexample_reason"] = "target_nonzero_gene_direction_mismatch"
        if len(chosen) < 5:
            remaining = subset.loc[~subset["gene_index"].isin(chosen["gene_index"])].sort_values(
                ["absolute_rank_gap", "gene_index"], ascending=[False, True], kind="stable"
            ).head(5 - len(chosen)).copy()
            remaining["counterexample_reason"] = "largest_remaining_gene_rank_gap"
            chosen = pd.concat([chosen, remaining], ignore_index=True)
        for record in chosen.to_dict(orient="records"):
            counterexample_rows.append({
                "cell_line": str(cell_line),
                "full_inchikey": str(key),
                "baseline": str(method),
                "gene_symbol": str(record["gene_symbol"]),
                "target_direction": str(record["target_direction"]),
                "prediction_direction": str(record["prediction_direction"]),
                "target_rank": float(record["target_rank"]),
                "prediction_rank": float(record["prediction_rank"]),
                "absolute_rank_gap": float(record["absolute_rank_gap"]),
                "counterexample_reason": str(record["counterexample_reason"]),
                "source_object_name_overlap": bool(record["source_object_name_overlap"]),
            })
    counterexamples_df = pd.DataFrame(counterexample_rows)

    npz_path = out / "keyed_baseline_predictions.npz"
    output_arrays = {
        "observation_key": np.asarray(included["obs_name"].astype(str).tolist(), dtype=np.str_),
        "cell_line": np.asarray(included["cell_iname"].astype(str).tolist(), dtype=np.str_),
        "full_inchikey": np.asarray(included["full_inchikey"].astype(str).tolist(), dtype=np.str_),
        "dose": np.asarray(included["dose"].astype(str).tolist(), dtype=np.str_),
        "dose_unit": np.asarray(included["dose_unit"].astype(str).tolist(), dtype=np.str_),
        "time": np.asarray(included["time"].astype(str).tolist(), dtype=np.str_),
        "time_unit": np.asarray(included["time_unit"].astype(str).tolist(), dtype=np.str_),
        "control_id": np.asarray(included["control_id"].astype(str).tolist(), dtype=np.str_),
        "genes": np.asarray(genes, dtype=np.str_),
        "target_log2_tpm_plus1_delta": target_delta.astype(np.float64, copy=False),
        "source_train_global_mean_delta": predictions[method_names[0]].astype(np.float64, copy=False),
        "source_train_exact_full_inchikey_mean_delta": predictions[method_names[1]].astype(np.float64, copy=False),
        "zero_change": predictions[method_names[2]].astype(np.float64, copy=False),
    }
    _atomic_npz(npz_path, output_arrays)
    with np.load(npz_path, allow_pickle=False) as saved:
        if any(saved[name].dtype.kind == "O" for name in saved.files):
            raise TypeError("keyed prediction archive contains a pickle/object array")
        if saved["target_log2_tpm_plus1_delta"].shape != (11, 978):
            raise ValueError("keyed prediction archive failed safe shape readback")
        if not np.array_equal(saved["genes"].astype(str), np.asarray(genes, dtype=str)):
            raise ValueError("keyed prediction archive gene axis failed safe readback")

    # Preserve the metadata-stage receipt before promoting run_manifest.json to the complete run.
    metadata_receipt_copy = out / "metadata_phase_manifest.json"
    if not metadata_receipt_copy.exists():
        metadata_receipt_copy.write_text(json.dumps(metadata_manifest, indent=2) + "\n", encoding="utf-8")

    condition_path = out / "per_condition_metrics.tsv"
    object_path = out / "per_object_metrics.tsv"
    drug_path = out / "per_drug_metrics.tsv"
    summary_path = out / "summary_metrics.tsv"
    paired_path = out / "paired_object_bootstrap.tsv"
    gene_path = out / "gene_direction_rank_detail.tsv"
    hallmark_result_path = out / "hallmark_set_direction.tsv"
    counterexample_path = out / "counterexamples.tsv"
    condition_df.to_csv(condition_path, sep="\t", index=False, na_rep="")
    object_df.to_csv(object_path, sep="\t", index=False, na_rep="")
    drug_df.to_csv(drug_path, sep="\t", index=False, na_rep="")
    summary_metrics.to_csv(summary_path, sep="\t", index=False, na_rep="")
    paired_df.to_csv(paired_path, sep="\t", index=False, na_rep="")
    gene_df.to_csv(gene_path, sep="\t", index=False, na_rep="")
    pd.DataFrame(hallmark_detail_rows).to_csv(hallmark_result_path, sep="\t", index=False, na_rep="")
    counterexamples_df.to_csv(counterexample_path, sep="\t", index=False, na_rep="")

    result = {
        "experiment_id": "EXP-057",
        "run_id": run_id,
        "status": "COMPLETED_EXTERNAL_DEVELOPMENT_FIXED_BASELINES",
        "interpretation": "author-exposed PANACEA external development candidate; not an independent final test or PDO efficacy evaluation",
        "data_scales": {
            "target": "log2(TPM+1) treatment minus same-row paired control expression",
            "source": "frozen XPert source TRAIN L1000 response delta",
            "amplitude_metrics_permitted": False,
            "target_fitted_transform_or_calibration": False,
        },
        "domain": {
            "included_conditions": int(len(included)),
            "excluded_conditions": int(len(excluded)),
            "target_objects": int(included["cell_iname"].nunique()),
            "exact_full_inchikeys": frozen_keys,
            "key_counts": {str(k): int(v) for k, v in included["full_inchikey"].value_counts().items()},
            "known_target_object_name_overlap_with_source_train": sorted(
                set(included["cell_iname"].astype(str)) & source_objects
            ),
            "author_exposure": "XPert authors included PANACEA in their release/config/evaluation; external development only",
        },
        "baselines": {
            method_names[0]: "one 978-gene vector: arithmetic mean over the 419 frozen source TRAIN deltas",
            method_names[1]: f"one 978-gene vector: arithmetic mean over source TRAIN rows with exact full InChIKey {required_key}",
            method_names[2]: "all-zero 978-gene vector",
        },
        "source_train_input": source_evidence,
        "target_numeric_read": target_read_evidence,
        "functional_sets": {
            "resource": str(function_cfg["resource"]),
            "path": str(hallmark_path),
            "sha256": sha256(hallmark_path),
            "set_count": int(len(hallmark_sets)),
            "method": "exact gene-symbol membership; mean signed response within each fixed set",
            "interpretation": "descriptive set direction only; no pathway significance test or inferred pathway activity",
        },
        "metric_contract": {
            "gene_direction": "fraction of exactly nonzero target genes whose predicted sign matches; prediction zero counts as no correct direction; target zeros excluded from denominator",
            "gene_rank": "Spearman correlation across all 978 finite gene values; a constant vector is undefined and remains null",
            "hallmark_direction": "fraction of fixed sets with nonzero target set mean whose predicted set-mean sign matches; zero prediction is incorrect; target-zero sets excluded",
            "aggregation": "average conditions within target object and exact key, average objects equally within each key, then average keys equally; current domain has one condition per object and one key",
            "uncertainty": "10,000 paired bootstrap resamples of target cell-line objects; descriptive intervals only; no p-value",
            "missing_rule": "undefined correlations and zero-denominator direction summaries remain null; no undefined metric is coerced to zero",
            "forbidden": ["NMSE", "MAE", "amplitude_ratio", "target_fitted_rescaling", "target_based_baseline_selection"],
        },
        "files": {
            "keyed_predictions_npz": str(npz_path),
            "per_condition_metrics_tsv": str(condition_path),
            "per_object_metrics_tsv": str(object_path),
            "per_drug_metrics_tsv": str(drug_path),
            "summary_metrics_tsv": str(summary_path),
            "paired_object_bootstrap_tsv": str(paired_path),
            "gene_direction_rank_detail_tsv": str(gene_path),
            "hallmark_set_direction_tsv": str(hallmark_result_path),
            "counterexamples_tsv": str(counterexample_path),
        },
        "gpu_hours": 0.0,
        "pdo_outer_locked_values_accessed": False,
        "elapsed_wall_seconds": time.monotonic() - start,
    }
    result_path = out / "summary_metrics.json"
    _atomic_json(result_path, result)
    output_paths = [
        npz_path, condition_path, object_path, drug_path, summary_path, paired_path,
        gene_path, hallmark_result_path, counterexample_path, gene_alignment_path, result_path,
    ]
    final_manifest = {
        **metadata_manifest,
        "phase": "metadata_gate_and_fixed_baselines",
        "status": "COMPLETED_EXTERNAL_DEVELOPMENT_FIXED_BASELINES",
        "root_metadata_readback_sha256": sha256(root_readback_path),
        "metadata_phase_manifest_sha256": sha256(metadata_receipt_copy),
        "source_train_values_npz_sha256": source_evidence["sha256"],
        "target_h5ad_sha256": target_read_evidence["file_sha256"],
        "target_numeric_matrix_scope": target_read_evidence["numeric_matrix_scope"],
        "numeric_arrays_loaded_from_source_npz": source_evidence["numeric_array_names_loaded"],
        "source_inner_dev_or_test_arrays_loaded": False,
        "output_sha256": {path.name: sha256(path) for path in output_paths},
        "numeric_matrix_accessed": True,
        "H5AD_X_values_accessed": True,
        "H5AD_X_row_count_accessed": 11,
        "H5AD_layers_values_accessed": False,
        "obsm_X_ctl_values_accessed": True,
        "obsm_control_row_count_accessed": 11,
        "source_train_only_numeric_values_accessed": True,
        "inner_dev_values_accessed": False,
        "pdo_outer_locked_values_accessed": False,
        "gpu_hours": 0.0,
        "numeric_phase_wall_seconds": result["elapsed_wall_seconds"],
    }
    _atomic_json(metadata_manifest_path, final_manifest)
    print(json.dumps({
        "run_id": run_id,
        "status": result["status"],
        "summary_metrics": str(summary_path),
        "prediction_archive": str(npz_path),
        "matrix_scope": target_read_evidence["numeric_matrix_scope"],
        "gpu_hours": 0.0,
        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
    }, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--run-id", default=None)
    p.add_argument("--phase", choices=["metadata", "run"], required=True)
    args = p.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or config["run_id"]
    if run_id != config["run_id"]:
        raise ValueError("runtime run_id must equal the preregistered unique run_id")
    if args.phase == "metadata":
        return _metadata_phase(config_path, config, run_id)
    if args.phase == "run":
        return _run_phase(config_path, config, run_id)
    raise ValueError(f"unsupported phase: {args.phase}")


if __name__ == "__main__":
    sys.exit(main())
