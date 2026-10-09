#!/usr/bin/env python3
"""Paired source-domain cell-line response comparison for EXP-050 section 3.2.

This is a descriptive analysis of the already-registered GSE70138 LINCS data.
It compares three basal breast cell lines with each other and with MCF7 under
the exact four-model, same-compound/dose/time, same-cell/plate/time-DMSO domain
frozen in the parent 3.2 package. It does not read PDO, outer, locked, or
GSE216540 sample-level metadata or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_breast_subtype_pairwise_comparison_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
PARENT = ROOT / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1"
SRC = ROOT / "data/raw/transfer_public_assets/gse_breast_drug_response/GSE70138"
GCTX_GZ = SRC / "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx.gz"
GCTX_TMP = ROOT / "tmp/exp050_public_transfer_intake_v1/GSE70138_Level2.gctx"
GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
IDENTITY = PARENT / "lincs_breast_cell_model_identity.tsv"
COMMON = PARENT / "source_domain_common_exposures.tsv"
PARENT_METRICS = PARENT / "source_domain_measured_comparison_metrics.tsv"
PARENT_RECEIPT = PARENT / "source_domain_comparison_receipt.json"
INPUT_MANIFEST = PARENT / "input_sha256_manifest.tsv"
MODEL_IDS = ["ACH-000536", "ACH-000148", "ACH-000768", "ACH-000019"]
MODEL_LABELS = {
    "ACH-000536": "BT20_basal_A",
    "ACH-000148": "HS578T_basal_B",
    "ACH-000768": "MDAMB231_basal_B",
    "ACH-000019": "MCF7_luminal",
}
MODEL_ORDER = list(MODEL_LABELS)
EXPOSURE_COLS = ["inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key"]
PAIRS = {
    "BT20_vs_HS578T": ("ACH-000536", "ACH-000148"),
    "BT20_vs_MDAMB231": ("ACH-000536", "ACH-000768"),
    "HS578T_vs_MDAMB231": ("ACH-000148", "ACH-000768"),
    "BT20_vs_MCF7": ("ACH-000536", "ACH-000019"),
    "HS578T_vs_MCF7": ("ACH-000148", "ACH-000019"),
    "MDAMB231_vs_MCF7": ("ACH-000768", "ACH-000019"),
}
PAIR_METRICS = [
    "gene_spearman_rho",
    "gene_nonzero_sign_agreement",
    "hallmark_mean_gene_delta_spearman_rho",
    "hallmark_nonzero_sign_agreement",
]
EXPECTED_GCTX_GZ_SHA256 = "aea95e65cdb6522a38e114d8ab024eba737d913531119c1342392f6f436b2b03"
EXPECTED_GCTX_UNCOMPRESSED_SHA256 = "a0ad938a8c5e7714480cae287815834176b479f5807810888d32076c25578464"
KEY_PATTERN = r"[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]"
PARENT_COMMON_EXPOSURES = 1297
PARENT_DISTINCT_KEY_STRINGS = 109
EXPECTED_COMMON_EXPOSURES = 1285
EXPECTED_KEYS = 108
EXPECTED_EXCLUDED_PLACEHOLDER = "RESTRICTED"
EXPECTED_EXCLUDED_EXPOSURES = 12
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261006


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False, suffix=".tmp") as stream:
        temp = Path(stream.name)
        frame.to_csv(stream, sep="\t", index=False, na_rep="")
    os.replace(temp, path)


def norm_num(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(number) else format(float(number), ".12g")


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _registered_inputs() -> dict[str, Path]:
    required = {
        "GSE70138_pert_info": SRC / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
        "GSE70138_cell_info": SRC / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
        "GSE70138_inst_info": SRC / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
        "GSE70138_gene_info": SRC / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
        "GSE70138_Level2_GEX_gctx_gz": GCTX_GZ,
        "fixed_Hallmark_2023_2": GMT,
        "root_verified_identity_map": IDENTITY,
        "parent_exact_exposure_domain": COMMON,
        "parent_measured_metrics": PARENT_METRICS,
        "parent_source_domain_receipt": PARENT_RECEIPT,
        "parent_input_hashes": INPUT_MANIFEST,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Registered or parent-verified inputs missing: {missing}")
    actual = {name: sha256_file(path) for name, path in required.items()}
    if actual["GSE70138_Level2_GEX_gctx_gz"] != EXPECTED_GCTX_GZ_SHA256:
        raise ValueError("Compressed registered GSE70138 Level 2 GCTX hash mismatch")
    return required


def ensure_gctx() -> tuple[str, float]:
    started = time.monotonic()
    if not GCTX_TMP.is_file():
        GCTX_TMP.parent.mkdir(parents=True, exist_ok=True)
        temp = GCTX_TMP.with_suffix(GCTX_TMP.suffix + ".partial")
        with temp.open("wb") as output:
            result = subprocess.run(["gzip", "-dc", str(GCTX_GZ)], stdout=output, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"gzip decompression failed: {result.stderr.decode('utf-8', errors='replace')[-1000:]}")
        os.replace(temp, GCTX_TMP)
    digest = sha256_file(GCTX_TMP)
    if digest != EXPECTED_GCTX_UNCOMPRESSED_SHA256:
        raise ValueError("Uncompressed GSE70138 Level 2 GCTX hash mismatch")
    return digest, time.monotonic() - started


def read_metadata() -> tuple[pd.DataFrame, dict[tuple[str, ...], dict[str, dict[str, Any]]], dict[str, str], list[str], dict[str, Any]]:
    p = _registered_inputs()
    pert = pd.read_csv(p["GSE70138_pert_info"], sep="\t", compression="gzip", dtype=str, low_memory=False)
    cell = pd.read_csv(p["GSE70138_cell_info"], sep="\t", compression="gzip", dtype=str, low_memory=False).drop_duplicates("cell_id")
    inst = pd.read_csv(p["GSE70138_inst_info"], sep="\t", compression="gzip", dtype=str, low_memory=False)
    gene_info = pd.read_csv(p["GSE70138_gene_info"], sep="\t", compression="gzip", dtype=str, low_memory=False)
    identity = pd.read_csv(IDENTITY, sep="\t", dtype=str)
    require_columns(identity, {"dataset", "LINCS_base_cell_id", "DepMap_ModelID", "LegacyMolecularSubtype"}, "root-verified LINCS identity map")
    identity = identity[identity.dataset.eq("GSE70138")].drop_duplicates("LINCS_base_cell_id")
    model_meta = identity.set_index("LINCS_base_cell_id").to_dict("index")
    if set(MODEL_IDS) - set(identity.DepMap_ModelID.dropna()):
        raise ValueError("Frozen four-model source identity map is incomplete")
    breast = cell[cell.primary_site.str.casefold().eq("breast")].copy()
    breast["DepMap_ModelID"] = breast.base_cell_id.map(lambda x: model_meta.get(x, {}).get("DepMap_ModelID"))
    breast = breast[breast.DepMap_ModelID.isin(MODEL_IDS)].copy()
    if set(MODEL_IDS) - set(breast.DepMap_ModelID):
        raise ValueError("At least one frozen cell model is missing from GSE70138 cell metadata")

    pert = pert[["pert_id", "inchi_key"]].drop_duplicates()
    pert["inchi_key"] = pert.inchi_key.astype("string").str.strip().str.upper()
    placeholder_pert = pert.loc[pert.inchi_key.eq(EXPECTED_EXCLUDED_PLACEHOLDER)].copy()
    placeholder_pert_ids = set(placeholder_pert.pert_id.astype(str))
    placeholder_pert_inames = sorted(inst.loc[inst.pert_id.astype(str).isin(placeholder_pert_ids), "pert_iname"].dropna().astype(str).unique())
    # Identity placeholders such as `restricted` are not chemical keys and
    # must not be merged into a pseudo-drug exposure group.
    pert = pert.loc[pert.inchi_key.str.fullmatch(KEY_PATTERN, na=False)].copy()
    treatments = inst.merge(pert, on="pert_id", how="inner", validate="many_to_one")
    treatments = treatments[treatments.pert_type.eq("trt_cp") & treatments.inchi_key.notna() & treatments.inchi_key.str.strip().ne("")].copy()
    treatments = treatments.merge(breast[["cell_id", "base_cell_id", "DepMap_ModelID"]], on="cell_id", how="inner", validate="many_to_one")
    treatments["dose_key"] = treatments.pert_dose.map(norm_num)
    treatments["dose_unit_key"] = treatments.pert_dose_unit.fillna("").str.casefold()
    treatments["time_key"] = treatments.pert_time.map(norm_num)
    treatments["time_unit_key"] = treatments.pert_time_unit.fillna("").str.casefold()

    vehicle = inst[(inst.pert_type.eq("ctl_vehicle")) & inst.pert_iname.fillna("").str.fullmatch("DMSO", case=False)].copy()
    vehicle["time_key"] = vehicle.pert_time.map(norm_num)
    vehicle["time_unit_key"] = vehicle.pert_time_unit.fillna("").str.casefold()
    vehicle_lookup: dict[tuple[str, str, str, str], list[str]] = {}
    for key, group in vehicle.groupby(["cell_id", "det_plate", "time_key", "time_unit_key"], dropna=False, sort=False):
        vehicle_lookup[tuple(map(str, key))] = group.inst_id.astype(str).tolist()

    group_cols = EXPOSURE_COLS + ["DepMap_ModelID"]
    model_groups: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    rejected_groups = 0
    for key, group in treatments.groupby(group_cols, dropna=False, sort=False):
        key = tuple(map(str, key))
        model_id = key[-1]
        plate_meta: list[dict[str, Any]] = []
        eligible = True
        for cell_id, cell_group in group.groupby("cell_id", sort=False):
            for plate, plate_group in cell_group.groupby("det_plate", sort=False):
                control_ids = vehicle_lookup.get((str(cell_id), str(plate), key[3], key[4]), [])
                if not control_ids:
                    eligible = False
                    break
                plate_meta.append({
                    "cell_id": str(cell_id), "plate": str(plate),
                    "treatment_ids": plate_group.inst_id.astype(str).tolist(),
                    "control_ids": control_ids,
                    "n_treatment_instances": int(plate_group.inst_id.nunique()),
                    "n_control_instances": len(control_ids),
                })
            if not eligible:
                break
        if not eligible or not plate_meta:
            rejected_groups += 1
            continue
        exp_key = key[:-1]
        model_groups.setdefault(exp_key, {})[model_id] = {
            "plate_meta": plate_meta,
            "n_plates": len(plate_meta),
            "n_treatment_instances": int(group.inst_id.nunique()),
            "n_control_instances": int(sum(int(r["n_control_instances"]) for r in plate_meta)),
            "pert_iname": sorted(group.pert_iname.dropna().astype(str).unique())[0],
        }
    common_keys = sorted(k for k, models in model_groups.items() if set(MODEL_IDS).issubset(models))
    common_parent = pd.read_csv(COMMON, sep="\t", dtype=str, keep_default_na=False)
    require_columns(common_parent, set(EXPOSURE_COLS + ["pert_iname", "required_four_models_complete"]), "parent source-domain exposure manifest")
    if len(common_parent) != PARENT_COMMON_EXPOSURES or common_parent.inchi_key.nunique() != PARENT_DISTINCT_KEY_STRINGS:
        raise ValueError("Root-verified parent source-domain raw counts changed")
    normalized_parent_key = common_parent.inchi_key.astype("string").str.strip().str.upper()
    valid_parent_mask = normalized_parent_key.str.fullmatch(KEY_PATTERN, na=False)
    excluded_parent = common_parent.loc[~valid_parent_mask].copy()
    excluded_keys = sorted(normalized_parent_key.loc[~valid_parent_mask].dropna().unique().tolist())
    if len(excluded_parent) != EXPECTED_EXCLUDED_EXPOSURES or excluded_keys != [EXPECTED_EXCLUDED_PLACEHOLDER]:
        raise ValueError(f"Unexpected invalid-key rows in parent exposure domain: rows={len(excluded_parent)}, keys={excluded_keys}")
    common = common_parent.loc[valid_parent_mask].copy()
    common["inchi_key"] = normalized_parent_key.loc[valid_parent_mask].astype(str).to_numpy()
    if len(common) != EXPECTED_COMMON_EXPOSURES or common.inchi_key.nunique() != EXPECTED_KEYS:
        raise ValueError("Strict full-InChIKey exposure domain has unexpected counts")
    if len(common_keys) != EXPECTED_COMMON_EXPOSURES:
        raise ValueError(f"Expected {EXPECTED_COMMON_EXPOSURES} strict exact shared exposures; found {len(common_keys)}")
    common_rows = common[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).tolist()
    if len(common_rows) != len(common_keys) or set(common_rows) != set(common_keys) or len(set(common_rows)) != len(common_rows):
        raise ValueError("Recomputed exact exposure domain differs from the root-verified parent 3.2 domain")
    if len(placeholder_pert_ids) != 2 or set(placeholder_pert_inames) != {"THZ-2-98-01", "WH-4-025"}:
        raise ValueError("The registered restricted placeholder no longer resolves to the audited two distinct source compound names")

    gene_info["pr_gene_id"] = gene_info.pr_gene_id.astype(str)
    symbol_map = gene_info.drop_duplicates("pr_gene_id").set_index("pr_gene_id").pr_gene_symbol.astype(str).to_dict()
    hallmark: dict[str, list[int]] = {}
    with GMT.open(encoding="utf-8") as stream:
        for line in stream:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].startswith("HALLMARK_"):
                hallmark[cols[0]] = []
    meta = {
        "common_keys": common_keys,
        "model_groups": model_groups,
        "symbol_map": symbol_map,
        "hallmark_names": sorted(hallmark),
        "rejected_groups_missing_same_cell_plate_time_DMSO": rejected_groups,
        "parent_domain_raw_exposures": len(common_parent),
        "parent_domain_raw_distinct_key_strings": int(common_parent.inchi_key.nunique()),
        "strict_full_key_exposures": len(common),
        "strict_full_key_count": int(common.inchi_key.nunique()),
        "excluded_placeholder_key": EXPECTED_EXCLUDED_PLACEHOLDER,
        "excluded_placeholder_exposure_count": len(excluded_parent),
        "excluded_placeholder_compound_names_from_source_metadata": placeholder_pert_inames,
        "excluded_placeholder_source_pert_ids": sorted(placeholder_pert_ids),
        "excluded_placeholder_source_pert_ids_count": len(placeholder_pert_ids),
        "excluded_placeholder_rows": excluded_parent[["inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key", "pert_iname"]].to_dict("records"),
        "registered_input_paths": {k: str(v.relative_to(ROOT)) if v.is_relative_to(ROOT) else str(v) for k, v in p.items()},
        "registered_input_sha256": {k: sha256_file(v) for k, v in p.items()},
    }
    return common, model_groups, symbol_map, sorted(hallmark), meta


def build_axes_and_hallmarks(gene_info_symbols: dict[str, str], hallmark_names: list[str], h5: h5py.File) -> tuple[list[str], dict[str, np.ndarray]]:
    gene_ids = [x.decode("utf-8") for x in h5["/0/META/ROW/id"][:]]
    if h5["/0/DATA/0/matrix"].shape != (345976, 978) or len(gene_ids) != 978 or len(set(gene_ids)) != 978:
        raise ValueError(f"Unexpected GSE70138 GCTX axes: {h5['/0/DATA/0/matrix'].shape}, {len(gene_ids)}")
    symbols = [gene_info_symbols.get(gene, "") for gene in gene_ids]
    if any(not gene for gene in symbols) or len(set(symbols)) != 978:
        raise ValueError("GSE70138 landmark genes do not map one-to-one to symbols")
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    gmt_sets: dict[str, set[str]] = {}
    with GMT.open(encoding="utf-8") as stream:
        for line in stream:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].startswith("HALLMARK_"):
                gmt_sets[cols[0]] = set(cols[2:])
    if sorted(gmt_sets) != hallmark_names or len(gmt_sets) != 50:
        raise ValueError("Fixed MSigDB Hallmark gene-set axis differs from the parent 3.2 source analysis")
    indices = {name: np.asarray(sorted({symbol_index[s] for s in gmt_sets[name] if s in symbol_index}), dtype=int) for name in hallmark_names}
    empty = [name for name, idx in indices.items() if idx.size == 0]
    if empty:
        raise ValueError(f"Fixed Hallmark sets with no measured genes: {empty}")
    return symbols, indices


def compute_vectors(common: pd.DataFrame, model_groups: dict[tuple[str, ...], dict[str, dict[str, Any]]], symbol_map: dict[str, str], hallmark_names: list[str], meta: dict[str, Any], phase: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    started = time.monotonic()
    exposures = meta["common_keys"] if phase == "full" else [meta["common_keys"][0]]
    needed_ids: set[str] = set()
    profile_per_exposure: dict[tuple[str, ...], set[str]] = {}
    for exp in exposures:
        ids: set[str] = set()
        for model_id in MODEL_IDS:
            for plate in model_groups[exp][model_id]["plate_meta"]:
                ids.update(plate["treatment_ids"])
                ids.update(plate["control_ids"])
        profile_per_exposure[exp] = ids
        needed_ids.update(ids)
    response_rows = []
    vector_rows: list[np.ndarray] = []
    with h5py.File(GCTX_TMP, "r") as h5:
        symbols, hallmark_idx = build_axes_and_hallmarks(symbol_map, hallmark_names, h5)
        matrix = h5["/0/DATA/0/matrix"]
        inst_ids = [x.decode("utf-8") for x in h5["/0/META/COL/id"][:]]
        if len(inst_ids) != 345976 or len(set(inst_ids)) != len(inst_ids):
            raise ValueError("GSE70138 GCTX instance axis is not unique or has unexpected length")
        index = {identifier: i for i, identifier in enumerate(inst_ids)}
        missing = sorted(needed_ids - set(index))
        if missing:
            raise ValueError(f"{len(missing)} selected source profiles absent from the registered GCTX")
        for exp in exposures:
            exp_info = dict(zip(EXPOSURE_COLS, exp))
            ids = profile_per_exposure[exp]
            ordered = sorted((index[i], i) for i in ids)
            profile_names = [x[1] for x in ordered]
            values = np.asarray(matrix[[x[0] for x in ordered], :], dtype=np.float64)
            if values.shape != (len(profile_names), 978) or not np.isfinite(values).all():
                raise ValueError("Selected source profile block has an invalid shape or non-finite value")
            profile = {name: values[i] for i, name in enumerate(profile_names)}
            response = {}
            for model_id in MODEL_IDS:
                plate_deltas = []
                for plate in model_groups[exp][model_id]["plate_meta"]:
                    treated = np.mean(np.vstack([profile[x] for x in plate["treatment_ids"]]), axis=0)
                    control = np.mean(np.vstack([profile[x] for x in plate["control_ids"]]), axis=0)
                    delta = treated - control
                    if not np.isfinite(delta).all():
                        raise ValueError(f"Non-finite delta for {exp} / {model_id}")
                    plate_deltas.append(delta)
                response[model_id] = np.mean(np.vstack(plate_deltas), axis=0)
            vector_rows.append(np.stack([response[mid] for mid in MODEL_IDS], axis=0))
            exposure_mask = common[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).map(lambda value: value == exp)
            base = common.loc[exposure_mask]
            if len(base) != 1:
                raise ValueError("Exact common exposure manifest is not one-to-one")
            for model_id in MODEL_IDS:
                detail = model_groups[exp][model_id]
                response_rows.append({
                    **exp_info,
                    "pert_iname": str(base.iloc[0]["pert_iname"]),
                    "DepMap_ModelID": model_id,
                    "cell_origin_label": MODEL_LABELS[model_id],
                    "n_source_plates": detail["n_plates"],
                    "n_treatment_instances": detail["n_treatment_instances"],
                    "n_same_cell_plate_time_DMSO_instances": detail["n_control_instances"],
                    "hallmark_nonempty_sets": len(hallmark_idx),
                })
    vectors = np.stack(vector_rows, axis=0)
    if vectors.shape != (len(exposures), 4, 978) or not np.isfinite(vectors).all():
        raise ValueError(f"Response vector tensor has unexpected shape/values: {vectors.shape}")
    stats = {
        "phase": phase,
        "n_exposures_processed": len(exposures),
        "n_unique_exact_keys_processed": len({e[0] for e in exposures}),
        "selected_profile_ids": len(needed_ids),
        "sum_profiles_across_exposures_if_no_reuse": int(sum(len(v) for v in profile_per_exposure.values())),
        "elapsed_vector_read_and_delta_seconds": round(time.monotonic() - started, 3),
        "shape": list(vectors.shape),
        "hallmark_set_count": len(hallmark_idx),
        "measured_gene_count": len(symbols),
    }
    return pd.DataFrame(response_rows), vectors, stats


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    result = float(spearmanr(left, right).statistic)
    return result if np.isfinite(result) else None


def sign_agreement(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    valid = np.isfinite(left) & np.isfinite(right) & (left != 0) & (right != 0)
    n = int(valid.sum())
    return (float((np.sign(left[valid]) == np.sign(right[valid])).mean()) if n else None, n)


def vector_metrics(left: np.ndarray, right: np.ndarray, hallmark_idx: dict[str, np.ndarray]) -> dict[str, Any]:
    agree, n_genes = sign_agreement(left, right)
    left_h = np.asarray([np.mean(left[hallmark_idx[name]]) for name in sorted(hallmark_idx)], dtype=float)
    right_h = np.asarray([np.mean(right[hallmark_idx[name]]) for name in sorted(hallmark_idx)], dtype=float)
    hagree, n_h = sign_agreement(left_h, right_h)
    return {
        "gene_spearman_rho": spearman(left, right),
        "gene_nonzero_sign_agreement": agree,
        "n_genes_both_nonzero": n_genes,
        "hallmark_mean_gene_delta_spearman_rho": spearman(left_h, right_h),
        "hallmark_nonzero_sign_agreement": hagree,
        "n_Hallmark_sets_both_nonzero": n_h,
        "n_Hallmark_sets_measured": len(hallmark_idx),
    }


def build_metrics(vectors: np.ndarray, common: pd.DataFrame, hallmark_idx: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = common.set_index(EXPOSURE_COLS, drop=False)
    condition_rows = []
    hallmark_rows = []
    for i, exposure in enumerate(sorted(common.index.tolist())):
        vec = {mid: vectors[i, j] for j, mid in enumerate(MODEL_ORDER)}
        base = common.loc[exposure]
        if isinstance(base, pd.DataFrame):
            raise ValueError("Parent common exposure table contains a duplicate key")
        row = {col: str(base[col]) for col in EXPOSURE_COLS}
        row.update({"pert_iname": str(base["pert_iname"]), "n_source_models": 4, "source_domain_only": True})
        for pair_name, (left_id, right_id) in PAIRS.items():
            metrics = vector_metrics(vec[left_id], vec[right_id], hallmark_idx)
            row[f"{pair_name}_gene_spearman_rho"] = metrics["gene_spearman_rho"]
            row[f"{pair_name}_gene_nonzero_sign_agreement"] = metrics["gene_nonzero_sign_agreement"]
            row[f"{pair_name}_n_genes_both_nonzero"] = metrics["n_genes_both_nonzero"]
            row[f"{pair_name}_hallmark_spearman_rho"] = metrics["hallmark_mean_gene_delta_spearman_rho"]
            row[f"{pair_name}_hallmark_nonzero_sign_agreement"] = metrics["hallmark_nonzero_sign_agreement"]
            row[f"{pair_name}_n_hallmarks_both_nonzero"] = metrics["n_Hallmark_sets_both_nonzero"]
        internal_rhos = [row[f"{p}_gene_spearman_rho"] for p in ["BT20_vs_HS578T", "BT20_vs_MDAMB231", "HS578T_vs_MDAMB231"]]
        external_rhos = [row[f"{p}_gene_spearman_rho"] for p in ["BT20_vs_MCF7", "HS578T_vs_MCF7", "MDAMB231_vs_MCF7"]]
        internal_signs = [row[f"{p}_gene_nonzero_sign_agreement"] for p in ["BT20_vs_HS578T", "BT20_vs_MDAMB231", "HS578T_vs_MDAMB231"]]
        external_signs = [row[f"{p}_gene_nonzero_sign_agreement"] for p in ["BT20_vs_MCF7", "HS578T_vs_MCF7", "MDAMB231_vs_MCF7"]]
        row["basal_internal_mean_gene_rho"] = float(np.mean([x for x in internal_rhos if x is not None])) if any(x is not None for x in internal_rhos) else None
        row["basal_vs_mcf7_mean_gene_rho"] = float(np.mean([x for x in external_rhos if x is not None])) if any(x is not None for x in external_rhos) else None
        row["basal_internal_minus_basal_mcf7_gene_rho"] = (row["basal_internal_mean_gene_rho"] - row["basal_vs_mcf7_mean_gene_rho"]) if row["basal_internal_mean_gene_rho"] is not None and row["basal_vs_mcf7_mean_gene_rho"] is not None else None
        row["basal_internal_mean_gene_sign_agreement"] = float(np.mean([x for x in internal_signs if x is not None])) if any(x is not None for x in internal_signs) else None
        row["basal_vs_mcf7_mean_gene_sign_agreement"] = float(np.mean([x for x in external_signs if x is not None])) if any(x is not None for x in external_signs) else None
        row["basal_internal_minus_basal_mcf7_gene_sign_agreement"] = (row["basal_internal_mean_gene_sign_agreement"] - row["basal_vs_mcf7_mean_gene_sign_agreement"]) if row["basal_internal_mean_gene_sign_agreement"] is not None and row["basal_vs_mcf7_mean_gene_sign_agreement"] is not None else None
        for metric_suffix in ["hallmark_spearman_rho", "hallmark_nonzero_sign_agreement"]:
            i_vals = [row[f"{p}_{metric_suffix}"] for p in ["BT20_vs_HS578T", "BT20_vs_MDAMB231", "HS578T_vs_MDAMB231"]]
            e_vals = [row[f"{p}_{metric_suffix}"] for p in ["BT20_vs_MCF7", "HS578T_vs_MCF7", "MDAMB231_vs_MCF7"]]
            im = float(np.mean([x for x in i_vals if x is not None])) if any(x is not None for x in i_vals) else None
            em = float(np.mean([x for x in e_vals if x is not None])) if any(x is not None for x in e_vals) else None
            row[f"basal_internal_mean_{metric_suffix}"] = im
            row[f"basal_vs_mcf7_mean_{metric_suffix}"] = em
            row[f"basal_internal_minus_basal_mcf7_{metric_suffix}"] = (im - em) if im is not None and em is not None else None
        for metric_suffix in PAIR_METRICS:
            if metric_suffix == "gene_spearman_rho":
                hs_mda_col = "HS578T_vs_MDAMB231_gene_spearman_rho"
                own_cols = ["HS578T_vs_MCF7_gene_spearman_rho", "MDAMB231_vs_MCF7_gene_spearman_rho"]
            elif metric_suffix == "gene_nonzero_sign_agreement":
                hs_mda_col = "HS578T_vs_MDAMB231_gene_nonzero_sign_agreement"
                own_cols = ["HS578T_vs_MCF7_gene_nonzero_sign_agreement", "MDAMB231_vs_MCF7_gene_nonzero_sign_agreement"]
            elif metric_suffix == "hallmark_mean_gene_delta_spearman_rho":
                hs_mda_col = "HS578T_vs_MDAMB231_hallmark_spearman_rho"
                own_cols = ["HS578T_vs_MCF7_hallmark_spearman_rho", "MDAMB231_vs_MCF7_hallmark_spearman_rho"]
            else:
                hs_mda_col = "HS578T_vs_MDAMB231_hallmark_nonzero_sign_agreement"
                own_cols = ["HS578T_vs_MCF7_hallmark_nonzero_sign_agreement", "MDAMB231_vs_MCF7_hallmark_nonzero_sign_agreement"]
            own_vals = [row[c] for c in own_cols if row[c] is not None]
            own_mean = float(np.mean(own_vals)) if own_vals else None
            delta = row[hs_mda_col] - own_mean if row[hs_mda_col] is not None and own_mean is not None else None
            row[f"HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__{metric_suffix}"] = delta
        # Hallmark directions are computed at the exposure level from the fixed set axis.
        for name in sorted(hallmark_idx):
            scores = {mid: float(np.mean(vec[mid][hallmark_idx[name]])) for mid in MODEL_ORDER}
            hs, mda, mcf7 = scores["ACH-000148"], scores["ACH-000768"], scores["ACH-000019"]
            valid = hs != 0 and mda != 0 and mcf7 != 0
            retain = bool(np.sign(hs) == np.sign(mda)) if hs != 0 and mda != 0 else None
            both_reverse = bool(np.sign(hs) == np.sign(mda) and np.sign(mcf7) == -np.sign(hs)) if valid else None
            internal_reverse = bool(np.sign(hs) != np.sign(mda)) if hs != 0 and mda != 0 else None
            hallmark_rows.append({
                **{c: str(base[c]) for c in EXPOSURE_COLS},
                "hallmark": name,
                "HS578T_delta": hs,
                "MDAMB231_delta": mda,
                "MCF7_delta": mcf7,
                "basal_B_both_nonzero": bool(hs != 0 and mda != 0),
                "basal_B_sign_retained": retain,
                "both_basal_B_same_direction_reversed_vs_MCF7": both_reverse,
                "basal_B_internal_sign_reversed": internal_reverse,
            })
        condition_rows.append(row)
    conditions = pd.DataFrame(condition_rows)
    # The vectors were emitted in sorted exact exposure order. Guard against pandas key-order drift.
    expected = sorted(common.index.tolist())
    observed = conditions[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).tolist()
    if observed != expected:
        raise ValueError("Metric exposure order differs from response vector order")
    condition_metrics = conditions

    raw_h = pd.DataFrame(hallmark_rows)
    key_h_rows = []
    for (key, hallmark), group in raw_h.groupby(["inchi_key", "hallmark"], sort=True):
        pair = group.loc[group.basal_B_both_nonzero]
        both = group.dropna(subset=["both_basal_B_same_direction_reversed_vs_MCF7"])
        key_h_rows.append({
            "inchi_key": key,
            "hallmark": hallmark,
            "n_exposures_total_for_key": len(group),
            "n_exposures_basal_B_both_nonzero": len(pair),
            "basal_B_sign_retention_rate_within_key": float(pair.basal_B_sign_retained.mean()) if len(pair) else None,
            "n_exposures_all_three_models_nonzero": len(both),
            "both_basal_B_same_direction_reversed_vs_MCF7_rate_within_key": float(both.both_basal_B_same_direction_reversed_vs_MCF7.mean()) if len(both) else None,
            "basal_B_internal_sign_reversal_rate_within_key": float(pair.basal_B_internal_sign_reversed.mean()) if len(pair) else None,
        })
    key_h = pd.DataFrame(key_h_rows)
    hall_summary = []
    for hallmark, group in key_h.groupby("hallmark", sort=True):
        support = group.dropna(subset=["basal_B_sign_retention_rate_within_key"])
        reverse_support = group.dropna(subset=["both_basal_B_same_direction_reversed_vs_MCF7_rate_within_key"])
        hall_summary.append({
            "hallmark": hallmark,
            "n_exact_InChIKeys_total": EXPECTED_KEYS,
            "n_keys_with_HS578T_MDAMB231_nonzero_support": len(support),
            "n_exposure_pairs_HS578T_MDAMB231_both_nonzero": int(group.n_exposures_basal_B_both_nonzero.sum()),
            "equal_key_mean_BasalB_sign_retention": float(support.basal_B_sign_retention_rate_within_key.mean()) if len(support) else None,
            "supporting_keys_BasalB_sign_retention": ";".join(support.sort_values("basal_B_sign_retention_rate_within_key", ascending=False).inchi_key.astype(str).tolist()),
            "n_keys_with_all_three_lines_nonzero_support": len(reverse_support),
            "n_exposure_triples_all_three_nonzero": int(group.n_exposures_all_three_models_nonzero.sum()),
            "equal_key_mean_both_BasalB_same_direction_reversed_vs_MCF7": float(reverse_support.both_basal_B_same_direction_reversed_vs_MCF7_rate_within_key.mean()) if len(reverse_support) else None,
            "supporting_keys_both_BasalB_reversed_vs_MCF7": ";".join(reverse_support.sort_values("both_basal_B_same_direction_reversed_vs_MCF7_rate_within_key", ascending=False).inchi_key.astype(str).tolist()),
            "n_keys_with_internal_reversal_support": len(support),
            "n_exposure_pairs_internal_sign_reversed": int((raw_h.loc[raw_h.hallmark.eq(hallmark), "basal_B_internal_sign_reversed"] == True).sum()),
            "equal_key_mean_BasalB_internal_sign_reversal": float(support.basal_B_internal_sign_reversal_rate_within_key.mean()) if len(support) else None,
        })
    return condition_metrics, key_h, pd.DataFrame(hall_summary)


def drug_equal_weight(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in condition_metrics.columns if c not in EXPOSURE_COLS + ["pert_iname", "n_source_models", "source_domain_only"]]
    rows = []
    for key, group in condition_metrics.groupby("inchi_key", sort=True):
        row = {
            "inchi_key": key,
            "pert_iname": ";".join(sorted(group.pert_iname.astype(str).unique())),
            "n_exact_dose_time_exposures": len(group),
            "aggregation": "equal mean across eligible exact dose/time exposures within full InChIKey",
        }
        for col in numeric:
            vals = pd.to_numeric(group[col], errors="coerce").dropna().to_numpy(dtype=float)
            row[col] = float(vals.mean()) if len(vals) else None
            if col.endswith("gene_spearman_rho") or col.endswith("gene_nonzero_sign_agreement"):
                row[f"n_exposures_valid__{col}"] = len(vals)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_KEYS or frame.inchi_key.nunique() != EXPECTED_KEYS:
        raise ValueError("Equal-key result does not contain the frozen 109 unique full chemical keys")
    return frame


def bootstrap_summary(drug_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        *[f"{pair}_{suffix}" for pair in PAIRS for suffix in ["gene_spearman_rho", "gene_nonzero_sign_agreement", "hallmark_spearman_rho", "hallmark_nonzero_sign_agreement"]],
        "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho",
        "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement",
        "basal_internal_mean_hallmark_spearman_rho", "basal_vs_mcf7_mean_hallmark_spearman_rho",
        "basal_internal_mean_hallmark_nonzero_sign_agreement", "basal_vs_mcf7_mean_hallmark_nonzero_sign_agreement",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_spearman_rho",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_nonzero_sign_agreement",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_mean_gene_delta_spearman_rho",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_nonzero_sign_agreement",
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for metric in metrics:
        if metric not in drug_metrics.columns:
            continue
        vals = pd.to_numeric(drug_metrics[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals):
            draw_idx = rng.integers(0, len(vals), size=(BOOTSTRAP_N, len(vals)))
            boot = vals[draw_idx].mean(axis=1)
            means.append({
                "comparison": metric.rsplit("_", 3)[0] if metric.startswith(tuple(PAIRS)) else metric,
                "metric": metric,
                "estimate_equal_key_mean": float(vals.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "n_full_InChIKeys": int(len(vals)),
                "bootstrap_replicates": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
                "interpretation": "descriptive resampling interval for the observed chemical-key domain; not a patient/subtype population interval",
            })
    contrasts = []
    contrast_specs = [
        ("basal_internal_minus_basal_vs_mcf7", "gene_spearman_rho", "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho"),
        ("basal_internal_minus_basal_vs_mcf7", "gene_nonzero_sign_agreement", "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement"),
        ("basal_internal_minus_basal_vs_mcf7", "hallmark_spearman_rho", "basal_internal_mean_hallmark_spearman_rho", "basal_vs_mcf7_mean_hallmark_spearman_rho"),
        ("basal_internal_minus_basal_vs_mcf7", "hallmark_nonzero_sign_agreement", "basal_internal_mean_hallmark_nonzero_sign_agreement", "basal_vs_mcf7_mean_hallmark_nonzero_sign_agreement"),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "gene_spearman_rho", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_spearman_rho", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "gene_nonzero_sign_agreement", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_nonzero_sign_agreement", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "hallmark_spearman_rho", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_mean_gene_delta_spearman_rho", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "hallmark_nonzero_sign_agreement", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_nonzero_sign_agreement", None),
    ]
    for label, metric_name, left_col, right_col in contrast_specs:
        left = pd.to_numeric(drug_metrics[left_col], errors="coerce")
        if right_col is None:
            vals = left.dropna().to_numpy(dtype=float)
        else:
            right = pd.to_numeric(drug_metrics[right_col], errors="coerce")
            valid = left.notna() & right.notna()
            vals = (left[valid] - right[valid]).to_numpy(dtype=float)
        if len(vals):
            draw_idx = rng.integers(0, len(vals), size=(BOOTSTRAP_N, len(vals)))
            boot = vals[draw_idx].mean(axis=1)
            contrasts.append({
                "comparison": label,
                "metric": metric_name,
                "estimate_equal_key_paired_difference": float(vals.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "n_paired_full_InChIKeys": int(len(vals)),
                "bootstrap_replicates": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
                "interpretation": "paired descriptive resampling interval over observed full chemical keys; not a patient/subtype population interval",
            })
    return pd.DataFrame(means), pd.DataFrame(contrasts)


def make_counterexamples(drug_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIRS:
        rho = f"{pair}_gene_spearman_rho"
        agreement_col = f"{pair}_gene_nonzero_sign_agreement"
        for metric in [rho, agreement_col]:
            subset = drug_metrics.dropna(subset=[metric]).sort_values(metric, ascending=True, kind="mergesort").head(5)
            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append({
                    "comparison": pair,
                    "rank_lowest": rank,
                    "metric": metric,
                    "estimate": float(row[metric]),
                    "inchi_key": row["inchi_key"],
                    "pert_iname": row["pert_iname"],
                    "n_exact_dose_time_exposures": int(row["n_exact_dose_time_exposures"]),
                    "interpretation": "low-similarity source-domain counterexample, equal-weighted within exact InChIKey",
                })
    return pd.DataFrame(rows)


def hallmark_indices(symbol_map: dict[str, str], names: list[str], gene_ids: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    symbols = [symbol_map.get(g, "") for g in gene_ids]
    if len(symbols) != 978 or any(not s for s in symbols) or len(set(symbols)) != 978:
        raise ValueError("Gene IDs do not map one-to-one to the frozen 978 landmark symbols")
    symbol_index = {s: i for i, s in enumerate(symbols)}
    sets = {}
    with GMT.open(encoding="utf-8") as stream:
        for line in stream:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].startswith("HALLMARK_"):
                sets[cols[0]] = set(cols[2:])
    if sorted(sets) != names or len(sets) != 50:
        raise ValueError("Hallmark names differ from fixed 50-set axis")
    indices = {name: np.asarray(sorted({symbol_index[s] for s in sets[name] if s in symbol_index}), dtype=int) for name in names}
    if any(not a.size for a in indices.values()):
        raise ValueError("At least one Hallmark set has no measured genes")
    return symbols, indices


def smoke_or_full(phase: str) -> None:
    wall_started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_dir = OUT / "smoke"
    full_dir = OUT / "full"
    target_dir = smoke_dir if phase == "smoke" else full_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    input_paths = _registered_inputs()
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    gctx_hash, decompression_seconds = ensure_gctx()
    common, model_groups, symbol_map, hallmark_names, meta = read_metadata()
    compute_start = time.monotonic()
    response_coverage, vectors, vector_stats = compute_vectors(common, model_groups, symbol_map, hallmark_names, meta, phase)
    common_subset = common.copy()
    if phase == "smoke":
        first = meta["common_keys"][0]
        exposure_mask = common[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).map(lambda value: value == first)
        common_subset = common[exposure_mask].copy()
    else:
        common_subset = common.sort_values(EXPOSURE_COLS, kind="mergesort").reset_index(drop=True)
    # Load the exact GCTX gene axis and fixed Hallmark map after vector calculation.
    with h5py.File(GCTX_TMP, "r") as h5:
        gene_ids = [x.decode("utf-8") for x in h5["/0/META/ROW/id"][:]]
    gene_symbols, hallmark_idx = hallmark_indices(symbol_map, hallmark_names, gene_ids)
    condition_metrics, key_hallmark, hall_summary = build_metrics(vectors, common_subset, hallmark_idx)
    if phase == "smoke":
        if len(condition_metrics) != 1 or response_coverage.DepMap_ModelID.nunique() != 4:
            raise ValueError("Single-exposure smoke did not include one exact exposure across all four frozen source lines")
        condition_metrics.insert(0, "smoke_only", True)
        write_tsv(target_dir / "smoke_metrics.tsv", condition_metrics)
        write_tsv(target_dir / "smoke_source_response_coverage.tsv", response_coverage)
        np.savez_compressed(
            target_dir / "one_exposure_vectors.npz",
            exposure=np.asarray([str(x) for x in meta["common_keys"][0]], dtype=str),
            model_ids=np.asarray(MODEL_ORDER, dtype=str),
            gene_symbols=np.asarray(gene_symbols, dtype=str),
            delta=vectors.astype(np.float32),
        )
        readback = pd.read_csv(target_dir / "smoke_metrics.tsv", sep="\t")
        arr = np.load(target_dir / "one_exposure_vectors.npz", allow_pickle=False)
        if len(readback) != 1 or arr["delta"].shape != (1, 4, 978) or not np.isfinite(arr["delta"]).all() or len(arr["gene_symbols"]) != 978:
            raise ValueError("Smoke output readback failed row, axis, shape, or finite checks")
        total_seconds = time.monotonic() - wall_started
        all_profile_ids = sum(len(set().union(*(set(plate["treatment_ids"] + plate["control_ids"]) for mid in MODEL_IDS for plate in model_groups[k][mid]["plate_meta"]))) for k in meta["common_keys"])
        smoke_profile_ids = vector_stats["sum_profiles_across_exposures_if_no_reuse"]
        profile_ratio = all_profile_ids / max(1, smoke_profile_ids)
        measured_per_exposure = vector_stats["elapsed_vector_read_and_delta_seconds"]
        parent_runtime = float(json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))["runtime_seconds"])
        # The prior source run covered 1297 raw exposure rows; 12 placeholder-key
        # rows are now excluded, leaving 1285 exact full-key exposures. Allow a
        # ten-fold multiplier for the additional pairwise,
        # pathway-direction, equal-key and bootstrap summaries, plus the
        # measured one-exposure/profile-volume projection from this smoke.
        projected = max(
            parent_runtime * 10.0 + decompression_seconds,
            total_seconds + measured_per_exposure * max(0.0, profile_ratio - 1.0),
        ) * 1.2
        receipt = {
            "status": "SMOKE_PASS",
            "run_id": RUN_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_scope": "GSE70138 LINCS source-domain only; no PDO, outer, locked, or GSE216540 sample metadata",
            "smoke_exposure": dict(zip(EXPOSURE_COLS, meta["common_keys"][0])),
            "source_models": MODEL_LABELS,
            "parent_raw_domain": {"exposures": meta["parent_domain_raw_exposures"], "distinct_key_strings": meta["parent_domain_raw_distinct_key_strings"]},
            "expected_strict_domain": {"exposures": EXPECTED_COMMON_EXPOSURES, "exact_InChIKeys": EXPECTED_KEYS, "measured_genes": 978, "Hallmark_sets": 50},
            "excluded_placeholder_identity": {"key_string": meta["excluded_placeholder_key"], "exposure_rows": meta["excluded_placeholder_exposure_count"], "source_pert_ids": meta["excluded_placeholder_source_pert_ids"], "source_compound_names": meta["excluded_placeholder_compound_names_from_source_metadata"]},
            "observed_vector_shape": list(vectors.shape),
            "observed_metric_rows": int(len(condition_metrics)),
            "profile_count_smoke": int(smoke_profile_ids),
            "profile_count_full_sum_by_exposure": int(all_profile_ids),
            "parent_full_domain_runtime_seconds": parent_runtime,
            "full_to_smoke_profile_count_ratio": round(profile_ratio, 3),
            "decompression_seconds": round(decompression_seconds, 3),
            "metadata_and_domain_build_seconds": round(vector_stats["elapsed_vector_read_and_delta_seconds"] * 0 + (compute_start - wall_started), 3),
            "smoke_vector_read_delta_seconds": vector_stats["elapsed_vector_read_and_delta_seconds"],
            "smoke_total_wall_seconds": round(total_seconds, 3),
            "projected_full_wall_seconds_with_20pct_margin": round(projected, 3),
            "projected_full_cpu_minutes_with_20pct_margin": round(projected / 60.0, 3),
            "cpu_cap_minutes": 90,
            "gpu_hours": 0,
            "readback": "PASS: strict complete-key domain, audited placeholder exclusions, exact exposure, four source identities, 978 gene axis, 50 Hallmarks, finite arrays and one-row TSV",
            "input_sha256": input_hashes,
            "uncompressed_GCTX_sha256": gctx_hash,
            "vector_stats": vector_stats,
        }
        write_tsv(target_dir / "excluded_placeholder_key_summary.tsv", pd.DataFrame([{
            "parent_raw_exposures": meta["parent_domain_raw_exposures"],
            "parent_raw_distinct_key_strings": meta["parent_domain_raw_distinct_key_strings"],
            "excluded_placeholder_key": meta["excluded_placeholder_key"],
            "excluded_placeholder_exposures": meta["excluded_placeholder_exposure_count"],
            "excluded_pert_ids": ";".join(meta["excluded_placeholder_source_pert_ids"]),
            "source_compound_names": ";".join(meta["excluded_placeholder_compound_names_from_source_metadata"]),
            "strict_exposures": meta["strict_full_key_exposures"],
            "strict_full_InChIKeys": meta["strict_full_key_count"],
        }]))
        write_tsv(target_dir / "excluded_placeholder_key_rows.tsv", pd.DataFrame(meta["excluded_placeholder_rows"]))
        excluded_summary_rb = pd.read_csv(target_dir / "excluded_placeholder_key_summary.tsv", sep="\t", dtype=str)
        excluded_rows_rb = pd.read_csv(target_dir / "excluded_placeholder_key_rows.tsv", sep="\t", dtype=str)
        if len(excluded_summary_rb) != 1 or int(excluded_summary_rb.iloc[0].strict_exposures) != EXPECTED_COMMON_EXPOSURES or int(excluded_summary_rb.iloc[0].strict_full_InChIKeys) != EXPECTED_KEYS or len(excluded_rows_rb) != EXPECTED_EXCLUDED_EXPOSURES or not set(excluded_rows_rb.pert_iname.astype(str)).issubset({"THZ-2-98-01", "WH-4-025"}) or set(excluded_summary_rb.iloc[0].source_compound_names.split(";")) != {"THZ-2-98-01", "WH-4-025"}:
            raise ValueError("Smoke readback of invalid placeholder-key exclusion audit failed")
        receipt["status"] = "SMOKE_PASS_COST_GATE_PASS" if projected <= 90 * 60 else "SMOKE_PASS_COST_GATE_FAIL_NO_FULL_RUN"
        atomic_json(target_dir / "smoke_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        if projected > 90 * 60:
            raise SystemExit(2)
        return

    if vectors.shape[0] != EXPECTED_COMMON_EXPOSURES:
        raise ValueError("Full run does not cover all frozen common exposures")
    np.savez_compressed(
        full_dir / "condition_vectors.npz",
        exposure_keys=np.asarray([[str(v) for v in e] for e in meta["common_keys"]], dtype=str),
        model_ids=np.asarray(MODEL_ORDER, dtype=str),
        gene_symbols=np.asarray(gene_symbols, dtype=str),
        delta=vectors.astype(np.float32),
    )
    write_tsv(full_dir / "condition_metrics.tsv", condition_metrics)
    write_tsv(full_dir / "drug_key_hallmark_directions.tsv", key_hallmark)
    write_tsv(full_dir / "hallmark_direction_summary.tsv", hall_summary)
    drug_metrics = drug_equal_weight(condition_metrics)
    write_tsv(full_dir / "drug_equal_weight_metrics.tsv", drug_metrics)
    mean_ci, paired_ci = bootstrap_summary(drug_metrics)
    write_tsv(full_dir / "bootstrap_equal_key_means.tsv", mean_ci)
    write_tsv(full_dir / "paired_bootstrap_ci.tsv", paired_ci)
    write_tsv(full_dir / "drug_counterexamples.tsv", make_counterexamples(drug_metrics))

    # Root-style readback within a fresh file read: unique keys, row domain, matrix axes, finite values.
    condition_rb = pd.read_csv(full_dir / "condition_metrics.tsv", sep="\t", dtype={"inchi_key": str})
    vector_rb = np.load(full_dir / "condition_vectors.npz", allow_pickle=False)
    parent_raw_rb = pd.read_csv(PARENT_METRICS, sep="\t", dtype={"inchi_key": str}).rename(columns={"dose": "dose_key", "dose_unit": "dose_unit_key", "time": "time_key", "time_unit": "time_unit_key"})
    parent_raw_rb["inchi_key"] = parent_raw_rb.inchi_key.astype("string").str.strip().str.upper()
    parent_valid_mask = parent_raw_rb.inchi_key.str.fullmatch(KEY_PATTERN, na=False)
    parent_rb = parent_raw_rb.loc[parent_valid_mask].copy()
    parent_rb = parent_rb.sort_values(EXPOSURE_COLS, kind="mergesort").reset_index(drop=True)
    condition_rb = condition_rb.sort_values(EXPOSURE_COLS, kind="mergesort").reset_index(drop=True)
    exposure_rb = vector_rb["exposure_keys"].astype(str)
    if len(parent_raw_rb) != PARENT_COMMON_EXPOSURES or len(parent_rb) != EXPECTED_COMMON_EXPOSURES or int(parent_raw_rb.inchi_key.nunique()) != PARENT_DISTINCT_KEY_STRINGS:
        raise ValueError("Parent raw/strict condition row counts differ from audited 1297/1285 domain")
    if len(condition_rb) != EXPECTED_COMMON_EXPOSURES or int(condition_rb.inchi_key.nunique()) != EXPECTED_KEYS:
        raise ValueError("New exact condition row counts differ from strict 1285/108 domain")
    if condition_rb[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).tolist() != parent_rb[EXPOSURE_COLS].astype(str).apply(tuple, axis=1).tolist():
        raise ValueError("Parent and new exact full-key condition domains differ")
    if vector_rb["delta"].shape != (EXPECTED_COMMON_EXPOSURES, 4, 978) or not np.isfinite(vector_rb["delta"]).all():
        raise ValueError("Saved response vector axis or values failed readback")
    if len(vector_rb["gene_symbols"]) != 978 or len(set(vector_rb["gene_symbols"].astype(str))) != 978:
        raise ValueError("Saved gene axis is not unique at 978 genes")
    if len(hall_summary) != 50 or len(hall_summary.hallmark.unique()) != 50:
        raise ValueError("Hallmark direction summary did not retain the fixed 50-set axis")
    max_baseline_abs_diff = 0.0
    for label, pair_prefix, base_prefix in [
        ("BT20_basal_A", "BT20_vs_MCF7", "BT20_basal_A_vs_MCF7"),
        ("HS578T_basal_B", "HS578T_vs_MCF7", "HS578T_basal_B_vs_MCF7"),
        ("MDAMB231_basal_B", "MDAMB231_vs_MCF7", "MDAMB231_basal_B_vs_MCF7"),
    ]:
        new_col = f"{pair_prefix}_gene_spearman_rho"
        old_col = f"{base_prefix}_gene_spearman_rho"
        if new_col not in condition_rb or old_col not in parent_rb:
            raise ValueError(f"Parent root metrics missing expected comparison field {old_col}")
        diffs = pd.to_numeric(condition_rb[new_col], errors="coerce") - pd.to_numeric(parent_rb[old_col], errors="coerce")
        if int(diffs.notna().sum()) != EXPECTED_COMMON_EXPOSURES or float(diffs.abs().max()) > 1e-10:
            raise ValueError(f"Independent baseline correlation readback mismatch: {label}; max_abs={diffs.abs().max()}")
        max_baseline_abs_diff = max(max_baseline_abs_diff, float(diffs.abs().max()))
    if not condition_rb.inchi_key.str.fullmatch(r"[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]").all() or condition_rb.inchi_key.nunique() != EXPECTED_KEYS:
        raise ValueError("Full exact chemical-key validation failed")

    summary = {
        "status": "COMPLETED_SOURCE_DOMAIN_PAIRWISE_DESCRIPTION_ROOT_READBACK_PASS",
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "registered GSE70138 LINCS Level 2 GEX source-domain asset",
        "scope": "Descriptive comparison among BT20, HS578T, MDA-MB-231 and MCF7 over the strict complete-full-InChIKey subset of the parent 3.2 shared exposure domain; parent rows with the non-identity placeholder restricted are excluded; not PDO migration, efficacy, or population inference.",
        "domain": {"parent_raw_exposure_count": meta["parent_domain_raw_exposures"], "parent_raw_distinct_key_string_count": meta["parent_domain_raw_distinct_key_strings"], "strict_exact_exposure_count": EXPECTED_COMMON_EXPOSURES, "strict_full_InChIKey_count": EXPECTED_KEYS, "excluded_placeholder_key": meta["excluded_placeholder_key"], "excluded_placeholder_exposure_count": meta["excluded_placeholder_exposure_count"], "excluded_placeholder_source_pert_ids": meta["excluded_placeholder_source_pert_ids"], "excluded_placeholder_source_compound_names": meta["excluded_placeholder_compound_names_from_source_metadata"], "measured_gene_count": 978, "Hallmark_set_count": 50},
        "independent_unit": "full exact InChIKey chemical cluster after equal averaging of its eligible dose/time exposures; source cell-line count is four fixed models and not a subtype population sample",
        "source_model_ids": MODEL_LABELS,
        "aggregation": "within each exact full InChIKey average all eligible exact dose/time exposure metrics; across keys weight each unique full key equally",
        "bootstrap": {"replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "full exact InChIKey", "interpretation": "descriptive uncertainty for the observed drug-domain only, not patient/subtype population inference"},
        "baseline_readback": {"parent_raw_rows": len(parent_raw_rb), "parent_strict_rows": len(parent_rb), "parent_excluded_placeholder_rows": len(parent_raw_rb) - len(parent_rb), "new_rows": len(condition_rb), "compared_per_cell_MCF7_rho_fields": 3, "max_abs_difference": max_baseline_abs_diff, "status": "PASS"},
        "vector_readback": {"shape": list(vector_rb["delta"].shape), "all_finite": bool(np.isfinite(vector_rb["delta"]).all()), "gene_axis_unique": len(set(vector_rb["gene_symbols"].astype(str))) == 978, "hallmark_axis_rows": len(hall_summary), "status": "PASS"},
        "n_negative_gene_rho_key_means": {pair: int((pd.to_numeric(drug_metrics[f"{pair}_gene_spearman_rho"], errors="coerce") < 0).sum()) for pair in PAIRS},
        "wall_seconds": round(time.monotonic() - wall_started, 3),
        "cpu_cap_minutes": 90,
        "gpu_hours": 0,
        "inference_boundary": "No p-values or causal statements; no clinical, patient, PDO, drug efficacy, or TNBC-generalization claim.",
        "input_sha256": input_hashes,
        "uncompressed_GCTX_sha256": gctx_hash,
        "outputs": [
            "condition_vectors.npz", "condition_metrics.tsv", "drug_equal_weight_metrics.tsv",
            "bootstrap_equal_key_means.tsv", "paired_bootstrap_ci.tsv", "drug_key_hallmark_directions.tsv",
            "hallmark_direction_summary.tsv", "drug_counterexamples.tsv",
            "excluded_placeholder_key_summary.tsv", "excluded_placeholder_key_rows.tsv",
        ],
    }
    atomic_json(full_dir / "summary.json", summary)
    manifest = {
        "run_id": RUN_ID,
        "experiment_id": "EXP-050",
        "protocol_parent": "experiments/records/EXP-050.md § 3.2; frozen supplement entry 2026-10-06 15:53 +08",
        "started_or_finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": "WSL2 Conda drugscreening-gpu",
        "command": "conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/analyze_exp050_breast_subtype_pairwise_comparison.py --run-id 20261006_breast_subtype_pairwise_comparison_v1 --phase full",
        "inputs": meta["registered_input_paths"],
        "input_sha256": input_hashes,
        "protocol_sha256": sha256_file(ROOT / "experiments/records/EXP-050.md"),
        "code_sha256": sha256_file(Path(__file__)),
        "status": summary["status"],
        "wall_seconds": summary["wall_seconds"],
        "cpu_cap_minutes": 90,
        "gpu_hours": 0,
        "privacy_and_label_boundary": "GSE70138 source LINCS/DepMap data only; no PDO values/labels, outer/locked values, or GSE216540 sample metadata read.",
    }
    atomic_json(OUT / "run_manifest.json", manifest)
    outputs = [p for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in {"sha256_manifest.tsv", "root_independent_readback_receipt.json", "console.log"}]
    hash_rows = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in outputs]
    write_tsv(OUT / "sha256_manifest.tsv", pd.DataFrame(hash_rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"Run identifier must be frozen as {RUN_ID}")
    smoke_or_full(args.phase)


if __name__ == "__main__":
    main()
