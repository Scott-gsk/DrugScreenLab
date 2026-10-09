"""EXP-062 response-blind source coverage expansion and fixed state-model comparison.

Phase ``metadata`` accesses only H5AD obs/var and chemical identity metadata.
Phase ``run`` is gated by the root readback receipt and reads only the frozen
source rows from H5AD X and obsm['X_ctl'].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import run_exp058_state_conditional_residual as exp058  # noqa: E402
import run_exp056_low_rank_state_drug_interaction as exp056  # noqa: E402
import run_exp053_drug_shared_state_residual as exp053  # noqa: E402

ARMS = ("nonlinear_state_residual", "capacity_matched_drug_only", "shuffled_state")
METHODS = ("zero_change", "per_drug_mean", "linear_state_residual", *ARMS)
FULL_KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def paired_bootstrap_spec(cfg: dict[str, Any]) -> tuple[int, int]:
    """Parse the already registered paired object-bootstrap recipe without changing it."""
    spec = str(cfg["metrics"].get("paired_interval", ""))
    match = re.fullmatch(r"source_object_bootstrap_(\d+)_seed_(\d+)", spec)
    if match is None:
        raise ValueError(f"unrecognized frozen paired interval specification: {spec!r}")
    return int(match.group(1)), int(match.group(2))


def config_and_paths(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-062" or cfg.get("run_id") != args.run_id:
        raise ValueError("EXP-062 config and run_id do not match the CLI")
    if cfg.get("data", {}).get("fixed_dose_uM") != 10 or cfg.get("data", {}).get("fixed_time_hours") != 24:
        raise ValueError("EXP-062 fixed exposure contract changed")
    out = ROOT / cfg["output"]["root"]
    if args.run_id not in str(out):
        raise ValueError("output path is not uniquely bound to this run_id")
    h5ad = Path(cfg["data"]["h5ad_path"])
    drug_info = Path(cfg["data"]["source_drug_info"])
    return cfg, cfg_path, out, h5ad


def _load_metadata_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                          dict[str, Any]]:
    import anndata as ad

    h5ad = Path(cfg["data"]["h5ad_path"])
    started = time.monotonic()
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        if adata.shape[1] != int(cfg["data"].get("genes", 978)):
            raise ValueError(f"H5AD gene count changed: {adata.shape[1]}")
        required = ["sample_id", "pert_id", "pert_idx", "cell_iname", "cell_idx",
                    "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
                    "pert_type", "project_code", "build_name", "split_cold_cell_1"]
        missing = [name for name in required if name not in adata.obs.columns]
        if missing:
            raise ValueError(f"H5AD metadata lacks identity/condition columns: {missing}")
        obs = adata.obs[required].copy().reset_index(drop=True)
        obs["obs_name"] = np.asarray(adata.obs_names.astype(str), dtype=str)
        obs["source_row"] = np.arange(len(obs), dtype=np.int64)
        var = pd.DataFrame({"h5ad_var_name": np.asarray(adata.var_names.astype(str), dtype=str)})
        shape = tuple(map(int, adata.shape))
        obsm_keys = list(adata.obsm.keys())
    finally:
        adata.file.close()

    if not obs["obs_name"].astype(str).is_unique:
        raise ValueError("H5AD observation names are not unique")
    if not obs["source_row"].is_unique:
        raise ValueError("H5AD metadata row indices are not unique")
    if "X_ctl" not in obsm_keys:
        raise ValueError("official XPert H5AD lacks paired obsm['X_ctl'] metadata key")
    if len(var) != 978 or not var["h5ad_var_name"].astype(str).is_unique:
        raise ValueError("official XPert var axis is not a unique 978-gene axis")

    feature_root = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
    gene_map_path = feature_root / "gene_axis_mapping.tsv"
    old_keys_path = feature_root / "response_blind_conditions.tsv"
    root_fp_receipt_path = feature_root / "root_ecfp4_identity_readback_receipt.json"
    for path in (gene_map_path, old_keys_path, root_fp_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"EXP-051 verified source contract missing: {path}")
    gene_map = pd.read_csv(gene_map_path, sep="\t", dtype=str, keep_default_na=False)
    if len(gene_map) != 978 or gene_map["axis_index"].astype(int).tolist() != list(range(978)):
        raise ValueError("registered EXP-051 gene-axis mapping is not ordered 978")
    if not np.array_equal(var["h5ad_var_name"].astype(str).to_numpy(),
                          gene_map["h5ad_var_name"].astype(str).to_numpy()):
        raise ValueError("official XPert H5AD gene order differs from the frozen EXP-051 axis")
    old_keys = pd.read_csv(old_keys_path, sep="\t", dtype=str, keep_default_na=False,
                           usecols=["source_object_id", "split_role"])
    old_roles = old_keys.groupby("source_object_id")["split_role"].agg(
        lambda x: sorted(set(map(str, x)))).to_dict()
    if any(not set(v).issubset({"train", "inner_dev"}) for v in old_roles.values()):
        raise ValueError("legacy source object roles contain an unregistered split label")
    panacea_path = ROOT / "artifacts/experiments/EXP-057/20261007_exp057_panacea_external_development_v6_20261007T0823Z/condition_metadata.tsv"
    if not panacea_path.is_file():
        raise FileNotFoundError("EXP-057 PANACEA metadata-only exposure inventory is absent")
    panacea = pd.read_csv(panacea_path, sep="\t", dtype=str, keep_default_na=False,
                          usecols=["cell_iname"])
    if "cell_iname" not in panacea:
        raise ValueError("EXP-057 PANACEA metadata inventory lacks cell_iname")
    panacea_names = set(panacea.cell_iname.astype(str).str.strip())

    obs["source_object_id"] = obs["cell_iname"].astype(str).str.strip()
    obs["pert_id"] = obs["pert_id"].astype(str).str.strip()
    obs["sample_id"] = obs["sample_id"].astype(str).str.strip()
    official = obs["split_cold_cell_1"].astype(str).str.strip().str.lower()
    if not set(official.unique()).issubset({"train", "test"}):
        raise ValueError(f"unrecognized official deterministic split labels: {sorted(set(official))}")
    obs["official_deterministic_role"] = official
    obs["legacy_project_roles"] = obs["source_object_id"].map(old_roles).map(
        lambda x: json.dumps(x if isinstance(x, (list, tuple, set)) else [],
                             ensure_ascii=False, separators=(",", ":")))
    obs["panacea_exact_name_overlap"] = obs["source_object_id"].isin(panacea_names)

    role_rows: list[dict[str, Any]] = []
    for obj, group in obs.groupby("source_object_id", sort=True, dropna=False):
        official_roles = set(group.official_deterministic_role.astype(str))
        if len(official_roles) != 1:
            raise ValueError(f"one source object has inconsistent official roles: {obj}")
        old_role_list = sorted({r for raw in group.legacy_project_roles.astype(str)
                                for r in json.loads(raw)})
        official_role = next(iter(official_roles))
        panacea_overlap = bool(group.panacea_exact_name_overlap.any())
        legacy_dev = "inner_dev" in old_role_list
        # Official test and any prior evaluation exposure are never admitted into TRAIN.
        effective = "inner_dev" if (official_role == "test" or legacy_dev or panacea_overlap) else "train"
        conflict = bool("train" in old_role_list and (official_role == "test" or legacy_dev or panacea_overlap))
        role_rows.append({"source_object_id": str(obj), "official_deterministic_role": official_role,
                          "legacy_project_roles": json.dumps(old_role_list, ensure_ascii=False),
                          "panacea_exact_name_overlap": panacea_overlap,
                          "effective_split_role": effective,
                          "legacy_official_conflict": conflict,
                          "role_resolution": ("inner_dev_priority_prior_evaluation_or_official_test"
                              if effective == "inner_dev" else "official_train_no_prior_evaluation")})
    role_table = pd.DataFrame(role_rows)
    if role_table.source_object_id.duplicated().any():
        raise ValueError("role table is not one row per source object")
    obs = obs.merge(role_table[["source_object_id", "effective_split_role",
                                "legacy_official_conflict"]], on="source_object_id",
                    how="left", validate="many_to_one")
    if obs.effective_split_role.isna().any():
        raise ValueError("some H5AD rows lack a frozen effective source role")

    meta_receipt = {
        "h5ad_shape": list(shape), "h5ad_obs_columns": list(obs.columns),
        "h5ad_var_columns": list(var.columns), "h5ad_obsm_keys_metadata_only": obsm_keys,
        "metadata_rows_read": int(len(obs)), "metadata_columns_read": list(obs.columns),
        "h5ad_X_values_read": False, "h5ad_layers_values_read": False,
        "h5ad_X_ctl_values_read": False, "any_expression_or_response_values_read": False,
        "role_counts_objects": role_table.effective_split_role.value_counts().to_dict(),
        "role_conflict_object_count": int(role_table.legacy_official_conflict.sum()),
        "panacea_exact_name_overlap_objects": sorted(role_table.loc[
            role_table.panacea_exact_name_overlap, "source_object_id"].astype(str).tolist()),
        "exp051_gene_axis_rows": int(len(gene_map)), "metadata_scan_wall_seconds": time.monotonic() - started,
    }
    return obs, var, role_table, meta_receipt


def _fingerprint_by_full_key(drug_info: pd.DataFrame, candidate_keys: set[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem

    table = drug_info.copy()
    table["pert_id"] = table["pert_id"].astype(str).str.strip()
    table["full_inchikey"] = table["inchi_key"].astype(str).str.strip().str.upper()
    table["canonical_smiles"] = table["canonical_smiles"].astype(str).str.strip()
    table = table.loc[table.full_inchikey.isin(candidate_keys)].copy()
    if table.empty:
        return {}, {"rdkit_version": rdBase.rdkitVersion, "audited_key_count": 0, "groups": []}
    output: dict[str, np.ndarray] = {}
    groups: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for key, group in table.groupby("full_inchikey", sort=True):
        if not (pd.Series([key]).str.fullmatch(FULL_KEY_RE).iloc[0]):
            excluded[key] = "malformed_full_inchikey"
            continue
        aliases = group.loc[:, ["pert_id", "canonical_smiles"]].drop_duplicates().sort_values(
            ["pert_id", "canonical_smiles"], kind="mergesort")
        hashes: dict[str, np.ndarray] = {}
        alias_receipts: list[dict[str, Any]] = []
        failed = None
        for row in aliases.itertuples(index=False):
            mol = Chem.MolFromSmiles(str(row.canonical_smiles))
            if mol is None:
                failed = "canonical_smiles_parse_failure"
                break
            computed = Chem.MolToInchiKey(mol)
            if computed != key:
                failed = "canonical_smiles_full_key_mismatch"
                break
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            bit_array = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, bit_array)
            bit_sha = hashlib.sha256(bit_array.tobytes()).hexdigest()
            hashes[bit_sha] = bit_array
            alias_receipts.append({"pert_id": str(row.pert_id), "canonical_smiles_sha256":
                                   hashlib.sha256(str(row.canonical_smiles).encode()).hexdigest(),
                                   "computed_full_inchikey": computed,
                                   "fingerprint_bits": 2048, "radius": 2,
                                   "fingerprint_sha256": bit_sha,
                                   "on_bits": int(bit_array.sum())})
        if failed:
            excluded[key] = failed
            continue
        if not hashes:
            excluded[key] = "no_canonical_smiles_alias"
            continue
        if len(hashes) != 1:
            excluded[key] = "aliases_do_not_have_identical_ecfp4"
            continue
        vector = next(iter(hashes.values()))
        output[key] = vector.astype(np.float32)
        groups.append({"full_inchikey": key, "pert_id_alias_count": int(aliases.pert_id.nunique()),
                       "canonical_smiles_alias_count": int(len(aliases)),
                       "unique_fingerprint_count": 1,
                       "fingerprint_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                       "on_bits": int(vector.sum()), "aliases": alias_receipts})
    return output, {"rdkit_version": rdBase.rdkitVersion,
                    "method": "EXP-051 verified contract: RDKit MolFromSmiles(canonical_smiles), exact MolToInchiKey, Morgan bit vector radius=2 nBits=2048, alias fingerprints must hash identically",
                    "audited_key_count": len(output), "groups": groups, "excluded_keys": excluded}


def metadata_phase(cfg: dict[str, Any], cfg_path: Path, out: Path, run_id: str) -> int:
    if out.exists():
        raise FileExistsError(f"EXP-062 unique run directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        obs, var, role_table, source_meta = _load_metadata_inputs(cfg)
        drug_path = Path(cfg["data"]["source_drug_info"])
        drug_info = pd.read_csv(drug_path, dtype=str, keep_default_na=False, low_memory=False,
                                usecols=["pert_id", "inchi_key", "canonical_smiles"])
        required = {"pert_id", "inchi_key", "canonical_smiles"}
        if not required.issubset(drug_info.columns):
            raise ValueError(f"official XPert drug table lacks identity fields: {sorted(required-set(drug_info.columns))}")
        drug_info["pert_id"] = drug_info.pert_id.astype(str).str.strip()
        drug_info["full_inchikey"] = drug_info.inchi_key.astype(str).str.strip().str.upper()
        valid_key = drug_info.full_inchikey.str.fullmatch(FULL_KEY_RE, na=False)
        drug_info["full_key_complete"] = valid_key
        pert_key_counts = drug_info.loc[valid_key].groupby("pert_id").full_inchikey.nunique()
        valid_pert_ids = set(pert_key_counts.index[pert_key_counts.eq(1)])
        perturbation = drug_info.loc[valid_key & drug_info.pert_id.isin(valid_pert_ids),
                                     ["pert_id", "full_inchikey"]].drop_duplicates()
        if perturbation.pert_id.duplicated().any():
            raise ValueError("official pert_id to full InChIKey map is not one-to-one")

        dose = pd.to_numeric(obs.pert_dose, errors="coerce")
        exposure_time = pd.to_numeric(obs.pert_time, errors="coerce")
        eligible_rows = obs.loc[
            dose.eq(float(cfg["data"]["fixed_dose_uM"]))
            & obs.pert_dose_unit.astype(str).str.strip().str.lower().eq("um")
            & exposure_time.eq(float(cfg["data"]["fixed_time_hours"]))
            & obs.pert_time_unit.astype(str).str.strip().str.lower().eq("h")
            & obs.pert_type.astype(str).str.strip().eq("trt_cp")
        ].copy()
        eligible_rows = eligible_rows.merge(perturbation, on="pert_id", how="inner", validate="many_to_one")
        if eligible_rows.empty:
            raise ValueError("no fixed-dose/time rows have complete official full-key identity")
        if (eligible_rows.groupby("source_object_id", dropna=False)["cell_idx"].nunique(dropna=False) > 1).any():
            raise ValueError("one H5AD source object maps to multiple cell_idx values")
        if (eligible_rows.groupby("cell_idx", dropna=False)["source_object_id"].nunique(dropna=False) > 1).any():
            raise ValueError("cell_idx maps to multiple source object identities")
        key_fields = ["source_object_id", "full_inchikey", "pert_id", "pert_dose", "pert_dose_unit",
                      "pert_time", "pert_time_unit", "project_code", "build_name", "sample_id", "obs_name"]
        for col in key_fields:
            eligible_rows[col] = eligible_rows[col].astype(str).fillna("").str.strip()
        eligible_rows["full_condition_key"] = eligible_rows[key_fields].agg("|".join, axis=1)
        if not eligible_rows.full_condition_key.is_unique:
            raise ValueError("source condition key is not unique; preserving duplicates and fail-closed")
        seed = int(cfg["data"]["selection_seed"])
        eligible_rows["selection_hash_sha256"] = eligible_rows.apply(
            lambda row: hashlib.sha256(
                f"{seed}|{row.full_inchikey}|{row.full_condition_key}".encode("utf-8")).hexdigest(), axis=1)
        metadata_columns = ["source_row", "obs_name", "sample_id", "source_object_id", "cell_idx",
                           "official_deterministic_role", "legacy_project_roles", "panacea_exact_name_overlap",
                           "effective_split_role", "legacy_official_conflict", "pert_id", "pert_idx",
                           "full_inchikey", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
                           "pert_type", "project_code", "build_name", "full_condition_key",
                           "selection_hash_sha256"]
        eligible_rows = eligible_rows[metadata_columns].copy()
        # Repeats within a source object and exact full key are resolved without reading response values.
        by_pair = eligible_rows.sort_values(["selection_hash_sha256", "full_condition_key"], kind="mergesort")\
            .drop_duplicates(["source_object_id", "full_inchikey"], keep="first")
        cover = by_pair.groupby("full_inchikey", sort=True).agg(
            source_objects=("source_object_id", "nunique"),
            train_objects=("effective_split_role", lambda x: int(x.eq("train").sum())),
            dev_objects=("effective_split_role", lambda x: int(x.eq("inner_dev").sum())),
            object_condition_rows=("source_object_id", "size"),
        ).reset_index()
        min_train = int(cfg["data"]["min_train_objects_per_drug"])
        candidates = cover.loc[cover.train_objects.ge(min_train)].copy()
        candidate_keys = set(candidates.full_inchikey.astype(str))
        feature_by_key, feature_audit = _fingerprint_by_full_key(drug_info, candidate_keys)
        candidates["fingerprint_qualified"] = candidates.full_inchikey.isin(feature_by_key)
        candidates["fingerprint_exclusion_reason"] = candidates.full_inchikey.map(
            lambda k: feature_audit.get("excluded_keys", {}).get(str(k), ""))
        candidates = candidates.loc[candidates.fingerprint_qualified].sort_values(
            ["train_objects", "full_inchikey"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
        candidates["precap_rank"] = np.arange(1, len(candidates) + 1, dtype=np.int64)
        max_per_drug = int(cfg["data"]["max_conditions_per_drug"])
        max_drugs = int(cfg["data"]["max_drugs"])
        max_total = int(cfg["data"]["max_total_conditions"])
        selected_parts: list[pd.DataFrame] = []
        selected_drugs: list[dict[str, Any]] = []
        total = 0
        for row in candidates.itertuples(index=False):
            key = str(row.full_inchikey)
            subset = by_pair.loc[by_pair.full_inchikey.astype(str).eq(key)].sort_values(
                ["selection_hash_sha256", "full_condition_key"], kind="mergesort").head(max_per_drug).copy()
            train_after = int(subset.effective_split_role.eq("train").sum())
            if train_after < min_train:
                continue
            if len(selected_drugs) >= max_drugs or total + len(subset) > max_total:
                continue
            rank = len(selected_drugs) + 1
            subset["drug_selection_rank"] = rank
            selected_parts.append(subset)
            selected_drugs.append({"drug_selection_rank": rank, "precap_rank": int(row.precap_rank),
                                   "full_inchikey": key, "train_objects_pre_cap": int(row.train_objects),
                                   "source_objects_pre_cap": int(row.source_objects),
                                   "selected_conditions": int(len(subset)),
                                   "selected_train_objects": train_after,
                                   "selected_dev_objects": int(subset.effective_split_role.eq("inner_dev").sum()),
                                   "selection_rule": "one hash-first row per object, then SHA256 order cap"})
            total += len(subset)
        if not selected_parts:
            raise ValueError("no complete-key drugs survive TRAIN coverage, fingerprint, and row-cap gates")
        selected = pd.concat(selected_parts, ignore_index=True)
        selected = selected.sort_values(["drug_selection_rank", "selection_hash_sha256",
                                         "full_condition_key"], kind="mergesort").reset_index(drop=True)
        selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
        train_objects = set(selected.loc[selected.effective_split_role.eq("train"), "source_object_id"].astype(str))
        dev_objects = set(selected.loc[selected.effective_split_role.eq("inner_dev"), "source_object_id"].astype(str))
        if train_objects & dev_objects:
            raise ValueError("source objects leak between TRAIN and inner-dev after role freezing")
        if len(selected) > max_total or (selected.groupby("full_inchikey").size() > max_per_drug).any():
            raise ValueError("condition caps were exceeded")
        # Preserve both official and legacy role provenance at object level.
        role_table.to_csv(out / "source_object_roles.tsv", sep="\t", index=False)
        candidates.to_csv(out / "response_blind_drug_candidates.tsv", sep="\t", index=False)
        selected.to_csv(out / "response_blind_conditions.tsv", sep="\t", index=False)
        pd.DataFrame(selected_drugs).to_csv(out / "selected_drugs.tsv", sep="\t", index=False)
        ordered_keys = [d["full_inchikey"] for d in selected_drugs]
        fp_matrix = np.stack([feature_by_key[k].astype(np.uint8) for k in ordered_keys])
        fp_path = out / "ecfp4_2048_by_full_inchikey.npz"
        np.savez_compressed(fp_path, full_inchikey=np.asarray(ordered_keys, dtype=str),
                            features=fp_matrix)
        atomic_json(out / "fingerprint_identity_audit.json", safe(feature_audit))
        source_files = {
            "h5ad": Path(cfg["data"]["h5ad_path"]),
            "drug_identity_table": Path(cfg["data"]["source_drug_info"]),
            "exp051_gene_axis_mapping": ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/gene_axis_mapping.tsv",
            "exp051_response_blind_conditions": ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/response_blind_conditions.tsv",
            "exp051_ecfp4_root_receipt": ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/root_ecfp4_identity_readback_receipt.json",
            "exp057_panacea_metadata": ROOT / "artifacts/experiments/EXP-057/20261007_exp057_panacea_external_development_v6_20261007T0823Z/condition_metadata.tsv",
            "config": cfg_path,
            "runner": Path(__file__).resolve(),
        }
        hashes = {name: sha256_file(path) for name, path in source_files.items()}
        file_hashes = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file()}
        elapsed = time.monotonic() - started
        receipt = {
            "experiment_id": "EXP-062", "run_id": run_id,
            "status": "PASS_METADATA_FREEZE_PENDING_ROOT_READBACK",
            "metadata_gate_scope": "response-blind metadata only; no X/X_ctl/layer values",
            "source_h5ad_sha256": hashes["h5ad"], "source_h5ad_shape": list(source_meta["h5ad_shape"]),
            "source_metadata": source_meta,
            "gene_axis_sha256": hashlib.sha256("\n".join(var.h5ad_var_name.astype(str)) .encode()).hexdigest(),
            "official_split_role_counts": role_table.effective_split_role.value_counts().to_dict(),
            "official_legacy_conflicts": int(role_table.legacy_official_conflict.sum()),
            "panacea_exact_name_overlaps": source_meta["panacea_exact_name_overlap_objects"],
            "dose_time_filter": "10 uM x 24 h; pert_type=trt_cp",
            "selection": {"min_train_objects_per_drug": min_train,
                          "selected_drug_count": len(selected_drugs),
                          "selected_conditions": int(len(selected)),
                          "selected_train_conditions": int(selected.effective_split_role.eq("train").sum()),
                          "selected_inner_dev_conditions": int(selected.effective_split_role.eq("inner_dev").sum()),
                          "selected_train_objects": len(train_objects),
                          "selected_inner_dev_objects": len(dev_objects),
                          "train_inner_dev_object_overlap": 0,
                          "full_inchikey_count": int(selected.full_inchikey.nunique()),
                          "gene_count": int(len(var)), "drug_ranking": "TRAIN unique source-object coverage descending, exact complete InChIKey ascending",
                          "condition_hash": "SHA256(UTF8(seed|full_InChIKey|full_condition_key)), ascending lowercase hex",
                          "selection_seed": seed, "row_cap_per_drug": max_per_drug,
                          "total_row_cap": max_total, "selected_drugs": selected_drugs},
            "fingerprint_method": feature_audit["method"],
            "fingerprint_rdkit_version": feature_audit["rdkit_version"],
            "fingerprint_matrix_shape": list(fp_matrix.shape),
            "fingerprint_file_sha256": sha256_file(fp_path),
            "source_asset_sha256": hashes,
            "frozen_files_sha256": file_hashes,
            "matrix_access": {"h5ad_X_read": False, "h5ad_X_ctl_read": False,
                              "h5ad_layers_read": False, "h5ad_obsm_numeric_values_read": False,
                              "panacea_numeric_values_read": False, "pdo_outer_locked_numeric_values_read": False},
            "elapsed_cpu_wall_seconds": elapsed,
            "gpu_hours": 0,
        }
        atomic_json(out / "metadata_freeze_manifest.json", safe(receipt))
        atomic_json(out / "run_manifest.json", {"experiment_id": "EXP-062", "run_id": run_id,
                    "phase": "metadata", "status": receipt["status"],
                    "config_sha256": hashes["config"], "runner_sha256": hashes["runner"],
                    "elapsed_cpu_wall_seconds": elapsed, "gpu_hours": 0,
                    "frozen_receipt_sha256": sha256_file(out / "metadata_freeze_manifest.json")})
        print(json.dumps({"status": receipt["status"], "run_id": run_id,
                          "selected_drugs": len(selected_drugs), "selected_conditions": len(selected),
                          "train_conditions": int(selected.effective_split_role.eq("train").sum()),
                          "inner_dev_conditions": int(selected.effective_split_role.eq("inner_dev").sum()),
                          "train_objects": len(train_objects), "inner_dev_objects": len(dev_objects),
                          "full_inchikeys": int(selected.full_inchikey.nunique()), "genes": int(len(var)),
                          "cpu_wall_seconds": elapsed, "gpu_hours": 0}, ensure_ascii=False))
        return 0
    except Exception as exc:
        receipt = {"experiment_id": "EXP-062", "run_id": run_id, "status": "FAILED_CLOSED_METADATA",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "elapsed_cpu_wall_seconds": time.monotonic()-started,
                   "gpu_hours": 0, "h5ad_X_values_read": False, "h5ad_X_ctl_values_read": False,
                   "pdo_outer_locked_numeric_values_read": False}
        atomic_json(out / "metadata_gate_failure.json", receipt)
        atomic_json(out / "run_manifest.json", {"experiment_id": "EXP-062", "run_id": run_id,
                    "phase": "metadata", "status": "FAILED_CLOSED_METADATA",
                    "elapsed_cpu_wall_seconds": receipt["elapsed_cpu_wall_seconds"], "gpu_hours": 0})
        raise


def _load_frozen_values(cfg: dict[str, Any], out: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray, np.ndarray,
                                                                  np.ndarray, dict[str, Any]]:
    import anndata as ad
    from scipy import sparse

    freeze_path = out / "metadata_freeze_manifest.json"
    root_receipt_path = out / "root_metadata_readback.json"
    key_path = out / "response_blind_conditions.tsv"
    fp_path = out / "ecfp4_2048_by_full_inchikey.npz"
    for path in (freeze_path, root_receipt_path, key_path, fp_path):
        if not path.is_file():
            raise FileNotFoundError(f"value-stage prerequisite missing: {path}")
    gate = json.loads(freeze_path.read_text(encoding="utf-8"))
    root_receipt = json.loads(root_receipt_path.read_text(encoding="utf-8"))
    if (gate.get("status") != "PASS_METADATA_FREEZE_PENDING_ROOT_READBACK"
            or root_receipt.get("status") != "PASS_ROOT_INDEPENDENT_METADATA_READBACK"
            or root_receipt.get("metadata_freeze_sha256") != sha256_file(freeze_path)):
        raise PermissionError("root metadata readback is not PASS for this exact immutable freeze")
    keys = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
    keys["source_row"] = pd.to_numeric(keys.source_row, errors="raise").astype(np.int64)
    keys["selection_rank"] = pd.to_numeric(keys.selection_rank, errors="raise").astype(np.int64)
    if not keys.full_condition_key.is_unique or keys.source_row.duplicated().any():
        raise ValueError("frozen condition or source-row keys are not unique")
    if not set(keys.effective_split_role).issubset({"train", "inner_dev"}):
        raise ValueError("forbidden role in frozen matrix selection")
    tr_obj = set(keys.loc[keys.effective_split_role.eq("train"), "source_object_id"])
    dv_obj = set(keys.loc[keys.effective_split_role.eq("inner_dev"), "source_object_id"])
    if tr_obj & dv_obj:
        raise ValueError("object identity overlap between TRAIN and inner-dev")
    current_sha = sha256_file(Path(cfg["data"]["h5ad_path"]))
    if current_sha != gate.get("source_h5ad_sha256"):
        raise ValueError("official H5AD bytes changed after metadata freeze")
    a = ad.read_h5ad(Path(cfg["data"]["h5ad_path"]), backed="r")
    try:
        genes = np.asarray(a.var_names.astype(str), dtype=str)
        map_path = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/gene_axis_mapping.tsv"
        gene_map = pd.read_csv(map_path, sep="\t", dtype=str, keep_default_na=False)
        if not np.array_equal(genes, gene_map.h5ad_var_name.astype(str).to_numpy()):
            raise ValueError("source var gene order differs from the exact verified EXP-051 gene axis")
        rows = keys.source_row.to_numpy(np.int64)
        order = np.argsort(rows, kind="mergesort")
        sorted_rows = rows[order]
        if len(np.unique(sorted_rows)) != len(sorted_rows):
            raise ValueError("selected H5AD rows are not unique")
        obs_names = np.asarray(a.obs_names[sorted_rows].astype(str), dtype=str)
        row_names_by_sorted = keys.obs_name.astype(str).to_numpy()[order]
        source_objects = np.asarray(a.obs["cell_iname"].iloc[sorted_rows].astype(str), dtype=str)
        row_objects_by_sorted = keys.source_object_id.astype(str).to_numpy()[order]
        pert_ids = np.asarray(a.obs["pert_id"].iloc[sorted_rows].astype(str), dtype=str)
        row_pert_by_sorted = keys.pert_id.astype(str).to_numpy()[order]
        if not np.array_equal(obs_names, row_names_by_sorted):
            raise ValueError("frozen source_row does not match exact obs_name")
        if not np.array_equal(source_objects, row_objects_by_sorted):
            raise ValueError("frozen source_row does not match exact source_object_id")
        if not np.array_equal(pert_ids, row_pert_by_sorted):
            raise ValueError("frozen source_row does not match exact pert_id")
        # This is the first and only stage permitted to read expression values.
        x_raw = a.X[sorted_rows, :]
        ctl_raw = a.obsm["X_ctl"][sorted_rows, :]
        if hasattr(x_raw, "to_memory"):
            x_raw = x_raw.to_memory()
        if hasattr(ctl_raw, "to_memory"):
            ctl_raw = ctl_raw.to_memory()
        if sparse.issparse(x_raw):
            x_raw = x_raw.toarray()
        if sparse.issparse(ctl_raw):
            ctl_raw = ctl_raw.toarray()
        x_sorted = np.asarray(x_raw, dtype=np.float32)
        ctl_sorted = np.asarray(ctl_raw, dtype=np.float32)
        x = np.empty_like(x_sorted)
        ctl = np.empty_like(ctl_sorted)
        x[order] = x_sorted
        ctl[order] = ctl_sorted
    finally:
        a.file.close()
    if x.shape != (len(keys), 978) or ctl.shape != x.shape:
        raise ValueError(f"selected X/X_ctl shape changed: {x.shape} / {ctl.shape}")
    if not np.isfinite(x).all() or not np.isfinite(ctl).all():
        raise FloatingPointError("selected official source X/X_ctl contains non-finite values")
    y = x - ctl
    if not np.isfinite(y).all() or not np.allclose(x, ctl + y, rtol=0, atol=2e-6):
        raise FloatingPointError("selected X-X_ctl delta failed finite/round-trip verification")
    fp_archive = np.load(fp_path, allow_pickle=False)
    fp_keys = fp_archive["full_inchikey"].astype(str)
    fp_mat = fp_archive["features"].astype(np.float32)
    if fp_mat.shape != (len(fp_keys), 2048) or not np.isfinite(fp_mat).all():
        raise ValueError("frozen drug fingerprint archive has invalid shape or non-finite values")
    if set(keys.full_inchikey.astype(str)) != set(fp_keys):
        raise ValueError("fingerprint complete-key set differs from selected condition keys")
    key_to_i = {k: i for i, k in enumerate(fp_keys)}
    drug_order = sorted(set(keys.full_inchikey.astype(str)))
    drug_index = {k: i for i, k in enumerate(drug_order)}
    features = fp_mat[[key_to_i[k] for k in drug_order]]
    key_to_fi = {k: i for i, k in enumerate(drug_order)}
    row_fp = np.stack([features[key_to_fi[k]] for k in keys.full_inchikey.astype(str)]).astype(np.float32)
    row_drug = np.asarray([drug_index[k] for k in keys.full_inchikey.astype(str)], dtype=np.int64)
    receipt = {"source_h5ad_sha256": current_sha, "metadata_freeze_sha256": sha256_file(freeze_path),
               "root_metadata_readback_sha256": sha256_file(root_receipt_path),
               "selected_condition_file_sha256": sha256_file(key_path),
               "selected_source_rows": len(keys), "train_conditions": int(keys.effective_split_role.eq("train").sum()),
               "inner_dev_conditions": int(keys.effective_split_role.eq("inner_dev").sum()),
               "train_objects": len(tr_obj), "inner_dev_objects": len(dv_obj),
               "full_inchikeys": len(drug_order), "genes": len(genes),
               "x_shape": list(x.shape), "x_ctl_shape": list(ctl.shape), "delta_shape": list(y.shape),
               "max_abs_x_equals_ctl_plus_delta_error": float(np.max(np.abs(x.astype(np.float64)-ctl.astype(np.float64)-y.astype(np.float64)))),
               "all_selected_x_finite": True, "all_selected_x_ctl_finite": True,
               "all_selected_delta_finite": True, "pdo_outer_locked_panacea_cdsdb_values_read": False,
               "matrix_rows_read_exactly_selected": True}
    return keys, genes, row_fp, row_drug, y.astype(np.float64), ctl.astype(np.float64), receipt


def _object_macro_metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                          gene_variance: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, rows = exp056.metrics_for_role(keys, truth, prediction, gene_variance)
    return summary, rows


def run_phase(cfg: dict[str, Any], cfg_path: Path, out: Path, run_id: str, attempt_id: str) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from sklearn.linear_model import Ridge

    if torch.cuda.is_available():
        raise RuntimeError("EXP-062 is CPU-only; refuse CUDA initialization")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)
    attempt = out / attempt_id
    if attempt.exists():
        raise FileExistsError(f"EXP-062 immutable attempt already exists: {attempt}")
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cap = int(cfg["resources"]["cpu_hard_cap_seconds"])
    prior = float(cfg["resources"]["inherited_exp055_cpu_settled_seconds"])
    inherited_cap = int(cfg["resources"]["inherited_exp055_cpu_cap_seconds"])
    metadata_manifest = json.loads((out / "metadata_freeze_manifest.json").read_text(encoding="utf-8"))
    root_metadata = json.loads((out / "root_metadata_readback.json").read_text(encoding="utf-8"))
    prior_failed_attempts = []
    for failure_path in sorted(out.parent.glob("*/metadata_gate_failure.json")):
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if failure.get("experiment_id") != "EXP-062" or failure.get("status") != "FAILED_CLOSED_METADATA":
            raise ValueError(f"unrecognized sibling EXP-062 metadata failure receipt: {failure_path}")
        if float(failure.get("gpu_hours", 0.0)) != 0.0:
            raise ValueError(f"metadata failure receipt reports nonzero GPU use: {failure_path}")
        prior_failed_attempts.append({"run_id": failure.get("run_id"),
                                      "cpu_wall_seconds": float(failure["elapsed_cpu_wall_seconds"]),
                                      "receipt": str(failure_path.relative_to(ROOT))})
    prior_failed_seconds = sum(item["cpu_wall_seconds"] for item in prior_failed_attempts)
    metadata_and_root_seconds = (prior_failed_seconds
                                 + float(metadata_manifest.get("elapsed_cpu_wall_seconds", 0.0))
                                 + float(root_metadata.get("elapsed_cpu_wall_seconds", 0.0)))
    prior_value_attempts = []
    for manifest_path in sorted(out.glob("fit_attempt_*/run_manifest.json")):
        if manifest_path.parent == attempt:
            continue
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("experiment_id") != "EXP-062" or previous.get("run_id") != run_id:
            raise ValueError(f"unrecognized prior EXP-062 value attempt manifest: {manifest_path}")
        if previous.get("status") == "RUNNING":
            raise RuntimeError(f"another EXP-062 value attempt is still active: {manifest_path.parent.name}")
        if previous.get("status") != "FAILED_CLOSED":
            raise FileExistsError(f"prior EXP-062 value attempt is not a closed failure: {manifest_path}")
        prior_value_attempts.append({"attempt_id": previous.get("attempt_id"),
                                     "package_cpu_seconds": float(previous["elapsed_cpu_wall_seconds"]),
                                     "receipt": str(manifest_path.relative_to(ROOT))})
    previous_value_seconds = max((item["package_cpu_seconds"] for item in prior_value_attempts), default=0.0)
    exp062_prior = max(metadata_and_root_seconds, previous_value_seconds)
    remaining_cap = cap - exp062_prior
    if remaining_cap <= 0 or prior + exp062_prior > inherited_cap:
        raise ValueError("EXP-062 package and inherited CPU limits are inconsistent")
    manifest = {"experiment_id": "EXP-062", "run_id": run_id, "attempt_id": attempt_id,
                "phase": "run", "status": "RUNNING", "config_sha256": sha256_file(cfg_path),
                "runner_sha256": sha256_file(Path(__file__).resolve()), "device": "cpu",
                "cuda_visible_devices": "", "prior_EXP055_CPU_seconds_charged": prior,
                "prior_EXP062_package_charged_seconds": exp062_prior,
                "EXP062_metadata_root_and_failed_metadata_seconds": metadata_and_root_seconds,
                "prior_failed_metadata_attempts": prior_failed_attempts,
                "prior_failed_metadata_attempt_cpu_seconds": prior_failed_seconds,
                "prior_failed_value_attempts": prior_value_attempts,
                "package_cpu_cap_seconds": cap, "remaining_EXP062_run_seconds": remaining_cap,
                "inherited_EXP055_cpu_cap_seconds": inherited_cap,
                "forbidden_numeric_inputs_opened": []}
    atomic_json(attempt / "run_manifest.json", manifest)
    try:
        keys, genes, fp_np, drug_np, y, ctl, input_receipt = _load_frozen_values(cfg, out)
        if len(genes) != 978 or len(set(genes)) != 978:
            raise ValueError("gene axis is not unique 978")
        train_mask = keys.effective_split_role.eq("train").to_numpy()
        dev_mask = keys.effective_split_role.eq("inner_dev").to_numpy()
        if not train_mask.any() or not dev_mask.any():
            raise ValueError("fixed comparison requires both TRAIN and inner-dev rows")
        tk, dk = keys.loc[train_mask].reset_index(drop=True), keys.loc[dev_mask].reset_index(drop=True)
        train_y, dev_y = y[train_mask], y[dev_mask]
        train_fp, dev_fp = fp_np[train_mask], fp_np[dev_mask]
        train_drug_np, dev_drug_np = drug_np[train_mask], drug_np[dev_mask]
        train_objects = sorted(set(tk.source_object_id.astype(str)))
        dev_objects = sorted(set(dk.source_object_id.astype(str)))
        if set(train_objects) & set(dev_objects):
            raise ValueError("source object overlap after value-stage load")
        drug_keys = sorted(set(tk.full_inchikey.astype(str)))
        if set(drug_keys) != set(dk.full_inchikey.astype(str)):
            # Missing development support is allowed; drug-key mean remains estimable from TRAIN.
            dev_only = set(dk.full_inchikey.astype(str)) - set(drug_keys)
            if dev_only:
                raise ValueError(f"inner-dev drug key lacks TRAIN support: {sorted(dev_only)}")
        means = {k: train_y[tk.full_inchikey.astype(str).to_numpy() == k].mean(axis=0, dtype=np.float64)
                 for k in drug_keys}
        train_base = np.stack([means[k] for k in tk.full_inchikey.astype(str)]).astype(np.float64)
        dev_base = np.stack([means[k] for k in dk.full_inchikey.astype(str)]).astype(np.float64)
        variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        if not np.isfinite(variance).all() or np.any(variance <= 1e-12):
            raise ValueError("TRAIN variance normalization contains zero/invalid gene variance")
        base_tr_t = torch.as_tensor(train_base, dtype=torch.float32)
        base_dv_t = torch.as_tensor(dev_base, dtype=torch.float32)
        train_y_t = torch.as_tensor(train_y, dtype=torch.float32)
        dev_y_t = torch.as_tensor(dev_y, dtype=torch.float32)
        variance_t = torch.as_tensor(variance.astype(np.float32), dtype=torch.float32)
        train_fp_t = torch.as_tensor(train_fp, dtype=torch.float32)
        dev_fp_t = torch.as_tensor(dev_fp, dtype=torch.float32)
        train_drug_t = torch.as_tensor(train_drug_np, dtype=torch.long)
        dev_drug_t = torch.as_tensor(dev_drug_np, dtype=torch.long)
        train_counts = tk.source_object_id.astype(str).value_counts().to_dict()
        weights = np.asarray([1.0 / train_counts[o] for o in tk.source_object_id.astype(str)], dtype=np.float32)
        weights /= weights.sum()
        weights_t = torch.as_tensor(weights, dtype=torch.float32)
        train_ctl, dev_ctl = ctl[train_mask], ctl[dev_mask]
        state_tr, state_dv, state_receipt = exp058.make_state_features(
            tk, dk, train_ctl, dev_ctl, n_components=16, shuffle_seed=None)
        shuf_tr, shuf_dv, shuffle_receipt = exp058.make_state_features(
            tk, dk, train_ctl, dev_ctl, n_components=16, shuffle_seed=58059)
        atomic_json(attempt / "input_receipt.json", safe(input_receipt))
        atomic_json(attempt / "state_transform_receipt.json", safe({"candidate": state_receipt,
                                                                      "shuffled_control": shuffle_receipt}))
        state_tr_t = torch.as_tensor(state_tr, dtype=torch.float32)
        state_dv_t = torch.as_tensor(state_dv, dtype=torch.float32)
        shuf_tr_t = torch.as_tensor(shuf_tr, dtype=torch.float32)
        shuf_dv_t = torch.as_tensor(shuf_dv, dtype=torch.float32)
        # EXP-053 fixed alpha=1 linear state-residual baseline, fitted only on this TRAIN domain.
        object_states = exp056.object_state_map(keys, np.concatenate([train_ctl, dev_ctl], axis=0))
        linear_dev, linear_receipt = exp053.fit_state_residual(
            tk, train_y, dk, train_base, dev_base, object_states, None, alpha=1.0)
        linear_train, _ = exp053.fit_state_residual(
            tk, train_y, tk, train_base, train_base, object_states, None, alpha=1.0)
        pred_tr: dict[str, np.ndarray] = {"zero_change": np.zeros_like(train_y),
            "per_drug_mean": train_base, "linear_state_residual": linear_train}
        pred_dv: dict[str, np.ndarray] = {"zero_change": np.zeros_like(dev_y),
            "per_drug_mean": dev_base, "linear_state_residual": linear_dev}
        models: dict[str, Any] = {}
        opts: dict[str, Any] = {}
        receipts: dict[str, Any] = {"linear_state_residual": linear_receipt}
        seeds = {"nonlinear_state_residual": 58058, "capacity_matched_drug_only": 58060,
                 "shuffled_state": 58059}
        st_tr = {"nonlinear_state_residual": state_tr_t,
                 "capacity_matched_drug_only": torch.zeros((len(tk), 16), dtype=torch.float32),
                 "shuffled_state": shuf_tr_t}
        st_dv = {"nonlinear_state_residual": state_dv_t,
                 "capacity_matched_drug_only": torch.zeros((len(dk), 16), dtype=torch.float32),
                 "shuffled_state": shuf_dv_t}
        for arm in ARMS:
            model, width, uses_state = exp058.build_model(torch, arm, len(drug_keys), 16, seeds[arm])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            models[arm], opts[arm] = model, optimizer
            receipts[arm] = {"seed": seeds[arm], "hidden_width": width, "uses_state": uses_state,
                             "parameter_count": sum(p.numel() for p in model.parameters()),
                             "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-3}
        residual = train_y_t - base_tr_t
        curves: list[dict[str, Any]] = []
        best_state: dict[str, dict[str, Any]] = {}
        best_epoch = {arm: 0 for arm in ARMS}
        best_nmse = {arm: math.inf for arm in ARMS}
        patience = {arm: 0 for arm in ARMS}
        active = set(ARMS)
        pre_epoch = time.monotonic() - started
        epoch_start = time.monotonic()
        for arm in ARMS:
            loss, step = exp058.train_epoch(
                torch, arm, models[arm], opts[arm], fp=torch.as_tensor(train_fp, dtype=torch.float32),
                drug=train_drug_t, state=st_tr[arm], residual_target=residual,
                row_weights=weights_t, gene_variance=variance_t, batch_size=64, seed=58058 + seeds[arm])
            dv_nmse = exp058.score_object_macro_nmse(
                torch, models[arm], base_dv_t, torch.as_tensor(dev_fp, dtype=torch.float32),
                dev_drug_t, st_dv[arm], dev_y_t, dk, variance_t)
            best_state[arm] = exp058.clone_state(torch, models[arm])
            best_epoch[arm], best_nmse[arm] = 1, dv_nmse
            exp058.save_checkpoint(torch, attempt / f"epoch1_{arm}.pt", arm,
                                   best_state[arm], 1, dv_nmse, seeds[arm])
            curves.append({"epoch": 1, "arm": arm,
                           "train_object_balanced_normalized_residual_mse": loss,
                           "inner_dev_source_object_macro_nmse": dv_nmse,
                           "improved_best": True, "patience": 0, "finite": True,
                           "train_batches": step["batches"]})
        epoch_seconds = time.monotonic() - epoch_start
        final_io_allowance = max(90.0, epoch_seconds * 0.12)
        projected = (pre_epoch + 300.0 * epoch_seconds + final_io_allowance) * 1.20
        package_wall = exp062_prior + projected
        cost_gate = {"status": "PASS" if projected <= remaining_cap and prior + package_wall <= inherited_cap else "FAIL_CLOSED_COST_GATE",
                     "cpu_package_cap_seconds": cap, "cpu_prior_EXP055_settled_seconds": prior,
                     "EXP062_metadata_and_root_readback_seconds": exp062_prior,
                     "remaining_EXP062_run_seconds": remaining_cap,
                     "EXP055_inherited_cap_seconds": inherited_cap,
                     "measured_pre_epoch_seconds": pre_epoch, "measured_all_arm_epoch1_seconds": epoch_seconds,
                     "three_trainable_arms": list(ARMS), "all_arm_epoch1_includes_train_eval_checkpoint": True,
                     "projected_full_300_epoch_schedule_seconds_before_margin": pre_epoch + 300*epoch_seconds + final_io_allowance,
                     "projection_margin_fraction": 0.20, "projected_package_seconds_with_margin": projected,
                     "projected_EXP062_package_seconds_with_margin": package_wall,
                     "projected_EXP055_cumulative_seconds_with_margin": prior + package_wall,
                     "package_cap_deficit_seconds": max(0.0, package_wall - cap),
                     "inherited_EXP055_cap_deficit_seconds": max(0.0, prior + package_wall - inherited_cap),
                     "continue_full_schedule": projected <= remaining_cap and prior + package_wall <= inherited_cap,
                     "schedule_shortened": False}
        atomic_json(attempt / "first_epoch_cost_gate.json", cost_gate)
        if not cost_gate["continue_full_schedule"]:
            # Preserve every first-epoch arm and baseline as cost evidence only.
            gate_train = {"zero_change": np.zeros_like(train_y), "per_drug_mean": train_base,
                          "linear_state_residual": linear_train}
            gate_dev = {"zero_change": np.zeros_like(dev_y), "per_drug_mean": dev_base,
                        "linear_state_residual": linear_dev}
            fp_all_tr = torch.as_tensor(train_fp, dtype=torch.float32)
            fp_all_dv = torch.as_tensor(dev_fp, dtype=torch.float32)
            with torch.no_grad():
                for arm in ARMS:
                    gate_train[arm] = (base_tr_t + models[arm](fp_all_tr, train_drug_t,
                        st_tr[arm])).cpu().numpy().astype(np.float64)
                    gate_dev[arm] = (base_dv_t + models[arm](fp_all_dv, dev_drug_t,
                        st_dv[arm])).cpu().numpy().astype(np.float64)
            np.savez_compressed(attempt / "first_epoch_cost_gate_predictions.npz",
                train_full_condition_key=tk.full_condition_key.astype(str).to_numpy(),
                inner_dev_full_condition_key=dk.full_condition_key.astype(str).to_numpy(),
                train_source_object_id=tk.source_object_id.astype(str).to_numpy(),
                inner_dev_source_object_id=dk.source_object_id.astype(str).to_numpy(),
                train_full_inchikey=tk.full_inchikey.astype(str).to_numpy(),
                inner_dev_full_inchikey=dk.full_inchikey.astype(str).to_numpy(),
                gene_axis=genes.astype(str), train_target_delta=train_y.astype(np.float32),
                inner_dev_target_delta=dev_y.astype(np.float32),
                **{f"{m}_train": gate_train[m].astype(np.float32) for m in METHODS},
                **{f"{m}_inner_dev": gate_dev[m].astype(np.float32) for m in METHODS})
            gate_metrics = {}
            for method in METHODS:
                tr_summary, _ = _object_macro_metrics(tk, train_y, gate_train[method], variance)
                dv_summary, _ = _object_macro_metrics(dk, dev_y, gate_dev[method], variance)
                gate_metrics[method] = {"train_in_sample": tr_summary,
                                        "inner_dev_epoch1_cost_gate_only": dv_summary}
            atomic_json(attempt / "first_epoch_cost_gate_metrics.json", safe(gate_metrics))
            atomic_json(attempt / "cost_gate_close.json", {"status": "CLOSED_AFTER_FIRST_EPOCH_COST_GATE",
                "candidate_score_is_final": False,
                "inner_dev_epoch1_use": "pre-registered early-stopping/cost-gate measurement; not a final score",
                "cost_gate": cost_gate, "no_schedule_shortening": True,
                "exact_package_deficit_seconds": cost_gate["package_cap_deficit_seconds"], "gpu_hours": 0})
            atomic_json(attempt / "run_manifest.json", {**manifest, "status": "CLOSED_AFTER_FIRST_EPOCH_COST_GATE",
                "elapsed_cpu_wall_seconds": exp062_prior + time.monotonic()-started,
                "EXP055_cumulative_cpu_seconds": prior + exp062_prior + time.monotonic()-started,
                "gpu_hours": 0})
            return 0
        all_fp_tr = torch.as_tensor(train_fp, dtype=torch.float32)
        all_fp_dv = torch.as_tensor(dev_fp, dtype=torch.float32)
        tr_pred_t = base_tr_t
        dv_pred_t = base_dv_t
        for epoch in range(2, 301):
            for arm in ARMS:
                if arm not in active:
                    continue
                loss, step = exp058.train_epoch(torch, arm, models[arm], opts[arm], fp=all_fp_tr,
                    drug=train_drug_t, state=st_tr[arm], residual_target=residual,
                    row_weights=weights_t, gene_variance=variance_t, batch_size=64,
                    seed=58058 + seeds[arm] + epoch)
                dv_nmse = exp058.score_object_macro_nmse(torch, models[arm], base_dv_t, all_fp_dv,
                    dev_drug_t, st_dv[arm], dev_y_t, dk, variance_t)
                improved = dv_nmse < best_nmse[arm] - 1e-12
                if improved:
                    best_state[arm] = exp058.clone_state(torch, models[arm])
                    best_epoch[arm], best_nmse[arm] = epoch, dv_nmse
                    patience[arm] = 0
                    exp058.save_checkpoint(torch, attempt / f"best_{arm}.pt", arm,
                                           best_state[arm], epoch, dv_nmse, seeds[arm])
                else:
                    patience[arm] += 1
                curves.append({"epoch": epoch, "arm": arm,
                    "train_object_balanced_normalized_residual_mse": loss,
                    "inner_dev_source_object_macro_nmse": dv_nmse, "improved_best": improved,
                    "patience": patience[arm], "finite": True, "train_batches": step["batches"]})
                if patience[arm] >= 30:
                    active.remove(arm)
            elapsed = time.monotonic()-started
            if exp062_prior + elapsed > cap or prior + exp062_prior + elapsed > inherited_cap:
                raise TimeoutError(f"EXP-062 cumulative CPU budget exceeded after epoch {epoch}")
            if not active:
                break
        train_prediction: dict[str, np.ndarray] = {
            "zero_change": np.zeros_like(train_y), "per_drug_mean": train_base,
            "linear_state_residual": linear_train}
        dev_prediction: dict[str, np.ndarray] = {
            "zero_change": np.zeros_like(dev_y), "per_drug_mean": dev_base,
            "linear_state_residual": linear_dev}
        for arm in ARMS:
            models[arm].load_state_dict(best_state[arm], strict=True)
            with torch.no_grad():
                train_prediction[arm] = (base_tr_t + models[arm](all_fp_tr, train_drug_t,
                    st_tr[arm])).cpu().numpy().astype(np.float64)
                dev_prediction[arm] = (base_dv_t + models[arm](all_fp_dv, dev_drug_t,
                    st_dv[arm])).cpu().numpy().astype(np.float64)
            exp058.save_checkpoint(torch, attempt / f"final_{arm}.pt", arm, best_state[arm],
                                   best_epoch[arm], best_nmse[arm], seeds[arm])
        if any(not np.isfinite(v).all() for v in (*train_prediction.values(), *dev_prediction.values())):
            raise FloatingPointError("a fixed comparison arm produced non-finite final predictions")
        metrics: dict[str, Any] = {}
        object_rows: list[dict[str, Any]] = []
        for method in METHODS:
            tr_metrics, tr_rows = _object_macro_metrics(tk, train_y, train_prediction[method], variance)
            dv_metrics, dv_rows = _object_macro_metrics(dk, dev_y, dev_prediction[method], variance)
            metrics[method] = {"train_in_sample": tr_metrics, "inner_dev": dv_metrics}
            object_rows.extend({"method": method, "role": "train_in_sample", **x} for x in tr_rows)
            object_rows.extend({"method": method, "role": "inner_dev", **x} for x in dv_rows)
        controls = ["per_drug_mean", "linear_state_residual", "capacity_matched_drug_only", "shuffled_state"]
        strongest = min(controls, key=lambda m: float(metrics[m]["inner_dev"]["object_macro_nmse"]))
        cand = metrics["nonlinear_state_residual"]["inner_dev"]
        ref = metrics[strongest]["inner_dev"]
        gain = (float(ref["object_macro_nmse"]) - float(cand["object_macro_nmse"])) / float(ref["object_macro_nmse"])
        candidate_scores = cand["object_scores"]
        reference_scores = ref["object_scores"]
        paired_ids = sorted(set(candidate_scores) & set(reference_scores))
        diff = np.asarray([reference_scores[o]["nmse"] - candidate_scores[o]["nmse"] for o in paired_ids])
        reps, bootstrap_seed = paired_bootstrap_spec(cfg)
        rng = np.random.default_rng(bootstrap_seed)
        draws = rng.integers(0, len(paired_ids), size=(reps, len(paired_ids)))
        boot = diff[draws].mean(axis=1)
        ci = np.quantile(boot, [0.025, 0.975]).astype(float).tolist()
        fraction = float(np.mean(diff > 0)) if len(diff) else None
        amp = cand.get("object_macro_amplitude_ratio")
        beats_all = all(float(cand["object_macro_nmse"]) < float(metrics[m]["inner_dev"]["object_macro_nmse"])
                        for m in controls)
        paired_comparisons: dict[str, Any] = {}
        for control in controls:
            control_scores = metrics[control]["inner_dev"]["object_scores"]
            common = sorted(set(candidate_scores) & set(control_scores))
            control_diff = np.asarray([control_scores[o]["nmse"] - candidate_scores[o]["nmse"]
                                       for o in common], dtype=np.float64)
            control_draws = rng.integers(0, len(common), size=(reps, len(common)))
            control_boot = control_diff[control_draws].mean(axis=1)
            paired_comparisons[control] = {
                "paired_objects": len(common), "control_minus_candidate_nmse": float(control_diff.mean()),
                "bootstrap_95pct_ci": np.quantile(control_boot, [0.025, 0.975]).astype(float).tolist(),
                "objects_improved": int((control_diff > 0).sum()),
                "fraction_objects_improved": float(np.mean(control_diff > 0)) if len(control_diff) else None}
        gate_pass = bool(gain >= 0.05 and len(ci) == 2 and ci[0] > 0 and fraction is not None
                         and fraction >= 0.60 and amp is not None and 0.5 <= float(amp) <= 2.0 and beats_all)
        decision = {"strongest_fixed_control": strongest, "candidate_inner_dev_object_macro_nmse": cand["object_macro_nmse"],
                    "strongest_control_inner_dev_object_macro_nmse": ref["object_macro_nmse"],
                    "relative_nmse_gain": gain, "paired_object_bootstrap_95pct_ci_absolute_gain": ci,
                    "objects_improved": int((diff > 0).sum()), "paired_objects": len(paired_ids),
                    "fraction_objects_improved": fraction, "candidate_amplitude_ratio": amp,
                    "paired_object_bootstrap_vs_each_fixed_control": paired_comparisons,
                    "beats_every_fixed_control": beats_all, "expansion_gate_pass": gate_pass,
                    "candidate_disposition": "QUALIFIES_FOR_AUTHORIZED_CONFIRMATION" if gate_pass else "CLOSED_NO_SEARCH_EXPANSION",
                    "claim_boundary": "fixed XPert source-development split only; no unseen drug, independent test, PDO efficacy, or clinical claim"}
        np.savez_compressed(attempt / "keyed_predictions.npz",
            train_full_condition_key=tk.full_condition_key.astype(str).to_numpy(),
            inner_dev_full_condition_key=dk.full_condition_key.astype(str).to_numpy(),
            train_source_object_id=tk.source_object_id.astype(str).to_numpy(),
            inner_dev_source_object_id=dk.source_object_id.astype(str).to_numpy(),
            train_full_inchikey=tk.full_inchikey.astype(str).to_numpy(),
            inner_dev_full_inchikey=dk.full_inchikey.astype(str).to_numpy(),
            gene_axis=genes.astype(str), train_target_delta=train_y.astype(np.float32),
            inner_dev_target_delta=dev_y.astype(np.float32),
            **{f"{m}_train": train_prediction[m].astype(np.float32) for m in METHODS},
            **{f"{m}_inner_dev": dev_prediction[m].astype(np.float32) for m in METHODS})
        pd.DataFrame(object_rows).to_csv(attempt / "object_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(curves).to_csv(attempt / "training_curves.tsv", sep="\t", index=False)
        drug_metric_rows: list[dict[str, Any]] = []
        for role, part_keys, target, predictions in (("train_in_sample", tk, train_y, train_prediction),
                                                       ("inner_dev", dk, dev_y, dev_prediction)):
            for key in sorted(set(part_keys.full_inchikey.astype(str))):
                ix = np.flatnonzero(part_keys.full_inchikey.astype(str).to_numpy() == key)
                subkeys = part_keys.iloc[ix].reset_index(drop=True)
                for method in METHODS:
                    summ, _ = _object_macro_metrics(subkeys, target[ix], predictions[method][ix], variance)
                    drug_metric_rows.append({"role": role, "full_inchikey": key, "method": method,
                                             **{k: v for k, v in summ.items() if k != "object_scores"}})
        pd.DataFrame(drug_metric_rows).to_csv(attempt / "per_drug_metrics.tsv", sep="\t", index=False)
        pd.DataFrame([{"source_object_id": o, "reference": strongest,
                       "candidate": "nonlinear_state_residual", "reference_minus_candidate_nmse": float(diff[i])}
                      for i, o in enumerate(paired_ids)]).to_csv(attempt / "paired_object_differences.tsv", sep="\t", index=False)
        summary = {"experiment_id": "EXP-062", "run_id": run_id, "attempt_id": attempt_id,
                   "status": "COMPLETED_FIXED_SOURCE_DEVELOPMENT_COMPARISON",
                   "cpu_wall_seconds": exp062_prior + time.monotonic()-started,
                   "EXP055_prior_cpu_settled_seconds": prior,
                   "EXP055_cumulative_cpu_seconds": prior + exp062_prior + time.monotonic()-started,
                   "cpu_cap_seconds": cap, "gpu_hours": 0,
                   "denominators": {"train_conditions": len(tk), "train_objects": len(train_objects),
                       "inner_dev_conditions": len(dk), "inner_dev_objects": len(dev_objects),
                       "object_overlap": 0, "full_inchikeys": len(drug_keys), "genes": len(genes),
                       "train_gene_variance_usable": int((variance > 1e-12).sum())},
                   "input_receipt": input_receipt, "cost_gate": cost_gate,
                   "paired_object_bootstrap": {"registered_spec": cfg["metrics"]["paired_interval"],
                                                "replicates": reps, "seed": bootstrap_seed},
                   "training_curves": len(curves), "best_epochs": best_epoch, "last_epoch": epoch,
                   "metrics": {m: {r: {k:v for k,v in metrics[m][r].items() if k != "object_scores"}
                                   for r in ("train_in_sample", "inner_dev")} for m in METHODS},
                   "decision": decision,
                   "forbidden_numeric_values_opened": {"pdo": False, "outer": False, "locked": False,
                                                        "panacea": False, "cdsdb": False}}
        atomic_json(attempt / "metrics_summary.json", safe(summary))
        atomic_json(attempt / "decision.json", safe(decision))
        output_hashes = {p.name: sha256_file(p) for p in attempt.iterdir() if p.is_file()}
        atomic_json(attempt / "root_readback_input_manifest.json", {"experiment_id": "EXP-062",
            "run_id": run_id, "attempt_id": attempt_id, "status": summary["status"],
            "output_sha256": output_hashes, "summary_sha256": sha256_file(attempt / "metrics_summary.json"),
            "prediction_sha256": sha256_file(attempt / "keyed_predictions.npz")})
        atomic_json(attempt / "run_manifest.json", {**manifest, "status": summary["status"],
            "elapsed_cpu_wall_seconds": exp062_prior + time.monotonic()-started,
            "EXP055_cumulative_cpu_seconds": prior + exp062_prior + time.monotonic()-started,
            "gpu_hours": 0, "outputs_sha256": output_hashes})
        print(json.dumps({"status": summary["status"], "decision": decision,
                          "denominators": summary["denominators"],
                          "cpu_wall_seconds": summary["cpu_wall_seconds"], "gpu_hours": 0}, ensure_ascii=False))
        return 0
    except Exception as exc:
        elapsed = time.monotonic()-started
        atomic_json(attempt / "failure_close_receipt.json", {"experiment_id": "EXP-062", "run_id": run_id,
            "attempt_id": attempt_id, "status": "FAILED_CLOSED", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "elapsed_cpu_wall_seconds": exp062_prior + elapsed,
            "EXP055_cumulative_cpu_seconds": prior + exp062_prior + elapsed, "gpu_hours": 0,
            "forbidden_numeric_values_opened": {"pdo": False, "outer": False, "locked": False,
                                                 "panacea": False, "cdsdb": False}})
        atomic_json(attempt / "run_manifest.json", {**manifest, "status": "FAILED_CLOSED",
            "elapsed_cpu_wall_seconds": exp062_prior + elapsed,
            "EXP055_cumulative_cpu_seconds": prior + exp062_prior + elapsed,
            "gpu_hours": 0})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True, choices=("metadata", "run"))
    parser.add_argument("--attempt-id", default="fit_attempt_001")
    args = parser.parse_args()
    cfg, cfg_path, out, _ = config_and_paths(args)
    if args.phase == "metadata":
        return metadata_phase(cfg, cfg_path, out, args.run_id)
    return run_phase(cfg, cfg_path, out, args.run_id, args.attempt_id)


if __name__ == "__main__":
    raise SystemExit(main())
