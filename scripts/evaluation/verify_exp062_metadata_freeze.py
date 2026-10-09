"""Independent root readback of the EXP-062 metadata-only freeze.

This verifier recomputes deterministic source roles, full-key coverage, hash
selection and selected ECFP4 vectors from obs/var and drug identity metadata.
It never accesses H5AD X, layers, or obsm['X_ctl'] values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def verified_bits(smiles: str, key: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.MolToInchiKey(mol) != key:
        raise ValueError(f"canonical_smiles does not reproduce exact full key: {key}")
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    out = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-062" or cfg.get("run_id") != args.run_id:
        raise ValueError("config/run_id mismatch")
    out = ROOT / cfg["output"]["root"]
    freeze_path = out / "metadata_freeze_manifest.json"
    selected_path = out / "response_blind_conditions.tsv"
    role_path = out / "source_object_roles.tsv"
    drug_candidate_path = out / "response_blind_drug_candidates.tsv"
    selected_drug_path = out / "selected_drugs.tsv"
    fp_path = out / "ecfp4_2048_by_full_inchikey.npz"
    required_paths = [freeze_path, selected_path, role_path, drug_candidate_path,
                      selected_drug_path, fp_path]
    if not all(p.is_file() for p in required_paths):
        raise FileNotFoundError("metadata freeze output set incomplete")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS_METADATA_FREEZE_PENDING_ROOT_READBACK":
        raise ValueError("source metadata stage did not produce a pending PASS freeze")
    forbidden = freeze.get("matrix_access", {})
    if any(forbidden.get(k) is not False for k in ("h5ad_X_read", "h5ad_X_ctl_read",
                                                     "h5ad_layers_read", "h5ad_obsm_numeric_values_read")):
        raise ValueError("metadata-stage manifest does not prove the numeric arrays remained unopened")
    h5ad = Path(cfg["data"]["h5ad_path"])
    drug_path = Path(cfg["data"]["source_drug_info"])
    a = ad.read_h5ad(h5ad, backed="r")
    try:
        metadata_columns = ["sample_id", "pert_id", "pert_idx", "cell_iname", "cell_idx",
                            "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
                            "pert_type", "project_code", "build_name", "split_cold_cell_1"]
        obs = a.obs[metadata_columns].copy().reset_index(drop=True)
        obs["obs_name"] = np.asarray(a.obs_names.astype(str), dtype=str)
        obs["source_row"] = np.arange(len(obs), dtype=np.int64)
        var_names = np.asarray(a.var_names.astype(str), dtype=str)
        shape = list(map(int, a.shape))
        obsm_keys = list(a.obsm.keys())
    finally:
        a.file.close()
    if shape != freeze["source_h5ad_shape"] or shape[1] != 978 or "X_ctl" not in obsm_keys:
        raise ValueError("official source H5AD metadata shape/keys changed")
    gene_map_path = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/gene_axis_mapping.tsv"
    old_path = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/response_blind_conditions.tsv"
    pan_path = ROOT / "artifacts/experiments/EXP-057/20261007_exp057_panacea_external_development_v6_20261007T0823Z/condition_metadata.tsv"
    gene_map = pd.read_csv(gene_map_path, sep="\t", dtype=str, keep_default_na=False)
    if len(gene_map) != 978 or not np.array_equal(var_names, gene_map.h5ad_var_name.astype(str).to_numpy()):
        raise ValueError("source gene axis differs from root-read EXP-051 exact 978 axis")
    old = pd.read_csv(old_path, sep="\t", dtype=str, keep_default_na=False,
                      usecols=["source_object_id", "split_role"])
    old_roles = old.groupby("source_object_id").split_role.agg(lambda v: set(map(str, v))).to_dict()
    pan = pd.read_csv(pan_path, sep="\t", dtype=str, keep_default_na=False,
                      usecols=["cell_iname"])
    pan_names = set(pan.cell_iname.astype(str).str.strip())
    obs["source_object_id"] = obs.cell_iname.astype(str).str.strip()
    obs["pert_id"] = obs.pert_id.astype(str).str.strip()
    obs["sample_id"] = obs.sample_id.astype(str).str.strip()
    obs["official_role"] = obs.split_cold_cell_1.astype(str).str.strip().str.lower()
    if not set(obs.official_role.unique()).issubset({"train", "test"}):
        raise ValueError("official split has an unrecognized role label")
    role_records = []
    for obj, group in obs.groupby("source_object_id", sort=True):
        official = set(group.official_role)
        if len(official) != 1:
            raise ValueError(f"official role changes within source object {obj}")
        official_role = next(iter(official))
        prior = sorted(old_roles.get(str(obj), set()))
        pan_overlap = str(obj) in pan_names
        effective = "inner_dev" if (official_role == "test" or "inner_dev" in prior or pan_overlap) else "train"
        role_records.append({"source_object_id": str(obj), "official_deterministic_role": official_role,
                             "legacy_project_roles": json.dumps(prior, ensure_ascii=False),
                             "panacea_exact_name_overlap": pan_overlap,
                             "effective_split_role": effective,
                             "legacy_official_conflict": bool("train" in prior and effective == "inner_dev")})
    role_recomputed = pd.DataFrame(role_records)
    role_frozen = pd.read_csv(role_path, sep="\t", dtype=str, keep_default_na=False)
    role_recomputed = role_recomputed.sort_values("source_object_id").reset_index(drop=True)
    role_frozen_sorted = role_frozen.sort_values("source_object_id").reset_index(drop=True)
    role_cols = ["source_object_id", "official_deterministic_role", "legacy_project_roles",
                 "panacea_exact_name_overlap", "effective_split_role", "legacy_official_conflict"]
    for c in role_cols:
        if c in {"panacea_exact_name_overlap", "legacy_official_conflict"}:
            left = role_recomputed[c].astype(str).str.lower()
            right = role_frozen_sorted[c].astype(str).str.lower()
        elif c == "legacy_project_roles":
            left = role_recomputed[c].astype(str)
            right = role_frozen_sorted[c].astype(str).map(lambda x: json.dumps(sorted(json.loads(x)), ensure_ascii=False))
        else:
            left, right = role_recomputed[c].astype(str), role_frozen_sorted[c].astype(str)
        if not left.equals(right):
            raise ValueError(f"root source-object role recomputation mismatch in {c}")
    obs = obs.merge(role_recomputed[["source_object_id", "effective_split_role"]],
                    on="source_object_id", how="left", validate="many_to_one")

    drug = pd.read_csv(drug_path, dtype=str, keep_default_na=False, low_memory=False,
                       usecols=["pert_id", "inchi_key", "canonical_smiles"])
    drug["pert_id"] = drug.pert_id.astype(str).str.strip()
    drug["full_inchikey"] = drug.inchi_key.astype(str).str.strip().str.upper()
    valid = drug.full_inchikey.str.fullmatch(KEY_RE, na=False)
    key_count = drug.loc[valid].groupby("pert_id").full_inchikey.nunique()
    valid_ids = set(key_count.index[key_count.eq(1)])
    pert_map = drug.loc[valid & drug.pert_id.isin(valid_ids), ["pert_id", "full_inchikey"]].drop_duplicates()
    if pert_map.pert_id.duplicated().any():
        raise ValueError("root exact pert_id-to-full-key mapping is not unique")
    fixed = obs.loc[
        pd.to_numeric(obs.pert_dose, errors="coerce").eq(10.0)
        & obs.pert_dose_unit.astype(str).str.strip().str.lower().eq("um")
        & pd.to_numeric(obs.pert_time, errors="coerce").eq(24.0)
        & obs.pert_time_unit.astype(str).str.strip().str.lower().eq("h")
        & obs.pert_type.astype(str).str.strip().eq("trt_cp")
    ].copy()
    fixed = fixed.merge(pert_map, on="pert_id", how="inner", validate="many_to_one")
    key_fields = ["source_object_id", "full_inchikey", "pert_id", "pert_dose", "pert_dose_unit",
                  "pert_time", "pert_time_unit", "project_code", "build_name", "sample_id", "obs_name"]
    fixed["full_condition_key"] = fixed[key_fields].astype(str).agg("|".join, axis=1)
    seed = int(cfg["data"]["selection_seed"])
    fixed["selection_hash_sha256"] = fixed.apply(lambda row: hashlib.sha256(
        f"{seed}|{row.full_inchikey}|{row.full_condition_key}".encode("utf-8")).hexdigest(), axis=1)
    one = fixed.sort_values(["selection_hash_sha256", "full_condition_key"], kind="mergesort").drop_duplicates(
        ["source_object_id", "full_inchikey"], keep="first")
    cover = one.groupby("full_inchikey", sort=True).agg(
        source_objects=("source_object_id", "nunique"),
        train_objects=("effective_split_role", lambda v: int(v.eq("train").sum())),
        dev_objects=("effective_split_role", lambda v: int(v.eq("inner_dev").sum()))).reset_index()
    candidate_keys = set(cover.loc[cover.train_objects.ge(int(cfg["data"]["min_train_objects_per_drug"])),
                                   "full_inchikey"].astype(str))
    fp_by_key: dict[str, np.ndarray] = {}
    invalid_fp: dict[str, str] = {}
    for key in sorted(candidate_keys):
        group = drug.loc[drug.full_inchikey.eq(key), ["pert_id", "canonical_smiles"]].drop_duplicates()
        if group.empty:
            invalid_fp[key] = "no_canonical_smiles"
            continue
        by_hash = {}
        try:
            for row in group.itertuples(index=False):
                bits = verified_bits(str(row.canonical_smiles), key)
                by_hash[hashlib.sha256(bits.tobytes()).hexdigest()] = bits
            if len(by_hash) != 1:
                invalid_fp[key] = "aliases_do_not_match"
            else:
                fp_by_key[key] = next(iter(by_hash.values()))
        except Exception as e:
            invalid_fp[key] = type(e).__name__ + ":" + str(e)
    candidates = cover.loc[cover.full_inchikey.astype(str).isin(fp_by_key)].copy().sort_values(
        ["train_objects", "full_inchikey"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    candidates["precap_rank"] = np.arange(1, len(candidates)+1, dtype=np.int64)
    selected_parts, selected_drugs = [], []
    max_per = int(cfg["data"]["max_conditions_per_drug"])
    for r in candidates.itertuples(index=False):
        key = str(r.full_inchikey)
        subset = one.loc[one.full_inchikey.astype(str).eq(key)].sort_values(
            ["selection_hash_sha256", "full_condition_key"], kind="mergesort").head(max_per).copy()
        if int(subset.effective_split_role.eq("train").sum()) < int(cfg["data"]["min_train_objects_per_drug"]):
            continue
        if len(selected_drugs) >= int(cfg["data"]["max_drugs"]):
            continue
        rank = len(selected_drugs)+1
        subset["drug_selection_rank"] = rank
        selected_parts.append(subset)
        selected_drugs.append(key)
    recomputed = pd.concat(selected_parts, ignore_index=True)
    recomputed = recomputed.sort_values(["drug_selection_rank", "selection_hash_sha256", "full_condition_key"],
                                        kind="mergesort").reset_index(drop=True)
    recomputed["selection_rank"] = np.arange(1, len(recomputed)+1, dtype=np.int64)
    frozen = pd.read_csv(selected_path, sep="\t", dtype=str, keep_default_na=False)
    frozen["source_row"] = pd.to_numeric(frozen.source_row, errors="raise").astype(np.int64)
    frozen["selection_rank"] = pd.to_numeric(frozen.selection_rank, errors="raise").astype(np.int64)
    frozen["drug_selection_rank"] = pd.to_numeric(frozen.drug_selection_rank, errors="raise").astype(np.int64)
    required_cmp = ["source_row", "obs_name", "source_object_id", "pert_id", "full_inchikey",
                    "full_condition_key", "selection_hash_sha256", "effective_split_role",
                    "selection_rank", "drug_selection_rank"]
    if len(recomputed) != len(frozen):
        raise ValueError(f"independent row selection count differs: {len(recomputed)} vs {len(frozen)}")
    for col in required_cmp:
        if not np.array_equal(recomputed[col].astype(str).to_numpy(), frozen[col].astype(str).to_numpy()):
            raise ValueError(f"independent selected-row recomputation differs in {col}")
    selected_drug_table = pd.read_csv(selected_drug_path, sep="\t", dtype=str, keep_default_na=False)
    if selected_drug_table.full_inchikey.astype(str).tolist() != selected_drugs:
        raise ValueError("independent selected-drug order differs from frozen selection")
    fp_archive = np.load(fp_path, allow_pickle=False)
    fp_keys = fp_archive["full_inchikey"].astype(str).tolist()
    fp_matrix = fp_archive["features"].astype(np.uint8)
    if fp_keys != selected_drugs or fp_matrix.shape != (len(fp_keys), 2048):
        raise ValueError("frozen feature archive keys/order/shape mismatch")
    for i, key in enumerate(fp_keys):
        if not np.array_equal(fp_matrix[i], fp_by_key[key]):
            raise ValueError(f"root ECFP4 recomputation differs for selected full key {key}")
    train_obj = sorted(set(frozen.loc[frozen.effective_split_role.eq("train"), "source_object_id"].astype(str)))
    dev_obj = sorted(set(frozen.loc[frozen.effective_split_role.eq("inner_dev"), "source_object_id"].astype(str)))
    if set(train_obj) & set(dev_obj):
        raise ValueError("frozen role overlap is nonzero")
    if len(frozen) > int(cfg["data"]["max_total_conditions"]) or (
            frozen.groupby("full_inchikey").size() > max_per).any():
        raise ValueError("frozen row caps exceeded")
    h5ad_sha = sha256(h5ad)
    if h5ad_sha != freeze.get("source_h5ad_sha256"):
        raise ValueError("frozen H5AD checksum differs from root readback")
    checks = {"role_assignment_recomputed": True, "exact_full_key_mapping_recomputed": True,
              "response_blind_drug_ranking_recomputed": True, "condition_hash_selection_recomputed": True,
              "selected_row_source_identity_recomputed": True, "2048_bit_ECFP4_recomputed": True,
              "gene_axis_exact_978_recomputed": True, "X_Xctl_layers_values_read": False,
              "selected_source_rows_only_for_future_value_phase": True,
              "pdo_outer_locked_panacea_cdsdb_numeric_values_read": False}
    receipt = {"experiment_id": "EXP-062", "run_id": args.run_id,
        "status": "PASS_ROOT_INDEPENDENT_METADATA_READBACK", "checks": checks,
        "metadata_freeze_sha256": sha256(freeze_path), "selected_condition_table_sha256": sha256(selected_path),
        "selected_drugs_sha256": sha256(selected_drug_path), "fingerprint_archive_sha256": sha256(fp_path),
        "source_h5ad_sha256": h5ad_sha, "official_source_shape": shape,
        "train_conditions": int(frozen.effective_split_role.eq("train").sum()),
        "train_objects": len(train_obj), "inner_dev_conditions": int(frozen.effective_split_role.eq("inner_dev").sum()),
        "inner_dev_objects": len(dev_obj), "train_inner_dev_object_overlap": 0,
        "full_inchikeys": len(fp_keys), "genes": len(var_names),
        "official_deterministic_objects": len(role_recomputed),
        "legacy_official_role_conflicts": int(role_recomputed.legacy_official_conflict.sum()),
        "panacea_exact_name_development_objects": sorted(role_recomputed.loc[
            role_recomputed.panacea_exact_name_overlap.astype(str).str.lower().eq("true"),
            "source_object_id"].astype(str).tolist()),
        "selected_drugs": selected_drugs, "selected_condition_rows": len(frozen),
        "fingerprint_method": "RDKit Morgan radius2, 2048 bits, official canonical_smiles exact full-key recomputation",
        "elapsed_cpu_wall_seconds": time.monotonic()-started, "gpu_hours": 0,
        "root_verifier_sha256": sha256(Path(__file__).resolve())}
    receipt_path = out / "root_metadata_readback.json"
    if receipt_path.exists():
        raise FileExistsError("root metadata readback receipt already exists; immutable")
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "selected_drugs": len(selected_drugs),
                      "selected_conditions": len(frozen), "train_conditions": receipt["train_conditions"],
                      "inner_dev_conditions": receipt["inner_dev_conditions"],
                      "train_objects": len(train_obj), "inner_dev_objects": len(dev_obj),
                      "cpu_wall_seconds": receipt["elapsed_cpu_wall_seconds"], "gpu_hours": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
