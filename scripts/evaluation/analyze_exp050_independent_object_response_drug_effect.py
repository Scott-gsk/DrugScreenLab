#!/usr/bin/env python3
"""Frozen CPU analysis of XPert response transfer versus CTRP/PRISM sensitivity.

Run in two phases. ``roles`` projects only identity/role/drug-key identifiers and
must finish before ``analyze`` reads any CTRP AAC or PRISM -LFC values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_independent_object_response_drug_effect_v1"
ATTEMPT_ID = "20261001_xpert_independent_object_fit_v1/attempts/attempt_001"
MIN_FULL_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
GENE_SETS = (
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
)
N_BOOT = 50000
BOOT_SEED = 20261001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace("", pd.NA)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks required columns: {sorted(missing)}")


def load_contract(root: Path) -> dict[str, Any]:
    record = (root / "experiments/records/EXP-050.md").read_text(encoding="utf-8")
    marker = "### 2026-10-01：冻结独立来源响应—药效关联CPU协议"
    if marker not in record:
        raise ValueError("EXP-050 section 3.4 CPU protocol marker is missing")
    config = __import__("yaml").safe_load((root / "configs/experiments/EXP-050.yaml").read_text(encoding="utf-8"))
    return {
        "run_id": RUN_ID,
        "protocol_source": "experiments/records/EXP-050.md, 2026-10-01 section 3.4",
        "execution_role": "evaluation_statistics_analyst; requested gpt-6-luna/xhigh",
        "cpu_limit_minutes": 120,
        "gpu_hours": 0,
        "response_source": "XPert attempt_001 frozen 467-condition prediction and measured response arrays",
        "source_objects": 10,
        "heldout_conditions": 467,
        "supported_object_full_inchikey_pairs": 462,
        "key": ["source_object_id", "full_inchikey"],
        "endpoints": {
            "CTRPv2": {"column": "aac_published", "higher_is_more_sensitive": True},
            "PRISM": {"column": "response", "manifest_name": "neg_extended_primary_lfc", "higher_is_more_sensitive": True},
        },
        "allowed_assay_roles": ["train", "val"],
        "forbidden": ["PDO treated labels", "outer/locked numeric values, labels, features, or scores", "model or feature selection from these labels"],
        "lincs_exposure": "10 uM x 24 h; assay exposures/platforms differ from CTRPv2 and PRISM",
        "aggregation": {
            "condition_magnitude": "mean absolute delta over the fixed 978-gene axis per condition",
            "repeated_conditions": "median of every frozen qualifying condition within source object x full InChIKey; no dose/time selection",
            "train_drug_mean": "per-key gene-wise signed mean over XPert TRAIN response rows only, then mean absolute value over the same gene set; excludes inner-dev, heldout, and direct assay values",
            "features": ["all_978_genes", *GENE_SETS],
        },
        "statistics": {
            "within_object": "Spearman across exact drug keys separately within each source object; report n and undefined reason; object macro average",
            "object_bootstrap_replicates": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "object_between": "pooled correlation before and after subtracting each exact-drug shared mean from both response magnitude and assay value; exploratory association only",
            "same_denominator_domains": ["measured_and_predicted_complete", "three_arm_common_with_train_mean"],
            "missing_policy": "no zero imputation",
        },
        "prior_label_exposure": "earlier single-source exploration included CTRPv2 22 exact drugs and PRISM 8 exact drugs for PT-pYG6Rc/MDAMB231; current results are not independent confirmation",
        "five_arm_gate": "metadata/key support only; no same-object fit-and-score; a single independent VAL object cannot establish generalization",
        "config_status": config.get("source_lincs_ctrp_association_exploratory", {}).get("status"),
    }


def frozen_roles(root: Path) -> pd.DataFrame:
    path = root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "IDENTITY_SPLIT_FROZEN":
        raise ValueError("EXP-007 identity/role mapping is not frozen")
    identity = pd.DataFrame(value.get("identity_rows", []))
    split = pd.DataFrame(value.get("split_assignments", []))
    require_columns(identity, {"lincs_cell_id", "depmap_id", "patient_id", "expression_present", "join_status"}, "EXP-007 identity_rows")
    require_columns(split, {"depmap_id", "split_role"}, "EXP-007 split_assignments")
    identity = identity.loc[
        identity.expression_present.astype(bool)
        & identity.join_status.eq("EXACT_STRIPPED_NAME_WITH_RNASEQ"),
        ["lincs_cell_id", "depmap_id", "patient_id", "join_status"],
    ].dropna().drop_duplicates()
    split = split[["depmap_id", "split_role"]].dropna().drop_duplicates()
    if identity.depmap_id.duplicated().any() or split.depmap_id.duplicated().any():
        raise ValueError("EXP-007 frozen identity/role keys are not unique")
    rows = identity.merge(split, on="depmap_id", how="inner", validate="one_to_one")
    if not rows.split_role.isin(["train", "val", "locked_eval"]).all():
        raise ValueError("EXP-007 contains unexpected source role")
    model = pd.read_csv(root / "data/raw/depmap/24q2/Model.csv", usecols=["ModelID", "PatientID"], dtype="string")
    model = model.dropna().drop_duplicates()
    check = rows.merge(model, left_on="depmap_id", right_on="ModelID", how="inner", validate="one_to_one")
    if len(check) != len(rows) or not check.patient_id.astype(str).eq(check.PatientID.astype(str)).all():
        raise ValueError("EXP-007 patient IDs disagree with DepMap 24Q2 Model.csv")
    return rows.rename(columns={"patient_id": "source_object_id"})[["source_object_id", "lincs_cell_id", "depmap_id", "split_role", "join_status"]]


def load_target_keys(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = root / "artifacts/experiments/EXP-050" / ATTEMPT_ID / "heldout_source_prediction_keys.tsv"
    receipt_path = path.parent / "attempt_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETED_ONE_FRESH_XPERT_FIT":
        raise ValueError("XPert attempt is not a completed frozen fit")
    if receipt.get("output_sha256", {}).get(path.name) != sha256(path):
        raise ValueError("XPert heldout key table checksum differs from attempt receipt")
    frame = pd.read_csv(path, sep="\t", usecols=["condition_id", "source_row", "source_object_id", "full_inchikey"], dtype={"source_object_id": "string", "full_inchikey": "string"})
    frame["full_inchikey"] = normalize(frame.full_inchikey)
    frame["condition_id"] = pd.to_numeric(frame.condition_id, errors="raise").astype(int)
    frame["source_row"] = pd.to_numeric(frame.source_row, errors="raise").astype(int)
    if len(frame) != 467 or frame.condition_id.duplicated().any() or frame.source_row.duplicated().any():
        raise ValueError("Frozen XPert target keys fail 467-condition unique identity check")
    if not frame.full_inchikey.str.fullmatch(MIN_FULL_KEY).all():
        raise ValueError("Frozen XPert target table has invalid complete InChIKey")
    pairs = frame[["source_object_id", "full_inchikey"]].drop_duplicates()
    if len(pairs) != 462 or frame.source_object_id.nunique() != 10:
        raise ValueError("Frozen XPert support differs from 10 objects/462 object-key pairs")
    return frame, receipt


def join_heldout_identity_whitelist(root: Path, targets: pd.DataFrame, receipt: dict[str, Any]) -> pd.DataFrame:
    """Restrict fit-condition identity metadata to the frozen 467 target IDs first."""
    path = root / "artifacts/experiments/EXP-050" / ATTEMPT_ID / "fit_condition_identity_key_map.tsv"
    if receipt.get("output_sha256", {}).get(path.name) != sha256(path):
        raise ValueError("XPert identity/key map checksum differs from attempt receipt")
    allowed_ids = set(targets.condition_id.astype(int))
    fields = ["condition_id", "source_row", "split_role", "source_object_id", "full_inchikey", "cell_id"]
    parts=[]
    for chunk in pd.read_csv(path, sep="\t", usecols=fields, dtype={"source_object_id":"string","full_inchikey":"string","cell_id":"string"}, chunksize=10000):
        chunk["condition_id"] = pd.to_numeric(chunk.condition_id, errors="raise").astype(int)
        selected=chunk.loc[chunk.condition_id.isin(allowed_ids)]
        if not selected.empty: parts.append(selected)
    mapped=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=fields)
    if len(mapped)!=len(targets) or mapped.condition_id.duplicated().any() or set(mapped.condition_id)!=allowed_ids:
        raise ValueError("Heldout condition ID whitelist did not yield exactly 467 unique identity rows")
    compare=targets[["condition_id","source_row","source_object_id","full_inchikey"]].merge(
        mapped[["condition_id","source_row","source_object_id","full_inchikey","split_role","cell_id"]],
        on="condition_id",how="outer",suffixes=("_target","_map"),validate="one_to_one",indicator=True)
    if not compare._merge.eq("both").all() or not compare.source_row_target.eq(compare.source_row_map).all() or not compare.source_object_id_target.eq(compare.source_object_id_map).all() or not compare.full_inchikey_target.eq(compare.full_inchikey_map).all() or not compare.split_role.eq("heldout_source").all():
        raise ValueError("Heldout keys and XPert condition identity/key map differ")
    identity = frozen_roles(root).rename(columns={"lincs_cell_id":"cell_id", "split_role":"drug_role"})
    joined=mapped.merge(identity[["cell_id","depmap_id","source_object_id","drug_role","join_status"]],on="cell_id",how="left",suffixes=("_map","_identity"),validate="many_to_one")
    if joined.depmap_id.isna().any() or joined.drug_role.isna().any() or not joined.source_object_id_map.eq(joined.source_object_id_identity).all():
        raise ValueError("Heldout XPert object to EXP-007 exact identity/role join failed")
    if not joined.drug_role.isin(["train","val"]).all():
        raise ValueError("Heldout object identity has unknown or locked role; do not read endpoint labels")
    return joined


def ctrp_identifiers(root: Path, objects: pd.DataFrame, target_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mapping_path = root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv"
    mapping = pd.read_csv(mapping_path, usecols=["cellid", "ModelID", "match_method", "has_depmap_expression"], dtype="string")
    mapping = mapping.loc[mapping.match_method.eq("exact_unique_cellosaurus_rrid") & mapping.has_depmap_expression.astype("string").str.lower().eq("true"), ["cellid", "ModelID"]].dropna().drop_duplicates()
    models = pd.read_csv(root / "data/raw/depmap/24q2/Model.csv", usecols=["ModelID", "PatientID"], dtype="string").dropna().drop_duplicates()
    if models.ModelID.duplicated().any():
        raise ValueError("DepMap ModelID is not unique for CTRP mapping")
    cellmap = mapping.merge(models, on="ModelID", how="inner", validate="many_to_one")
    cellmap = cellmap.merge(objects[["source_object_id", "split_role"]], left_on="PatientID", right_on="source_object_id", how="inner", validate="many_to_one")
    cellmap = cellmap.loc[cellmap.split_role.isin(["train", "val"])].copy()
    drugs = pd.read_csv(root / "data/processed/exp013/ctrp_v2/drugs.csv", usecols=["drugid", "inchikey"], dtype="string")
    drugs["inchikey"] = normalize(drugs.inchikey)
    drugs = drugs.dropna(subset=["drugid", "inchikey"])
    key_counts = drugs.groupby("inchikey").drugid.nunique()
    drugs = drugs.loc[drugs.inchikey.map(key_counts).eq(1), ["drugid", "inchikey"]].drop_duplicates()
    if drugs.drugid.duplicated().any():
        raise ValueError("CTRP drugid maps to multiple full InChIKeys")
    target_sets = target_pairs.groupby("source_object_id").full_inchikey.agg(set).to_dict()
    cells = set(cellmap.cellid.astype(str))
    chunks = []
    for chunk in pd.read_csv(root / "data/processed/exp013/ctrp_v2/responses.csv", usecols=["cellid", "drugid"], dtype="string", chunksize=100_000):
        chunk = chunk.loc[chunk.cellid.astype(str).isin(cells)]
        if not chunk.empty:
            chunks.append(chunk.drop_duplicates())
    ids = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["cellid", "drugid"])
    key_rows = ids.merge(cellmap[["cellid", "source_object_id", "split_role"]], on="cellid", how="inner", validate="many_to_one").merge(drugs, on="drugid", how="inner", validate="many_to_one")
    key_rows = key_rows.loc[key_rows.apply(lambda r: r.inchikey in target_sets.get(str(r.source_object_id), set()), axis=1), ["source_object_id", "split_role", "inchikey"]].drop_duplicates()
    key_rows = key_rows.rename(columns={"inchikey": "full_inchikey"})
    summary = {
        "mapped_train_val_objects": int(cellmap.source_object_id.nunique()),
        "mapped_train_val_cell_lines": int(cellmap.cellid.nunique()),
        "objects_with_target_drug_pairs": int(key_rows.source_object_id.nunique()),
        "full_inchikeys": int(key_rows.full_inchikey.nunique()),
        "object_full_inchikey_pairs": int(len(key_rows)),
        "per_role": {str(k): {"objects": int(key_rows.loc[key_rows.split_role.eq(k), "source_object_id"].nunique()), "drugs": int(key_rows.loc[key_rows.split_role.eq(k), "full_inchikey"].nunique()), "pairs": int(len(key_rows.loc[key_rows.split_role.eq(k)]))} for k in ["train", "val"]},
    }
    return cellmap, key_rows, summary


def prism_identifiers(root: Path, objects: pd.DataFrame, target_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = root / "data/processed/exp012/prism_extended_primary_samples.parquet"
    samples = pd.read_parquet(path, columns=["sample_id", "cell_line_id"])
    samples["sample_id"] = normalize(samples.sample_id)
    samples["cell_line_id"] = normalize(samples.cell_line_id)
    model = frozen_roles(root).rename(columns={"depmap_id": "model_id"})
    model = model.loc[model.source_object_id.isin(set(objects.source_object_id)) & model.split_role.isin(["train", "val"]), ["model_id", "source_object_id", "split_role"]].drop_duplicates()
    matched = samples.merge(model, left_on="sample_id", right_on="model_id", how="inner", validate="one_to_one")
    if not matched.cell_line_id.eq(matched.model_id).all():
        raise ValueError("PRISM sample_id/cell_line_id differs from the frozen exact DepMap ModelID")
    matched = matched.loc[matched.split_role.isin(["train", "val"])].copy()
    drug_path = root / "data/processed/exp012/prism_extended_primary_drugs.parquet"
    drugs = pd.read_parquet(drug_path, columns=["drug_id", "inchikey"])
    drugs["drug_id"] = normalize(drugs.drug_id)
    drugs["inchikey"] = normalize(drugs.inchikey)
    drugs = drugs.dropna(subset=["drug_id", "inchikey"])
    drugs = drugs.loc[drugs.inchikey.str.fullmatch(MIN_FULL_KEY)]
    id_n = drugs.groupby("drug_id").inchikey.nunique()
    key_n = drugs.groupby("inchikey").drug_id.nunique()
    drugs = drugs.loc[drugs.drug_id.map(id_n).eq(1) & drugs.inchikey.map(key_n).eq(1)].drop_duplicates()
    allowed_sample_ids = set(matched.sample_id.astype(str))
    pair_path = root / "data/processed/exp012/prism_extended_primary_pairs.parquet"
    dataset = ds.dataset(str(pair_path), format="parquet")
    identifiers = dataset.to_table(columns=["sample_id", "drug_id"], filter=ds.field("sample_id").isin(sorted(allowed_sample_ids))).to_pandas()
    identifiers["sample_id"] = normalize(identifiers.sample_id)
    identifiers["drug_id"] = normalize(identifiers.drug_id)
    ids = identifiers.drop_duplicates().merge(matched[["sample_id", "source_object_id", "split_role"]], on="sample_id", how="inner", validate="many_to_one").merge(drugs, on="drug_id", how="inner", validate="many_to_one")
    target_sets = target_pairs.groupby("source_object_id").full_inchikey.agg(set).to_dict()
    key_rows = ids.loc[ids.apply(lambda r: r.inchikey in target_sets.get(str(r.source_object_id), set()), axis=1), ["source_object_id", "split_role", "inchikey"]].drop_duplicates().rename(columns={"inchikey": "full_inchikey"})
    summary = {
        "mapped_train_val_objects": int(matched.source_object_id.nunique()),
        "mapped_train_val_samples": int(matched.sample_id.nunique()),
        "objects_with_target_drug_pairs": int(key_rows.source_object_id.nunique()),
        "full_inchikeys": int(key_rows.full_inchikey.nunique()),
        "object_full_inchikey_pairs": int(len(key_rows)),
        "per_role": {str(k): {"objects": int(key_rows.loc[key_rows.split_role.eq(k), "source_object_id"].nunique()), "drugs": int(key_rows.loc[key_rows.split_role.eq(k), "full_inchikey"].nunique()), "pairs": int(len(key_rows.loc[key_rows.split_role.eq(k)]))} for k in ["train", "val"]},
        "identifier_scan": "Arrow filter restricted to TRAIN/VAL sample IDs; response column not projected",
    }
    return matched, key_rows, summary


def role_preflight(root: Path, out: Path) -> dict[str, Any]:
    targets, receipt = load_target_keys(root)
    heldout_identity = join_heldout_identity_whitelist(root, targets, receipt)
    roles = frozen_roles(root)
    role_conflicts = roles.groupby("source_object_id").split_role.nunique()
    if role_conflicts.gt(1).any():
        raise ValueError("one heldout patient identity maps to conflicting frozen TRAIN/VAL roles")
    roles_by_object = roles.groupby("source_object_id", as_index=False).agg(
        lincs_cell_id=("lincs_cell_id", lambda x: ";".join(sorted(set(map(str, x))))),
        depmap_id=("depmap_id", lambda x: ";".join(sorted(set(map(str, x))))),
        split_role=("split_role", "first"), join_status=("join_status", "first"))
    objects = targets.groupby("source_object_id", as_index=False).agg(
        xpert_conditions=("condition_id", "nunique"), xpert_full_inchikeys=("full_inchikey", "nunique"))
    pairs = targets[["source_object_id", "full_inchikey"]].drop_duplicates()
    objects = objects.merge(pairs.groupby("source_object_id").size().rename("xpert_object_drug_pairs"), on="source_object_id", how="left")
    objects = objects.merge(roles_by_object, on="source_object_id", how="left", validate="one_to_one")
    objects["role_available"] = objects.split_role.isin(["train", "val"])
    ctrp_cells, ctrp_keys, ctrp_summary = ctrp_identifiers(root, objects, pairs)
    prism_samples, prism_keys, prism_summary = prism_identifiers(root, objects, pairs)
    for assay, key_frame in (("CTRPv2", ctrp_keys), ("PRISM", prism_keys)):
        counts = key_frame.groupby("source_object_id").agg(**{f"{assay}_exact_object_drug_pairs": ("full_inchikey", "nunique"), f"{assay}_exact_drugs": ("full_inchikey", "nunique")})
        objects = objects.merge(counts, on="source_object_id", how="left")
        objects[f"{assay}_exact_object_drug_pairs"] = objects[f"{assay}_exact_object_drug_pairs"].fillna(0).astype(int)
        objects[f"{assay}_exact_drugs"] = objects[f"{assay}_exact_drugs"].fillna(0).astype(int)
        present = key_frame.groupby("source_object_id").size().gt(0)
        objects[f"{assay}_object_present_with_target_key"] = objects.source_object_id.isin(present.index[present])
    for assay, key_frame in (("CTRPv2", ctrp_keys), ("PRISM", prism_keys)):
        role_col = f"{assay}_split_role"
        objects[role_col] = objects.split_role.where(objects[f"{assay}_object_present_with_target_key"], "NO_EXACT_TARGET_DRUG_SUPPORT")
    objects["prior_label_exposure"] = objects.source_object_id.eq("PT-pYG6Rc")
    objects["prior_exposure_note"] = np.where(objects.prior_label_exposure, "single-object exploration previously read CTRPv2 22 drugs and PRISM 8 drugs", "no prior label exposure recorded in EXP-050 §3.4")
    out.mkdir(parents=True, exist_ok=True)
    roles_path = out / "object_roles_key_support.tsv"
    objects.sort_values("source_object_id").to_csv(roles_path, sep="\t", index=False, na_rep="")
    long_rows = []
    for assay, frame in (("CTRPv2", ctrp_keys), ("PRISM", prism_keys)):
        for row in frame.itertuples(index=False):
            long_rows.append({"assay": assay, "source_object_id": row.source_object_id, "split_role": row.split_role, "full_inchikey": row.full_inchikey})
    support_path = out / "train_val_exact_object_drug_key_support.tsv"
    pd.DataFrame(long_rows, columns=["assay", "source_object_id", "split_role", "full_inchikey"]).sort_values(["assay", "source_object_id", "full_inchikey"]).to_csv(support_path, sep="\t", index=False)
    n_train = int(objects.split_role.eq("train").sum())
    n_val = int(objects.split_role.eq("val").sum())
    both_endpoint_train = sorted(set(ctrp_keys.loc[ctrp_keys.split_role.eq("train"), "source_object_id"]) & set(prism_keys.loc[prism_keys.split_role.eq("train"), "source_object_id"]))
    both_endpoint_val = sorted(set(ctrp_keys.loc[ctrp_keys.split_role.eq("val"), "source_object_id"]) & set(prism_keys.loc[prism_keys.split_role.eq("val"), "source_object_id"]))
    ctrp_val_keys = set(ctrp_keys.loc[ctrp_keys.split_role.eq("val"), "full_inchikey"])
    prism_val_keys = set(prism_keys.loc[prism_keys.split_role.eq("val"), "full_inchikey"])
    target_val_keys = set(pairs.loc[pairs.source_object_id.isin(objects.loc[objects.split_role.eq("val"), "source_object_id"]), "full_inchikey"])
    gate = {
        "status": "NO_GO_LOCAL_FIVE_ARM_GENERALIZATION_INSUFFICIENT_INDEPENDENT_VALIDATION_OBJECTS",
        "reason": "only one independent VAL source object among the ten; TRAIN and VAL are role-disjoint but one VAL object cannot support a generalization claim",
        "xpert_target_objects": len(objects), "train_objects": n_train, "val_objects": n_val,
        "train_val_object_overlap": sorted(set(objects.loc[objects.split_role.eq("train"), "source_object_id"]) & set(objects.loc[objects.split_role.eq("val"), "source_object_id"])),
        "ctprv2_trainval_objects_with_target_key_support": ctrp_summary["objects_with_target_drug_pairs"],
        "prism_trainval_objects_with_target_key_support": prism_summary["objects_with_target_drug_pairs"],
        "ctprv2_trainval_common_objects": len(both_endpoint_train) + len(both_endpoint_val),
        "ctprv2_prism_train_objects_common": len(both_endpoint_train),
        "ctprv2_prism_val_objects_common": len(both_endpoint_val),
        "val_target_drugs": len(target_val_keys), "ctrp_val_drugs": len(ctrp_val_keys), "prism_val_drugs": len(prism_val_keys),
        "val_common_target_drugs_across_ctrp_prism": len(target_val_keys & ctrp_val_keys & prism_val_keys),
        "no_five_arm_fit_or_score_run": True,
    }
    summary = {
        "status": "ROLE_AND_IDENTIFIER_PREFLIGHT_COMPLETE",
        "target_support": {"conditions": int(len(targets)), "objects": int(objects.source_object_id.nunique()), "object_full_inchikey_pairs": int(len(pairs)), "full_inchikeys": int(pairs.full_inchikey.nunique())},
        "roles": {str(k): int(v) for k, v in objects.split_role.value_counts(dropna=False).items()},
        "heldout_identity_whitelist_join": {"condition_rows": int(len(heldout_identity)), "unique_objects": int(heldout_identity.source_object_id_map.nunique()), "heldout_fit_role": sorted(heldout_identity.split_role.unique().astype(str).tolist()), "direct_assay_roles": {str(k): int(v) for k, v in heldout_identity[["source_object_id_map","drug_role"]].drop_duplicates().drug_role.value_counts().items()}, "cell_id_patient_id_role_join": "condition_id whitelist -> XPert fit identity map -> exact LINCS cell_id -> frozen EXP-007 identity and role; source_object_id required to equal EXP-007 PatientID"},
        "CTRPv2": ctrp_summary, "PRISM": prism_summary,
        "five_arm_gate": gate,
        "data_boundary": {"assay_numeric_values_read": False, "locked_or_outer_values_labels_features_scores_read": False, "PDO_labels_read": False, "GPU_hours": 0},
        "outputs": {roles_path.name: sha256(roles_path), support_path.name: sha256(support_path)},
        "input_sha256": {"XPert_keys": sha256(root / "artifacts/experiments/EXP-050" / ATTEMPT_ID / "heldout_source_prediction_keys.tsv"), "XPert_attempt_receipt": sha256(root / "artifacts/experiments/EXP-050" / ATTEMPT_ID / "attempt_receipt.json"), "EXP007_identity_split": sha256(root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json")},
    }
    write_json(out / "role_preflight.json", summary)
    write_json(out / "five_arm_gate.json", gate)
    return summary


def assert_attempt_outputs(root: Path) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    directory = root / "artifacts/experiments/EXP-050" / ATTEMPT_ID
    receipt = json.loads((directory / "attempt_receipt.json").read_text(encoding="utf-8"))
    for name in ["heldout_source_prediction_keys.tsv", "heldout_source_prediction_target_order.npy", "heldout_source_true_delta_target_order.npy", "fit_condition_identity_key_map.tsv", "training_protocol_snapshot.json"]:
        expected = receipt.get("output_sha256", {}).get(name)
        if not expected or sha256(directory / name) != expected:
            raise ValueError(f"XPert frozen output hash mismatch: {name}")
    mapping = pd.read_csv(directory / "fit_condition_identity_key_map.tsv", sep="\t", dtype={"source_object_id": "string", "full_inchikey": "string"})
    require_columns(mapping, {"condition_id", "source_row", "split_role", "source_object_id", "full_inchikey"}, "XPert fit condition map")
    if not mapping.split_role.eq("train").any():
        raise ValueError("XPert fit map has no training conditions")
    return directory, receipt, mapping


def response_values(root: Path, objects: pd.DataFrame, target_pairs: pd.DataFrame, assay: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed = objects.loc[objects.split_role.isin(["train", "val"]), ["source_object_id", "split_role"]]
    targets = target_pairs.set_index("source_object_id").full_inchikey.groupby(level=0).agg(set).to_dict()
    if assay == "CTRPv2":
        mapping = pd.read_csv(root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv", usecols=["cellid", "ModelID", "match_method", "has_depmap_expression"], dtype="string")
        mapping = mapping.loc[mapping.match_method.eq("exact_unique_cellosaurus_rrid") & mapping.has_depmap_expression.astype("string").str.lower().eq("true"), ["cellid", "ModelID"]].dropna().drop_duplicates()
        models = pd.read_csv(root / "data/raw/depmap/24q2/Model.csv", usecols=["ModelID", "PatientID"], dtype="string").dropna().drop_duplicates()
        cellmap = mapping.merge(models, on="ModelID", how="inner", validate="many_to_one").merge(allowed, left_on="PatientID", right_on="source_object_id", how="inner", validate="many_to_one")
        cellmap = cellmap.loc[cellmap.split_role.isin(["train", "val"])]
        drugs = pd.read_csv(root / "data/processed/exp013/ctrp_v2/drugs.csv", usecols=["drugid", "inchikey"], dtype="string")
        drugs["inchikey"] = normalize(drugs.inchikey)
        drugs = drugs.dropna(subset=["drugid", "inchikey"])
        counts = drugs.groupby("inchikey").drugid.nunique()
        drugs = drugs.loc[drugs.inchikey.map(counts).eq(1), ["drugid", "inchikey"]].drop_duplicates()
        cells = set(cellmap.cellid.astype(str))
        chunks = []
        for chunk in pd.read_csv(root / "data/processed/exp013/ctrp_v2/responses.csv", usecols=["cellid", "drugid", "aac_published"], dtype={"cellid": "string", "drugid": "string"}, chunksize=100_000):
            chunk = chunk.loc[chunk.cellid.astype(str).isin(cells)]
            if not chunk.empty:
                chunks.append(chunk)
        raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["cellid", "drugid", "aac_published"])
        raw["sensitivity"] = pd.to_numeric(raw.aac_published, errors="coerce")
        raw = raw.loc[np.isfinite(raw.sensitivity)]
        raw = raw.merge(cellmap[["cellid", "source_object_id", "split_role"]], on="cellid", how="inner", validate="many_to_one").merge(drugs, on="drugid", how="inner", validate="many_to_one")
        raw = raw.loc[raw.apply(lambda r: r.inchikey in targets.get(str(r.source_object_id), set()), axis=1)]
        result = raw.groupby(["source_object_id", "split_role", "inchikey"], as_index=False).agg(sensitivity=("sensitivity", "median"), n_response_rows=("sensitivity", "size"), n_cell_lines=("cellid", "nunique")).rename(columns={"inchikey": "full_inchikey"})
        return result, {"numeric_column_projected": "aac_published", "higher_is_more_sensitive": True, "finite_label_rows": int(len(raw)), "object_key_pairs": int(len(result)), "role_filter": sorted(set(raw.split_role.astype(str))), "aggregation": "median across replicate rows and exact mapped cell lines sharing one PatientID"}
    if assay == "PRISM":
        path = root / "data/processed/exp012/prism_extended_primary_samples.parquet"
        samples = pd.read_parquet(path, columns=["sample_id", "cell_line_id"])
        samples["sample_id"] = normalize(samples.sample_id)
        samples["cell_line_id"] = normalize(samples.cell_line_id)
        model = objects.rename(columns={"depmap_id": "model_id"})[["model_id", "source_object_id", "split_role"]]
        matched = samples.merge(model, left_on="sample_id", right_on="model_id", how="inner", validate="one_to_one")
        if not matched.cell_line_id.eq(matched.model_id).all():
            raise ValueError("PRISM sample key no longer matches frozen DepMap ID")
        matched = matched.loc[matched.split_role.isin(["train", "val"])].copy()
        allowed_ids = set(matched.sample_id.astype(str))
        dataset = ds.dataset(str(root / "data/processed/exp012/prism_extended_primary_pairs.parquet"), format="parquet")
        table = dataset.to_table(columns=["sample_id", "drug_id", "response"], filter=ds.field("sample_id").isin(sorted(allowed_ids)))
        raw = table.to_pandas()
        raw["sample_id"] = normalize(raw.sample_id)
        raw["drug_id"] = normalize(raw.drug_id)
        if set(raw.sample_id.dropna().astype(str)) - allowed_ids:
            raise ValueError("PRISM numeric projection returned non-TRAIN/VAL samples")
        raw["sensitivity"] = pd.to_numeric(raw.response, errors="coerce")
        raw = raw.loc[np.isfinite(raw.sensitivity)]
        drugs = pd.read_parquet(root / "data/processed/exp012/prism_extended_primary_drugs.parquet", columns=["drug_id", "inchikey"])
        drugs["drug_id"] = normalize(drugs.drug_id)
        drugs["inchikey"] = normalize(drugs.inchikey)
        drugs = drugs.dropna(subset=["drug_id", "inchikey"])
        drugs = drugs.loc[drugs.inchikey.str.fullmatch(MIN_FULL_KEY)]
        id_counts = drugs.groupby("drug_id").inchikey.nunique(); key_counts = drugs.groupby("inchikey").drug_id.nunique()
        drugs = drugs.loc[drugs.drug_id.map(id_counts).eq(1) & drugs.inchikey.map(key_counts).eq(1)].drop_duplicates()
        cell = raw.groupby(["sample_id", "drug_id"], as_index=False).agg(sensitivity=("sensitivity", "median"), n_response_rows=("sensitivity", "size"))
        cell = cell.merge(matched[["sample_id", "source_object_id", "split_role"]], on="sample_id", how="inner", validate="many_to_one").merge(drugs, on="drug_id", how="inner", validate="many_to_one")
        cell = cell.loc[cell.apply(lambda r: r.inchikey in targets.get(str(r.source_object_id), set()), axis=1)]
        result = cell.groupby(["source_object_id", "split_role", "inchikey"], as_index=False).agg(sensitivity=("sensitivity", "median"), n_response_rows=("n_response_rows", "sum"), n_cell_lines=("sample_id", "nunique")).rename(columns={"inchikey": "full_inchikey"})
        return result, {"numeric_column_projected": "response (EXP-012 neg_extended_primary_lfc)", "higher_is_more_sensitive": True, "finite_label_rows": int(len(raw)), "object_key_pairs": int(len(result)), "role_filter": sorted(set(raw.merge(matched[["sample_id", "split_role"]], on="sample_id", how="left").split_role.astype(str))), "arrow_filter": "sample_id in exact TRAIN/VAL allowlist before response column returned", "aggregation": "median within sample x drug, then across exact mapped samples sharing PatientID"}
    raise ValueError(assay)


def gene_feature_indices(root: Path) -> dict[str, np.ndarray]:
    target_genes = json.loads((root / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json").read_text(encoding="utf-8"))
    if len(target_genes) != 978 or len(set(target_genes)) != 978:
        raise ValueError("frozen XPert target gene axis must contain 978 unique Entrez IDs")
    gene_info_path = root / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
    info = pd.read_csv(gene_info_path, sep="\t", compression="gzip", usecols=["pr_gene_id", "pr_gene_symbol"], dtype="string")
    info = info.dropna().drop_duplicates()
    info["pr_gene_id"] = info.pr_gene_id.astype(str)
    info["pr_gene_symbol"] = info.pr_gene_symbol.astype(str).str.strip().str.upper()
    symbol_by_id = info.drop_duplicates("pr_gene_id").set_index("pr_gene_id").pr_gene_symbol.to_dict()
    target_symbols = [symbol_by_id.get(str(gene), "") for gene in target_genes]
    gmt = (root / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt").read_text(encoding="utf-8")
    members: dict[str, set[str]] = {}
    for line in gmt.splitlines():
        parts = line.split("\t")
        if parts and parts[0] in GENE_SETS:
            members[parts[0]] = {x.strip().upper() for x in parts[2:] if x.strip()}
    if set(members) != set(GENE_SETS):
        raise ValueError("MSigDB GMT is missing a frozen Hallmark set")
    out: dict[str, np.ndarray] = {"all_978_genes": np.arange(978, dtype=np.int64)}
    for name in GENE_SETS:
        idx = np.asarray([i for i, symbol in enumerate(target_symbols) if symbol and symbol in members[name]], dtype=np.int64)
        if not len(idx):
            raise ValueError(f"frozen Hallmark set has no genes on XPert axis: {name}")
        out[name] = idx
    return out


def magnitude_features(matrix: np.ndarray, indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.abs(matrix[:, cols]).mean(axis=1) for name, cols in indices.items()}


def response_feature_table(root: Path, targets: pd.DataFrame, receipt: dict[str, Any], mapping: pd.DataFrame, attempt_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred_path = attempt_dir / "heldout_source_prediction_target_order.npy"
    true_path = attempt_dir / "heldout_source_true_delta_target_order.npy"
    pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    true = np.load(true_path, mmap_mode="r", allow_pickle=False)
    if pred.shape != (len(targets), 978) or true.shape != pred.shape or not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("XPert frozen predicted/observed response arrays fail shape/finite validation")
    index = gene_feature_indices(root)
    pred_feats = magnitude_features(pred, index)
    true_feats = magnitude_features(true, index)
    per_cond = targets[["condition_id", "source_row", "source_object_id", "full_inchikey"]].copy()
    for name in index:
        per_cond[f"observed__{name}"] = true_feats[name]
        per_cond[f"xpert__{name}"] = pred_feats[name]
    per_pair = per_cond.groupby(["source_object_id", "full_inchikey"], as_index=False).agg(
        **{f"observed__{name}": (f"observed__{name}", "median") for name in index},
        **{f"xpert__{name}": (f"xpert__{name}", "median") for name in index},
        n_lincs_conditions=("condition_id", "nunique"),
    )
    train = mapping.loc[mapping.split_role.eq("train") & mapping.full_inchikey.isin(set(targets.full_inchikey))].copy()
    if train.empty:
        raise ValueError("No exact target-key TRAIN response rows available for training mean arm")
    train.sort_values(["full_inchikey", "condition_id"], inplace=True)
    protocol = json.loads((attempt_dir / "training_protocol_snapshot.json").read_text(encoding="utf-8"))
    gene_map_rel = root / "artifacts/experiments/EXP-037/20260923_xpert_asset_audit/source_to_target_gene_indices.json"
    declared_gene_map = protocol.get("source_hashes", {}).get("source_to_target_gene_map", {}).get("sha256")
    if not gene_map_rel.is_file() or not declared_gene_map or sha256(gene_map_rel) != declared_gene_map:
        raise ValueError("XPert source-to-target gene map path/hash differs from frozen attempt snapshot")
    gene_map = np.asarray(json.loads(gene_map_rel.read_text(encoding="utf-8")), dtype=np.int64)
    if gene_map.shape != (978,) or not np.array_equal(np.sort(gene_map), np.arange(978)):
        raise ValueError("XPert frozen source-to-target gene map is not a 978 permutation")
    delta_path = root / "data/processed/lincs/GSE92742/continuous_conditions_v2/condition_delta.npy"
    delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if delta.ndim != 2 or delta.shape[1] != 978 or int(train.source_row.max()) >= delta.shape[0]:
        raise ValueError("LINCS response matrix does not match frozen train source rows")
    train_values = np.asarray(delta[train.source_row.to_numpy(dtype=np.int64)], dtype=np.float32)[:, np.argsort(gene_map)]
    if not np.isfinite(train_values).all():
        raise ValueError("XPert TRAIN responses contain nonfinite values")
    train_feature_rows = []
    train_feats = magnitude_features(train_values, index)
    train_meta = train[["condition_id", "source_row", "source_object_id", "full_inchikey"]].copy().reset_index(drop=True)
    for name in index:
        train_meta[f"training_mean__{name}"] = train_feats[name]
    # Canonical XPert baseline: average signed gene response by key in TRAIN, then
    # apply the preregistered mean-absolute-gene magnitude formula.
    train_vector_feats = {name: {} for name in index}
    for key, positions in train_meta.groupby("full_inchikey", sort=True).indices.items():
        mean_vector = train_values[np.asarray(positions, dtype=np.int64)].mean(axis=0, dtype=np.float64)
        for name, cols in index.items():
            train_vector_feats[name][str(key)] = float(np.abs(mean_vector[cols]).mean())
        train_feature_rows.append({"full_inchikey": str(key), "n_train_conditions": int(len(positions)), "n_train_objects": int(train_meta.iloc[list(positions)].source_object_id.nunique()), **{f"training_mean__{name}": train_vector_feats[name][str(key)] for name in index}})
    train_frame = pd.DataFrame(train_feature_rows)
    for name in index:
        train_frame[f"train_mean_n__{name}"] = train_frame.n_train_conditions
    per_pair = per_pair.merge(train_frame, on="full_inchikey", how="left", validate="many_to_one")
    info = {
        "feature_gene_counts": {name: int(len(cols)) for name, cols in index.items()},
        "observed_prediction_conditions": int(len(targets)),
        "observed_prediction_object_key_pairs": int(len(per_pair)),
        "train_mean_available_keys": int(train_frame.full_inchikey.nunique()),
        "train_mean_available_conditions": int(len(train)),
        "train_mean_objects": int(train.source_object_id.nunique()),
        "training_mean_input_roles": sorted(train.split_role.unique().astype(str).tolist()),
        "inner_dev_or_heldout_rows_used_in_train_mean": False,
        "direct_sensitivity_values_used_in_train_mean": False,
        "gene_axis_reorder": "target_order = source_order[:, argsort(frozen source_to_target_gene_indices)]",
        "train_mean_formula": "for each exact InChIKey, mean signed 978-axis delta across XPert TRAIN conditions; then mean absolute gene value (same absolute-magnitude feature)",
    }
    return per_pair, info


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str | None]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y); x=x[finite]; y=y[finite]
    if len(x) < 2:
        return None, "fewer_than_2_drugs"
    if np.unique(x).size < 2:
        return None, "constant_response_feature"
    if np.unique(y).size < 2:
        return None, "constant_sensitivity"
    value = float(spearmanr(x, y).statistic)
    return (value, None) if math.isfinite(value) else (None, "spearman_undefined")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [None, None]
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def object_scores(pairs: pd.DataFrame, endpoint: str, domains: dict[str, pd.DataFrame], features: list[str], rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores=[]; summaries=[]; contrasts=[]
    for domain, frame in domains.items():
        arms = ["observed", "xpert"] + (["training_mean"] if domain == "three_arm_common_with_train_mean" else [])
        for feature in features:
            cols={arm: f"{arm}__{feature}" if arm != "training_mean" else f"training_mean__{feature}" for arm in arms}
            by_object=[]
            for obj, group in frame.groupby("source_object_id", sort=True):
                valid=group.dropna(subset=["sensitivity", *cols.values()]).sort_values("full_inchikey")
                for arm in arms:
                    rho, reason=corr(valid[cols[arm]].to_numpy(), valid.sensitivity.to_numpy())
                    by_object.append({"endpoint":endpoint,"domain":domain,"feature":feature,"arm":arm,"source_object_id":obj,"split_role":str(valid.split_role.iloc[0]) if len(valid) else str(group.split_role.iloc[0]),"n_drugs":int(len(valid)),"spearman_rho":rho,"undefined_reason":reason,"sparse_n_lt_3":bool(len(valid)<3)})
            per=pd.DataFrame(by_object)
            scores.extend(by_object)
            for arm, group in per.groupby("arm", sort=False):
                vals=group.spearman_rho.dropna().to_numpy(dtype=float)
                ci=bootstrap_mean(vals,rng)
                summaries.append({"endpoint":endpoint,"support_domain":domain,"feature":feature,"arm":arm,"n_source_objects_in_domain":int(frame.source_object_id.nunique()),"n_objects_with_defined_rho":int(len(vals)),"defined_object_fraction":float(len(vals)/max(frame.source_object_id.nunique(),1)),"mean_object_spearman":float(vals.mean()) if len(vals) else None,"median_object_spearman":float(np.median(vals)) if len(vals) else None,"bootstrap_95_percentile_ci":ci,"bootstrap_unit":"source object","bootstrap_replicates":N_BOOT if len(vals)>=2 else 0,"n_pairs_total":int(group.n_drugs.sum()),"median_drugs_per_object":float(group.n_drugs.median()) if len(group) else None,"min_drugs_per_object":int(group.n_drugs.min()) if len(group) else 0,"max_drugs_per_object":int(group.n_drugs.max()) if len(group) else 0,"higher_sensitivity_direction":"higher is more sensitive; positive rho means stronger transcriptional change tends to pair with greater sensitivity"})
            pivot=per.pivot(index="source_object_id",columns="arm",values="spearman_rho")
            comparisons=[]
            if "training_mean" in arms:
                comparisons=[("observed","xpert"),("observed","training_mean"),("xpert","training_mean")]
            else:
                comparisons=[("observed","xpert")]
            for a,b in comparisons:
                diff=(pivot[a]-pivot[b]).dropna().to_numpy(dtype=float)
                contrasts.append({"endpoint":endpoint,"support_domain":domain,"feature":feature,"contrast":f"{a}_minus_{b}","n_paired_objects":int(len(diff)),"mean_paired_rho_difference":float(diff.mean()) if len(diff) else None,"bootstrap_95_percentile_ci":bootstrap_mean(diff,rng),"interpretation":"descriptive difference on the same objects and object-key support; not a selection rule"})
    return pd.DataFrame(scores),pd.DataFrame(summaries),pd.DataFrame(contrasts)


def object_between(pairs: pd.DataFrame, endpoint: str, domains: dict[str,pd.DataFrame], features: list[str], rng: np.random.Generator, n_boot: int = 5000) -> pd.DataFrame:
    rows=[]
    for domain, frame in domains.items():
        arms=["observed","xpert"]+(["training_mean"] if domain=="three_arm_common_with_train_mean" else [])
        for feature in features:
            for arm in arms:
                col=f"{arm}__{feature}" if arm!="training_mean" else f"training_mean__{feature}"
                clean=frame.dropna(subset=[col,"sensitivity","source_object_id","full_inchikey"]).copy()
                raw_r,raw_reason=corr(clean[col].to_numpy(),clean.sensitivity.to_numpy())
                centered=clean.copy()
                centered[col]=centered[col]-centered.groupby("full_inchikey")[col].transform("mean")
                centered["sensitivity"]-=centered.groupby("full_inchikey").sensitivity.transform("mean")
                adj_r,adj_reason=corr(centered[col].to_numpy(),centered.sensitivity.to_numpy())
                objects=clean.source_object_id.drop_duplicates().to_numpy()
                # Object bootstrap: sample independent objects, then recenter each drug
                # inside each bootstrap draw. Retry-free fixed-size bootstrap; undefined
                # draws are omitted and counted.
                raw_boot=[]; adj_boot=[]
                if len(objects)>=2:
                    draws=rng.integers(0,len(objects),size=(n_boot,len(objects)))
                    indexed={str(o):clean.loc[clean.source_object_id.eq(o)] for o in objects}
                    for draw in draws:
                        sampled=pd.concat([indexed[str(objects[i])] for i in draw],ignore_index=True)
                        rv,_=corr(sampled[col].to_numpy(),sampled.sensitivity.to_numpy())
                        if rv is not None: raw_boot.append(rv)
                        sampled[col]=sampled[col]-sampled.groupby("full_inchikey")[col].transform("mean")
                        sampled["sensitivity"]=sampled.sensitivity-sampled.groupby("full_inchikey").sensitivity.transform("mean")
                        av,_=corr(sampled[col].to_numpy(),sampled.sensitivity.to_numpy())
                        if av is not None: adj_boot.append(av)
                rows.append({"endpoint":endpoint,"support_domain":domain,"feature":feature,"arm":arm,"n_objects":int(len(objects)),"n_drugs":int(clean.full_inchikey.nunique()),"n_pairs":int(len(clean)),"pooled_spearman_before_drug_centering":raw_r,"before_undefined_reason":raw_reason,"object_bootstrap_ci_before":list(map(float,np.quantile(raw_boot,[.025,.975]))) if len(raw_boot)>=100 else [None,None],"valid_bootstrap_draws_before":len(raw_boot),"pooled_spearman_after_drug_centering":adj_r,"after_undefined_reason":adj_reason,"object_bootstrap_ci_after":list(map(float,np.quantile(adj_boot,[.025,.975]))) if len(adj_boot)>=100 else [None,None],"valid_bootstrap_draws_after":len(adj_boot),"bootstrap_replicates_requested":n_boot if len(objects)>=2 else 0,"bootstrap_seed_fixed":BOOT_SEED,"interpretation":"pooled descriptive association; after-centering controls shared exact-drug means. Any centering uses these assay labels and is exploratory, not independent prediction."})
    return pd.DataFrame(rows)


def counterexamples(pairs: pd.DataFrame, endpoint: str, domains: dict[str,pd.DataFrame], features: list[str]) -> pd.DataFrame:
    rows=[]
    for domain,frame in domains.items():
        arms=["observed","xpert"]+(["training_mean"] if domain=="three_arm_common_with_train_mean" else [])
        for feature in features:
            for obj,g in frame.groupby("source_object_id",sort=True):
                for arm in arms:
                    col=f"{arm}__{feature}" if arm!="training_mean" else f"training_mean__{feature}"
                    v=g.dropna(subset=[col,"sensitivity"]).copy()
                    if len(v)<3: continue
                    v["response_rank_fraction"]=v[col].rank(method="average",pct=True)
                    v["sensitivity_rank_fraction"]=v.sensitivity.rank(method="average",pct=True)
                    v["absolute_rank_gap"]=(v.response_rank_fraction-v.sensitivity_rank_fraction).abs()
                    row=v.sort_values(["absolute_rank_gap","full_inchikey"],ascending=[False,True]).iloc[0]
                    rows.append({"endpoint":endpoint,"support_domain":domain,"feature":feature,"arm":arm,"source_object_id":obj,"n_drugs":int(len(v)),"full_inchikey":row.full_inchikey,"response_magnitude":float(row[col]),"sensitivity":float(row.sensitivity),"response_rank_fraction":float(row.response_rank_fraction),"sensitivity_rank_fraction":float(row.sensitivity_rank_fraction),"absolute_rank_gap":float(row.absolute_rank_gap),"counterexample_rule":"largest absolute within-object percentile-rank gap; ties resolved by sorted full InChIKey"})
    return pd.DataFrame(rows)


def analyze(root: Path,out: Path) -> dict[str,Any]:
    t0=time.monotonic(); contract=load_contract(root)
    roles=role_preflight(root,out)
    role_objects=pd.read_csv(out/"object_roles_key_support.tsv",sep="\t",dtype={"source_object_id":"string"})
    attempt_dir,receipt,mapping=assert_attempt_outputs(root)
    targets,_=load_target_keys(root)
    target_pairs=targets[["source_object_id","full_inchikey"]].drop_duplicates()
    pair_roles=target_pairs.merge(role_objects[["source_object_id","split_role"]],on="source_object_id",how="left",validate="many_to_one")
    if pair_roles.split_role.isna().any() or not pair_roles.split_role.isin(["train","val"]).all():
        raise ValueError("XPert source object roles include unknown/locked objects; numeric join stopped")
    response_features,feature_info=response_feature_table(root,targets,receipt,mapping,attempt_dir)
    all_features=["all_978_genes",*GENE_SETS]
    all_obj_scores=[]; all_summaries=[]; all_contrasts=[]; all_between=[]; all_counter=[]; paired_outputs=[]; endpoint_info={}
    rng=np.random.default_rng(BOOT_SEED)
    for assay in ["CTRPv2","PRISM"]:
        direct,direct_info=response_values(root,role_objects,target_pairs,assay)
        if not direct.split_role.isin(["train","val"]).all():
            raise ValueError(f"{assay} direct rows include forbidden role")
        pairs=direct.merge(response_features,on=["source_object_id","full_inchikey"],how="inner",validate="one_to_one")
        pairs["endpoint"]=assay
        pairs["sensitivity_higher_is_more_sensitive"]=True
        pairs["lincs_exposure"]=contract["lincs_exposure"]
        train_available=pairs["training_mean__all_978_genes"].notna()
        full=pairs.dropna(subset=["observed__all_978_genes","xpert__all_978_genes"]).copy()
        # One common key list for all measured and predicted features; do not fill missing values.
        full_domain={"measured_and_predicted_complete":full}
        common=full.loc[train_available.reindex(full.index,fill_value=False)].copy()
        # Ensure each arm/feature sees an identical exact key denominator.
        feature_cols=[f"observed__{f}" for f in all_features]+[f"xpert__{f}" for f in all_features]+[f"training_mean__{f}" for f in all_features]
        if not full.empty:
            full=full.dropna(subset=[*feature_cols[:len(all_features)],*feature_cols[len(all_features):2*len(all_features)]])
            full_domain={"measured_and_predicted_complete":full}
        common=full.dropna(subset=[f"training_mean__{f}" for f in all_features]).copy()
        domains={"measured_and_predicted_complete":full,"three_arm_common_with_train_mean":common}
        for d,frame in domains.items():
            frame["support_domain"]=d
            frame["endpoint"]=assay
        if full.duplicated(["source_object_id","full_inchikey"]).any() or common.duplicated(["source_object_id","full_inchikey"]).any():
            raise ValueError("Direct endpoint object/key join is not unique")
        if not common.empty and not common["training_mean__all_978_genes"].notna().all():
            raise ValueError("Three-arm support has a missing train-mean key")
        paired_outputs.append(pairs)
        scores,summary,contrasts=object_scores(pairs,assay,domains,all_features,rng)
        all_obj_scores.append(scores); all_summaries.append(summary); all_contrasts.append(contrasts)
        all_between.append(object_between(pairs,assay,domains,all_features,rng,n_boot=5000))
        all_counter.append(counterexamples(pairs,assay,domains,all_features))
        endpoint_info[assay]={"numeric_values_read":True,"direct_value_column":direct_info["numeric_column_projected"],"direct_aggregation":direct_info["aggregation"],"endpoint_rows_after_exact_identity_structure_join":int(len(pairs)),"endpoint_objects":int(pairs.source_object_id.nunique()),"endpoint_drugs":int(pairs.full_inchikey.nunique()),"endpoint_pairs":int(len(pairs)),"full_response_support":{"objects":int(full.source_object_id.nunique()),"drugs":int(full.full_inchikey.nunique()),"pairs":int(len(full))},"train_mean_common_support":{"objects":int(common.source_object_id.nunique()),"drugs":int(common.full_inchikey.nunique()),"pairs":int(len(common))},**direct_info}
    out.mkdir(parents=True,exist_ok=True)
    outputs={}
    def save(name:str,frame:pd.DataFrame)->None:
        path=out/name; frame.to_csv(path,sep="\t",index=False,na_rep=""); outputs[name]=sha256(path)
    save("paired_object_drug_values.tsv",pd.concat(paired_outputs,ignore_index=True,sort=False))
    save("per_object_association.tsv",pd.concat(all_obj_scores,ignore_index=True,sort=False))
    save("association_summary.tsv",pd.concat(all_summaries,ignore_index=True,sort=False))
    save("paired_arm_contrasts.tsv",pd.concat(all_contrasts,ignore_index=True,sort=False))
    save("object_between_shared_drug_control.tsv",pd.concat(all_between,ignore_index=True,sort=False))
    save("counterexamples_largest_rank_gaps.tsv",pd.concat(all_counter,ignore_index=True,sort=False))
    summary={"run_id":RUN_ID,"status":"COMPLETED_EXPLORATORY_CPU_ONLY","protocol":contract,"analysis":{"endpoints":endpoint_info,"feature_axis":feature_info,"interpretation":"all labels were used only for exploratory association after model checkpoint, predictions, and feature set were frozen; labels do not select model, feature, sign, or candidate. Direct sensitivity is a reference for response transfer and is not PDO efficacy."},"budget":{"CPU_limit_minutes":120,"elapsed_minutes":(time.monotonic()-t0)/60,"GPU_hours":0.0,"training_or_download_started":False},"data_boundary":{"CTRP_values_read_for_roles_train_val_only":True,"PRISM_values_read_for_exact_train_val_samples_only":True,"locked_or_outer_numeric_values_labels_features_scores_read":False,"PDO_treated_labels_read":False,"model_retrained_or_changed":False,"feature_direction_selected_from_results":False},"five_arm_gate":json.loads((out/"five_arm_gate.json").read_text(encoding="utf-8")),"outputs_sha256":outputs}
    write_json(out/"summary.json",summary)
    inputs=[root/"experiments/records/EXP-050.md",root/"scripts/evaluation/analyze_exp050_independent_object_response_drug_effect.py",root/"artifacts/experiments/EXP-050"/ATTEMPT_ID/"attempt_receipt.json",root/"artifacts/experiments/EXP-050"/ATTEMPT_ID/"heldout_source_prediction_keys.tsv",root/"artifacts/experiments/EXP-050"/ATTEMPT_ID/"heldout_source_prediction_target_order.npy",root/"artifacts/experiments/EXP-050"/ATTEMPT_ID/"heldout_source_true_delta_target_order.npy",root/"artifacts/experiments/EXP-050"/ATTEMPT_ID/"fit_condition_identity_key_map.tsv",root/"artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json",root/"data/processed/exp013/ctrp_v2/responses.csv",root/"data/processed/exp013/ctrp_v2/drugs.csv",root/"data/processed/exp012/prism_extended_primary_pairs.parquet",root/"data/processed/exp012/prism_extended_primary_samples.parquet",root/"data/processed/exp012/prism_extended_primary_drugs.parquet",root/"data/processed/lincs/GSE92742/continuous_conditions_v2/condition_delta.npy",root/"data/processed/lincs/GSE92742/continuous_contract_v2/genes.json",root/"data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",root/"data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"]
    manifest={"run_id":RUN_ID,"status":"COMPLETED_EXPLORATORY_CPU_ONLY","started_at_utc":datetime.now(timezone.utc).isoformat(),"exit_code":0,"execution_environment":"WSL2 Conda drugscreening-gpu","python_executable":os.sys.executable,"python_version":os.sys.version,"compute":{"device":"CPU-only","gpu_hours":0.0,"elapsed_minutes":summary["budget"]["elapsed_minutes"]},"command":"wsl.exe -d Ubuntu --cd /mnt/d/Code/DrugScreenLab -- /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/analyze_exp050_independent_object_response_drug_effect.py --phase analyze --run-id 20261001_independent_object_response_drug_effect_v1","locked_or_outer_numeric_values_labels_features_scores_read":False,"PDO_treated_labels_read":False,"response_values_read":True,"inputs_sha256":{str(p.relative_to(root)).replace("\\","/"):sha256(p) for p in inputs if p.is_file()},"outputs_sha256":outputs}
    write_json(out/"run_manifest.json",manifest)
    return summary


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--phase",choices=["roles","analyze"],required=True)
    parser.add_argument("--run-id",default=RUN_ID)
    args=parser.parse_args(); root=args.root.resolve()
    if args.run_id!=RUN_ID: raise ValueError("run_id differs from frozen EXP-050 §3.4")
    out=root/"artifacts/experiments/EXP-050"/RUN_ID
    if args.phase=="roles":
        if (out/"summary.json").exists(): raise FileExistsError("analysis package already completed")
        result=role_preflight(root,out)
        print(json.dumps({"phase":"roles","status":result["status"],"roles":result["roles"],"CTRPv2":result["CTRPv2"],"PRISM":result["PRISM"],"five_arm_gate":result["five_arm_gate"],"outputs":result["outputs"]},ensure_ascii=False))
    else:
        if (out/"run_manifest.json").exists(): raise FileExistsError("run manifest exists; refusing overwrite")
        result=analyze(root,out)
        print(json.dumps({"phase":"analyze","status":result["status"],"elapsed_minutes":result["budget"]["elapsed_minutes"],"endpoints":result["analysis"]["endpoints"],"five_arm_gate":result["five_arm_gate"]},ensure_ascii=False))


if __name__=="__main__": main()
