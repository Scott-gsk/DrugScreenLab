#!/usr/bin/env python3
"""ChemPert contract v2 supplement (EXP-023 Task 4.5 pre-training audit).

Purpose
-------
The frozen v1 contract marks delta_supervision=true only for responses whose
Expression_ID resolves to an explicitly control-labeled profile (78 profiles /
143 responses). Pre-training usability re-audit found written evidence that the
496 shared LINCS plate-level profiles (shared_multi_chemical_baseline,
77,676 responses) are control/baseline profiles of the *initial cellular
state* and were conservatively excluded by a naming-pattern rule:

1. ChemPert official documentation defines the expression-profile resource as
   "gene expression profiles of initial cellular states"
   (https://chempert.uni.lu/information, retrieved 2026-09-19).
2. Structural necessity: each shared profile is referenced by 2..265 distinct
   Chemical_IDs (median 36.5). A post-treatment profile cannot be the
   treatment outcome of >1 distinct compound.
3. LINCS cross-validation: all 272 GSE92742-group profile colnames match
   LINCS GSE92742 inst_info rna_plate exactly (100%), those plates carry
   ctl_vehicle (DMSO) wells; all 225 GSE70138-group profiles carry explicit
   well positions (e.g. LJP005_HA1E_24H_X1_B19) and are shared by >=16
   distinct compounds each.

Rule (v2 supplement, strictly additive):
    delta_supervision_v2 = true iff expression_profile_kind in
        {control_labeled, shared_multi_chemical_baseline}
single_chemical_profile rows (4,451 responses; one-to-one profiles named with
compound+dose+duration, incl. PCLB001_HA1E_24H_X3, the only GSE92742-group
single profile shared by 4 responses of a single compound on a mixed plate)
stay false: their semantics remain ambiguous and no structural evidence
supports treating them as controls.

Outputs (data/processed/chempert/, v1 FROZEN files untouched):
  - chempert_delta_supervision_v2.parquet  per-response v2 flag + kind
  - chempert_v2_control_expression_slice.parquet  795 landmark genes x all
    v2-true Expression_IDs (dense, decoded from expression RDS)
  - chempert_v2_response_table.parquet  v2-true responses + up/down gene lists
  - chempert_v2_chem_smiles.parquet  Chemical_ID -> LINCS canonical_smiles
    bridge (name-matched, same channel as the frozen leakage keys)
  - chempert_v2_manifest.json  checksums + evidence statistics

Read-only over data/raw.
"""
from __future__ import annotations

import argparse
import gzip
import csv
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
from rdata.parser._parser import RObjectType

RAW_REL = Path("raw/resource_acquisition_20260916/improve_chempert")
LINCS_REL = Path("raw/lincs/GSE92742")
OUT_REL = Path("processed/chempert")
CONTROL_PATTERN = re.compile(r"dmso|vehicle|ctrl|control|untrt|untreated|naive|mock", re.IGNORECASE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def r_char_list(robj) -> list[str]:
    out = []
    for c in robj.value:
        v = c.value
        while hasattr(v, "value"):
            v = v.value
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def r_sym_name(tag) -> str | None:
    if tag is None:
        return None
    v = tag.value
    while hasattr(v, "value"):
        v = v.value
    return v.decode() if isinstance(v, bytes) else str(v)


def r_iter_pairlist(node):
    while node is not None and node.info.type == RObjectType.LIST:
        car, cdr = node.value[0], node.value[1]
        yield r_sym_name(node.tag), car
        node = cdr


def load_landmark_genes(lincs_dir: Path) -> list[str]:
    # Same extraction as freeze_chempert_contract.load_lincs_training_compounds
    with gzip.open(lincs_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", "rt",
                   encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return [r["pr_gene_symbol"] for r in rows if r.get("pr_is_lm", "") == "1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    args = ap.parse_args()
    root = args.root.resolve()
    raw = root / RAW_REL
    lincs_dir = root / LINCS_REL
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    pk = pd.read_parquet(out / "chempert_primary_keys_FROZEN.parquet")
    expr_meta = pd.read_csv(raw / "chempert_information_for_gene_expression.csv", dtype=str)
    responses = pd.read_csv(raw / "chempert_information_for_transcriptional_responses.csv", dtype="string")

    # ------------------------------------------------- evidence recomputation
    rid2chem = responses.set_index("Response_ID")["Chemical_ID"].to_dict()
    ev_rows = []
    for _, r in expr_meta.iterrows():
        rids = [x.strip() for x in str(r["Response_ID"]).split(";") if x.strip()]
        chems = {rid2chem.get(x) for x in rids}
        ev_rows.append((r["Expression_ID"], r["Matrix_Colname"], r["Data_Accession"],
                        len(rids), len(chems)))
    ev = pd.DataFrame(ev_rows, columns=["eid", "colname", "accession", "n_resp", "n_chem"])
    ev["kind"] = np.where(
        ev["colname"].str.contains(CONTROL_PATTERN, na=False),
        "control_labeled",
        np.where(ev["n_chem"] >= 2, "shared_multi_chemical_baseline", "single_chemical_profile"),
    )
    shared = ev[ev["kind"] == "shared_multi_chemical_baseline"]
    shared_by_acc = shared.groupby("accession").size().to_dict()

    # LINCS plate cross-validation (GSE92742 group)
    plate_set = set()
    with gzip.open(lincs_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", "rt",
                   encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            plate_set.add(row["rna_plate"])
    g9 = ev[ev["accession"] == "GSE92742"].copy()
    g9["bare"] = g9["colname"].str.replace(r"__GSE92742$", "", regex=True)
    n_plate_match = int(g9["bare"].isin(plate_set).sum())

    # ------------------------------------------------------ v2 flags
    upgrade_kinds = {"control_labeled", "shared_multi_chemical_baseline"}
    flags = pk[["Response_ID", "Expression_ID", "expression_profile_kind", "accession_group"]].copy()
    flags["delta_supervision_v2"] = flags["expression_profile_kind"].isin(upgrade_kinds)
    flags["v2_change"] = flags["delta_supervision_v2"] & ~flags["Expression_ID"].isin(
        set(expr_meta.loc[expr_meta["Matrix_Colname"].str.contains(CONTROL_PATTERN, na=False),
                          "Expression_ID"])
    )
    n_true_v2 = int(flags["delta_supervision_v2"].sum())

    # ------------------------------------------- control expression slice
    v2_true = pk[pk["expression_profile_kind"].isin(upgrade_kinds)].copy()
    delta_eids = sorted(v2_true["Expression_ID"].unique())
    em = expr_meta.set_index("Expression_ID")
    delta_cols = [em.loc[e, "Matrix_Colname"] for e in delta_eids]

    print(f"parsing expression RDS for {len(delta_eids)} v2 control profiles ...", flush=True)
    parsed = rdata.parser.parse_file(raw / "chempert_gene_expression_profiles.rds")
    slots = dict(r_iter_pairlist(parsed.object.attributes))
    dim = [int(x) for x in slots["Dim"].value]
    gene_names = r_char_list(slots["Dimnames"].value[0])
    col_names = r_char_list(slots["Dimnames"].value[1])
    i_idx = np.asarray(slots["i"].value, dtype=np.int64)
    p_ptr = np.asarray(slots["p"].value, dtype=np.int64)
    x_val = np.asarray(slots["x"].value, dtype=np.float64)

    landmark = load_landmark_genes(lincs_dir)
    gene_pos = {g: i for i, g in enumerate(gene_names)}
    inter_genes = [g for g in landmark if g in gene_pos]
    gene_rows = np.array([gene_pos[g] for g in inter_genes], dtype=np.int64)
    row_lookup = {r: k for k, r in enumerate(gene_rows)}
    row_sel = np.zeros(len(gene_names), dtype=bool)
    row_sel[gene_rows] = True

    col_pos = {c: j for j, c in enumerate(col_names)}
    missing_cols = [c for c in delta_cols if c not in col_pos]
    if missing_cols:
        raise ValueError(f"v2 control colnames missing from matrix: {missing_cols[:5]}")

    slice_mat = np.zeros((len(inter_genes), len(delta_eids)), dtype=np.float64)
    for out_col, cname in enumerate(delta_cols):
        j = col_pos[cname]
        lo, hi = int(p_ptr[j]), int(p_ptr[j + 1])
        for k in range(lo, hi):
            r = int(i_idx[k])
            if row_sel[r]:
                slice_mat[row_lookup[r], out_col] = x_val[k]
    slice_df = pd.DataFrame(slice_mat, index=inter_genes, columns=delta_eids)
    slice_df.index.name = "gene_symbol"

    # ------------------------------------------------- response up/down lists
    print("parsing responses RDS for up/down gene lists ...", flush=True)
    r_parsed = rdata.parser.parse_file(raw / "chempert_transcriptional_responses.rds")
    r_obj = r_parsed.object
    r_slots = dict(r_iter_pairlist(r_obj.attributes))
    r_cols = r_char_list(r_slots["names"])
    if r_cols != ["Response ID", "Up-regulation number", "Up-regulation",
                  "Down-regulation number", "Down-regulation"]:
        raise ValueError(f"unexpected responses RDS columns: {r_cols}")
    rid_col, up_col, down_col = r_obj.value[0], r_obj.value[2], r_obj.value[4]
    rds_rids = r_char_list(rid_col)
    ups = r_char_list(up_col)
    downs = r_char_list(down_col)
    if len(rds_rids) != len(responses) or len(ups) != len(responses):
        raise ValueError("responses RDS row count != CSV row count")
    # Row-position alignment: 14 RDS rows carry versioned ids (RIDxxxxx.1) where the
    # CSV renumbered them; the CSV is the authoritative identity source (schema audit).
    pos_mismatch = sum(1 for a, b in zip(responses["Response_ID"], rds_rids) if a != b)
    resp_lists = pd.DataFrame({
        "Response_ID": responses["Response_ID"].to_numpy(),
        "up_genes": ups,
        "down_genes": downs,
    })

    v2_table = v2_true.merge(resp_lists, on="Response_ID", how="left", validate="one_to_one")
    if v2_table["up_genes"].isna().any():
        raise ValueError("v2 responses missing up/down lists")

    # --------------------------------------------------- SMILES bridge
    smi_map: dict[str, dict] = {}
    with gzip.open(lincs_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", "rt",
                   encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("pert_type") != "trt_cp":
                continue
            name = (row.get("pert_iname") or "").strip().lower()
            smi = (row.get("canonical_smiles") or "").strip()
            if not name or not smi or smi in ("", "-666"):
                continue
            if name not in smi_map:
                smi_map[name] = {
                    "canonical_smiles": smi,
                    "inchi_key": row.get("inchi_key", ""),
                    "pubchem_cid": row.get("pubchem_cid", ""),
                }
    chem = pk[["Chemical_ID", "Chemical_Compound"]].drop_duplicates("Chemical_ID")
    chem["key"] = chem["Chemical_Compound"].str.strip().str.lower()
    chem = chem.merge(
        pd.DataFrame.from_dict(smi_map, orient="index").rename_axis("key").reset_index(),
        on="key", how="left",
    )
    n_chem_smiles = int(chem["canonical_smiles"].notna().sum())
    v2_resp_smiles = int(v2_table.merge(chem[["Chemical_ID", "canonical_smiles"]],
                                        on="Chemical_ID", how="left")["canonical_smiles"].notna().sum())

    # ------------------------------------------------------------- write
    flags_path = out / "chempert_delta_supervision_v2.parquet"
    flags.to_parquet(flags_path, index=False)
    slice_path = out / "chempert_v2_control_expression_slice.parquet"
    slice_df.reset_index().to_parquet(slice_path, index=False)
    table_path = out / "chempert_v2_response_table.parquet"
    v2_table.to_parquet(table_path, index=False)
    smiles_path = out / "chempert_v2_chem_smiles.parquet"
    chem.drop(columns=["key"]).to_parquet(smiles_path, index=False)

    outputs = {
        "chempert_delta_supervision_v2.parquet": {
            "path": f"processed/chempert/chempert_delta_supervision_v2.parquet",
            "bytes": flags_path.stat().st_size,
            "sha256": sha256(flags_path),
        },
        "chempert_v2_control_expression_slice.parquet": {
            "path": "processed/chempert/chempert_v2_control_expression_slice.parquet",
            "bytes": slice_path.stat().st_size,
            "sha256": sha256(slice_path),
        },
        "chempert_v2_response_table.parquet": {
            "path": "processed/chempert/chempert_v2_response_table.parquet",
            "bytes": table_path.stat().st_size,
            "sha256": sha256(table_path),
        },
        "chempert_v2_chem_smiles.parquet": {
            "path": "processed/chempert/chempert_v2_chem_smiles.parquet",
            "bytes": smiles_path.stat().st_size,
            "sha256": sha256(smiles_path),
        },
    }

    manifest = {
        "status": "FROZEN_SUPPLEMENT",
        "version": "v2_supplement",
        "supplement_tag": "chempert_contract_v2_supplement_20260919",
        "created_at": stamp,
        "created_by": "scripts/data/chempert_contract_v2_supplement.py",
        "creation_environment": f"WSL2; conda drugscreening-gpu; Python {platform.python_version()}",
        "raw_inputs_read_only": True,
        "parent_contract": {
            "freeze_tag": "chempert_contract_freeze_20260919",
            "report": "docs/reports/chempert_contract_freeze_20260919.md",
            "v1_files_untouched": True,
        },
        "supplement_rule": (
            "delta_supervision_v2 = true iff expression_profile_kind in "
            "{control_labeled, shared_multi_chemical_baseline}. Strictly additive over v1: "
            "single_chemical_profile rows (incl. PCLB001_HA1E_24H_X3) remain false."
        ),
        "evidence": {
            "official_documentation": {
                "url": "https://chempert.uni.lu/information",
                "retrieved": "2026-09-19",
                "quote": "the gene expression profiles of initial cellular states",
                "interpretation": (
                    "ChemPert defines its expression-profile resource as profiles of the "
                    "initial (pre-perturbation) cellular state, i.e. control/baseline profiles."
                ),
            },
            "structural_necessity": {
                "n_shared_profiles": int(len(shared)),
                "shared_profiles_by_accession": {k: int(v) for k, v in shared_by_acc.items()},
                "distinct_compounds_per_shared_profile": {
                    "min": int(shared["n_chem"].min()),
                    "median": float(shared["n_chem"].median()),
                    "max": int(shared["n_chem"].max()),
                },
                "argument": (
                    "A post-treatment profile cannot be the treatment outcome of two or more "
                    "distinct compounds; every shared_multi_chemical_baseline profile is "
                    "referenced by >=2 distinct Chemical_IDs, so none can be a treatment profile."
                ),
            },
            "lincs_cross_validation": {
                "gse92742_profiles_checked": int(len(g9)),
                "gse92742_profiles_matching_lincs_plate": n_plate_match,
                "gse70138_profiles_all_shared_min_chem": int(shared[shared["accession"] == "GSE70138"]["n_chem"].min()),
                "detail": (
                    "GSE92742-group colnames match LINCS inst_info rna_plate exactly "
                    "(plates carry ctl_vehicle DMSO wells); GSE70138-group colnames carry "
                    "explicit well positions (e.g. LJP005_HA1E_24H_X1_B19) and are each shared "
                    "by >=16 distinct compounds."
                ),
            },
            "conservative_exclusions": (
                "single_chemical_profile (4,451 responses): one-to-one profiles named with "
                "compound+dose+duration (tg-GATEs-style, e.g. "
                "rat_hepatocyte_E-MTAB-797_acetamidofluorene_24h_10micromolar); semantics "
                "ambiguous, no structural evidence of being controls; PCLB001_HA1E_24H_X3 "
                "(4 responses of one compound on a mixed plate) likewise excluded."
            ),
        },
        "counts": {
            "n_responses_total": int(len(pk)),
            "n_delta_true_v1": 143,
            "n_delta_true_v2": n_true_v2,
            "n_delta_true_v2_by_group": {
                k: int(v) for k, v in flags[flags["delta_supervision_v2"]]["accession_group"].value_counts().items()
            },
            "n_v2_control_profiles_sliced": len(delta_eids),
            "landmark_intersection_genes": len(inter_genes),
            "n_chempert_chemicals": int(len(chem)),
            "n_chemicals_with_lincs_smiles": n_chem_smiles,
            "n_v2_responses_with_smiles": v2_resp_smiles,
            "responses_rds_csv_position_mismatches": int(pos_mismatch),
        },
        "smiles_bridge": {
            "channel": "lowercased Chemical_Compound == lowercased LINCS pert_iname (trt_cp) "
                       "-> GSE92742 pert_info canonical_smiles; same name-matching channel as "
                       "the frozen leakage keys",
            "caveat": "synonym mismatches possible in both directions; chemicals without a "
                      "matched canonical_smiles are excluded from the ctrl_morgan arm only",
        },
        "data_boundary": "Post-perturbation transcriptional reference only; no viability/AUC/IC50 "
                         "labels; strictly separate from the IMPROVE CSA drug-sensitivity benchmark.",
        "outputs": outputs,
    }
    mpath = out / "chempert_v2_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"[v2-supplement] wrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
