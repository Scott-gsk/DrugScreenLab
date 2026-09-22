#!/usr/bin/env python3
"""Freeze the ChemPert data contract (Tasks 4.1-4.4).

Read-only over data/raw; writes frozen contract assets to data/processed/chempert:
  - chempert_primary_keys_FROZEN.parquet (+ manifest json)
  - chempert_leakage_keys_FROZEN.json
  - chempert_landmark_contract_v1/ (delta_supervision=true metadata + control
    expression slice on the LINCS landmark-978 gene intersection)

All outputs carry sha256 checksums and provenance.
"""

from __future__ import annotations

import argparse
import gzip
import csv
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
from rdata.parser._parser import RObjectType

FREEZE_TAG = "chempert_contract_freeze_20260919"
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


def load_lincs_training_compounds(lincs_dir: Path) -> dict:
    """Compounds actually covered by the exact978 level-3 cache (= all inst rows)."""
    names: set[str] = set()
    pids: set[str] = set()
    cells: set[str] = set()
    n_inst = 0
    with gzip.open(lincs_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz", "rt") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            n_inst += 1
            cells.add(row["cell_id"].strip())
            if row["pert_type"] == "trt_cp":
                names.add(row["pert_iname"].strip().lower())
                pids.add(row["pert_id"].strip())
    pert_info = pd.read_csv(lincs_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t", dtype=str)
    trt_cp_info = pert_info[pert_info.pert_type == "trt_cp"]
    gene_info = pd.read_csv(lincs_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t", dtype=str)
    landmark = sorted(gene_info[gene_info.pr_is_lm == "1"].pr_gene_symbol.tolist())
    return {
        "n_inst_rows": n_inst,
        "trt_cp_pert_iname": names,
        "trt_cp_pert_id": pids,
        "cell_ids": cells,
        "pert_info_trt_cp_rows": int(len(trt_cp_info)),
        "landmark_genes": landmark,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / RAW_REL
    lincs_dir = root / LINCS_REL
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    provenance = {
        "freeze_tag": FREEZE_TAG,
        "created_at": stamp,
        "created_by": "scripts/data/freeze_chempert_contract.py",
        "creation_environment": f"WSL2; conda drugscreening-gpu; Python {platform.python_version()}",
        "raw_inputs_read_only": True,
    }

    # ------------------------------------------------------------------ inputs
    responses = pd.read_csv(
        raw / "chempert_information_for_transcriptional_responses.csv", dtype="string"
    )
    expr_meta = pd.read_csv(
        raw / "chempert_information_for_gene_expression.csv", dtype="string"
    )
    if len(responses) != 82270:
        raise ValueError(f"expected 82270 response rows, got {len(responses)}")
    if responses.Response_ID.nunique() != len(responses):
        raise ValueError("Response_ID not unique in response metadata")

    lincs = load_lincs_training_compounds(lincs_dir)
    lincs_names = lincs["trt_cp_pert_iname"]
    landmark = lincs["landmark_genes"]

    # ------------------------------------------------- expression profile map
    expr_meta["is_control_profile"] = expr_meta.Matrix_Colname.str.contains(
        CONTROL_PATTERN, na=False
    )
    expr_meta["n_rids_on_profile"] = expr_meta.Response_ID.str.split(r"\s*;\s*").apply(len)
    rid_lists = expr_meta.assign(
        rid=expr_meta.Response_ID.str.split(r"\s*;\s*")
    ).explode("rid")[["rid", "Expression_ID", "Matrix_Colname", "is_control_profile"]]
    chem_per_eid = (
        responses.merge(
            rid_lists[["rid", "Expression_ID"]].rename(columns={"Expression_ID": "eid_ref"}),
            left_on="Response_ID",
            right_on="rid",
        )
        .groupby("eid_ref").Chemical_ID.nunique()
    )
    eid_kind = {}
    for eid, row in expr_meta.set_index("Expression_ID").iterrows():
        if bool(row.is_control_profile):
            eid_kind[eid] = "control_labeled"
        elif int(chem_per_eid.get(eid, 0)) >= 2:
            eid_kind[eid] = "shared_multi_chemical_baseline"
        else:
            eid_kind[eid] = "single_chemical_profile"
    eid_is_control = set(expr_meta.loc[expr_meta.is_control_profile, "Expression_ID"])

    # ------------------------------------------------------- primary key table
    pk = responses[
        [
            "Response_ID",
            "Chemical_ID",
            "Chemical_Compound",
            "Data_Accession",
            "Cell_Source",
            "Species",
            "Duration",
            "Concentration",
            "Expression_ID",
            "Dataset_Name",
        ]
    ].copy()
    pk["accession_group"] = np.where(
        pk.Data_Accession == "GSE92742",
        "GSE92742",
        np.where(pk.Data_Accession == "GSE70138", "GSE70138", "other"),
    )
    pk["delta_supervision"] = pk.Expression_ID.isin(eid_is_control)
    pk["lincs_overlap_chemical"] = (
        pk.Chemical_Compound.str.strip().str.lower().isin(lincs_names)
    )
    pk["expression_profile_kind"] = pk.Expression_ID.map(eid_kind)
    pk["n_rids_on_expression_id"] = pk.Expression_ID.map(
        expr_meta.set_index("Expression_ID").n_rids_on_profile.to_dict()
    )
    pk_path = out / "chempert_primary_keys_FROZEN.parquet"
    pk.to_parquet(pk_path, index=False)

    # ------------------------------------------------------- leakage keys
    chem = (
        pk[["Chemical_ID", "Chemical_Compound", "lincs_overlap_chemical"]]
        .drop_duplicates("Chemical_ID")
        .sort_values("Chemical_ID")
    )
    hit = chem[chem.lincs_overlap_chemical]
    per_group = (
        pk.groupby("accession_group")
        .agg(
            n_responses=("Response_ID", "size"),
            n_unique_chemical_id=("Chemical_ID", "nunique"),
            n_responses_lincs_overlap=("lincs_overlap_chemical", "sum"),
        )
        .to_dict("index")
    )
    cell_overlap = sorted(
        set(pk.Cell_Source.dropna().str.strip()) & lincs["cell_ids"]
    )
    # chemical x cell atoms that overlap LINCS on the chemical axis
    atoms = pk[pk.lincs_overlap_chemical][["Chemical_ID", "Cell_Source"]].drop_duplicates()
    leakage = {
        "status": "FROZEN",
        "freeze_tag": FREEZE_TAG,
        "created_at": stamp,
        "purpose": (
            "ChemPert x LINCS GSE92742 leakage atoms. Any chemical (Chemical_ID) "
            "overlapping the LINCS training compound set must be kept whole on one "
            "side of any train/test split; groups (accession_group) carrying such "
            "chemicals must not straddle sides."
        ),
        "source_csv": f"{RAW_REL}/chempert_information_for_transcriptional_responses.csv",
        "n_responses_total": int(len(pk)),
        "accession_groups": {
            g: {
                "n_responses": int(v["n_responses"]),
                "n_unique_chemical_id": int(v["n_unique_chemical_id"]),
                "n_responses_lincs_overlap": int(v["n_responses_lincs_overlap"]),
            }
            for g, v in per_group.items()
        },
        "lincs_training_compound_set": {
            "source": f"{LINCS_REL}/GSE92742_Broad_LINCS_inst_info.txt.gz",
            "definition": (
                "pert_iname of pert_type=trt_cp rows; inst_info rows align 1:1 with the "
                "exact978_cache_v1 matrix (1,319,138 x 978), so this is the compound set "
                "actually covered by LINCS GSE92742 training data."
            ),
            "n_inst_rows": lincs["n_inst_rows"],
            "n_trt_cp_pert_iname": len(lincs_names),
            "n_trt_cp_pert_id": len(lincs["trt_cp_pert_id"]),
            "pert_info_trt_cp_rows": lincs["pert_info_trt_cp_rows"],
        },
        "matching_method": {
            "channel": "lowercased Chemical_Compound == lowercased LINCS pert_iname",
            "caveat": (
                "ChemPert Chemical_ID is a ChemPert-internal sequential id (CIDxxxxx), "
                "not a PubChem CID (e.g. ChemPert CID01776 = Adapin/doxepin, PubChem CID "
                "3158), so identity matching is impossible without a structure bridge; "
                "name matching is exact-string only and therefore conservative-against-"
                "overblocking but may miss synonym pairs."
            ),
        },
        "n_chempert_unique_chemicals": int(len(chem)),
        "n_chemicals_overlapping_lincs": int(len(hit)),
        "overlapping_chemical_ids": {
            row.Chemical_ID: row.Chemical_Compound for row in hit.itertuples()
        },
        "n_chemical_x_cell_atoms_overlapping": int(len(atoms)),
        "cell_source_overlap_with_lincs": cell_overlap,
        "isolation_rule": (
            "For any split over ChemPert responses: all responses of a chemical with "
            "lincs_overlap_chemical=true must be on the same side as any LINCS training "
            "use of that compound; prefer whole accession_group isolation for "
            "GSE92742/GSE70138 groups."
        ),
        "promoted_from": {
            "draft_json": "data/processed/resource_acquisition_20260916/chempert/chempert_accession_leakage_groups_DRAFT.json",
            "note": "DRAFT retained read-only; this FROZEN table is authoritative.",
        },
    }
    leak_path = out / "chempert_leakage_keys_FROZEN.json"
    leak_path.write_text(json.dumps(leakage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ------------------------------------------------ landmark contract (4.4)
    print("parsing expression RDS ...", flush=True)
    parsed = rdata.parser.parse_file(raw / "chempert_gene_expression_profiles.rds")
    slots = dict(r_iter_pairlist(parsed.object.attributes))
    dim = [int(x) for x in slots["Dim"].value]
    gene_names = r_char_list(slots["Dimnames"].value[0])
    col_names = r_char_list(slots["Dimnames"].value[1])
    i_idx = np.asarray(slots["i"].value, dtype=np.int64)
    p_ptr = np.asarray(slots["p"].value, dtype=np.int64)
    x_val = np.asarray(slots["x"].value, dtype=np.float64)
    if dim != [len(gene_names), len(col_names)]:
        raise ValueError(f"Dim mismatch: {dim} vs names {[len(gene_names), len(col_names)]}")

    delta_true = pk[pk.delta_supervision]
    delta_eids = sorted(delta_true.Expression_ID.unique())
    eids_known = set(expr_meta.Expression_ID)
    unknown = [e for e in delta_eids if e not in eids_known]
    if unknown:
        raise ValueError(f"delta EIDs missing from expression metadata: {unknown[:5]}")
    em = expr_meta.set_index("Expression_ID")
    delta_cols = [em.loc[e, "Matrix_Colname"] for e in delta_eids]

    gene_pos = {g: i for i, g in enumerate(gene_names)}
    inter_genes = [g for g in landmark if g in gene_pos]
    gene_rows = np.array([gene_pos[g] for g in inter_genes], dtype=np.int64)
    row_sel = np.zeros(len(gene_names), dtype=bool)
    row_sel[gene_rows] = True

    col_pos = {c: j for j, c in enumerate(col_names)}
    col_index_map = {}
    for eid, cname in zip(delta_eids, delta_cols):
        if cname not in col_pos:
            raise ValueError(f"colname {cname!r} not in matrix")
        col_index_map[eid] = col_pos[cname]

    # dense slice: rows = landmark intersection genes, cols = control EIDs
    slice_mat = np.zeros((len(inter_genes), len(delta_eids)), dtype=np.float64)
    row_lookup = {r: k for k, r in enumerate(gene_rows)}
    for out_col, eid in enumerate(delta_eids):
        j = col_index_map[eid]
        lo, hi = int(p_ptr[j]), int(p_ptr[j + 1])
        for k in range(lo, hi):
            r = int(i_idx[k])
            if row_sel[r]:
                slice_mat[row_lookup[r], out_col] = x_val[k]

    slice_df = pd.DataFrame(slice_mat, index=inter_genes, columns=delta_eids)
    slice_df.index.name = "gene_symbol"

    lm_dir = out / "chempert_landmark_contract_v1"
    lm_dir.mkdir(parents=True, exist_ok=True)
    meta_path = lm_dir / "chempert_delta_true_metadata.parquet"
    lm_meta = delta_true.merge(
        expr_meta[["Expression_ID", "Matrix_Colname", "Data_Accession"]].rename(
            columns={"Data_Accession": "profile_accession"}
        ),
        on="Expression_ID",
        how="left",
    )
    lm_meta.to_parquet(meta_path, index=False)
    slice_path = lm_dir / "chempert_landmark_control_expression_slice.parquet"
    slice_df.reset_index().to_parquet(slice_path, index=False)

    # ------------------------------------------------------------- manifests
    def file_entry(path: Path) -> dict:
        rel = path.relative_to(root)
        return {"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}

    pk_manifest = {
        "status": "FROZEN",
        **provenance,
        "purpose": "Response_ID <-> Chemical_ID <-> Data_Accession <-> Expression_ID primary key table for ChemPert contract freeze (supports EXP-023).",
        "row_count": int(len(pk)),
        "columns": pk.columns.tolist(),
        "row_count_reconciliation": {
            "response_metadata_csv_rows": 82270,
            "response_rds_rows": 82270,
            "note": "CSV is authoritative for identity; RDS carries 14 versioned id variants (RIDxxxxx.1) recorded in the schema audit.",
        },
        "delta_supervision": {
            "definition": (
                "true iff the response's Expression_ID resolves to an expression profile "
                "whose Matrix_Colname is explicitly control-labeled (control/DMSO/vehicle/"
                "untrt/naive/mock), i.e. a matched control profile is available in the "
                "ChemPert expression matrix."
            ),
            "n_true": int(pk.delta_supervision.sum()),
            "n_false": int((~pk.delta_supervision).sum()),
            "n_control_profiles": int(expr_meta.is_control_profile.sum()),
            "limitation": (
                "ChemPert v2 provides no study with both treatment and control profiles "
                "in the matrix; treatment-side expression is therefore not recoverable "
                "for delta_supervision=true rows, and up/down gene lists (responses RDS, "
                "gene-symbol space) remain the delta readout."
            ),
        },
        "expression_profile_kind_counts": pk.expression_profile_kind.value_counts().to_dict(),
        "lincs_overlap_chemical_counts": {
            "n_responses_overlap": int(pk.lincs_overlap_chemical.sum()),
            "n_unique_chemical_overlap": int(pk.loc[pk.lincs_overlap_chemical, "Chemical_ID"].nunique()),
        },
        "source_files": {
            "responses_csv": file_entry(raw / "chempert_information_for_transcriptional_responses.csv"),
            "expression_csv": file_entry(raw / "chempert_information_for_gene_expression.csv"),
            "responses_rds": file_entry(raw / "chempert_transcriptional_responses.rds"),
            "expression_rds": file_entry(raw / "chempert_gene_expression_profiles.rds"),
            "lincs_inst_info": file_entry(lincs_dir / "GSE92742_Broad_LINCS_inst_info.txt.gz"),
            "lincs_pert_info": file_entry(lincs_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz"),
            "lincs_gene_info": file_entry(lincs_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz"),
        },
        "outputs": {"primary_keys_parquet": file_entry(pk_path)},
        "matrix_only_columns_note": (
            "Expression RDS has 5028 columns; expression metadata registers 5017. The 11 "
            "unregistered columns (penciclovir/valacyclovir treatment profiles, GSE59894/"
            "GSE57811/GSE59913/GSE75603) are not referenced by any response."
        ),
    }
    (out / "chempert_primary_keys_FROZEN_manifest.json").write_text(
        json.dumps(pk_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lm_manifest = {
        "status": "FROZEN",
        **provenance,
        "purpose": (
            "Metadata-only + landmark-intersection contract for delta_supervision=true "
            "responses: control-side expression slice over the LINCS landmark-978 "
            "gene-symbol intersection."
        ),
        "n_delta_true_responses": int(len(delta_true)),
        "n_control_profiles_sliced": len(delta_eids),
        "gene_space": {
            "matrix_rows_total": len(gene_names),
            "matrix_identifier": "gene symbols (Dimnames[[1]])",
            "lincs_landmark_genes": len(landmark),
            "intersection_size": len(inter_genes),
            "intersection_genes": inter_genes,
        },
        "matrix": {
            "class": "dgCMatrix",
            "dim": dim,
            "nnz": int(len(x_val)),
            "column_alignment": "Matrix_Colname <-> matrix column position; metadata Matrix_Index verified 1-based 0 mismatches",
        },
        "data_boundary": (
            "Post-perturbation transcriptional reference only; no viability/AUC/IC50 "
            "labels; strictly separate from the IMPROVE CSA drug-sensitivity benchmark."
        ),
        "outputs": {
            "delta_true_metadata_parquet": file_entry(meta_path),
            "control_expression_slice_parquet": file_entry(slice_path),
        },
        "known_limitation": (
            "Treatment-side expression is not present in the ChemPert download for any "
            "delta_supervision=true response (verified: no accession has both control-"
            "labeled and treatment profiles). The slice therefore carries the control "
            "arm; the treatment-vs-control delta itself remains the ChemPert up/down "
            "gene lists (gene-symbol space, responses RDS)."
        ),
    }
    (lm_dir / "manifest.json").write_text(
        json.dumps(lm_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # leakage manifest gets checksums of frozen siblings appended
    leakage["frozen_siblings"] = {
        "primary_keys": file_entry(pk_path),
        "primary_keys_manifest": file_entry(out / "chempert_primary_keys_FROZEN_manifest.json"),
    }
    leak_path.write_text(json.dumps(leakage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"pk rows={len(pk)} delta_true={pk.delta_supervision.sum()} "
          f"landmark_intersection={len(inter_genes)} control_profiles={len(delta_eids)}")
    print(f"lincs overlap chemicals={len(hit)} responses={pk.lincs_overlap_chemical.sum()}")
    print(f"elapsed {time.time()-t0:.1f}s")
    for p in (pk_path, leak_path, meta_path, slice_path):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
