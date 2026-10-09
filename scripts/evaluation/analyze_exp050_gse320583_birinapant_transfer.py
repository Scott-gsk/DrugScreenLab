#!/usr/bin/env python3
"""Compare exact-exposure GSE70138 MCF7 and GSE320583 PDO transcriptional responses."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_exp050_frozen_functional_transfer import load_gmt, sign, write_tsv


RUN_ID = "20261001_gse320583_birinapant_transfer_v1"
TARGET_KEYS = {
    "doxorubicin": "AOJJSUZBOXZQNB-TZSSRYMLSA-N",
    "bortezomib": "GXJABQQUPOEUTA-RDJZCZTQSA-N",
    "birinapant": "PKWRMUKBEYJEIX-DXXQBUJASA-N",
}
HALLMARK_PATH = "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
TAG_MIN_COUNT = 10
TAG_MIN_RATIO = 3.0
SOFT_CONDITION_TO_CODE = {"non-treated": "DMSO", "Bi": "Bi", "BiE": "BiE", "H+BiE": "H+BiE"}
CONDITION_LABELS = {
    "DMSO": "DMSO, 24 h",
    "Bi": "Birinapant, 10 µM, 24 h",
    "BiE": "Birinapant + Emricasan, composite exposure, 24 h",
    "H+BiE": "HOIPIN-8 pretreatment then Birinapant + Emricasan + HOIPIN-8, composite exposure",
}
SOURCE_DOSE_UM = 10.0
SOURCE_TIME_H = 24.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha512(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(pearsonr(left, right).statistic)
    return value if np.isfinite(value) else None


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def read_tsv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", dtype=str, low_memory=False)


def parse_geo_sample_map(soft_path: Path) -> tuple[dict[int, str], dict[int, dict[str, str]]]:
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as stream:
        text = stream.read()
    model_line = next((line for line in text.splitlines() if line.startswith("!Sample_characteristics_ch1 = cell line:")), None)
    tag_line = next((line for line in text.splitlines() if line.startswith("!Sample_characteristics_ch1 = hto:")), None)
    if model_line is None or tag_line is None:
        raise ValueError("GSE320583 GEO SOFT is missing the model or HTO map")
    model_map = {int(donor): model for donor, model in re.findall(r"#(\d+)\s+(HUB-C2-[A-Za-z0-9-]+)", model_line)}
    if model_map != {1: "HUB-C2-120", 2: "HUB-C2-152", 3: "HUB-C2-123"}:
        raise ValueError(f"Unexpected GEO donor-to-model annotation: {model_map}")
    matches = re.findall(r"SampleTag(\d+)_hs\s+Donor #(\d+)\s+(H\+BiE|BiE|Bi|non-treated)", tag_line)
    if len(matches) != 12 or {int(tag) for tag, _, _ in matches} != set(range(1, 13)):
        raise ValueError(f"GEO SOFT did not provide the expected 12 tag assignments: {len(matches)}")
    result: dict[int, dict[str, str]] = {}
    for tag, donor, condition in matches:
        donor_number = int(donor)
        result[int(tag)] = {
            "donor_number": donor_number,
            "model_id": model_map[donor_number],
            "condition_code": SOFT_CONDITION_TO_CODE[condition],
            "condition_label": CONDITION_LABELS[SOFT_CONDITION_TO_CODE[condition]],
        }
    return model_map, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--expression-chunk-rows", type=int, default=256)
    args = parser.parse_args()
    if args.expression_chunk_rows < 32 or args.expression_chunk_rows > 1024:
        raise ValueError("--expression-chunk-rows must be between 32 and 1024")

    root = args.repo_root.resolve()
    started_clock = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    out = root / "artifacts/experiments/EXP-050" / args.run_id
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run directory: {out}")
    out.mkdir(parents=True)

    raw_root = root / "data/raw/transfer_public_assets/gse_breast_drug_response"
    source_root = raw_root / "GSE70138"
    pdo_root = raw_root / "GSE320583"
    temp_gctx = root / "tmp/exp050_public_transfer_intake_v1/GSE70138_Level2.gctx"
    gmt_path = root / HALLMARK_PATH
    source_paths = {
        "pert": source_root / "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz",
        "cell": source_root / "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz",
        "inst": source_root / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz",
        "gene": source_root / "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz",
        "sig": source_root / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
        "metrics": source_root / "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
        "level2": source_root / "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx.gz",
        "source_checksum": source_root / "GSE70138_SHA512SUMS.txt.gz",
        "gex": pdo_root / "GSM9546005_Combined_scCITE-WTA-Brustkrebsorganoide_RSEC_MolsPerCell.csv.gz",
        "hto": pdo_root / "GSM9546006_scCITE-WTA-Brustkrebsorganoide_Sample_Tag_ReadsPerCell.csv.gz",
        "soft": pdo_root / "GSE320583_family.soft.gz",
        "filelist": pdo_root / "GSE320583_filelist.txt",
        "raw_archive": pdo_root / "GSE320583_RAW.tar",
        "gmt": gmt_path,
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing ({name}): {path}")
    if not temp_gctx.is_file():
        raise FileNotFoundError(f"Decompressed GCTX needed for keyed slice is missing: {temp_gctx}")

    # Exact full InChIKey binding; names are used only to label the resulting row.
    pert = read_tsv_gz(source_paths["pert"])
    cell = read_tsv_gz(source_paths["cell"])
    inst = read_tsv_gz(source_paths["inst"])
    gene_info = read_tsv_gz(source_paths["gene"])
    required_pert = {"pert_id", "pert_iname", "inchi_key", "canonical_smiles"}
    if not required_pert.issubset(pert.columns):
        raise ValueError(f"GSE70138 pert_info schema changed: {sorted(pert.columns)}")
    compounds = pert.loc[pert["inchi_key"].isin(TARGET_KEYS.values())].copy()
    compound_by_key = compounds.groupby("inchi_key")["pert_id"].agg(list).to_dict()
    if set(compound_by_key) != set(TARGET_KEYS.values()) or any(len(v) != 1 for v in compound_by_key.values()):
        raise ValueError(f"Exact chemical identity did not resolve one-to-one: {compound_by_key}")
    key_to_name = {v: k for k, v in TARGET_KEYS.items()}
    compounds["analysis_drug"] = compounds["inchi_key"].map(key_to_name)
    key_rows = compounds[["pert_id", "inchi_key", "analysis_drug"]].copy()
    if key_rows["pert_id"].duplicated().any():
        raise ValueError("GSE70138 exact-key pert_id mapping is not unique")
    required_cell = {"cell_id", "base_cell_id", "primary_site", "subtype", "cell_type"}
    if not required_cell.issubset(cell.columns):
        raise ValueError(f"GSE70138 cell_info schema changed: {sorted(cell.columns)}")
    cell_meta = cell[list(required_cell)].drop_duplicates()
    breast_cells = cell_meta.loc[cell_meta["primary_site"].str.casefold().eq("breast")].copy()
    source = inst.merge(key_rows, on="pert_id", how="inner", validate="many_to_one").merge(
        cell_meta, on="cell_id", how="left", validate="many_to_one", suffixes=("", "_cell")
    )
    source["dose_numeric"] = pd.to_numeric(source["pert_dose"], errors="coerce")
    source["time_numeric"] = pd.to_numeric(source["pert_time"], errors="coerce")
    breast_target = source.loc[
        source["pert_type"].eq("trt_cp") & source["primary_site"].str.casefold().eq("breast")
    ].copy()
    vehicle = inst.loc[
        inst["pert_type"].eq("ctl_vehicle") & inst["pert_iname"].fillna("").str.fullmatch("DMSO", case=False)
    ].copy()
    support_rows: list[dict[str, Any]] = []
    for key, group in breast_target.groupby(
        ["analysis_drug", "inchi_key", "pert_id", "cell_id", "base_cell_id", "primary_site", "subtype", "pert_dose_unit", "pert_dose", "pert_time", "pert_time_unit"],
        dropna=False,
        sort=True,
    ):
        drug, inchi, pert_id, cell_id, base_cell_id, site, subtype, dose_unit, dose, time_h, time_unit = key
        plate_ids = sorted(group["det_plate"].dropna().unique().tolist())
        matched_controls = vehicle.loc[
            vehicle["cell_id"].eq(cell_id)
            & vehicle["det_plate"].isin(plate_ids)
            & vehicle["pert_time"].eq(time_h)
            & vehicle["pert_time_unit"].eq(time_unit)
        ]
        matched_plate_ids = set(matched_controls["det_plate"])
        support_rows.append({
            "drug_label": drug,
            "pert_iname": group["pert_iname"].iloc[0],
            "full_inchi_key": inchi,
            "pert_id": pert_id,
            "cell_id": cell_id,
            "base_cell_id": base_cell_id,
            "primary_site": site,
            "subtype_annotation": subtype,
            "dose": dose,
            "dose_unit": dose_unit,
            "time": time_h,
            "time_unit": time_unit,
            "target_instances": int(group["inst_id"].nunique()),
            "target_plates": len(plate_ids),
            "plates_with_same_cell_time_DMSO": len(matched_plate_ids),
            "same_cell_time_DMSO_instances": int(matched_controls["inst_id"].nunique()),
            "all_target_plates_have_DMSO": bool(set(plate_ids).issubset(matched_plate_ids)),
        })
    support = pd.DataFrame(support_rows)
    if support.empty or not support["all_target_plates_have_DMSO"].all():
        raise ValueError("A breast-source target condition lacks same-cell/plate/time DMSO control")
    write_tsv(out / "gse70138_exact_key_breast_exposure_support.tsv", support.to_dict(orient="records"))

    # Fixed exact exposure shared with GSE320583. Other drug arms are source QC references only.
    source_exact = breast_target.loc[
        np.isclose(breast_target["dose_numeric"], SOURCE_DOSE_UM)
        & np.isclose(breast_target["time_numeric"], SOURCE_TIME_H)
        & breast_target["pert_dose_unit"].str.casefold().eq("um")
        & breast_target["pert_time_unit"].str.casefold().eq("h")
    ].copy()
    if source_exact.empty:
        raise ValueError("No exact 10 uM x 24 h source condition in breast cell metadata")

    # Parse sample tags and model IDs from the official GEO family record.
    model_map, tag_map = parse_geo_sample_map(source_paths["soft"])
    hto = pd.read_csv(source_paths["hto"], compression="gzip", skiprows=7, dtype=str)
    hto_cols = [c for c in hto.columns if re.match(r"SampleTag\d+_hs\|stAbO", c)]
    if len(hto_cols) != 12 or hto["Cell_Index"].duplicated().any():
        raise ValueError("GSE320583 HTO table has unexpected columns or duplicate Cell_Index")
    tag_values = hto[hto_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int64)
    sorted_values = np.sort(tag_values, axis=1)
    top_values, second_values = sorted_values[:, -1], sorted_values[:, -2]
    top_indices = tag_values.argmax(axis=1)
    unique_top = (tag_values == top_values[:, None]).sum(axis=1) == 1
    ratios = top_values / np.maximum(second_values, 1)
    assignment_rows: list[dict[str, Any]] = []
    assignment_by_cell: dict[str, dict[str, str]] = {}
    hto_cell_ids = set(hto["Cell_Index"].astype(str))
    for i, cell_index in enumerate(hto["Cell_Index"].astype(str)):
        tag_no = int(top_indices[i]) + 1
        accepted = bool(unique_top[i] and top_values[i] >= TAG_MIN_COUNT and ratios[i] >= TAG_MIN_RATIO)
        metadata = tag_map[tag_no]
        reason = "accepted_unique_tag" if accepted else (
            "tie_for_top_tag" if not unique_top[i] else "tag_count_below_minimum" if top_values[i] < TAG_MIN_COUNT else "top_to_second_ratio_below_minimum"
        )
        assignment_rows.append({
            "Cell_Index": cell_index,
            "sample_tag": f"SampleTag{tag_no:02d}_hs",
            "tag_reads": int(top_values[i]),
            "second_tag_reads": int(second_values[i]),
            "top_second_ratio": float(ratios[i]),
            "assigned": accepted,
            "assignment_reason": reason,
            **metadata,
        })
        if accepted:
            assignment_by_cell[cell_index] = metadata
    assignments = pd.DataFrame(assignment_rows)
    assignments.to_csv(out / "gse320583_response_blind_cell_tag_assignment.tsv.gz", sep="\t", index=False, compression="gzip")

    # Freeze the response analysis contract before reading RNA expression values.
    protocol = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "preregistered_question": "Do exact-key, 10 uM x 24 h breast cell-line responses to Birinapant resemble same-condition measured responses across independently identified GSE320583 breast organoid models?",
        "identity": "GSE70138 pert_info full InChIKey joined to registered DOX/BTZ/Birinapant keys. GSE320583 compound identity is the official GEO sample/condition label; its processed metadata do not provide a full InChIKey. Donor numbers map through GEO SOFT model labels.",
        "source_selection": "Include every GSE70138 breast base_cell_id with exact-key target and same-cell/same-plate/same-time DMSO; no target response values are used to choose a source.",
        "source_exact_comparison": "For each exact-key 10 uM x 24 h treatment instance, subtract the mean same-cell/same-plate/time DMSO Level2 GEX. Preserve each plate response, then average eligible plate vectors per drug.",
        "target_cell_assignment": f"Response-blind HTO tag call: unique top tag count >= {TAG_MIN_COUNT} and top/second count ratio >= {TAG_MIN_RATIO}; sample-tag mapping is taken from GSE320583_family.soft.gz.",
        "target_normalization": "Aggregate RSEC molecule counts by model-condition over assigned HTO singlets; exclude 20 ADT features from RNA library totals; compute log2(CPM+0.5) and within-model treatment-minus-DMSO deltas.",
        "primary_target_arm": "Single-agent Birinapant (Bi) versus same-model 1.5% DMSO, both 24 h; exact 10 uM dose.",
        "secondary_target_arms": "BiE and H+BiE remain separately labeled composite exposures; no component-isolating control, so no single-drug attribution.",
        "comparison": "Compare all exact-symbol-overlap GSE70138 landmark genes (minimum 900 required) by rank and direction; compute descriptive Spearman/Pearson and nonzero sign agreement. Summarize the predeclared 50 Hallmark sets and coverage. Do not compare absolute magnitudes across L1000 and RNA-seq.",
        "inference_rule": "Report model-level values, signs and coverage only. No cell-as-replicate p-values or patient-level intervals. Random-source panel is run only when >=2 eligible independent breast base_cell_ids; otherwise report the one-line limitation.",
        "resource_and_data_boundary": "0 GPU hours; no training, direct drug-sensitivity values, PDO sensitivity labels, outer/locked values, external contact, wet lab or FASTQ; processed matrix input <=5 GiB.",
        "frozen_before_expression_values": True,
        "frozen_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    protocol_path = out / "protocol_snapshot.json"
    protocol_bytes = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    protocol_path.write_bytes(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()

    # Read only tagged RNA expression rows and aggregate molecule counts by model/condition.
    gex_header = pd.read_csv(source_paths["gex"], compression="gzip", skiprows=7, nrows=0)
    if len(gex_header.columns) < 1000 or gex_header.columns[0] != "Cell_Index":
        raise ValueError("GSE320583 GEX matrix schema is not cell-by-feature")
    feature_columns = list(gex_header.columns[21:])  # first 20 features are ADTs, followed by RNA genes
    if not feature_columns:
        raise ValueError("GSE320583 GEX matrix has no RNA gene features after the 20 ADT columns")
    unique_gene_symbols = list(dict.fromkeys(feature_columns))
    first_index: dict[str, int] = {}
    unique_gene_index = {symbol: index for index, symbol in enumerate(unique_gene_symbols)}
    duplicate_pairs: list[tuple[int, int]] = []
    for index, symbol in enumerate(feature_columns):
        if symbol in first_index:
            duplicate_pairs.append((first_index[symbol], index))
        else:
            first_index[symbol] = index
    unique_positions = np.array([first_index[s] for s in unique_gene_symbols], dtype=int)
    group_keys = [(model, condition) for model in sorted(model_map.values()) for condition in ["DMSO", "Bi", "BiE", "H+BiE"]]
    group_counts = {key: np.zeros(len(unique_gene_symbols), dtype=np.int64) for key in group_keys}
    group_library_total = {key: 0 for key in group_keys}
    group_cell_count = {key: 0 for key in group_keys}
    assigned_counts_by_cell = {cell_id: (meta["model_id"], meta["condition_code"]) for cell_id, meta in assignment_by_cell.items()}
    gene_dtypes = {name: np.int32 for name in feature_columns}
    gene_dtypes["Cell_Index"] = "int64"
    seen_gex_cells: set[str] = set()
    gex_rows = 0
    joined_rows = 0
    matched_assigned_rows = 0
    for chunk in pd.read_csv(
        source_paths["gex"], compression="gzip", skiprows=7, dtype=gene_dtypes,
        chunksize=args.expression_chunk_rows, low_memory=False,
    ):
        ids = chunk["Cell_Index"].astype(str).to_numpy()
        duplicates = set(ids).intersection(seen_gex_cells)
        if duplicates:
            raise ValueError(f"Duplicate GSE320583 GEX cell IDs across chunks: {sorted(duplicates)[:5]}")
        seen_gex_cells.update(ids)
        gex_rows += len(chunk)
        cell_data = chunk[feature_columns].to_numpy(dtype=np.int64, copy=False)
        if (cell_data < 0).any():
            raise ValueError("Negative RSEC molecule count found in GSE320583 expression matrix")
        cell_totals = cell_data.sum(axis=1, dtype=np.int64)
        unique_data = cell_data[:, unique_positions].copy()
        for first, duplicate in duplicate_pairs:
            unique_data[:, unique_gene_index[feature_columns[first]]] += cell_data[:, duplicate]
        labels: list[tuple[str, str] | None] = [assigned_counts_by_cell.get(cell_id) for cell_id in ids]
        for label in set(labels):
            if label is None:
                continue
            row_mask = np.fromiter((v == label for v in labels), dtype=bool, count=len(labels))
            if not row_mask.any():
                continue
            key = (str(label[0]), str(label[1]))
            if key not in group_counts:
                raise ValueError(f"Unexpected model/condition tag assignment: {key}")
            group_counts[key] += unique_data[row_mask].sum(axis=0, dtype=np.int64)
            group_library_total[key] += int(cell_totals[row_mask].sum(dtype=np.int64))
            group_cell_count[key] += int(row_mask.sum())
            matched_assigned_rows += int(row_mask.sum())
        joined_rows += sum(cell_id in hto_cell_ids for cell_id in ids)
    if not seen_gex_cells:
        raise ValueError("GSE320583 GEX matrix parsed zero cells")
    missing_tags = hto_cell_ids - seen_gex_cells
    unmatched_tags = seen_gex_cells - hto_cell_ids
    if len(unmatched_tags) > 0.01 * len(seen_gex_cells):
        raise ValueError(f"More than 1% of GEX cells lack an HTO row: {len(unmatched_tags)}/{len(seen_gex_cells)}")

    group_manifest_rows = []
    expression_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    log_expression: dict[tuple[str, str], np.ndarray] = {}
    for key in group_keys:
        model, condition = key
        cells = group_cell_count[key]
        total = group_library_total[key]
        if cells == 0 or total <= 0:
            raise ValueError(f"No expression support for tagged group {key}")
        if cells < 50:
            raise ValueError(f"Fewer than 50 high-confidence cells in tagged group {key}: {cells}")
        cpm = group_counts[key].astype(np.float64) / total * 1_000_000.0
        log_cpm = np.log2(cpm + 0.5)
        if not np.isfinite(log_cpm).all():
            raise ValueError(f"Non-finite pseudobulk log2 CPM for {key}")
        log_expression[key] = log_cpm
        group_manifest_rows.append({
            "model_id": model,
            "condition_code": condition,
            "condition_label": CONDITION_LABELS[condition],
            "high_confidence_cells": cells,
            "library_total_rsec_molecules": total,
            "rna_gene_features": len(unique_gene_symbols),
        })
        for gene, raw_count, value in zip(unique_gene_symbols, group_counts[key], log_cpm, strict=True):
            expression_rows.append({
                "model_id": model,
                "condition_code": condition,
                "gene_symbol": gene,
                "sum_rsec_molecules": int(raw_count),
                "high_confidence_cells": cells,
                "log2_cpm_plus_0.5": float(value),
            })
    control_code = "DMSO"
    for model in sorted(model_map.values()):
        for treatment in ["Bi", "BiE", "H+BiE"]:
            treatment_key = (model, treatment)
            control_key = (model, control_code)
            delta = log_expression[treatment_key] - log_expression[control_key]
            for gene, treated, control, value in zip(
                unique_gene_symbols, log_expression[treatment_key], log_expression[control_key], delta, strict=True
            ):
                delta_rows.append({
                    "model_id": model,
                    "condition_code": treatment,
                    "condition_label": CONDITION_LABELS[treatment],
                    "matched_control": "1.5% DMSO, 24 h",
                    "gene_symbol": gene,
                    "treated_log2_cpm_plus_0.5": float(treated),
                    "control_log2_cpm_plus_0.5": float(control),
                    "delta_log2_cpm": float(value),
                    "treated_high_confidence_cells": group_cell_count[treatment_key],
                    "control_high_confidence_cells": group_cell_count[control_key],
                })
    pd.DataFrame(expression_rows).to_csv(out / "gse320583_model_condition_pseudobulk_expression.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(delta_rows).to_csv(out / "gse320583_model_response_gene_deltas.tsv.gz", sep="\t", index=False, compression="gzip")
    write_tsv(out / "gse320583_model_condition_cell_counts.tsv", group_manifest_rows)

    # Read the keyed, exact 10 uM x 24 h GSE70138 profiles and their same-plate MCF7 DMSO controls.
    source_exact = source_exact.loc[source_exact["cell_id"].isin(breast_cells["cell_id"])].copy()
    exact_target_ids = source_exact["inst_id"].astype(str).tolist()
    exact_plate_ids = sorted(source_exact["det_plate"].dropna().unique().tolist())
    source_exact_controls = vehicle.loc[
        vehicle["cell_id"].isin(breast_cells["cell_id"])
        & vehicle["det_plate"].isin(exact_plate_ids)
        & vehicle["pert_time"].astype(str).eq(str(SOURCE_TIME_H))
        & vehicle["pert_time_unit"].str.casefold().eq("h")
    ].copy()
    selected_source = pd.concat([source_exact, source_exact_controls], ignore_index=True).drop_duplicates("inst_id")
    matrix_inst_to_index: dict[str, int] = {}
    source_gene_ids: list[str] = []
    with h5py.File(temp_gctx, "r") as h5:
        matrix = h5["/0/DATA/0/matrix"]
        matrix_inst_ids = [x.decode("utf-8") for x in h5["/0/META/COL/id"][:]]
        source_gene_ids = [x.decode("utf-8") for x in h5["/0/META/ROW/id"][:]]
        if matrix.shape != (345_976, 978) or len(matrix_inst_ids) != 345_976 or len(source_gene_ids) != 978:
            raise ValueError(f"Unexpected GSE70138 Level2 GCTX axes: {matrix.shape}")
        if len(set(matrix_inst_ids)) != len(matrix_inst_ids) or len(set(source_gene_ids)) != 978:
            raise ValueError("GSE70138 GCTX contains duplicate profile or gene keys")
        matrix_inst_to_index = {inst_id: i for i, inst_id in enumerate(matrix_inst_ids)}
        missing_profiles = sorted(set(selected_source["inst_id"].astype(str)) - set(matrix_inst_to_index))
        if missing_profiles:
            raise ValueError(f"Selected GSE70138 profiles missing from Level2 matrix: {len(missing_profiles)}")
        gene_info["pr_gene_id"] = gene_info["pr_gene_id"].astype(str)
        gene_map = gene_info.drop_duplicates("pr_gene_id").set_index("pr_gene_id")["pr_gene_symbol"].astype(str).to_dict()
        source_symbols = [gene_map.get(gene_id, "") for gene_id in source_gene_ids]
        if any(not symbol for symbol in source_symbols) or len(set(source_symbols)) != 978:
            raise ValueError("GSE70138 978 landmark axis does not map one-to-one to gene symbols")
        profile_indices = sorted(matrix_inst_to_index[i] for i in selected_source["inst_id"].astype(str))
        profile_ids_sorted = [matrix_inst_ids[i] for i in profile_indices]
        profile_values = np.asarray(matrix[profile_indices, :], dtype=np.float64)
    profile_by_id = {inst_id: profile_values[i] for i, inst_id in enumerate(profile_ids_sorted)}
    source_plate_vectors: dict[tuple[str, str], np.ndarray] = {}
    source_plate_rows: list[dict[str, Any]] = []
    source_matrix_rows: list[dict[str, Any]] = []
    exposure_to_targets: dict[str, pd.DataFrame] = {}
    for drug in sorted(TARGET_KEYS):
        group_all = source_exact.loc[source_exact["analysis_drug"].eq(drug)].copy()
        exposure_to_targets[drug] = group_all
        if group_all.empty:
            continue
        plates_for_drug = sorted(group_all["det_plate"].dropna().unique())
        for plate in plates_for_drug:
            treatment_rows = group_all.loc[group_all["det_plate"].eq(plate)]
            ctl_rows = source_exact_controls.loc[
                source_exact_controls["det_plate"].eq(plate)
                & source_exact_controls["cell_id"].eq(treatment_rows["cell_id"].iloc[0])
            ]
            if treatment_rows.empty or ctl_rows.empty:
                raise ValueError(f"Missing exact-dose source treatment or matched MCF7 control: {drug} {plate}")
            treatment_matrix = np.vstack([profile_by_id[str(i)] for i in treatment_rows["inst_id"]]).mean(axis=0)
            control_matrix = np.vstack([profile_by_id[str(i)] for i in ctl_rows["inst_id"]]).mean(axis=0)
            response = treatment_matrix - control_matrix
            source_plate_vectors[(drug, plate)] = response
            source_plate_rows.append({
                "drug_label": drug,
                "full_inchi_key": TARGET_KEYS[drug],
                "source_cell_id": treatment_rows["cell_id"].iloc[0],
                "source_base_cell_id": treatment_rows["base_cell_id"].iloc[0],
                "dose_um": SOURCE_DOSE_UM,
                "time_h": SOURCE_TIME_H,
                "plate": plate,
                "target_instances": int(treatment_rows["inst_id"].nunique()),
                "same_plate_time_DMSO_instances": int(ctl_rows["inst_id"].nunique()),
            })
            for gene, value in zip(source_symbols, response, strict=True):
                source_matrix_rows.append({
                    "drug_label": drug,
                    "source_cell_id": treatment_rows["cell_id"].iloc[0],
                    "dose_um": SOURCE_DOSE_UM,
                    "time_h": SOURCE_TIME_H,
                    "plate": plate,
                    "gene_symbol": gene,
                    "treatment_minus_same_plate_DMSO_level2": float(value),
                })
    if not source_plate_vectors:
        raise ValueError("No source plate-level treatment-minus-DMSO vectors were created")
    pd.DataFrame(source_plate_rows).to_csv(out / "gse70138_exact_key_plate_replicates.tsv", sep="\t", index=False)
    pd.DataFrame(source_matrix_rows).to_csv(out / "gse70138_exact_key_plate_response_vectors.tsv.gz", sep="\t", index=False, compression="gzip")

    source_mean: dict[str, np.ndarray] = {}
    source_sd: dict[str, np.ndarray] = {}
    source_cell_count: dict[str, int] = {}
    for drug in sorted({d for d, _ in source_plate_vectors}):
        vectors = [v for (d, _), v in sorted(source_plate_vectors.items()) if d == drug]
        source_mean[drug] = np.mean(vectors, axis=0)
        source_sd[drug] = np.std(vectors, axis=0, ddof=1) if len(vectors) > 1 else np.full(978, np.nan)
        source_cell_count[drug] = len(vectors)

    # Per-model target response vs the source exact same full key, dose, and time.
    gene_to_source_index = {symbol: i for i, symbol in enumerate(source_symbols)}
    delta_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for row in delta_rows:
        delta_lookup[(row["model_id"] + "|" + row["condition_code"], row["gene_symbol"])] = (
            row["delta_log2_cpm"], row["treated_log2_cpm_plus_0.5"]
        )
    common_symbols = [gene for gene in source_symbols if gene in unique_gene_symbols]
    if len(common_symbols) < 900:
        raise ValueError(f"GSE320583 exact-symbol overlap with GSE70138 landmarks is below the frozen 900-gene minimum: {len(common_symbols)}/978")
    source_birinapant = source_mean.get("birinapant")
    if source_birinapant is None:
        raise ValueError("The Birinapant source vector is missing despite the exact-key metadata match")
    cross_system_rows: list[dict[str, Any]] = []
    gene_counterexamples: list[dict[str, Any]] = []
    target_birinapant_vectors: dict[str, np.ndarray] = {}
    for model in sorted(model_map.values()):
        target = np.array([delta_lookup[(model + "|Bi", gene)][0] for gene in common_symbols], dtype=float)
        target_birinapant_vectors[model] = target
        source_common = np.array([source_birinapant[gene_to_source_index[gene]] for gene in common_symbols], dtype=float)
        nonzero = (target != 0) & (source_common != 0)
        same = np.sign(target[nonzero]) == np.sign(source_common[nonzero])
        source_ranks = rankdata(source_common, method="average")
        target_ranks = rankdata(target, method="average")
        for i, gene in enumerate(common_symbols):
            gene_counterexamples.append({
                "model_id": model,
                "drug_label": "birinapant",
                "full_inchi_key": TARGET_KEYS["birinapant"],
                "gene_symbol": gene,
                "source_level2_delta": float(source_common[i]),
                "source_delta_sign": sign(float(source_common[i])),
                "pdo_delta_log2_cpm": float(target[i]),
                "pdo_delta_sign": sign(float(target[i])),
                "rank_gap_abs": float(abs(source_ranks[i] - target_ranks[i])),
                "direction_discordant": bool(source_common[i] * target[i] < 0),
            })
        cross_system_rows.append({
            "model_id": model,
            "target_dataset": "GSE320583",
            "source_dataset": "GSE70138",
            "source_cell_id": "MCF7",
            "drug_label": "birinapant",
            "full_inchi_key": TARGET_KEYS["birinapant"],
            "dose_um": SOURCE_DOSE_UM,
            "time_h": SOURCE_TIME_H,
            "source_plate_replicates": source_cell_count["birinapant"],
            "source_target_instances": int(exposure_to_targets["birinapant"]["inst_id"].nunique()),
            "pdo_model_condition_cells": group_cell_count[(model, "Bi")],
            "pdo_matched_control_cells": group_cell_count[(model, "DMSO")],
            "n_common_landmark_genes": len(common_symbols),
            "gene_spearman_rho": spearman(source_common, target),
            "gene_pearson_r": pearson(source_common, target),
            "nonzero_gene_sign_agreement": float(same.mean()) if len(same) else None,
            "nonzero_gene_sign_denominator": int(nonzero.sum()),
            "source_scale": "GSE70138 Level2 GEX; each source plate is treatment minus its same-cell/same-plate/time DMSO mean",
            "target_scale": "GSE320583 RSEC molecule counts pseudobulked over response-blind HTO singlets; log2 CPM + 0.5 treatment minus same-model DMSO",
            "absolute_amplitudes_comparable": False,
            "inferential_p_value": None,
        })
    write_tsv(out / "gse320583_birinapant_cross_system_gene_agreement.tsv", cross_system_rows)
    counterexamples = pd.DataFrame(gene_counterexamples)
    counterexamples = counterexamples.sort_values(["model_id", "rank_gap_abs"], ascending=[True, False])
    counterexamples.to_csv(out / "gse320583_birinapant_gene_counterexamples.tsv.gz", sep="\t", index=False, compression="gzip")

    gene_sets, _ = load_gmt(gmt_path)
    target_hallmark_rows: list[dict[str, Any]] = []
    cross_hallmark_rows: list[dict[str, Any]] = []
    for hallmark, members in gene_sets.items():
        for model in sorted(model_map.values()):
            for treatment in ["Bi", "BiE", "H+BiE"]:
                obs = [delta_lookup[(model + "|" + treatment, gene)][0] for gene in members if (model + "|" + treatment, gene) in delta_lookup]
                target_hallmark_rows.append({
                    "model_id": model,
                    "condition_code": treatment,
                    "condition_label": CONDITION_LABELS[treatment],
                    "hallmark_set": hallmark,
                    "gmt_gene_count": len(members),
                    "observed_target_gene_count": len(obs),
                    "target_gene_coverage": len(obs) / len(members),
                    "mean_target_delta_log2_cpm": float(np.mean(obs)) if obs else None,
                    "target_direction": sign(float(np.mean(obs))) if obs else "unavailable",
                })
        common = [g for g in members if g in gene_to_source_index and g in unique_gene_symbols]
        for model in sorted(model_map.values()):
            target = np.array([delta_lookup[(model + "|Bi", gene)][0] for gene in common], dtype=float)
            source_vec = np.array([source_birinapant[gene_to_source_index[gene]] for gene in common], dtype=float)
            target_mean = float(target.mean()) if len(target) else None
            source_mean_set = float(source_vec.mean()) if len(source_vec) else None
            cross_hallmark_rows.append({
                "model_id": model,
                "hallmark_set": hallmark,
                "gmt_gene_count": len(members),
                "source_common_gene_count": len(common),
                "source_gene_coverage_of_gmt": len(common) / len(members),
                "target_mean_delta_log2_cpm": target_mean,
                "target_direction": sign(target_mean) if target_mean is not None else "unavailable",
                "source_mean_delta_level2": source_mean_set,
                "source_direction": sign(source_mean_set) if source_mean_set is not None else "unavailable",
                "direction_relation": "same_sign" if target_mean and source_mean_set and np.sign(target_mean) == np.sign(source_mean_set) else "opposite_sign" if target_mean and source_mean_set else "cannot_judge_zero_or_unavailable",
                "within_set_gene_spearman_rho": spearman(source_vec, target) if len(common) >= 3 else None,
            })
    write_tsv(out / "gse320583_hallmark_target_directions.tsv", target_hallmark_rows)
    write_tsv(out / "gse320583_hallmark_cross_system_directions.tsv", cross_hallmark_rows)

    # Count cross-model direction consistency at the model level, never at the cell level.
    hallmark_consistency_rows: list[dict[str, Any]] = []
    hallmark_target = pd.DataFrame(target_hallmark_rows)
    for hallmark, group in hallmark_target.loc[hallmark_target["condition_code"].eq("Bi")].groupby("hallmark_set", sort=True):
        signs = group.set_index("model_id")["target_direction"].to_dict()
        resolved = [v for v in signs.values() if v in {"positive", "negative"}]
        same = len(set(resolved)) <= 1
        hallmark_consistency_rows.append({
            "condition_code": "Bi",
            "hallmark_set": hallmark,
            "resolved_models": len(resolved),
            "positive_models": sum(v == "positive" for v in resolved),
            "negative_models": sum(v == "negative" for v in resolved),
            "zero_or_unavailable_models": len(signs) - len(resolved),
            "model_directions": json.dumps(signs, ensure_ascii=False, sort_keys=True),
            "cross_model_direction_consistent": bool(same and len(resolved) == len(signs)),
        })
    write_tsv(out / "gse320583_birinapant_hallmark_model_consistency.tsv", hallmark_consistency_rows)

    source_unique_lines = int(source_exact["base_cell_id"].nunique())
    summary = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "started_at_local": started_at,
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_cpu_wall_minutes": (time.monotonic() - started_clock) / 60,
        "gpu_hours": 0,
        "gse70138_exact_key_support": {
            "target_drugs": TARGET_KEYS,
            "breast_source_base_cell_ids": sorted(breast_target["base_cell_id"].dropna().unique().tolist()),
            "breast_source_cell_count": int(breast_target["base_cell_id"].nunique()),
            "exact_key_breast_exposure_condition_rows": int(len(support)),
            "all_exact_key_breast_conditions_have_same_cell_plate_time_DMSO": bool(support["all_target_plates_have_DMSO"].all()),
            "exact_10um_24h_source_counts_by_drug": {
                drug: {
                    "source_cells": sorted(group["cell_id"].unique().tolist()),
                    "target_instances": int(group["inst_id"].nunique()),
                    "plates": int(group["det_plate"].nunique()),
                }
                for drug, group in source_exact.groupby("analysis_drug", sort=True)
            },
            "all_source_means_random_panel_rule": "No multi-source average or count-matched random source panel: GSE70138 has one eligible breast base_cell_id (MCF7) for these exact-key compounds. This does not prevent the separately scoped one-line exact-exposure Birinapant comparison.",
        },
        "gse320583_response": {
            "model_map_from_official_GEO": model_map,
            "tag_map_from_official_GEO": {str(k): v for k, v in tag_map.items()},
            "GEX_cells": gex_rows,
            "HTO_cells": len(hto),
            "unique_GEX_cells": len(seen_gex_cells),
            "unique_HTO_cells": len(hto_cell_ids),
            "HTO_GEX_joined_cells": joined_rows,
            "HTO_cells_missing_GEX": len(missing_tags),
            "GEX_cells_missing_HTO": len(unmatched_tags),
            "high_confidence_cells_used": matched_assigned_rows,
            "tag_rule": f"Unique largest SampleTag count >= {TAG_MIN_COUNT} and largest/second-largest ratio >= {TAG_MIN_RATIO}; defined before expression values were analyzed.",
            "condition_cell_counts": group_manifest_rows,
            "primary_single_agent": "Bi: 10 uM Birinapant, 24 h versus the same-model 1.5% DMSO, 24 h control.",
            "combination_arms": "BiE (Birinapant+Emricasan, 24 h) and H+BiE (30 uM HOIPIN-8 pretreatment 30 min, then Birinapant+Emricasan+HOIPIN-8 for 24 h) are separate composite exposures; no component-isolating control is present.",
            "count_normalization": "Pseudobulk sum of RSEC molecules across response-blind HTO singlet cells per model-condition; CPM over RNA genes only; log2(CPM+0.5).",
        },
        "matched_exposure_transfer": {
            "source": "One GSE70138 MCF7 breast cell line; 10 uM x 24 h exact-key Birinapant minus same-plate/same-time DMSO, averaged across all eligible plate profiles.",
            "target": "Three GEO-mapped metastatic breast organoid model IDs from GSE320583; 10 uM x 24 h single-agent Birinapant versus 1.5% DMSO.",
            "target_identity_basis": "Official GSE320583 treatment/sample-tag annotation identifies Birinapant; a target-side full InChIKey is not supplied. The target/source match is therefore same-name, dose- and time-matched, not a two-sided structure-key match.",
            "gene_axis": f"{len(common_symbols)} exact-symbol-overlap genes from the 978 GSE70138 L1000 landmarks; the predeclared minimum is 900.",
            "per_model_agreement": cross_system_rows,
            "absolute_amplitudes_comparable": False,
            "biological_n_for_gene_correlation": "Three target organoid model-level contrasts and one source cell line. Genes and cells are not independent biological replicates.",
            "inference": "No p-values, cell-level confidence intervals, efficacy claim, or patient-level prediction. Descriptive gene rank/sign agreement only.",
        },
        "hallmark": {
            "sets": len(gene_sets),
            "target_full_gene_axis_rows": len(target_hallmark_rows),
            "cross_system_model_hallmark_rows": len(cross_hallmark_rows),
            "model_direction_consistency_rows": hallmark_consistency_rows,
            "interpretation": "Hallmark means summarize predefined gene sets; cells, genes and sets are not independent biological replicates. Source L1000 and target RNA-seq values have different scales, so only gene ranks/signs and separate within-system directions are compared.",
        },
        "boundaries": [
            "No direct drug-sensitivity numeric labels, PDO sensitivity labels, outer/locked labels, model fitting, GPU, external contact, procurement, or wet-lab action.",
            "GSE70138 contributes one breast source cell line (MCF7); the predeclared multi-source average and sample-count-matched random panel are not applicable.",
            "GSE320583 provides one pooled library per model-condition; HTO-demultiplexed cells are subsamples, not replicate organoids.",
            "The source compound is bound by full InChIKey. The target matrix identifies Birinapant by its official GEO condition label but does not provide a full InChIKey; dose and time match by annotation, and cross-platform absolute expression magnitudes are not comparable.",
            "The two GSE320583 combination arms are retained as composite exposures, not attributed to Birinapant alone.",
        ],
        "outputs": {p.name: p.relative_to(root).as_posix() for p in sorted(out.iterdir()) if p.is_file()},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    notes = [
        "# GSE70138–GSE320583 同药转录响应比较",
        "",
        f"运行：`{args.run_id}`。仅用官方处理矩阵和响应盲样本标签，CPU-only，新增 GPU 账 0 小时。",
        "",
        "## 数据与身份",
        "",
        f"GSE70138 的化合物按完整 InChIKey 连接 DOX、BTZ 和 Birinapant。对每个精确键，覆盖全部乳腺来源细胞的剂量/时间条件，并逐板寻找同细胞、同板、同时间 DMSO 对照。矩阵为 Level2 GEX；每个处理孔减去所在板的 DMSO 均值。",
        f"该版本对三种药物都只找到一个合格乳腺来源背景：MCF7。Birinapant、BTZ、DOX 的 10 µM×24 h 条件分别有 {source_cell_count.get('birinapant',0)}、{source_cell_count.get('bortezomib',0)}、{source_cell_count.get('doxorubicin',0)} 个来源板重复。由于只有一个独立来源细胞，不做多来源均值或来源数量匹配随机面板；这不阻止保留该单一来源的精准同药、同剂量、同时间比较。",
        f"GSE320583 的 GEO sample-tag 表将 Donor #1/2/3 映射到 {', '.join(model_map[k] for k in sorted(model_map))}。HTO 单细胞标签用预先固定的规则分配：唯一最高标签至少 10 reads，且最高/次高至少 3。高置信度细胞只作模型×处理表达汇总，不作为独立重复。",
        "",
        "## 实测响应迁移",
        "",
        f"主要比较为 GSE70138 MCF7 与三个 GSE320583 类器官模型的 Birinapant 10 µM、24 h 同名药响应。来源药物身份按完整 InChIKey 核验；目标侧只有 GEO 样本标签和官方 SOFT 处理注释，没有完整结构键，按其 Birinapant 单药标签及 response-blind HTO 标签绑定。来源为同板 DMSO 归一差，PDO 侧为按模型汇总的 log2(CPM+0.5) 与同模型 DMSO 的差。跨平台只比较 {len(common_symbols)} 个共同 landmark 基因的排序和方向，不比较绝对幅度。",
        "",
        f"逐模型 Pearson/Spearman、非零基因方向一致率、Hallmark 方向和覆盖、差异最大的基因反例见本目录的 TSV。50 个 Hallmark 集合中，{sum(bool(row['cross_model_direction_consistent']) for row in hallmark_consistency_rows)} 个在三个模型均有相同方向，其余存在模型间方向差异；来源与目标的模型×集合方向对照按去重后的 {len(cross_hallmark_rows)} 行报告。Hallmark 平均值仅作描述，不构成患者级统计检验。",
        "",
        "## 组合处理和结论边界",
        "",
        "BiE（Birinapant+Emricasan）与 H+BiE（HOIPIN-8 预处理后再联合处理）单独比较同模型 DMSO，并保留组合成分及顺序；由于没有成分分离对照，只能说明整段组合暴露响应。",
        "",
        "公开的三个模型标签对应各模型×处理的一份 pooled library；库内细胞数只表示测量覆盖，不能替代独立培养重复。分析未读取药敏数值、PDO 敏感/耐药标签、outer/locked 数据，也没有训练模型、运行 GPU、联系外部人员或执行湿实验。",
        "",
        "## 下一步",
        "",
        "将这组三模型实测响应迁移证据与既有 GSE287429 DOX/BTZ 结果并列综合，保持各来源合同和分母独立；只有新的、可精确绑定的实测 PDO 响应矩阵进入同一阶段结论，不基于当前方向追加候选或调参。",
    ]
    (out / "analysis_result_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    input_manifest = {}
    for name, path in source_paths.items():
        input_manifest[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    input_manifest[temp_gctx.relative_to(root).as_posix()] = {
        "bytes": temp_gctx.stat().st_size,
        "sha256": sha256(temp_gctx),
        "derived_from": source_paths["level2"].relative_to(root).as_posix(),
    }
    source_official_checksums = {
        "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx.gz": "8fbb94560545a779141a2fc857998934f4d9bc421024152f28fdd0d3b136d4e287b120f5876a5b6f4d7f687924f736b15ef92455d9c34e324ec838e3b8add1d6",
        "GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz": "b2df105ebf7a69799b23130d03b7da769bfd6ab83fd69cbb419c91c78ae069ae3e5d474e09d36e3ac4271f3b7fc56b24caaf1ac5ece9554ea7738b15535dfefd",
        "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz": "bb162b68fbe94dde005246ba893ba7a43f3eb1f4930339ff137311fd18e05f52a0c1a0c8c15708da4ccee88762fd5327a51a6a9301179300cf9a792739414ae0",
        "GSE70138_Broad_LINCS_pert_info_2017-03-06.txt.gz": "b0df8aaa5653c581bd69a3dddd6eaa64e6296a70f40f50b1b80a2854abbee17a9d66e874dc00657c045be738df521f2e673115c1aae9f5972e820e88f0032ddc",
        "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz": "8d16dfc3a8fdfba572797f2519f9bc96b687c3666c78d386b53fc997164dd2a8ab5953c0b323795ec7f62de5de4ddbd0285e79f8be48487156479dc8ee0128c9",
        "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz": "b9a79bc8835aa19bb46a824e1458d15ecc6d5ceff16afee1f02e759f495a7c2e8b4c99761e1d00ca8db10fc4789a6ba65958f9aabf3daa557681fd542a5ab89a",
        "GSE70138_Broad_LINCS_cell_info_2017-04-28.txt.gz": "75b5d4cf9b6f3955bfe888ce22ae85134dd9fd053111c0f0d7b02298640269454267b0bdfcb2dc9ab562e0281b332703eb1d94337e57355945ccc3505b609b4a",
    }
    actual_source_checksums = {name: sha512(source_root / name) for name in source_official_checksums}
    if actual_source_checksums != source_official_checksums:
        raise ValueError("Downloaded GSE70138 source asset failed an official SHA-512 value")
    output_paths = [p for p in out.iterdir() if p.is_file()]
    run_manifest = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "started_at_local": started_at,
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_wall_minutes": (time.monotonic() - started_clock) / 60,
        "command": f"conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/analyze_exp050_gse320583_birinapant_transfer.py --run-id {args.run_id} --expression-chunk-rows {args.expression_chunk_rows}",
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "scope": "Response-blind metadata-keyed processed-matrix comparison; no FASTQ, drug-sensitivity labels, PDO response classes, outer/locked data, training or external contact.",
        "gse70138_official_sha512": {name: {"expected": expected, "observed": actual_source_checksums[name], "match": expected == actual_source_checksums[name]} for name, expected in source_official_checksums.items()},
        "input_hashes": input_manifest,
        "code": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
            "reused_helper": "scripts/evaluation/compare_exp050_frozen_functional_transfer.py",
            "reused_helper_sha256": sha256(Path(__file__).resolve().parent / "compare_exp050_frozen_functional_transfer.py"),
        },
        "protocol_snapshot": {"path": "protocol_snapshot.json", "sha256": sha256(out / "protocol_snapshot.json")},
        "outputs": {p.name: {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in output_paths},
        "exit_code": 0,
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["run_manifest"] = (out / "run_manifest.json").relative_to(root).as_posix()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest["outputs"]["summary.json"] = {
        "path": (out / "summary.json").relative_to(root).as_posix(),
        "bytes": (out / "summary.json").stat().st_size,
        "sha256": sha256(out / "summary.json"),
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
