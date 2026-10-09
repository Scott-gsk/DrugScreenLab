#!/usr/bin/env python3
"""Add breast-LINCS identity, lineage, exposure, and transcript-evidence coverage to the frozen 20-drug frame."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


RUN_ID = "20261001_candidate_breast_source_coverage_v3"
ROOT_CANDIDATES = Path("artifacts/experiments/EXP-050/20261001_source_sensitivity_candidate_frame_v1/candidate_panel_source_reference.tsv")
ROOT_GSE288_CHEM = Path("artifacts/experiments/EXP-050/20261001_gse288798_compound_eligibility_v1/summary.json")
ROOT_GSE303509_CASES = Path("artifacts/experiments/EXP-050/20261001_gse303509_published_case_crosswalk_v1/summary.json")
ROOT_GSE287_XWALK = Path("artifacts/experiments/EXP-050/20260930_keyed_response_v7/EXP050_drug_identity_crosswalk.tsv")
ROOT_GSE287_RESULT = Path("artifacts/experiments/EXP-050/20261001_frozen_functional_transfer_v2/summary.json")
PERT_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
CELL_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz")
CONDITIONS = Path("data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet")
CONDITIONS_VERIFY = Path("data/processed/lincs/GSE92742/continuous_conditions_v2/verification.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def dose_time_string(rows: pd.DataFrame) -> str:
    keys = rows[["pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]].drop_duplicates()
    parts = [
        f"{r.pert_dose} {r.pert_dose_unit} × {r.pert_time} {r.pert_time_unit}"
        for r in keys.sort_values(["pert_time_unit", "pert_time", "pert_dose_unit", "pert_dose"]).itertuples(index=False)
    ]
    return "; ".join(parts)


def main() -> None:
    root = Path.cwd().resolve()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    candidates = pd.read_csv(root / ROOT_CANDIDATES, sep="\t", dtype=str).fillna("")
    chem = json.loads((root / ROOT_GSE288_CHEM).read_text(encoding="utf-8"))
    published_cases = json.loads((root / ROOT_GSE303509_CASES).read_text(encoding="utf-8"))
    target_keys = {
        str(item["pubchem_record"]["InChIKey"]).upper(): f"GSE288798 {label} (single-organoid, scale gate failed)"
        for label, item in chem["compound_identity_crosswalks"].items()
    }
    measured = pd.read_csv(root / ROOT_GSE287_XWALK, sep="\t", dtype=str).fillna("")
    for row in measured.to_dict(orient="records"):
        if row.get("exact_full_key_match", "").lower() == "true":
            target_keys[str(row["PubChem_full_InChIKey"]).upper()] = f"GSE287429 {row['drug']} measured PDO response"

    perturbagens = pd.read_csv(root / PERT_INFO, sep="\t", dtype=str, low_memory=False).fillna("")
    cells = pd.read_csv(root / CELL_INFO, sep="\t", dtype=str, low_memory=False).fillna("")
    conditions = pd.read_parquet(root / CONDITIONS)
    verification = json.loads((root / CONDITIONS_VERIFY).read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or len(conditions) != verification.get("n_conditions"):
        raise ValueError("LINCS continuous condition table does not satisfy its registered read-back receipt")
    if len(candidates) != 20 or candidates["inchikey"].nunique() != 20:
        raise ValueError("The frozen candidate frame must contain exactly 20 unique full InChIKeys")

    breast = cells.loc[
        (cells["cell_type"].str.lower() == "cell line")
        & (cells["sample_type"].str.lower() == "tumor")
        & (cells["primary_site"].str.lower() == "breast")
    ].copy()
    if len(breast) != 7:
        raise ValueError(f"Expected seven official LINCS breast-tumor cell-line entries, found {len(breast)}")
    breast_ids = set(breast["cell_id"])
    cell_by_id = breast.set_index("cell_id").to_dict(orient="index")
    breast_conditions = conditions.loc[conditions["cell_id"].isin(breast_ids)].copy()
    for col in ["pert_dose", "pert_time"]:
        breast_conditions[col] = breast_conditions[col].astype(str)

    rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict(orient="records"):
        key = str(candidate["inchikey"]).upper()
        perturbagen_matches = perturbagens.loc[perturbagens["inchi_key"].str.upper() == key]
        pert_ids = sorted(perturbagen_matches["pert_id"].dropna().astype(str).unique().tolist())
        matched = breast_conditions.loc[breast_conditions["pert_id"].isin(pert_ids)].copy()
        line_details: list[str] = []
        for cell_id, line_group in matched.groupby("cell_id", sort=True):
            metadata = cell_by_id[str(cell_id)]
            base_id = str(metadata.get("base_cell_id") or cell_id)
            subtype = str(metadata.get("subtype") or "not recorded")
            line_details.append(f"{cell_id} [base={base_id}; GEO/LINCS subtype field={subtype}; n_condition={len(line_group)}]")
            for item in line_group.to_dict(orient="records"):
                exposure_rows.append({
                    "frame_role": "source_sensitivity_candidate",
                    "tier": candidate["source_sensitivity_tier"],
                    "rank_within_tier": candidate["rank_within_tier"],
                    "ctrp_compound_name": candidate["ctrp_compound_name"],
                    "full_inchikey": key,
                    "exact_lincs_pert_ids": ";".join(pert_ids),
                    "cell_id": str(item["cell_id"]),
                    "base_cell_id": base_id,
                    "official_subtype_field": subtype,
                    "condition_id": str(item["condition_id"]),
                    "dose": str(item["pert_dose"]),
                    "dose_unit": str(item["pert_dose_unit"]),
                    "time": str(item["pert_time"]),
                    "time_unit": str(item["pert_time_unit"]),
                    "n_treatment_instances": int(item["n_treatment_instances"]),
                    "n_matched_control_groups": int(item["n_control_groups"]),
                })
        exact_pdo_refs = [label for target_key, label in target_keys.items() if target_key == key]
        if "DOX" in candidate["gse287429_exact_key_drug"] or "BTZ" in candidate["gse287429_exact_key_drug"]:
            exact_pdo_refs.append("GSE287429 same-key measured transcript response")
        if exact_pdo_refs:
            transcript_evidence = "; ".join(sorted(set(exact_pdo_refs)))
        else:
            transcript_evidence = "No exact full-key match to currently registered measured-PDO transcript-response drugs (GSE287429 DOX/BTZ or GSE288798 JQ1/MS177). GSE216540 Fulvestrant remains structurally unresolved; GSE303509 gives four selected same-paper drug-sensitivity cases, not an RNA-response screen."
        if candidate["ctrp_compound_name"].strip().lower() == "veliparib":
            direct_reference = "The published GSE303509 article's label-level VP combination (cisplatin + veliparib) response cases are relevant direct-sensitivity context. This is not an isolated veliparib response, no exact product-key crosswalk to the candidate was made, and no treated RNA or numeric curve values are available."
        else:
            direct_reference = "No exact-key direct-sensitivity case mapped to this candidate in the current reviewed assets."
        rows.append({
            "frame_role": "general_source_sensitivity_candidate",
            "source_sensitivity_tier": candidate["source_sensitivity_tier"],
            "rank_within_tier": int(candidate["rank_within_tier"]),
            "ctrp_compound_name": candidate["ctrp_compound_name"],
            "full_inchikey": key,
            "n_source_train_origins_for_aac_selection": int(candidate["n_train_source_origins"]),
            "median_aac_train_unchanged": float(candidate["median_aac_train"]),
            "aac_selection_order_changed": False,
            "exact_pert_info_full_key_match": bool(len(perturbagen_matches)),
            "exact_lincs_pert_ids": ";".join(pert_ids),
            "breast_line_entries_with_qc_condition": int(matched["cell_id"].nunique()),
            "breast_base_cell_origins_with_qc_condition": int(matched["cell_id"].map(lambda x: cell_by_id[x].get("base_cell_id") or x).nunique()) if len(matched) else 0,
            "official_breast_lines_and_recorded_subtypes": "; ".join(line_details) if line_details else "No exact-key QC-passing condition among the seven official breast tumor cell-line entries.",
            "n_exact_key_qc_passing_breast_exposure_conditions": int(matched["condition_id"].nunique()),
            "n_treatment_instances_across_conditions": int(matched["n_treatment_instances"].sum()) if len(matched) else 0,
            "n_matched_control_groups_across_conditions": int(matched["n_control_groups"].sum()) if len(matched) else 0,
            "all_recorded_breast_dose_time_conditions": dose_time_string(matched) if len(matched) else "none",
            "same_drug_pdo_transcript_key_match": bool(exact_pdo_refs),
            "same_drug_pdo_transcript_function_evidence": transcript_evidence,
            "label_level_direct_sensitivity_literature_reference": direct_reference,
            "functional_evidence_boundary": "A breast LINCS condition means source-cell transcript measurements exist under the registered QC contract; no target PDO match is established unless an exact-key row is explicitly named above. No candidate was selected/re-ranked using these fields.",
            "procurement_verification_required": "Before any future wet-lab planning: supplier and catalog number; exact parent/salt/solvate and stereochemistry against this InChIKey; purity and lot CoA; stock solvent, storage and stability; organoid-medium solubility and vehicle limit; available amount and institutional purchasing/access constraints. No purchasing or supplier contact performed.",
            "pdo_prediction_status": "not predicted; prospective measurement required",
            "clinical_interpretation": "none; source-cell sensitivity frame only",
        })

    # Add independent DOX/BTZ mechanism references, without rank or selection fields.
    reference_specs = [
        ("doxorubicin", "AOJJSUZBOXZQNB-TZSSRYMLSA-N", 18, 36),
        ("bortezomib", "GXJABQQUPOEUTA-RDJZCZTQSA-N", 10, 20),
    ]
    for name, key, conditions_per_target, fixed_comparisons in reference_specs:
        pm = perturbagens.loc[perturbagens["inchi_key"].str.upper() == key]
        pert_ids = sorted(pm["pert_id"].dropna().astype(str).unique().tolist())
        matched = breast_conditions.loc[breast_conditions["pert_id"].isin(pert_ids)]
        target_match = bool(any(x.upper() == key for x in target_keys))
        details = []
        for cell_id, line_group in matched.groupby("cell_id", sort=True):
            md = cell_by_id[str(cell_id)]
            details.append(f"{cell_id} [base={md.get('base_cell_id') or cell_id}; subtype={md.get('subtype')}; conditions={len(line_group)}]")
        rows.append({
            "frame_role": "mechanism_reference_only_not_in_candidate_rank",
            "source_sensitivity_tier": "reference",
            "rank_within_tier": "",
            "ctrp_compound_name": name,
            "full_inchikey": key,
            "n_source_train_origins_for_aac_selection": "",
            "median_aac_train_unchanged": "",
            "aac_selection_order_changed": False,
            "exact_pert_info_full_key_match": bool(len(pm)),
            "exact_lincs_pert_ids": ";".join(pert_ids),
            "breast_line_entries_with_qc_condition": int(matched["cell_id"].nunique()),
            "breast_base_cell_origins_with_qc_condition": int(matched["cell_id"].map(lambda x: cell_by_id[x].get("base_cell_id") or x).nunique()) if len(matched) else 0,
            "official_breast_lines_and_recorded_subtypes": "; ".join(details),
            "n_exact_key_qc_passing_breast_exposure_conditions": int(matched["condition_id"].nunique()),
            "n_treatment_instances_across_conditions": int(matched["n_treatment_instances"].sum()) if len(matched) else 0,
            "n_matched_control_groups_across_conditions": int(matched["n_control_groups"].sum()) if len(matched) else 0,
            "all_recorded_breast_dose_time_conditions": dose_time_string(matched) if len(matched) else "none",
            "same_drug_pdo_transcript_function_evidence": f"GSE287429 exact-key measured response; fixed existing source/target comparison frame has {fixed_comparisons} records. Used only as mechanism/transfer reference; not substituted into the source-AAC candidate ranking.",
            "functional_evidence_boundary": "This row is an explicitly separate reference arm, not one of the 20 candidate ranks.",
            "procurement_verification_required": "Not a candidate procurement instruction.",
            "pdo_prediction_status": "No new PDO prediction; historical observed transcript-response data only.",
            "clinical_interpretation": "none; mechanism/transfer reference only",
        })
        for item in matched.to_dict(orient="records"):
            metadata = cell_by_id[str(item["cell_id"])]
            exposure_rows.append({
                "frame_role": "mechanism_reference_only_not_in_candidate_rank",
                "tier": "reference",
                "rank_within_tier": "",
                "ctrp_compound_name": name,
                "full_inchikey": key,
                "exact_lincs_pert_ids": ";".join(pert_ids),
                "cell_id": str(item["cell_id"]),
                "base_cell_id": str(metadata.get("base_cell_id") or item["cell_id"]),
                "official_subtype_field": str(metadata.get("subtype", "")),
                "condition_id": str(item["condition_id"]),
                "dose": str(item["pert_dose"]),
                "dose_unit": str(item["pert_dose_unit"]),
                "time": str(item["pert_time"]),
                "time_unit": str(item["pert_time_unit"]),
                "n_treatment_instances": int(item["n_treatment_instances"]),
                "n_matched_control_groups": int(item["n_control_groups"]),
            })

    out = root / "artifacts/experiments/EXP-050" / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "candidate_panel_breast_source_coverage.tsv"
    exposure_path = out / "candidate_breast_source_exposure_conditions.tsv"
    summary_path = out / "summary.json"
    manifest_path = out / "run_manifest.json"
    write_tsv(candidate_path, rows)
    write_tsv(exposure_path, exposure_rows)

    candidate_only = [r for r in rows if r["frame_role"] == "general_source_sensitivity_candidate"]
    summary = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "completed_exact_key_breast_source_coverage",
        "source_frame": "Frozen 20-drug AAC high/low frame (10 per tier); frame rank, endpoint and order are unchanged.",
        "candidate_count": len(candidate_only),
        "additional_mechanism_reference_count": 2,
        "official_breast_tumor_cell_line_entries": breast[["cell_id", "base_cell_id", "subtype"]].to_dict(orient="records"),
        "official_line_entry_count": int(len(breast)),
        "source_base_cell_origins_across_official_registry": int(breast["base_cell_id"].replace("", pd.NA).fillna(breast["cell_id"]).nunique()),
        "lincs_conditions_source": "continuous_conditions_v2, exact full perturbagen key match; registered table passes structural/finite-control readback",
        "candidate_exact_full_key_matches_in_pert_info": sum(r["exact_pert_info_full_key_match"] for r in candidate_only),
        "candidate_with_at_least_one_breast_qc_condition": sum(r["n_exact_key_qc_passing_breast_exposure_conditions"] > 0 for r in candidate_only),
        "candidates_with_same_drug_registered_PDO_RNA_response_key": sum(r["same_drug_pdo_transcript_key_match"] for r in candidate_only),
        "breast_subtype_boundary": "The official LINCS cell_info subtype field is broad carcinoma/adenocarcinoma; intrinsic molecular subtype is not provided and is not imputed.",
        "procurement_boundary": "No supplier/catalog lookup, purchase, or supplier contact; requirements are listed for later independent feasibility review.",
        "interpretation": "This table establishes source-cell availability only. It does not establish breast-PDO sensitivity, target-patient efficacy, or a validated breast-cancer panel. Candidate rank was frozen by general-source train AAC and was not changed using breast source coverage or transcript data. DOX/BTZ are separate mechanism references.",
        "gpu_hours": 0,
        "outputs": {
            "candidate_coverage": candidate_path.relative_to(root).as_posix(),
            "exposure_conditions": exposure_path.relative_to(root).as_posix(),
            "summary": summary_path.relative_to(root).as_posix(),
            "manifest": manifest_path.relative_to(root).as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inputs = [root / p for p in [ROOT_CANDIDATES, ROOT_GSE288_CHEM, ROOT_GSE303509_CASES, ROOT_GSE287_XWALK, ROOT_GSE287_RESULT, PERT_INFO, CELL_INFO, CONDITIONS, CONDITIONS_VERIFY]]
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "started_at_local": started,
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/audit_exp050_candidate_breast_source_coverage.py",
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "protocol": "Exact full InChIKey join from the unchanged 20-candidate frame to LINCS pert_info; exact LINCS pert_id and existing QC condition-table lookup; source origins counted by official base_cell_id, not modified derivative count.",
        "inputs": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in inputs},
        "outputs": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in [candidate_path, exposure_path, summary_path]},
        "code": {"path": "scripts/evaluation/audit_exp050_candidate_breast_source_coverage.py", "sha256": sha256_file(Path(__file__).resolve())},
        "exit_code": 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
