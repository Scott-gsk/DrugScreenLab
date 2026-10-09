#!/usr/bin/env python3
"""Build an exact-key, TRAIN-only breast/basal-annotation sensitivity frame.

The frozen EXP-050 source_pairs projection is read only after a response-blind
identity join and official Model.csv lineage/subtype annotation. No PDO numeric
response values or locked-evaluation labels are accessed.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_breast_subtype_panel_v5"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
V1 = ROOT / "artifacts/experiments/EXP-050/20261006_breast_subtype_panel_v1"
SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from analyze_exp050_source_lincs_ctrp_association import read_frozen_cells  # noqa: E402

PAIRS = ROOT / "artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3/patient_level_pairs.parquet"
SPLIT = ROOT / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"
MODEL = ROOT / "data/raw/depmap/24q2/Model.csv"
DRUGS = ROOT / "data/processed/exp013/ctrp_v2/drugs.csv"
OLD = ROOT / "artifacts/experiments/EXP-050/20261001_source_sensitivity_candidate_frame_v1/candidate_panel_source_reference.tsv"
RESP = ROOT / "artifacts/experiments/EXP-050/20260930_keyed_response_v7/EXP050_drug_identity_crosswalk.tsv"
KEY_RE = re.compile(r"^[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace("", pd.NA)


def write_tsv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUT / name, sep="\t", index=False, na_rep="")


def classify_consistency(calls: list[str] | pd.Series, n_origins: int) -> str:
    counts = pd.Series(list(calls), dtype="string").value_counts().to_dict()
    high = int(counts.get("high", 0))
    low = int(counts.get("low", 0))
    if n_origins < 2:
        return "single_source"
    if high >= 2 and low == 0:
        return "concordant_high"
    if low >= 2 and high == 0:
        return "concordant_low"
    if high > 0 and low > 0:
        return "discordant"
    return "indeterminate_neutral_or_mixed"


def select_diverse_extremes(group: pd.DataFrame, call: str, ascending: bool) -> pd.DataFrame:
    ranked = group.loc[group["cross_origin_consistency"].eq(call)].sort_values(
        ["median_aac", "inchikey"], ascending=[ascending, True], kind="mergesort"
    )
    if ranked.empty:
        return ranked.copy()
    chosen_indices: list[int] = []
    used_mechanisms: set[str] = set()
    # First pass: one per exact, non-conflicting target annotation. Unknown is
    # one explicit group; repeated/unknown classes are filled by AAC rank next.
    for index, row in ranked.iterrows():
        mechanism = str(row["mechanism_group"])
        if mechanism not in used_mechanisms:
            chosen_indices.append(index)
            used_mechanisms.add(mechanism)
        if len(chosen_indices) == 10:
            break
    for index in ranked.index:
        if len(chosen_indices) == 10:
            break
        if index not in chosen_indices:
            chosen_indices.append(index)
    selected = ranked.loc[chosen_indices].copy()
    selected["candidate_tier"] = call
    selected["candidate_rank_in_tier"] = np.arange(1, len(selected) + 1)
    return selected


def main() -> None:
    global OUT
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    started = datetime.now(timezone.utc)

    protocol_path = V1 / "protocol_snapshot.json"
    receipt_path = V1 / "protocol_frozen_receipt.json"
    if not protocol_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("v1 frozen protocol snapshot/receipt is missing")
    protocol_hash = sha(protocol_path)
    freeze_receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if str(freeze_receipt.get("protocol_snapshot_sha256", "")).casefold() != protocol_hash.casefold():
        raise ValueError("Frozen protocol receipt does not match the snapshot hash")
    if freeze_receipt.get("status") != "FROZEN_BEFORE_OUTCOME_READ":
        raise ValueError("Frozen protocol receipt is not outcome-blind")
    shutil.copyfile(protocol_path, OUT / "protocol_snapshot.json")
    shutil.copyfile(receipt_path, OUT / "protocol_frozen_receipt.json")

    # Hand checks for the predeclared agreement rule; these synthetic calls are
    # implementation checks only and are not scientific observations.
    assert classify_consistency(["high", "high", "neutral"], 3) == "concordant_high"
    assert classify_consistency(["low", "low", "neutral"], 3) == "concordant_low"
    assert classify_consistency(["high", "low", "neutral"], 3) == "discordant"
    assert classify_consistency(["high"], 1) == "single_source"

    required_inputs = [PAIRS, SPLIT, MODEL, DRUGS, OLD, RESP]
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required registered inputs missing: {missing}")

    # Resolve the frozen identity split, then drop validation objects before
    # pairing the cohort to the official, complete DepMap Model.csv identity.
    cells = read_frozen_cells(ROOT)
    train_cells = cells.loc[cells["split_role"].eq("train")].copy()
    if train_cells.empty or train_cells["PatientID"].isna().any():
        raise ValueError("Frozen identity join returned no complete TRAIN source origins")
    model_columns = [
        "ModelID", "PatientID", "CellLineName", "OncotreeLineage",
        "OncotreePrimaryDisease", "OncotreeSubtype", "OncotreeCode",
        "LegacyMolecularSubtype", "LegacySubSubtype", "PatientMolecularSubtype",
    ]
    model = pd.read_csv(MODEL, usecols=model_columns, dtype="string")
    model = model.dropna(subset=["ModelID", "PatientID"]).drop_duplicates()
    if model["ModelID"].duplicated().any():
        raise ValueError("Official Model.csv ModelID is not unique")
    objects = train_cells.merge(
        model, on=["ModelID", "PatientID"], how="left", validate="many_to_one", indicator=True
    )
    join_failures = int(objects["_merge"].ne("both").sum())
    if join_failures:
        raise ValueError(f"{join_failures} frozen TRAIN identities failed exact Model.csv join")
    objects = objects.drop(columns="_merge")
    lineage = objects["OncotreeLineage"].fillna("").str.casefold()
    primary = objects["OncotreePrimaryDisease"].fillna("").str.casefold()
    objects["official_breast_lineage"] = lineage.str.contains("breast", regex=False) | primary.str.contains("breast", regex=False)
    legacy = objects["LegacyMolecularSubtype"].fillna("").str.strip().str.casefold()
    molecular = (
        objects["OncotreeSubtype"].fillna("") + " " + objects["OncotreeCode"].fillna("") + " "
        + objects["LegacyMolecularSubtype"].fillna("") + " " + objects["LegacySubSubtype"].fillna("")
        + " " + objects["PatientMolecularSubtype"].fillna("")
    ).str.casefold()
    objects["explicit_basal_like_or_tnbc"] = molecular.str.contains(
        r"basal-like|basal like|triple negative|triple-negative|tnbc", regex=True
    )
    objects["legacy_basal_b"] = legacy.eq("basal_b")
    objects["ER_status"] = "unknown_not_in_model_annotation"
    objects["HER2_status"] = "unknown_not_directly_measured_in_model_annotation"
    objects["PR_status"] = "unknown_not_in_model_annotation"

    # Collapse cell-line derivatives to the independent frozen PatientID origin.
    def joined_unique(series: pd.Series) -> str:
        return ";".join(sorted(set(series.dropna().astype(str))))

    def one_complete_label(series: pd.Series, expected_n: int, allowed: set[str]) -> bool:
        nonempty = series.dropna().astype(str).str.strip()
        return len(nonempty) == expected_n and len(set(nonempty.str.casefold())) == 1 and next(iter(set(nonempty.str.casefold())), "") in allowed

    rows = []
    for patient, group in objects.groupby("PatientID", sort=True):
        n_models = int(group["ModelID"].nunique())
        legacy_values = set(group["LegacyMolecularSubtype"].dropna().astype(str).str.strip().str.casefold())
        basal_b = one_complete_label(group["LegacyMolecularSubtype"], n_models, {"basal_b"})
        explicit_basal = bool(group["explicit_basal_like_or_tnbc"].all())
        breast_flags = group["official_breast_lineage"].dropna()
        breast = bool(len(breast_flags) == n_models and breast_flags.all())
        rows.append({
            "source_origin_id": str(patient),
            "n_exact_model_ids": n_models,
            "model_ids": joined_unique(group["ModelID"]),
            "cell_line_names": joined_unique(group["CellLineName"]),
            "official_lineages": joined_unique(group["OncotreeLineage"]),
            "official_primary_diseases": joined_unique(group["OncotreePrimaryDisease"]),
            "official_subtypes": joined_unique(group["OncotreeSubtype"]),
            "oncotree_codes": joined_unique(group["OncotreeCode"]),
            "legacy_molecular_subtypes": joined_unique(group["LegacyMolecularSubtype"]),
            "legacy_subsubtypes": joined_unique(group["LegacySubSubtype"]),
            "patient_molecular_subtypes": joined_unique(group["PatientMolecularSubtype"]),
            "official_breast_lineage": breast,
            "legacy_basal_b_exact_annotation": basal_b,
            "explicit_basal_like_or_tnbc": explicit_basal,
            "subtype_annotation_complete": bool(len(legacy_values) == 1 and all(group["LegacyMolecularSubtype"].notna())),
            "ER_status": "unknown_not_in_model_annotation",
            "HER2_status": "unknown_not_directly_measured_in_model_annotation",
            "PR_status": "unknown_not_in_model_annotation",
        })
    origin_meta = pd.DataFrame(rows)
    breast_origins = set(origin_meta.loc[origin_meta["official_breast_lineage"], "source_origin_id"])
    basal_origins = set(origin_meta.loc[
        origin_meta["official_breast_lineage"]
        & (origin_meta["legacy_basal_b_exact_annotation"] | origin_meta["explicit_basal_like_or_tnbc"]),
        "source_origin_id",
    ])
    if not breast_origins:
        raise ValueError("No breast-lineage object in frozen TRAIN identities")
    write_tsv("train_object_identity_subtype.tsv", objects)
    write_tsv("train_source_origin_subtype.tsv", origin_meta)

    # TRAIN-only AAC projection for exact breast origins. No validation or
    # locked-evaluation source key or endpoint column is returned by this scan.
    source = pq.read_table(
        PAIRS,
        columns=["source_origin_id", "inchikey", "aac"],
        filters=[("source_origin_id", "in", sorted(breast_origins))],
    ).to_pandas()
    source["source_origin_id"] = source["source_origin_id"].astype("string")
    source["inchikey"] = normalize_key(source["inchikey"])
    source["aac"] = pd.to_numeric(source["aac"], errors="coerce")
    if not set(source["source_origin_id"].dropna().astype(str)).issubset(breast_origins):
        raise ValueError("TRAIN source projection returned an identity outside official breast lineage")
    if not source["inchikey"].map(lambda value: bool(KEY_RE.fullmatch(str(value)))).all():
        raise ValueError("TRAIN source projection contains a missing/malformed complete InChIKey")
    if source.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Duplicate TRAIN source-origin x full InChIKey")
    if not np.isfinite(source["aac"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite TRAIN AAC value in exact breast projection")
    origin_medians = source.groupby("source_origin_id")["aac"].median()
    source["train_origin_median_aac"] = source["source_origin_id"].map(origin_medians)
    source["relative_call"] = np.select(
        [source["aac"].gt(source["train_origin_median_aac"]), source["aac"].lt(source["train_origin_median_aac"])],
        ["high", "low"], default="neutral",
    )
    source = source.merge(origin_meta, on="source_origin_id", how="left", validate="many_to_one")
    source["stratum"] = np.where(
        source["source_origin_id"].isin(basal_origins), "basal_like_annotation_basal_B_legacy", "breast_all"
    )

    drug_columns = ["drugid", "cpd_name", "inchikey", "gene_symbol_of_protein_target", "target_or_activity_of_compound"]
    drugs = pd.read_csv(DRUGS, usecols=drug_columns, dtype="string")
    drugs["inchikey"] = normalize_key(drugs["inchikey"])
    drugs = drugs.dropna(subset=["inchikey"])
    for column in ["gene_symbol_of_protein_target", "target_or_activity_of_compound", "cpd_name", "drugid"]:
        drugs[column] = drugs[column].astype("string").str.strip()

    annotation_rows = []
    for key, group in drugs.groupby("inchikey", sort=True):
        target_values = sorted(set(group["gene_symbol_of_protein_target"].dropna().loc[lambda value: value.ne("")].str.casefold()))
        target_original = sorted(set(group.loc[group["gene_symbol_of_protein_target"].notna() & group["gene_symbol_of_protein_target"].ne(""), "gene_symbol_of_protein_target"].astype(str)))
        activity_rows = group["target_or_activity_of_compound"].astype("string").str.strip()
        activity_values = sorted(set(activity_rows.dropna().loc[lambda value: value.ne("")].str.casefold()))
        activities = sorted(set(activity_rows.dropna().loc[lambda value: value.ne("")].astype(str)))
        all_rows_have_activity = bool(activity_rows.notna().all() and activity_rows.ne("").all())
        all_rows_have_target = bool(group["gene_symbol_of_protein_target"].notna().all() and group["gene_symbol_of_protein_target"].str.strip().ne("").all())
        target_conflict = len(target_values) != 1 or not all_rows_have_target
        activity_conflict = len(activity_values) > 1 or not all_rows_have_activity
        if target_conflict or activity_conflict:
            reasons = []
            if target_conflict:
                reasons.append("target_empty_or_conflicting")
            if activity_conflict:
                reasons.append("activity_empty_or_conflicting")
            mechanism_group = "unknown_" + "__and__".join(reasons)
            mechanism_status = "unknown_" + "__and__".join(reasons)
        else:
            mechanism_group = "target:" + target_values[0] + "|activity:" + activity_values[0]
            mechanism_status = "unique_exact_key_target_and_activity_annotation"
        annotation_rows.append({
            "inchikey": str(key),
            "drug_ids": ";".join(sorted(set(group["drugid"].dropna().astype(str)))),
            "drug_names": ";".join(sorted(set(group["cpd_name"].dropna().astype(str)))),
            "gene_symbol_of_protein_target": ";".join(target_original),
            "target_or_activity_of_compound": ";".join(activities),
            "activity_annotation_status": "unknown_empty_or_conflicting" if activity_conflict else "unique_exact_key_activity_annotation",
            "mechanism_group": mechanism_group,
            "mechanism_annotation_status": mechanism_status,
            "activity_annotation_conflict": activity_conflict,
            "n_ctrp_annotation_rows": int(len(group)),
        })
    annotations = pd.DataFrame(annotation_rows)
    annotation_names = annotations[["inchikey", "drug_ids", "drug_names", "gene_symbol_of_protein_target", "target_or_activity_of_compound", "activity_annotation_status", "mechanism_group", "mechanism_annotation_status", "activity_annotation_conflict", "n_ctrp_annotation_rows"]]

    old = pd.read_csv(OLD, sep="\t", dtype="string")
    old_key_col = next((column for column in old.columns if "inchikey" in column.casefold()), None)
    if old_key_col is None:
        raise ValueError("Frozen general-source candidate table has no full InChIKey column")
    old_keys = set(normalize_key(old[old_key_col]).dropna().astype(str))
    response = pd.read_csv(RESP, sep="\t", dtype="string")
    response_rows = response.loc[response["exact_full_key_match"].astype("string").str.casefold().eq("true")].copy()
    response_key_col = "PubChem_full_InChIKey"
    if response_key_col not in response_rows:
        raise ValueError("Frozen PDO-response identity crosswalk has no PubChem full-key field")
    response_rows["response_full_key"] = normalize_key(response_rows[response_key_col])
    transcript_datasets: dict[str, list[str]] = {}
    for _, row in response_rows.dropna(subset=["response_full_key"]).iterrows():
        transcript_datasets.setdefault(str(row["response_full_key"]), []).append("GSE287429_exact_key_transcript_response")
    transcript_keys = set(transcript_datasets)

    expected_by_stratum = {
        "breast_all": sorted(breast_origins),
        "basal_like_annotation_basal_B_legacy": sorted(basal_origins),
    }
    summaries: list[dict] = []
    for stratum, source_origins in expected_by_stratum.items():
        frame = source.loc[source["source_origin_id"].isin(source_origins)].copy()
        for key, group in frame.groupby("inchikey", sort=True):
            calls = group["relative_call"].astype(str).tolist()
            n = int(group["source_origin_id"].nunique())
            missing_ids = sorted(set(source_origins) - set(group["source_origin_id"].astype(str)))
            values = group["aac"].astype(float)
            summaries.append({
                "stratum": stratum,
                "inchikey": str(key),
                "n_expected_independent_train_origins": len(source_origins),
                "n_independent_train_origins": n,
                "missing_source_origin_ids": ";".join(missing_ids),
                "n_missing_source_origins": len(missing_ids),
                "observed_source_origin_ids": ";".join(sorted(set(group["source_origin_id"].astype(str)))),
                "observed_cell_line_names": ";".join(sorted(set(group["cell_line_names"].dropna().astype(str)))),
                "median_aac": float(values.median()),
                "mean_aac": float(values.mean()),
                "q25_aac": float(values.quantile(0.25)),
                "q75_aac": float(values.quantile(0.75)),
                "min_aac": float(values.min()),
                "max_aac": float(values.max()),
                "n_relatively_high": int(sum(call == "high" for call in calls)),
                "n_relatively_low": int(sum(call == "low" for call in calls)),
                "n_neutral": int(sum(call == "neutral" for call in calls)),
                "cross_origin_consistency": classify_consistency(calls, n),
                "measured_transcript_exact_key_match": str(key) in transcript_keys,
                "measured_transcript_datasets": ";".join(sorted(set(transcript_datasets.get(str(key), [])))),
                "overlap_frozen_general_20": str(key) in old_keys,
            })
    summary = pd.DataFrame(summaries)
    summary = summary.merge(annotation_names, on="inchikey", how="left", validate="many_to_one")
    summary["mechanism_group"] = summary["mechanism_group"].fillna("unknown_no_exact_key_annotation")
    summary["candidate_tier"] = "not_selected"
    summary["candidate_rank_in_tier"] = pd.NA
    summary["candidate_exclusion_reason"] = "not_a_concordant_high_or_low_extreme"
    summary.loc[summary["cross_origin_consistency"].eq("single_source"), "candidate_exclusion_reason"] = "only_one_independent_source"
    summary.loc[summary["cross_origin_consistency"].eq("discordant"), "candidate_exclusion_reason"] = "cross_source_direction_disagreement"
    summary.loc[summary["cross_origin_consistency"].eq("indeterminate_neutral_or_mixed"), "candidate_exclusion_reason"] = "fewer_than_two_non_neutral_sources_in_same_direction"
    for stratum in summary["stratum"].drop_duplicates():
        subset = summary.loc[summary["stratum"].eq(stratum)]
        for call, ascending in [("concordant_high", False), ("concordant_low", True)]:
            selected = select_diverse_extremes(subset, call, ascending)
            if not selected.empty:
                summary.loc[selected.index, "candidate_tier"] = call
                summary.loc[selected.index, "candidate_rank_in_tier"] = selected["candidate_rank_in_tier"]
                summary.loc[selected.index, "candidate_exclusion_reason"] = ""
    write_tsv("breast_subtype_candidate_table.tsv", summary.sort_values(
        ["stratum", "candidate_tier", "candidate_rank_in_tier", "inchikey"], na_position="last"
    ))
    write_tsv("breast_subtype_all_drug_summaries.tsv", summary.sort_values(
        ["stratum", "median_aac", "inchikey"], ascending=[True, False, True]
    ))
    source_cols = [
        "source_origin_id", "model_ids", "cell_line_names", "official_lineages", "official_subtypes",
        "legacy_molecular_subtypes", "stratum", "inchikey", "aac", "train_origin_median_aac", "relative_call",
    ]
    write_tsv("train_breast_source_drug_key_evidence.tsv", source[source_cols].sort_values(["stratum", "inchikey", "source_origin_id"]))

    availability = []
    for stratum, origin_ids in expected_by_stratum.items():
        subset = source.loc[source["source_origin_id"].isin(origin_ids)]
        availability.append({
            "stratum": stratum,
            "n_expected_train_origins": len(origin_ids),
            "train_origin_ids": ";".join(origin_ids),
            "n_origin_drug_rows": len(subset),
            "n_exact_drug_keys": int(subset["inchikey"].nunique()),
            "scope_or_gap": "official breast lineage; no subtype inference" if stratum == "breast_all" else "exact legacy basal_B subtype on Model.csv; not a verified TNBC/PAM50/ER/HER2/PR label",
        })
    write_tsv("stratum_source_availability.tsv", pd.DataFrame(availability))

    # Three anchors are separate. DOX/BTZ use the frozen PDO response crosswalk;
    # Birinapant has no exact target full key in the registered local crosswalk.
    anchor_keys = {
        "Doxorubicin": next((str(value) for value in response_rows.loc[response_rows["drug"].str.casefold().eq("doxorubicin"), "response_full_key"].dropna().unique()), None),
        "Bortezomib": next((str(value) for value in response_rows.loc[response_rows["drug"].str.casefold().eq("bortezomib"), "response_full_key"].dropna().unique()), None),
        "Birinapant": None,
    }
    anchors = []
    for anchor, key in anchor_keys.items():
        for stratum, origin_ids in expected_by_stratum.items():
            selection = source.loc[source["source_origin_id"].isin(origin_ids) & source["inchikey"].eq(key)] if key else source.iloc[0:0]
            calls = selection["relative_call"].astype(str).tolist()
            counts = pd.Series(calls, dtype="string").value_counts().to_dict()
            n = int(selection["source_origin_id"].nunique())
            anchors.append({
                "anchor": anchor,
                "identity_source": "EXP050_drug_identity_crosswalk_exact_full_key" if key else "no_exact_full_key_for_target_dataset_in_registered_crosswalk",
                "inchikey": key,
                "stratum": stratum,
                "n_independent_train_origins": n,
                "source_aac_values_by_origin": ";".join(f"{row.source_origin_id}:{row.aac:g}" for row in selection.itertuples()),
                "median_aac": float(selection["aac"].median()) if n else None,
                "relative_good_sensitivity_support_n": int(counts.get("high", 0)),
                "relative_poor_sensitivity_support_n": int(counts.get("low", 0)),
                "good_end_supported_by_at_least_2_origins": int(counts.get("high", 0)) >= 2,
                "poor_end_supported_by_at_least_2_origins": int(counts.get("low", 0)) >= 2,
                "exact_measured_pdo_transcript_key_match": bool(key and key in transcript_keys),
                "interpretation": "Relative source AAC calls only; no absolute sensitivity threshold and no PDO drug-response value.",
            })
    write_tsv("anchor_Dox_BTZ_Birinapant_train_support.tsv", pd.DataFrame(anchors))

    # Compact prospective PDO intake design. Actual dose/time/patient count stay
    # pending the experimental team and a patient-level power calculation.
    design = [
        {"topic":"public_existing_response_data","current_asset":"GSE287429 DOX/BTZ treated RNA at 23/47 h; GSE320583 Birinapant 10 uM/24 h RNA labels","available_evidence":"Transcript observations only; GSE287429 dose/patient identity unresolved; GSE320583 full compound key and numeric PDO sensitivity are absent from this task scope.","next_requirement":"Do not infer dose or efficacy from these RNA records; acquire same-origin paired PDO dose-response and RNA under a frozen design."},
        {"topic":"drug_dose","current_asset":"None fixed for PDOs","available_evidence":"The LINCS dose/time exposures describe source cell lines and do not define PDO conditions.","next_requirement":"Experimental team selects per-drug log-spaced concentration range after solubility/viability feasibility; predefine primary dose-response summary before discovery labels."},
        {"topic":"transcript_collection_time","current_asset":"No validated PDO time for the newly selected panel","available_evidence":"Existing public 23/47 h and 24 h examples are descriptive and not transferable dose-time prescriptions.","next_requirement":"Team fixes an early RNA collection time in a separate feasibility pilot before validation outcomes are opened; use identical time-matched vehicle."},
        {"topic":"paired_vehicle","current_asset":"Prospective requirement","available_evidence":"Every drug x dose x time needs same independent PDO origin, passage, medium, matrix, culture batch, and matched solvent control.","next_requirement":"Key samples by patient/origin, PDO line, drug full InChIKey, dose, time, passage, batch, treatment/vehicle, and replicate; retain raw curves and RNA counts."},
        {"topic":"independent_origins","current_asset":"Current source evidence has 4 breast TRAIN origins, including 2 LegacyMolecularSubtype=basal_B origins","available_evidence":"Cell lines are not PDO patients and cannot set PDO sample size.","next_requirement":"Use patient/origin as the statistical unit; clones, passages, wells, and cells remain nested. Determine sample size from pilot variance and a prespecified target effect."},
        {"topic":"discovery_validation","current_asset":"No PDO response cohort evaluated in this package","available_evidence":"No PDO numeric values were read.","next_requirement":"Assign whole patient origins to discovery/development and independent validation before response or treated RNA review. Keep every derivative of one patient in one cohort; freeze predictions before validation labels."},
    ]
    write_tsv("pdo_intake_design.tsv", pd.DataFrame(design))

    selected = summary[summary["candidate_tier"].ne("not_selected")]
    selected_unique_keys = selected.drop_duplicates("inchikey")
    selected_unique_mechanisms = selected_unique_keys["mechanism_annotation_status"]
    selected_unique_transcript_match = selected_unique_keys["measured_transcript_exact_key_match"]
    counts = {
        "frozen_train_source_origins_total": int(train_cells["PatientID"].nunique()),
        "train_identity_model_rows": int(len(objects)),
        "exact_model_csv_join_failures": join_failures,
        "official_breast_train_origins": int(len(breast_origins)),
        "legacy_basal_B_or_explicit_basal_like_train_origins": int(len(basal_origins)),
        "breast_source_pair_rows": int(len(source)),
        "breast_unique_exact_drug_keys": int(source["inchikey"].nunique()),
        "basal_source_pair_rows": int(source["source_origin_id"].isin(basal_origins).sum()),
        "basal_unique_exact_drug_keys": int(source.loc[source["source_origin_id"].isin(basal_origins), "inchikey"].nunique()),
        "candidate_rows_breast_all": int(selected["stratum"].eq("breast_all").sum()),
        "candidate_rows_basal_metadata": int(selected["stratum"].eq("basal_like_annotation_basal_B_legacy").sum()),
        "candidate_high_breast": int((selected["stratum"].eq("breast_all") & selected["candidate_tier"].eq("concordant_high")).sum()),
        "candidate_low_breast": int((selected["stratum"].eq("breast_all") & selected["candidate_tier"].eq("concordant_low")).sum()),
        "candidate_high_basal": int((selected["stratum"].eq("basal_like_annotation_basal_B_legacy") & selected["candidate_tier"].eq("concordant_high")).sum()),
        "candidate_low_basal": int((selected["stratum"].eq("basal_like_annotation_basal_B_legacy") & selected["candidate_tier"].eq("concordant_low")).sum()),
        "single_source_drug_keys_breast": int(summary.loc[summary["stratum"].eq("breast_all"), "cross_origin_consistency"].eq("single_source").sum()),
        "discordant_drug_keys_breast": int(summary.loc[summary["stratum"].eq("breast_all"), "cross_origin_consistency"].eq("discordant").sum()),
        "single_source_drug_keys_basal": int(summary.loc[summary["stratum"].eq("basal_like_annotation_basal_B_legacy"), "cross_origin_consistency"].eq("single_source").sum()),
        "discordant_drug_keys_basal": int(summary.loc[summary["stratum"].eq("basal_like_annotation_basal_B_legacy"), "cross_origin_consistency"].eq("discordant").sum()),
        "old_frozen_20_keys_with_any_breast_train_aac": int(len(old_keys & set(source["inchikey"].astype(str)))),
        "old_frozen_20_denominator": int(len(old_keys)),
        "candidate_stratum_rows_with_exact_pdo_transcript_response_match": int(selected["measured_transcript_exact_key_match"].sum()),
        "candidate_stratum_rows_without_exact_pdo_transcript_response_match": int((~selected["measured_transcript_exact_key_match"]).sum()),
        "candidate_unique_keys_with_exact_pdo_transcript_response_match": int(selected_unique_transcript_match.sum()),
        "candidate_unique_keys_without_exact_pdo_transcript_response_match": int((~selected_unique_transcript_match).sum()),
        "candidate_selected_stratum_rows": int(len(selected)),
        "candidate_unique_exact_keys": int(selected_unique_keys["inchikey"].nunique()),
        "candidate_unique_keys_with_known_target_and_activity_mechanism": int(
            selected_unique_mechanisms.eq("unique_exact_key_target_and_activity_annotation").sum()
        ),
        "candidate_unique_keys_mechanism_unknown": int(
            selected_unique_mechanisms.ne("unique_exact_key_target_and_activity_annotation").sum()
        ),
        "er_her2_pr_status": "unknown; registered Model.csv fields do not provide directly measured ER/PR and HER2 status for these records",
    }
    completed = datetime.now(timezone.utc)
    outputs = [
        "train_object_identity_subtype.tsv", "train_source_origin_subtype.tsv",
        "train_breast_source_drug_key_evidence.tsv", "stratum_source_availability.tsv",
        "breast_subtype_candidate_table.tsv", "breast_subtype_all_drug_summaries.tsv",
        "anchor_Dox_BTZ_Birinapant_train_support.tsv", "pdo_intake_design.tsv",
        "protocol_snapshot.json", "protocol_frozen_receipt.json",
    ]
    run_manifest = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "TRAIN_BREAST_SUBTYPE_CANDIDATE_FRAME_COMPLETED",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "protocol_snapshot_sha256": protocol_hash,
        "runner_sha256": sha(SCRIPT_PATH),
        "cpu_process_seconds": time.process_time() - cpu0,
        "wall_seconds": time.perf_counter() - t0,
        "cumulative_confirmed_runner_cpu_seconds_v1_v2_verifier_v3_v4_v5": 0.5440072 + 1.4487485 + 0.0246062 + 1.722376 + 1.6542714 + time.process_time() - cpu0,
        "prior_v2_whole_shell_command_cpu_seconds_inclusive": 7.51,
        "gpu_hours": 0,
        "response_access": "Only the frozen source_pairs aac projection for exact breast-lineage TRAIN PatientID origins was read; validation and locked_eval values were not projected.",
        "pdo_numeric_response_access": 0,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in required_inputs + [protocol_path, receipt_path, SCRIPT_PATH]},
        "counts": counts,
        "outputs_sha256": {name: sha(OUT / name) for name in outputs},
        "v1_supersession": {
            "superseded_run_id": "20261006_breast_subtype_panel_v1",
            "reason": "v1 runner omitted available exact-key target annotations, implemented the frozen non-neutral consensus rule too strictly, used a name-resolved DOX anchor rather than the frozen PDO response full key, and did not include the explicit LegacyMolecularSubtype=basal_B category. v1 files are preserved and not used for selection.",
        },
        "v2_supersession": {
            "superseded_run_id": "20261006_breast_subtype_panel_v2",
            "reason": "v2 remains immutable; v3 corrects mechanism group to require non-empty, non-conflicting target AND activity annotation, and reports mechanism annotation counts over unique complete InChIKeys rather than duplicated stratum rows. Candidate key selection is rerun under the same frozen protocol.",
        },
        "v3_supersession": {
            "superseded_run_id": "20261006_breast_subtype_panel_v3",
            "reason": "v3 corrected mechanism classification/counts, but labeled transcript coverage as candidate keys while counting repeated breast/basal_B stratum rows. v4 separates stratum-row and unique-InChIKey counts; scientific selection rule and frozen source data are unchanged.",
        },
        "v4_supersession": {
            "superseded_run_id": "20261006_breast_subtype_panel_v4",
            "reason": "v4 corrected unique-key transcript coverage counts and passed readback, but its cumulative CPU ledger omitted the confirmed v3 run. v5 retains the same frozen data selection and records all confirmed runner process CPU use to date.",
        },
        "interpretation": "Descriptive source cell-line AAC reference only. This is not PDO efficacy, clinical sensitivity, or a validated breast/PDO ranking. Basal_B remains its official legacy subtype label and does not imply TNBC/PAM50 or ER/HER2/PR status.",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_json = {
        "run_id": RUN_ID,
        "status": run_manifest["status"],
        "counts": counts,
        "selection_rule": "At least two independent source origins; each origin's AAC is compared with its TRAIN breast-origin median; concordant high/low requires at least two non-neutral origins all on the same side. Rank the top/bottom 10 by across-origin median AAC within each stratum, greedily prefer distinct non-conflicting exact-key target annotations, then fill by AAC rank. Report discordant and single-origin keys separately.",
        "subtype_boundary": "Model.csv official lineage and LegacyMolecularSubtype=basal_B identify two basal_B origins. TNBC, PAM50, ER, HER2 protein, and PR statuses are not established by this annotation and are marked unknown.",
        "mechanism_diversity": "CTRPv2 gene_symbol_of_protein_target and target_or_activity_of_compound are joined by unique complete InChIKey. A mechanism is known only when every exact-key CTRPv2 annotation row has a non-empty, non-conflicting target and activity value; either missing or conflicting field places the key in an explicit unknown mechanism group. Selection prefers distinct known target-and-activity groups, then fills by frozen AAC rank.",
        "pdo_interpretation": "No PDO sensitivity values or post-treatment migration scores were accessed. All AAC values are authorized TRAIN cell-line source references only.",
        "dataset_sources": [
            "EXP-007 frozen response-blind identity split and exact CTRPv2/DepMap identity mapping",
            "EXP-050 20261001_lincs_ctrp_transfer_association_v3/patient_level_pairs.parquet, filtered to TRAIN breast origins and projected to source_origin_id, inchikey, aac",
            "DepMap 24Q2 Model.csv official lineage/subtype annotation",
            "CTRPv2 drugs.csv exact-key drug display/target annotation",
            "EXP-050 frozen general-source 20-drug table and exact PDO drug-identity crosswalk, used as descriptive key references only",
        ],
        "outputs_sha256": run_manifest["outputs_sha256"],
        "runner_sha256": run_manifest["runner_sha256"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # The manifest also binds the final summary and its own runner after the
    # initial atomic result write, so verification can recompute these hashes.
    run_manifest["outputs_sha256"]["summary.json"] = sha(OUT / "summary.json")
    run_manifest["outputs_sha256"]["run_manifest.json"] = "self_hash_omitted"
    (OUT / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "status": run_manifest["status"], "counts": counts,
                      "cpu_process_seconds": run_manifest["cpu_process_seconds"], "gpu_hours": 0,
                      "output_dir": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
