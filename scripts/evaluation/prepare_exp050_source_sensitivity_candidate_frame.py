#!/usr/bin/env python3
"""Prepare the authorized source-sensitivity high/low frame for EXP-050 section 3.6.

Selection uses only train-role DepMap PatientID objects and exact full InChIKeys.
The output is a prospective coverage frame, not a PDO response prediction or a
clinical drug ranking. Chen's audited label list is read from its summary only;
numeric IC50 cells are never accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

try:
    from .analyze_exp050_source_lincs_ctrp_association import read_frozen_cells
except ImportError:
    from analyze_exp050_source_lincs_ctrp_association import read_frozen_cells


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUN_ID = "20261001_source_sensitivity_candidate_frame_v1"
MIN_TRAIN_ORIGINS = 10
N_HIGH = 10
N_LOW = 10
KEY_PATTERN = re.compile(r"\b[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tsv") as stream:
        temp = Path(stream.name)
        frame.to_csv(stream, sep="\t", index=False, na_rep="")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def normalized_name(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split()) if pd.notna(value) else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config.get("source_sensitivity_candidate_frame_v1", {})
    if args.run_id != EXPECTED_RUN_ID or contract.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Run ID does not match the frozen EXP-050 source candidate-frame contract")
    if contract.get("status") != "preregistered_exploratory_pending_execution":
        raise ValueError("Source sensitivity candidate-frame protocol is not pending execution")
    if int(contract.get("gpu_hours", -1)) != 0:
        raise ValueError("Candidate-frame work package must remain CPU-only")

    out_dir = root / contract["outputs"]["directory"]
    if out_dir.exists():
        raise FileExistsError(f"Candidate-frame output already exists: {out_dir}")
    input_paths = {
        "source_pairs": root / contract["inputs"]["source_pairs"],
        "identity_split": root / contract["inputs"]["identity_split"],
        "ctrp_drug_names": root / contract["inputs"]["ctrp_drug_names"],
        "transcript_migration": root / contract["inputs"]["transcript_migration"],
        "chen_drug_label_coverage": root / contract["inputs"]["chen_drug_label_coverage"],
        "config": config_path,
        "identity_helper": Path(__file__).resolve().parents[0] / "analyze_exp050_source_lincs_ctrp_association.py",
        "analysis_script": Path(__file__).resolve(),
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required candidate-frame input(s) missing: {missing}")

    cells = read_frozen_cells(root)
    if cells["split_role"].eq("locked_eval").any():
        raise ValueError("Identity helper unexpectedly returned locked_eval rows")
    source_origins = cells[["PatientID", "split_role"]].drop_duplicates().copy()
    if source_origins["PatientID"].duplicated().any():
        raise ValueError("A source origin has multiple frozen split roles")
    train_origins = sorted(source_origins.loc[source_origins["split_role"].eq("train"), "PatientID"].astype(str))
    if len(train_origins) < MIN_TRAIN_ORIGINS:
        raise ValueError(f"Insufficient frozen train origins for candidate frame: {len(train_origins)}")

    # Filter at the parquet scan before returning any AAC values. Validation and
    # locked-evaluation outcome columns do not enter this candidate selection.
    pair_table = pq.read_table(
        input_paths["source_pairs"],
        columns=["source_origin_id", "inchikey", "dose_uM", "time_h", "aac"],
        filters=[("source_origin_id", "in", train_origins)],
    )
    pairs = pair_table.to_pandas()
    pairs["source_origin_id"] = pairs["source_origin_id"].astype("string")
    pairs["inchikey"] = pairs["inchikey"].astype("string").str.strip().str.upper()
    if not set(pairs["source_origin_id"].astype(str)).issubset(set(train_origins)):
        raise ValueError("Parquet source filter returned a non-train source origin")
    if pairs.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Source response table has duplicate train-origin x exact-drug keys")
    values = pairs[["dose_uM", "time_h", "aac"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Selected train-only source pairs contain non-finite dose/time/AAC values")

    counts = pairs.groupby("inchikey")["source_origin_id"].nunique()
    eligible_keys = sorted(counts.loc[counts.ge(MIN_TRAIN_ORIGINS)].index.astype(str))
    eligible = pairs.loc[pairs["inchikey"].isin(eligible_keys)].copy()
    if len(eligible_keys) < N_HIGH + N_LOW:
        raise ValueError(f"Only {len(eligible_keys)} drugs meet the train-origin gate; cannot form 10+10")
    exposures = eligible.groupby("inchikey")[["dose_uM", "time_h"]].nunique()
    if (exposures > 1).any().any():
        raise ValueError("Frozen source table has multiple LINCS selected dose/time values per exact drug")

    summary = (
        eligible.groupby("inchikey", as_index=False, sort=True)
        .agg(
            n_train_source_origins=("source_origin_id", "nunique"),
            median_aac_train=("aac", "median"),
            mean_aac_train=("aac", "mean"),
            q25_aac_train=("aac", lambda x: x.quantile(0.25)),
            q75_aac_train=("aac", lambda x: x.quantile(0.75)),
            min_aac_train=("aac", "min"),
            max_aac_train=("aac", "max"),
            selected_lincs_dose_uM=("dose_uM", "first"),
            selected_lincs_time_h=("time_h", "first"),
        )
    )

    ctrp = pd.read_csv(
        input_paths["ctrp_drug_names"],
        usecols=["drugid", "cpd_name", "inchikey"],
        dtype="string",
    )
    ctrp["inchikey"] = ctrp["inchikey"].astype("string").str.strip().str.upper()
    ctrp = ctrp.loc[ctrp["inchikey"].isin(set(summary["inchikey"]))].dropna(subset=["inchikey"])
    if ctrp.groupby("inchikey")["drugid"].nunique().gt(1).any():
        raise ValueError("An exact InChIKey maps to multiple CTRPv2 drug IDs")
    names = (
        ctrp.groupby("inchikey")["cpd_name"]
        .agg(lambda values: " | ".join(sorted({str(v).strip() for v in values.dropna() if str(v).strip()})))
        .rename("ctrp_compound_name")
        .reset_index()
    )
    summary = summary.merge(names, on="inchikey", how="left", validate="one_to_one")
    if summary["ctrp_compound_name"].isna().any():
        raise ValueError("A selected exact drug key has no registered CTRPv2 compound name")

    # Only the two fixed GSE287429 transfer keys may label exact measured-response coverage.
    target_key_contract = config["inputs"]["cell_line_response"]["drug_structure_crosswalk"]
    target_keys: dict[str, str] = {}
    for drug in ("DOX", "BTZ"):
        matches = KEY_PATTERN.findall(str(target_key_contract[drug]))
        if len(matches) != 1:
            raise ValueError(f"Expected one full structure key for frozen GSE287429 drug {drug}")
        target_keys[drug] = matches[0]
    target_key_to_name = {key: name for name, key in target_keys.items()}
    migration_rows = pd.read_csv(input_paths["transcript_migration"], sep="\t")
    if len(migration_rows) != 56 or set(migration_rows["target_drug"].astype(str).str.lower()) != {"bortezomib", "doxorubicin"}:
        raise ValueError("Fixed GSE287429 migration evidence no longer matches 56 DOX/BTZ conditions")
    migration_summary = (
        migration_rows.assign(target_time_hours=pd.to_numeric(migration_rows["target_time_hours"], errors="raise"))
        .groupby(["target_drug", "target_time_hours"], as_index=False)
        .agg(
            n_source_conditions=("pearson_r", "size"),
            median_pearson=("pearson_r", "median"),
            median_spearman=("spearman_rho", "median"),
            median_direction_agreement=("sign_agreement", "median"),
        )
    )
    migration_text: dict[str, str] = {}
    for drug_name in ("bortezomib", "doxorubicin"):
        rows = migration_summary.loc[migration_summary["target_drug"].astype(str).str.lower().eq(drug_name)]
        migration_text[drug_name.upper()] = "; ".join(
            f"{int(r.target_time_hours)}h: Pearson {r.median_pearson:.4f}, Spearman {r.median_spearman:.4f}, nonzero sign agreement {r.median_direction_agreement:.4f} (n={int(r.n_source_conditions)} source conditions)"
            for r in rows.itertuples(index=False)
        ) + "; MCF7 source only, target dose unknown; descriptive fixed-condition similarity"

    chen = json.loads(input_paths["chen_drug_label_coverage"].read_text(encoding="utf-8"))
    if chen.get("status") != "COVERAGE_ONLY_COMPLETE":
        raise ValueError("Chen input is not the verified coverage-only summary")
    chen_names = {normalized_name(value) for value in chen["source"]["drug_labels"]}
    summary["normalized_ctrp_name"] = summary["ctrp_compound_name"].map(normalized_name)
    summary["chen_exact_name_label_match"] = summary["normalized_ctrp_name"].isin(chen_names)
    summary["chen_name_match_boundary"] = np.where(
        summary["chen_exact_name_label_match"],
        "exact drug-label name only; no sample crosswalk or numeric IC50 accessed",
        "no exact case-insensitive trimmed drug-label match",
    )
    summary["gse287429_exact_key_drug"] = summary["inchikey"].map(target_key_to_name)
    summary["gse287429_transfer_evidence"] = summary["gse287429_exact_key_drug"].map(
        {"BTZ": migration_text["BORTEZOMIB"], "DOX": migration_text["DOXORUBICIN"]}
    ).fillna("No exact full-key match to the two measured GSE287429 response drugs")

    # Median AAC is the frozen sensitivity direction; complete key breaks ties.
    high = summary.sort_values(["median_aac_train", "inchikey"], ascending=[False, True], kind="mergesort").head(N_HIGH).copy()
    low = summary.sort_values(["median_aac_train", "inchikey"], ascending=[True, True], kind="mergesort").head(N_LOW).copy()
    high["source_sensitivity_tier"] = "higher_train_AAC"
    high["rank_within_tier"] = np.arange(1, N_HIGH + 1)
    low["source_sensitivity_tier"] = "lower_train_AAC"
    low["rank_within_tier"] = np.arange(1, N_LOW + 1)
    chosen = pd.concat([high, low], ignore_index=True)
    if chosen["inchikey"].duplicated().any():
        raise ValueError("High and low train-AAC extremes overlap; cannot form 20 unique exact drugs")
    chosen["selection_rule"] = "median AAC among train-role PatientID source origins only; higher AAC means more sensitive; ties by ascending full InChIKey"
    chosen["pdo_prediction_status"] = "not predicted; independent PDO response measurement required"
    chosen["clinical_interpretation"] = "none; source-cell-line sensitivity frame only"
    chosen = chosen.sort_values(["source_sensitivity_tier", "rank_within_tier"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "RUNNING_CPU_ONLY",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": "WSL2 Conda drugscreening-gpu",
        "python": platform.python_version(),
        "protocol_source": "configs/experiments/EXP-050.yaml:source_sensitivity_candidate_frame_v1",
        "supervision": "gpt-6.1-sol/medium directed source-side 10-high/10-low candidate-frame completion within the already approved EXP-050 section 3.6 scope",
        "resource_use": {"gpu_hours": 0.0, "cpu_only": True},
        "outcome_filtering": {
            "source_split_role": "train only",
            "n_train_origins": int(len(train_origins)),
            "validation_outcomes_returned_from_parquet": 0,
            "locked_eval_outcomes_returned_from_parquet": 0,
            "chen_numeric_ic50_values_read": 0,
        },
        "candidate_rule": "Top and bottom ten by median train-source AAC among exact-key drugs with at least ten train-role PatientID origins; ties by ascending full InChIKey.",
        "forbidden_selection_inputs_used": {
            "prior_per_drug_rho": False,
            "Hallmark_or_transcript_features": False,
            "validation_or_locked_eval_AAC": False,
            "Chen_IC50_values": False,
            "name_only_transfer_join": False,
        },
        "inputs": {},
        "outputs": {},
    }
    for label, path in input_paths.items():
        manifest["inputs"][label] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "read_scope": "coverage/identity metadata only" if label == "chen_drug_label_coverage" else "frozen protocol input",
        }
    atomic_json(out_dir / "run_manifest.json", manifest)

    try:
        candidate_path = out_dir / "candidate_panel_source_reference.tsv"
        summary_path = out_dir / "candidate_coverage_summary.json"
        atomic_tsv(candidate_path, chosen)
        target_match_counts = chosen["gse287429_exact_key_drug"].value_counts(dropna=True).to_dict()
        chen_hits = chosen.loc[chosen["chen_exact_name_label_match"], "ctrp_compound_name"].astype(str).tolist()
        candidates = {
            "run_id": args.run_id,
            "status": "COMPLETED_EXPLORATORY_SOURCE_CANDIDATE_FRAME",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_frame": {
                "n_train_source_origins": int(len(train_origins)),
                "eligible_exact_key_drugs_n_ge_10_train_origins": int(len(summary)),
                "candidate_drugs": int(len(chosen)),
                "higher_train_aac_candidates": int(chosen["source_sensitivity_tier"].eq("higher_train_AAC").sum()),
                "lower_train_aac_candidates": int(chosen["source_sensitivity_tier"].eq("lower_train_AAC").sum()),
                "train_aac_higher_direction": "Higher AAC indicates greater source-cell-line sensitivity in the registered CTRPv2 endpoint.",
                "higher_tier_train_median_aac_range": [float(high["median_aac_train"].min()), float(high["median_aac_train"].max())],
                "lower_tier_train_median_aac_range": [float(low["median_aac_train"].min()), float(low["median_aac_train"].max())],
                "source_origin_counts_in_candidate_rows": {
                    "minimum": int(chosen["n_train_source_origins"].min()),
                    "maximum": int(chosen["n_train_source_origins"].max()),
                },
            },
            "transcription_response_coverage": {
                "exact_key_candidates_matching_GSE287429_DOX_or_BTZ": int(chosen["gse287429_exact_key_drug"].notna().sum()),
                "exact_key_matches_by_drug": {str(key): int(value) for key, value in target_match_counts.items()},
                "same_drug_evidence": "The source-cell-line evidence is only the fixed GSE287429 MCF7 condition-level comparison; target dose is unreported and source 6/24-hour exposures are not matched to target 23/47 hours.",
                "candidate_selection_used_transcript_response": False,
            },
            "chen_2021_label_coverage": {
                "candidate_names_with_exact_case_insensitive_trimmed_label": int(len(chen_hits)),
                "matching_names": chen_hits,
                "numeric_ic50_values_read": 0,
                "sample_crosswalk": "none established; drug-label overlap alone is not paired PDO response evidence",
            },
            "interpretation_boundary": [
                "This is a source-cell-line sensitivity high/low frame for prospective experimental planning; it is not a validated PDO panel or a drug efficacy ranking.",
                "Only train-role source origins selected the candidates; validation and locked-evaluation AAC outcomes were excluded before the source-pair scan returned values.",
                "Transcript-response evidence for the measured GSE287429 drugs is sparse and exposure-mismatched; it did not influence candidate selection.",
                "Exact drug-name labels in Chen 2021 do not create a sample crosswalk; numeric IC50 values were not read.",
                "Independent PDO drug-response measurements and paired treated/control RNA are required before testing transfer or drug-effect hypotheses.",
            ],
        }
        atomic_json(summary_path, candidates)
        manifest["status"] = "COMPLETED_EXPLORATORY_SOURCE_CANDIDATE_FRAME"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["candidate_counts"] = candidates["source_frame"]
        manifest["outputs"] = {
            candidate_path.name: {"bytes": candidate_path.stat().st_size, "sha256": sha256_file(candidate_path)},
            summary_path.name: {"bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        }
        atomic_json(out_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "FAILED_NO_ACCEPTED_RESULT"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_json(out_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
