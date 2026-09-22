"""Read-only identity and assay-quality audit for the public Kryeziu mCRC PDO v3 bundle.

This program never rewrites raw or processed inputs.  It creates a new audit
record that preserves source values and makes each admission decision explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from drug_screen.data.mcrc_source import rebuild_mcrc, sha256


SOURCE_FILES = ("Data S1.xlsx", "Data S2.xlsx", "Data S3.xlsx", "Data S4.xlsx", "Data S5.xlsx")
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13293968/"


def _native(value: Any) -> Any:
    """Convert pandas/numpy values to deterministic JSON primitives."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    view = frame if columns is None else frame.loc[:, columns]
    return [{str(k): _native(v) for k, v in row.items()} for row in view.to_dict(orient="records")]


def _same_frame(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> bool:
    """Equality that preserves list-valued chemistry provenance in parquet."""
    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result.select_dtypes(include="object"):
            result[column] = result[column].map(
                lambda value: tuple(value.tolist()) if isinstance(value, np.ndarray)
                else tuple(value) if isinstance(value, list) else value
            )
        return result.sort_values(keys).reset_index(drop=True)
    return normalized(left).equals(normalized(right))


def _stats(series: pd.Series) -> dict[str, Any]:
    value = pd.to_numeric(series, errors="coerce")
    finite = value[np.isfinite(value)]
    if finite.empty:
        return {"n": int(len(value)), "finite": 0, "min": None, "max": None, "median": None}
    return {
        "n": int(len(value)), "finite": int(len(finite)), "min": float(finite.min()),
        "max": float(finite.max()), "median": float(finite.median()),
    }


def _outlier_count_by_drug(pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Flag IQR outliers for review only; values are retained and never altered."""
    annotated = pairs.copy()
    annotated["iqr_outlier"] = False
    summaries: list[dict[str, Any]] = []
    for name, index in annotated.groupby("drug_name", sort=True).groups.items():
        values = annotated.loc[index, "response"].astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        flags = (values < lower) | (values > upper) if iqr > 0 else pd.Series(False, index=values.index)
        annotated.loc[index, "iqr_outlier"] = flags
        summaries.append({
            "drug_name": str(name), "n_pairs": int(len(values)), "q1": float(q1), "q3": float(q3),
            "iqr": iqr, "review_lower": lower, "review_upper": upper, "iqr_outlier_count": int(flags.sum()),
        })
    return annotated, summaries


def _drug_key(name: object) -> str:
    """Match source labels while retaining the deposited labels in all outputs."""
    return re.sub(r"[._\-\s]+", "", str(name).casefold())


def _raw_dose_audit(
    raw_s4: pd.DataFrame, pair_samples: set[str], pair_drugs: set[str], retained_dss_keys: set[tuple[str, str]]
) -> dict[str, Any]:
    required = {
        "sample_id", "sample_id_drums", "run_id", "assay_no", "library_id", "compound_name",
        "compound_drums", "compound_type", "concentration", "concentration_unit", "plate", "viability",
    }
    missing = sorted(required.difference(raw_s4.columns))
    if missing:
        raise ValueError(f"Data S4 missing required columns: {missing}")

    raw = raw_s4.copy()
    for column in ("sample_id", "sample_id_drums", "compound_name", "compound_drums", "compound_type", "library_id"):
        raw[column] = raw[column].astype("string").str.strip()
    raw["concentration"] = pd.to_numeric(raw["concentration"], errors="coerce")
    raw["viability"] = pd.to_numeric(raw["viability"], errors="coerce")

    # Data S5 uses the public PDO IDs and drug labels.  *_drums fields instead
    # hold run-specific/internal identifiers and must not be used for joining.
    raw["drug_key"] = raw.compound_name.map(_drug_key)
    pair_key_to_name = {_drug_key(name): name for name in pair_drugs}
    raw["processed_drug_name"] = raw.drug_key.map(pair_key_to_name)
    source_pair_rows = raw[raw.sample_id.isin(pair_samples) & raw.processed_drug_name.notna()].copy()
    controls = raw[raw.compound_type.str.contains("control", case=False, na=False)].copy()
    # Multiple source runs can be the parallel technical measurements.  Do not
    # collapse them: only count their occurrence for QC.
    dose_key = ["sample_id", "library_id", "processed_drug_name", "concentration"]
    dose_groups = source_pair_rows.groupby(dose_key, dropna=False, sort=False)
    replicate_n = dose_groups.size()
    unique_doses = source_pair_rows.groupby(["sample_id", "processed_drug_name"], sort=True).concentration.nunique(dropna=True)
    source_pair_keys = set(zip(source_pair_rows.sample_id.astype(str), source_pair_rows.processed_drug_name.astype(str)))

    expected = {(sample, drug) for sample in pair_samples for drug in pair_drugs}
    # Only pairs with an author DSS value require a raw curve, not the full Cartesian product.
    missing_raw_for_dss = sorted(retained_dss_keys.difference(source_pair_keys))
    extra_raw_without_dss = sorted(source_pair_keys.difference(retained_dss_keys))
    missing_viability_curve_keys = set(
        zip(
            source_pair_rows.loc[source_pair_rows.viability.isna(), "sample_id"].astype(str),
            source_pair_rows.loc[source_pair_rows.viability.isna(), "processed_drug_name"].astype(str),
        )
    )
    return {
        "raw_rows": int(len(raw)),
        "raw_rows_for_retained_dss_pairs": int(len(source_pair_rows)),
        "raw_sample_id_count": int(raw.sample_id.nunique()),
        "raw_run_specific_sample_id_drums_count": int(raw.sample_id_drums.nunique()),
        "raw_compound_name_count": int(raw.compound_name.nunique()),
        "raw_internal_compound_drums_count": int(raw.compound_drums.nunique()),
        "source_pair_keys_observed": int(len(source_pair_keys)),
        "retained_dss_pair_keys": int(len(retained_dss_keys)),
        "retained_dss_pairs_missing_raw_curve": [f"{sample}|{drug}" for sample, drug in missing_raw_for_dss],
        "raw_curve_keys_without_retained_dss": [f"{sample}|{drug}" for sample, drug in extra_raw_without_dss],
        "cartesian_pair_count_not_used_as_requirement": int(len(expected)),
        "missing_concentration_rows_retained_pairs": int(source_pair_rows.concentration.isna().sum()),
        "nonpositive_concentration_rows_retained_pairs": int((source_pair_rows.concentration <= 0).sum()),
        "concentration_units": sorted(str(x) for x in source_pair_rows.concentration_unit.dropna().unique()),
        "viability": {
            **_stats(source_pair_rows.viability),
            "missing": int(source_pair_rows.viability.isna().sum()),
            "negative": int((source_pair_rows.viability < 0).sum()),
            "above_one": int((source_pair_rows.viability > 1).sum()),
        },
        "technical_replicate_observed_per_dose": {
            "min": int(replicate_n.min()) if not replicate_n.empty else None,
            "max": int(replicate_n.max()) if not replicate_n.empty else None,
            "groups_below_two": int((replicate_n < 2).sum()),
            "groups_exactly_two": int((replicate_n == 2).sum()),
            "groups_above_two": int((replicate_n > 2).sum()),
        },
        "dose_count_per_sample_drug": {
            "min": int(unique_doses.min()) if not unique_doses.empty else None,
            "max": int(unique_doses.max()) if not unique_doses.empty else None,
            "pairs_with_fewer_than_two_doses": int((unique_doses < 2).sum()),
            "distribution": {str(k): int(v) for k, v in unique_doses.value_counts().sort_index().items()},
        },
        "controls": {
            "compound_types": sorted(str(x) for x in controls.compound_type.dropna().unique()),
            "compound_names": sorted(str(x) for x in controls.compound_name.dropna().unique()),
            "rows": int(len(controls)),
            "author_protocol": "paper reports 100 uM benzethonium chloride positive and 0.1% DMSO negative controls; Data S4 labels them control_positive and control_negative in compound_type",
        },
        "raw_duplicate_rows_exact_all_columns": int(raw.duplicated().sum()),
        "join_fields": {"Data_S5": ["sample_id", "drug name after _lib suffix removal"], "Data_S4": ["sample_id", "compound_name"], "normalization": "casefold plus removal of '.', '_' and '-' solely for label comparison; source values retained"},
        "normalized_label_collisions_in_S5": sorted(name for key, name in pair_key_to_name.items() if sum(_drug_key(x) == key for x in pair_drugs) > 1),
        "unmatched_retained_drug_names_in_raw": sorted(name for name in pair_drugs if _drug_key(name) not in set(raw.drug_key)),
        "unmatched_retained_sample_ids_in_raw": sorted(pair_samples.difference(set(raw.sample_id.dropna().astype(str)))),
        "retained_curve_keys_with_missing_viability": sorted({f"{sample}|{drug}" for sample, drug in missing_viability_curve_keys}),
        "interpretation": [
            "Data S4 stores normalized viability, concentration, run/assay/library and plate fields, so raw curve re-fitting is technically possible.",
            "This audit does not re-fit curves or replace author DSS; it only checks whether the deposited records are structurally adequate for that later, separately specified analysis.",
        ],
    }


def audit(root: Path) -> dict[str, Any]:
    raw_dir = root / "raw/mcrc_pdo_2026/v3"
    processed_dir = root / "processed/transfer/mcrc_v3"
    legacy = root / "external/exp010/pubchem/mcrc_drug_structures.json"
    fixed = root / "raw/transfer_public_assets/pubchem/mcrc_isomeric_20260912.json"
    metadata = json.loads((raw_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    expected_hashes = {item["filename"]: item["content_details"]["sha256_hash"] for item in metadata["files"]}
    hashes = {name: sha256(raw_dir / name) for name in SOURCE_FILES}
    hash_pass = all(hashes[name] == expected_hashes[name] for name in SOURCE_FILES)

    s1 = pd.read_excel(raw_dir / "Data S1.xlsx")
    s2 = pd.read_excel(raw_dir / "Data S2.xlsx")
    s3 = pd.read_excel(raw_dir / "Data S3.xlsx")
    s4 = pd.read_excel(raw_dir / "Data S4.xlsx")
    s5 = pd.read_excel(raw_dir / "Data S5.xlsx")
    rebuilt = rebuild_mcrc(s1, s2, s5, legacy, fixed)
    samples = pd.read_parquet(processed_dir / "samples.parquet")
    drugs = pd.read_parquet(processed_dir / "drugs.parquet")
    pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    with np.load(processed_dir / "indexed.npz", allow_pickle=False) as data:
        expression = data["expression"]
        ordered_samples = tuple(data["sample_id"].astype(str))
        ordered_genes = tuple(data["ordered_genes"].astype(str))

    exact = {
        "expression_max_abs_error": float(np.max(np.abs(expression - rebuilt.expression))),
        "sample_order_exact": ordered_samples == rebuilt.sample_ids,
        "gene_order_exact": ordered_genes == rebuilt.gene_ids,
        "sample_rows_exact": _same_frame(samples, rebuilt.samples, ["sample_id"]),
        "drug_rows_exact": _same_frame(drugs, rebuilt.drugs, ["drug_id"]),
        "pair_rows_exact": _same_frame(pairs, rebuilt.pairs, ["sample_id", "drug_id"]),
    }
    pair_samples = set(pairs.sample_id.astype(str))
    pair_drugs = set(pairs.drug_name.astype(str))

    s1_records = s1.loc[:, ["sample_id", "patient", "sample_type"]].copy()
    s1_records = s1_records[s1_records.sample_id.isin(samples.sample_id)]
    s1_pdo = s1_records[s1_records.sample_type.astype(str).eq("PDO")]
    s3_ids = set(s3["PDO_id"].dropna().astype(str))
    identity = samples.loc[:, ["sample_id", "patient_id", "identity_status", "patient_id_provenance", "source_sample_types"]].copy()
    identity["in_data_s3_protein_table"] = identity.sample_id.isin(s3_ids)
    identity["in_raw_dss_table"] = identity.sample_id.isin(set(s5.sample_id.astype(str)))
    identity["in_processed_pairs"] = identity.sample_id.isin(pair_samples)
    identity["source_s1_pdo_rows"] = identity.sample_id.map(s1_pdo.groupby("sample_id").size()).fillna(0).astype(int)
    identity["identity_mapping_verdict"] = np.where(
        identity.identity_status.eq("S1_PDO_patient_verified"),
        "PASS_S1_PDO_patient_column_matches_sample_prefix",
        "CONDITIONAL_prefix_group_only_no_independent_patient_column_in_S1-S5",
    )

    annotated_pairs, drug_outliers = _outlier_count_by_drug(pairs)
    per_drug = annotated_pairs.groupby("drug_name", sort=True).agg(
        dss_pairs=("response", "size"), dss_samples=("sample_id", "nunique"), dss_min=("response", "min"),
        dss_max=("response", "max"), dss_median=("response", "median"), missing_dss=("response", lambda x: int(x.isna().sum())),
        iqr_outlier_count=("iqr_outlier", "sum"),
    ).reset_index()
    per_sample = annotated_pairs.groupby("sample_id", sort=True).agg(
        patient_id=("patient_id", "first") if "patient_id" in annotated_pairs.columns else ("response", "size"),
        dss_drugs=("drug_name", "nunique"), dss_min=("response", "min"), dss_max=("response", "max"),
        dss_median=("response", "median"), iqr_outlier_count=("iqr_outlier", "sum"),
    ).reset_index()
    # Pair table intentionally has no patient_id; join it only for reporting, not calculation.
    per_sample = per_sample.drop(columns=["patient_id"], errors="ignore").merge(samples[["sample_id", "patient_id", "identity_status"]], on="sample_id", how="left")
    pair_duplicate_count = int(pairs.duplicated(["sample_id", "drug_id"]).sum())
    dss_range = _stats(pairs.response)
    retained_dss_keys = set(zip(pairs.sample_id.astype(str), pairs.drug_name.astype(str)))
    dose = _raw_dose_audit(s4, pair_samples, pair_drugs, retained_dss_keys)
    oxaliplatin = {
        "status": "QUARANTINED",
        "is_monotherapy_in_source": True,
        "source_columns": [column for column in s5.columns if str(column).startswith("Oxaliplatin")],
        "raw_curve_rows": int(s4.compound_drums.astype(str).eq("Oxaliplatin").sum()),
        "reason": "the frozen small-molecule standardizer rejects the deposited Pt coordination-complex SMILES; no identity-preserving representation approved for the current model contract",
        "not_a_reason": "not classified as a combination or biologic",
        "effect_on_counts": "excluded before the 48 structure-confirmed single-drug / 3,183-pair processed contract",
    }
    identity_counts = {str(k): int(v) for k, v in identity.identity_status.value_counts().items()}
    assay_pass = (
        dose["missing_concentration_rows_retained_pairs"] == 0
        and dose["nonpositive_concentration_rows_retained_pairs"] == 0
        and dose["retained_dss_pairs_missing_raw_curve"] == []
        and dose["unmatched_retained_drug_names_in_raw"] == []
        and dose["unmatched_retained_sample_ids_in_raw"] == []
        and pair_duplicate_count == 0
        and dss_range["finite"] == len(pairs)
    )
    reconstruction_pass = hash_pass and exact["expression_max_abs_error"] == 0.0 and all(
        value for key, value in exact.items() if key != "expression_max_abs_error"
    )

    return {
        "audit_id": "kryeziu_identity_and_assay_20260916",
        "cohort_id": "mcrc_pdo_v3",
        "scope": "read-only audit; no raw or processed source value was modified, removed, imputed, re-fit, or re-normalized",
        "source": {
            "mendeley_version": "hr94h42xdc.3",
            "license": "CC BY 4.0",
            "raw_file_sha256": hashes,
            "source_hashes_match_mendeley_metadata": hash_pass,
            "article": PAPER_URL,
        },
        "data_contract": {
            "role": "WITHIN_STUDY_PATIENT_GROUPED_PDO_DRUG_RANKING only after G0 identity admission",
            "expression": "Data S2 RNA-seq counts -> frozen log2(CPM+1), 19,421 HGNC genes, 92 PDO sample IDs",
            "response": "Data S5 author-calculated DSS; higher score means greater ex-vivo sensitivity; S4 stores raw normalized viability curves",
            "drug_identity": "48 retained monotherapies have fixed PubChem CID, full InChIKey and canonical isomeric SMILES",
            "sample_context": "metastatic colorectal cancer liver-metastasis PDOs; paper reports 96 h CellTiter-Glo 3D readout",
            "control": "paper protocol: 0.1% DMSO negative and 100 uM benzethonium chloride positive control; Data S4 records DMSO/BzCl and compound_type control_negative/control_positive",
            "split": "patient_id is the mandatory grouping key; no PDO of a held-out patient may appear in training, tuning, auxiliary fitting, feature selection, or threshold selection",
            "forbidden": [
                "clinical treatment-effect or external-validation claim",
                "automatic use of the 32 prefix-only patient groupings before independent mapping is added",
                "Oxaliplatin in current SMILES graph inputs",
                "using Data S5 labels to select an external test or tune a model claimed as external",
            ],
        },
        "reconstruction": {
            "status": "PASS" if reconstruction_pass else "BLOCKED",
            "checks": exact,
            "counts": {
                "expression_samples": int(len(samples)), "response_paired_samples": int(len(pair_samples)),
                "patients": int(samples.patient_id.nunique()), "response_paired_patients": int(samples[samples.sample_id.isin(pair_samples)].patient_id.nunique()),
                "structure_confirmed_single_drugs": int(len(drugs)), "dss_pairs": int(len(pairs)),
                "samples_without_dss": sorted(set(samples.sample_id) - pair_samples),
            },
        },
        "identity_audit": {
            "status": "CONDITIONAL",
            "counts": identity_counts,
            "s1_pdo_rows_for_expression_samples": int(len(s1_pdo)),
            "data_s1_to_s5_stronger_mapping": "Data S1 supplies explicit patient values only for the 60 S1-PDO-verified samples. Data S2-S5 and Data S3 preserve PDO IDs but add no patient column for the other 32, so they do not independently establish those patient groupings.",
            "paper_supplement_assessment": "The paper states that Table S2 lists PDOs with associated patient characteristics, and its key-resources table lists PDO IDs, but neither accessible paper text supplies a machine-readable independent patient-ID crosswalk for the 32 prefix-only expression samples. A verified Table S2 crosswalk or author confirmation is still required.",
            "identity_by_expression_sample": _records(identity.sort_values("sample_id")),
        },
        "assay_quality": {
            "status": "CONDITIONAL_STRUCTURAL_QC_AUTHOR_DSS_COMPLETE_RAW_CURVE_REVIEW_REMAINING",
            "author_protocol_evidence": {
                "paper_methods": "libraries: lib1 41 drugs at 9 concentrations and lib2 47 drugs at 7 concentrations; two parallel technical replicates; 96 h CTG-3D viability; normalized against positive/negative controls; logistic curve fit then drums DSS",
                "article_url": PAPER_URL,
            },
            "dss": {
                "response_name": "DSS", "higher_is_sensitive": True, "rows": int(len(pairs)),
                "missing": int(pairs.response.isna().sum()), "nonfinite": int((~np.isfinite(pairs.response)).sum()),
                "duplicate_sample_drug_rows": pair_duplicate_count, "observed_range": dss_range,
                "outlier_rule": "per-drug Tukey 1.5*IQR review flag; flags are retained source values, not exclusions",
                "per_drug": _records(per_drug.sort_values("drug_name")),
                "per_sample": _records(per_sample.sort_values("sample_id")),
                "outlier_thresholds": drug_outliers,
            },
            "raw_dose_response": dose,
            "unresolved": [
                "No independent curve refit, fit diagnostics, or repeat-noise estimate was performed here; that would require a separately frozen curve-fitting contract.",
                "Twelve raw viability points across 12 retained sample-drug curves are missing. Normalization also yields some values below 0 or above 1; these source values are retained as review flags and do not make the already-complete author DSS table missing.",
                "Raw Data S4 lacks an explicit elapsed-time column; 96 h comes from the paper method. Its control rows are explicitly labelled by compound_type.",
            ],
        },
        "oxaliplatin": oxaliplatin,
        "decision": {
            "data_status": "DATA_PARTIAL" if reconstruction_pass and assay_pass else "DATA_BLOCKED",
            "admission": "CONDITIONAL" if reconstruction_pass and assay_pass else "BLOCKED",
            "allowed_now": [
                "read-only coverage and provenance analysis",
                "same-study descriptive DSS analysis without treating prefix-only groups as independently verified patients",
            ],
            "blocked_now": [
                "formal patient-grouped model development or evaluation using all 52 groups",
                "any external-test or clinical-efficacy conclusion",
            ],
            "release_conditions": [
                "obtain and checksum a Table S2 or author-supplied PDO-to-patient crosswalk that independently covers the 32 prefix-only samples",
                "approve a separate raw-curve refit/repeat-noise data contract before replacing or qualifying author DSS",
                "add a chemically valid, identity-preserving coordination-complex representation before admitting Oxaliplatin",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/artifacts/resource_acquisition_20260916/kryeziu_audit/kryeziu_identity_assay_audit.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit evidence: {args.output}")
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "data_status": result["decision"]["data_status"], "admission": result["decision"]["admission"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
