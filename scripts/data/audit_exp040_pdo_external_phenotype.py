"""Audit structure and patient coverage for the EXP-040 external phenotype test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import inchi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid registered drug SMILES")
    return inchi.MolToInchiKey(molecule)


def audit(pdo_drugs_path: Path, pdo_pairs_path: Path, pdo_samples_path: Path,
          pdo_patient_pairs_path: Path, pdo_input_manifest_path: Path,
          lincs_drugs_path: Path, panel_path: Path) -> dict:
    target = pd.read_parquet(pdo_drugs_path)
    source = pd.read_parquet(lincs_drugs_path)
    panel = pd.read_parquet(panel_path)
    target_pairs = pd.read_parquet(pdo_pairs_path, columns=["sample_id", "drug_id"])
    target_samples = pd.read_parquet(pdo_samples_path, columns=["sample_id", "patient_id"])
    patient_pairs = pd.read_parquet(pdo_patient_pairs_path, columns=["patient_id", "drug_id"])
    pdo_input_manifest = json.loads(pdo_input_manifest_path.read_text(encoding="utf-8"))
    if target["drug_id"].duplicated().any() or source["pert_id"].duplicated().any():
        raise ValueError("PDO drug IDs or LINCS perturbation IDs are not unique")
    if target_samples["sample_id"].duplicated().any() or target_samples["patient_id"].isna().any():
        raise ValueError("PDO sample/patient identities are not complete and unique")
    if target_pairs.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("PDO response contract contains duplicate sample/drug keys")
    if patient_pairs.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("patient-level PDO response keys are not unique")
    if pdo_input_manifest.get("status") != "PHARMAFORMER_TARGET_INPUTS_MATERIALIZED_PENDING_INDEPENDENT_VERIFY":
        raise ValueError("patient-level response table is not from the frozen EXP-036 input contract")
    if pdo_input_manifest.get("files_sha256", {}).get("patient_pairs.parquet") != sha256(pdo_patient_pairs_path):
        raise ValueError("patient-level response table checksum differs from its input manifest")

    target["canonical_key"] = target["canonical_isomeric_smiles"].map(structure_key)
    source["canonical_key"] = source["canonical_smiles"].map(structure_key)
    if target["canonical_key"].duplicated().any():
        raise ValueError("PDO drug panel contains multiple IDs for one canonical structure")
    mapping = target[["drug_id", "drug_name", "canonical_key"]].merge(
        source[["pert_id", "canonical_key", "inchi_key"]],
        on="canonical_key", how="inner", validate="one_to_many")
    if mapping.empty:
        raise ValueError("no PDO drug has an exact canonical structure match in LINCS")
    drug_mappings = []
    for row in target.sort_values("drug_id").itertuples(index=False):
        source_rows = source.loc[source["canonical_key"].eq(row.canonical_key)]
        if source_rows.empty:
            continue
        drug_mappings.append({
            "pdo_drug_id": row.drug_id,
            "pdo_drug_name": row.drug_name,
            "canonical_inchikey_recomputed": row.canonical_key,
            "lincs_pert_ids": sorted(source_rows["pert_id"].tolist()),
            "lincs_stored_inchikey_by_pert_id": {
                str(item.pert_id): str(item.inchi_key)
                for item in source_rows[["pert_id", "inchi_key"]].itertuples(index=False)},
            "lincs_stored_inchikey_matches_recomputed": bool(
                source_rows["inchi_key"].eq(source_rows["canonical_key"]).all()),
        })

    eligible = panel.loc[panel["panel_status"].eq("ELIGIBLE")].copy()
    panel_matches = eligible.merge(mapping[["drug_id", "pert_id"]], on="pert_id",
                                   how="inner", validate="many_to_many")
    panel_counts = (panel_matches.groupby("drug_id", sort=True)
                    .agg(n_lincs_pert_ids=("pert_id", "nunique"),
                         n_source_cells=("cell_id", "nunique"),
                         n_source_conditions=("condition_id", "nunique"))
                    .reset_index()
                    .merge(target[["drug_id", "drug_name"]], on="drug_id",
                           how="left", validate="one_to_one"))

    patient_labels = target_pairs.merge(target_samples, on="sample_id", how="left",
                                        validate="many_to_one")
    if patient_labels["patient_id"].isna().any():
        raise ValueError("a PDO response sample has no registered patient identity")
    matched_ids = set(mapping["drug_id"])
    all_patients = pd.Index(target_samples["patient_id"].unique(), name="patient_id")
    if set(patient_pairs["patient_id"]) != set(all_patients):
        raise ValueError("patient-level response table differs from the admitted patient axis")
    raw_patient_counts = (patient_labels.loc[patient_labels["drug_id"].isin(matched_ids)]
                          .groupby("patient_id")["drug_id"].nunique()
                          .reindex(all_patients, fill_value=0))
    patient_counts = (patient_pairs.loc[patient_pairs["drug_id"].isin(matched_ids)]
                      .groupby("patient_id")["drug_id"].nunique()
                      .reindex(all_patients, fill_value=0))
    if not raw_patient_counts.equals(patient_counts):
        raise ValueError("patient-level response aggregation changes matched drug coverage")
    eligible_patients = int(patient_counts.ge(5).sum())

    stored_key_mismatches = source.loc[
        source["inchi_key"].fillna("").ne(source["canonical_key"]),
        ["pert_id", "inchi_key", "canonical_key"]]
    if panel_counts["drug_id"].nunique() != len(mapping["drug_id"].unique()):
        missing = sorted(set(mapping["drug_id"]) - set(panel_counts["drug_id"]))
    else:
        missing = []
    if missing:
        raise ValueError(f"structure-matched PDO drugs absent from the frozen source panel: {missing}")
    if eligible_patients != len(all_patients) or patient_counts.min() < 5:
        raise ValueError("not every admitted PDO patient has five evaluable matched drug labels")

    files = (pdo_drugs_path, pdo_pairs_path, pdo_samples_path,
             pdo_patient_pairs_path, pdo_input_manifest_path,
             lincs_drugs_path, panel_path)
    return {
        "experiment_id": "EXP-040",
        "status": "PDO_EXTERNAL_PHENOTYPE_DATA_GATE_PASS",
        "scope": "structure-key and response-coverage audit only; response values are not read",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": sha256(Path(__file__).resolve()),
        "matching_rule": "recompute InChIKey from registered canonical SMILES with the same RDKit version on both datasets",
        "rdkit_version": rdBase.rdkitVersion,
        "n_pdo_drugs": int(target["drug_id"].nunique()),
        "n_exact_structure_matched_pdo_drugs": int(mapping["drug_id"].nunique()),
        "n_matched_lincs_pert_ids": int(mapping["pert_id"].nunique()),
        "unmatched_pdo_drugs": target.loc[~target["drug_id"].isin(matched_ids),
                                           ["drug_id", "drug_name", "canonical_key"]].to_dict("records"),
        "drug_structure_mappings": drug_mappings,
        "n_admitted_pdo_patients": int(len(all_patients)),
        "n_patients_with_at_least_five_matched_drug_labels": eligible_patients,
        "matched_drugs_per_patient_min_max": [int(patient_counts.min()), int(patient_counts.max())],
        "patient_level_response_aggregation_verified_against_sample_keys": True,
        "patient_level_input_manifest_sha256": sha256(pdo_input_manifest_path),
        "n_frozen_source_cells_with_eligible_matched_panel": int(eligible["cell_id"].nunique()),
        "matched_source_panel_coverage": panel_counts.to_dict("records"),
        "stored_lincs_inchikey_disagreement_count": int(len(stored_key_mismatches)),
        "stored_lincs_inchikey_disagreements_for_matched_drugs": int(
            mapping[["pert_id", "inchi_key", "canonical_key"]].drop_duplicates()
            .assign(disagrees=lambda frame: frame["inchi_key"].ne(frame["canonical_key"]))["disagrees"].sum()),
        "stored_lincs_inchikey_disagreements": stored_key_mismatches.to_dict("records"),
        "sha256": {str(path): sha256(path) for path in files},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdo-drugs", type=Path, required=True)
    parser.add_argument("--pdo-pairs", type=Path, required=True)
    parser.add_argument("--pdo-samples", type=Path, required=True)
    parser.add_argument("--pdo-patient-pairs", type=Path, required=True)
    parser.add_argument("--pdo-input-manifest", type=Path, required=True)
    parser.add_argument("--lincs-drugs", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    result = audit(args.pdo_drugs, args.pdo_pairs, args.pdo_samples,
                   args.pdo_patient_pairs, args.pdo_input_manifest,
                   args.lincs_drugs, args.panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "experiment_id", "status", "n_exact_structure_matched_pdo_drugs",
        "n_matched_lincs_pert_ids", "n_admitted_pdo_patients",
        "matched_drugs_per_patient_min_max", "stored_lincs_inchikey_disagreement_count")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
