"""Response-blind identity, exposure, and fold contract for EXP-066/067."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(value: Any) -> str:
    return str(value).strip().upper()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _true_mask(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def _assign_folds(objects: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    if objects["source_object_id"].duplicated().any():
        raise ValueError("Object load table must have one row per PatientID")
    ordered = objects.copy()
    ordered["tie"] = ordered["source_object_id"].map(
        lambda x: _stable_hash(f"{seed}|fold|{x}")
    )
    ordered = ordered.sort_values(
        ["n_conditions", "tie"], ascending=[False, True], kind="mergesort"
    )
    loads = [0] * n_folds
    counts = [0] * n_folds
    result: dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        fold = min(range(n_folds), key=lambda i: (loads[i], counts[i], i))
        result[str(row.source_object_id)] = fold
        loads[fold] += int(row.n_conditions)
        counts[fold] += 1
    if len(result) < n_folds or any(count == 0 for count in counts):
        raise ValueError("At least one complete PatientID is required in each fixed fold")
    return result


def _object_derangements(
    condition_frame: pd.DataFrame, n_folds: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    object_folds = condition_frame[
        ["source_object_id", "fold_id"]
    ].drop_duplicates()
    if object_folds["source_object_id"].duplicated().any():
        raise ValueError("A PatientID was assigned to multiple folds")
    object_to_fold = dict(
        zip(object_folds["source_object_id"].astype(str), object_folds["fold_id"].astype(int))
    )
    cell_column = "lincs_cell_id" if "lincs_cell_id" in condition_frame else "cell_id"
    if cell_column not in condition_frame:
        raise ValueError("Frozen condition frame lacks its LINCS cell identity")
    model_by_cell = dict(
        zip(condition_frame[cell_column].astype(str), condition_frame["depmap_model_id"].astype(str))
    )
    row_bindings = condition_frame[["depmap_model_id", "rna_row_index"]].drop_duplicates()
    if row_bindings["depmap_model_id"].astype(str).duplicated().any():
        raise ValueError("DepMap model to RNA row mapping is not one-to-one")
    rna_row_by_model = dict(
        zip(row_bindings["depmap_model_id"].astype(str), row_bindings["rna_row_index"].astype(int))
    )
    patient_by_cell = dict(
        zip(condition_frame[cell_column].astype(str), condition_frame["source_object_id"].astype(str))
    )
    cells_by_object: dict[str, list[str]] = {}
    for cell, patient in patient_by_cell.items():
        cells_by_object.setdefault(patient, []).append(cell)
    for patient in cells_by_object:
        cells_by_object[patient] = sorted(set(cells_by_object[patient]))

    rows: list[dict[str, Any]] = []
    matrix = np.full((n_folds, len(condition_frame)), -1, dtype=np.int64)
    donor_matrix = np.empty((n_folds, len(condition_frame)), dtype="U64")
    for fold in range(n_folds):
        for partition, patients in (
            ("train", sorted(p for p, f in object_to_fold.items() if f != fold)),
            ("heldout", sorted(p for p, f in object_to_fold.items() if f == fold)),
        ):
            shuffled = sorted(
                patients,
                key=lambda patient: _stable_hash(f"{seed}|shuffle|{fold}|{partition}|{patient}"),
            )
            if len(shuffled) < 2:
                raise ValueError(f"Cannot create a fold-local object derangement for {partition}")
            donor = {patient: shuffled[(i + 1) % len(shuffled)] for i, patient in enumerate(shuffled)}
            for patient in patients:
                donor_patient = donor[patient]
                target_cells = cells_by_object[patient]
                donor_cells = cells_by_object[donor_patient]
                for cell in target_cells:
                    target_position = target_cells.index(cell)
                    donor_cell = donor_cells[target_position % len(donor_cells)]
                    donor_model = model_by_cell[donor_cell]
                    rows.append(
                        {
                            "fold_id": fold,
                            "partition": partition,
                            "source_object_id": patient,
                            "target_lincs_cell_id": cell,
                            "donor_object_id": donor_patient,
                            "donor_lincs_cell_id": donor_cell,
                            "donor_depmap_model_id": donor_model,
                        }
                    )
        lookup = {
            (r["fold_id"], r["target_lincs_cell_id"]): r["donor_depmap_model_id"]
            for r in rows
            if r["fold_id"] == fold
        }
        for i, condition in enumerate(condition_frame.itertuples(index=False)):
            donor_model = lookup.get((fold, str(getattr(condition, cell_column))))
            if donor_model is None:
                raise ValueError("Missing fold-local shuffled context mapping")
            matrix[fold, i] = int(rna_row_by_model[donor_model])
            donor_matrix[fold, i] = donor_model
    return pd.DataFrame(rows), matrix, donor_matrix


def freeze_metadata(root: Path, config_path: Path, run_id: str) -> Path:
    """Create a response-blind frozen domain. Does not open LINCS/CTRP value arrays."""
    started = time.monotonic()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_ids") != ["EXP-066", "EXP-067"]:
        raise ValueError("Shared response benefit config must explicitly name EXP-066 and EXP-067")
    outdir = root / config["outputs"]["shared_dir"] / run_id
    outdir.mkdir(parents=True, exist_ok=False)

    rel = config["data"]
    paths = {name: root / value for name, value in rel.items() if name.endswith("_path")}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required metadata source {name} not found: {path}")

    conditions_path = paths["lincs_conditions_path"]
    conditions = pd.read_parquet(conditions_path)
    genes = json.loads(paths["lincs_gene_axis_path"].read_text(encoding="utf-8"))
    actual_axis_sha256 = hashlib.sha256(("\n".join(map(str, genes)) + "\n").encode("utf-8")).hexdigest()
    if len(genes) != 978 or len(set(map(str, genes))) != 978 or actual_axis_sha256 != config["data"]["target_axis_sha256"]:
        raise ValueError("Frozen exact978 target axis failed ID/count/hash verification")
    _require_columns(
        conditions,
        {
            "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
            "pert_time", "pert_time_unit", "n_treatment_instances", "n_control_groups",
        },
        "LINCS condition metadata",
    )
    conditions = conditions.copy()
    conditions["condition_id"] = pd.to_numeric(conditions["condition_id"], errors="raise").astype(np.int64)
    if conditions["condition_id"].duplicated().any():
        raise ValueError("LINCS condition_id is not unique")

    treatment_meta = pd.read_parquet(paths["lincs_treatment_instances_path"])
    control_meta = pd.read_parquet(paths["lincs_control_groups_path"])
    _require_columns(
        treatment_meta,
        {"inst_id", "control_group_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "rna_plate", "treatment_finite", "all_controls_finite", "n_matched_controls"},
        "LINCS treatment metadata",
    )
    _require_columns(
        control_meta,
        {"inst_id", "control_group_id", "cell_id", "pert_time", "pert_time_unit", "rna_plate", "control_finite"},
        "LINCS control metadata",
    )
    condition_keys = ["cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
    treatment_meta["_valid_source"] = (
        _true_mask(treatment_meta["treatment_finite"])
        & _true_mask(treatment_meta["all_controls_finite"])
        & pd.to_numeric(treatment_meta["n_matched_controls"], errors="coerce").gt(0)
    )
    treatment_groups = treatment_meta.groupby(condition_keys, as_index=False, dropna=False).agg(
        source_treatment_count=("inst_id", "size"),
        source_treatment_all_valid=("_valid_source", "all"),
        source_treatment_ids=("inst_id", lambda x: ";".join(sorted(set(x.astype(str))))),
        source_control_group_ids=("control_group_id", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
    )
    conditions = conditions.merge(treatment_groups, on=condition_keys, how="inner", validate="one_to_one")
    conditions = conditions.loc[
        conditions["source_treatment_all_valid"]
        & conditions["source_treatment_count"].eq(pd.to_numeric(conditions["n_treatment_instances"], errors="coerce"))
    ].copy()
    if conditions.empty:
        raise ValueError("No condition remains after matched-control finiteness and source-instance checks")

    ctrp_map = pd.read_csv(paths["ctrp_identity_path"], dtype="string")
    _require_columns(ctrp_map, {"cellid", "ModelID", "match_method", "has_depmap_expression"}, "CTRP identity map")
    ctrp_map = ctrp_map.loc[
        ctrp_map["match_method"].eq("exact_unique_cellosaurus_rrid")
        & ctrp_map["has_depmap_expression"].astype("string").str.lower().eq("true"),
        ["cellid", "ModelID"],
    ].dropna().drop_duplicates()
    if ctrp_map["cellid"].duplicated().any() or ctrp_map["ModelID"].duplicated().any():
        raise ValueError("CTRP exact RRID mapping is not one-to-one")

    models = pd.read_csv(paths["depmap_model_path"], usecols=["ModelID", "PatientID"], dtype="string")
    models = models.dropna(subset=["ModelID", "PatientID"]).drop_duplicates()
    if models["ModelID"].duplicated().any():
        raise ValueError("DepMap ModelID does not map uniquely to PatientID")
    depmap_map = json.loads(paths["depmap_exact978_mapping_path"].read_text(encoding="utf-8"))
    if not isinstance(depmap_map, list):
        raise ValueError("DepMap exact978 row mapping must be a list")
    depmap_map = pd.DataFrame(depmap_map)
    _require_columns(depmap_map, {"depmap_id", "exact978_row"}, "DepMap exact978 row mapping")
    depmap_map = depmap_map[["depmap_id", "exact978_row"]].copy()
    depmap_map["exact978_row"] = pd.to_numeric(depmap_map["exact978_row"], errors="raise").astype(int)
    if depmap_map["depmap_id"].duplicated().any() or depmap_map["exact978_row"].duplicated().any():
        raise ValueError("DepMap exact978 row mapping is not one-to-one")

    identity = json.loads(paths["identity_split_path"].read_text(encoding="utf-8"))
    if identity.get("status") != "IDENTITY_SPLIT_FROZEN":
        raise ValueError("EXP-007 identity split is not frozen")
    identity_rows = pd.DataFrame(identity.get("identity_rows", []))
    _require_columns(
        identity_rows,
        {"lincs_cell_id", "depmap_id", "patient_id", "expression_present", "join_status"},
        "EXP-007 identity rows",
    )
    identity_rows = identity_rows.loc[
        identity_rows["expression_present"].astype(bool)
        & identity_rows["join_status"].eq("EXACT_STRIPPED_NAME_WITH_RNASEQ"),
        ["lincs_cell_id", "depmap_id", "patient_id"],
    ].dropna().drop_duplicates()
    if identity_rows["lincs_cell_id"].duplicated().any() or identity_rows["depmap_id"].duplicated().any():
        raise ValueError("EXP-007 exact identity map is not one-to-one")
    split = pd.DataFrame(identity.get("split_assignments", []))
    _require_columns(split, {"depmap_id", "split_role"}, "EXP-007 split assignments")
    split = split[["depmap_id", "split_role"]].dropna().drop_duplicates()
    if split["depmap_id"].duplicated().any() or not split["split_role"].isin(["train", "val", "locked_eval"]).all():
        raise ValueError("EXP-007 roles are not unique/recognized")
    cell_map = identity_rows.merge(split, on="depmap_id", validate="one_to_one")
    cell_map = cell_map.loc[cell_map["split_role"].isin(["train", "val"])].copy()
    cell_map = cell_map.merge(models, left_on="depmap_id", right_on="ModelID", validate="one_to_one")
    if not cell_map["patient_id"].astype(str).eq(cell_map["PatientID"].astype(str)).all():
        raise ValueError("EXP-007 PatientID differs from DepMap Model.csv")
    cell_map = cell_map.merge(depmap_map, left_on="depmap_id", right_on="depmap_id", validate="one_to_one")
    cell_map = cell_map.merge(ctrp_map, left_on="ModelID", right_on="ModelID", how="inner", validate="one_to_one")
    cell_map = cell_map.rename(
        columns={
            "lincs_cell_id": "cell_id",
            "ModelID": "depmap_model_id",
            "PatientID": "source_object_id",
            "patient_id": "exp007_patient_id",
            "split_role": "source_split_role",
            "exact978_row": "rna_row_index",
            "cellid": "ctrp_cell_id",
        }
    )
    cell_map = cell_map[
        ["cell_id", "depmap_model_id", "source_object_id", "source_split_role", "rna_row_index", "ctrp_cell_id"]
    ].drop_duplicates()
    if cell_map["cell_id"].duplicated().any() or cell_map["depmap_model_id"].duplicated().any():
        raise ValueError("Shared exact cell map is not one-to-one")
    if cell_map.empty:
        raise ValueError("No exact non-locked DepMap/LINCS/CTRP identity intersection")

    ctrp_drugs = pd.read_csv(
        paths["ctrp_drugs_path"], usecols=["drugid", "inchikey", "smiles"], dtype="string"
    )
    ctrp_drugs["full_inchikey"] = ctrp_drugs["inchikey"].map(_canonical_id)
    ctrp_drugs = ctrp_drugs.dropna(subset=["drugid", "full_inchikey", "smiles"])
    ctrp_key_counts = ctrp_drugs.groupby("full_inchikey")["drugid"].nunique()
    ctrp_drugs = ctrp_drugs.loc[ctrp_drugs["full_inchikey"].map(ctrp_key_counts).eq(1)].drop_duplicates()
    ctrp_drugs = ctrp_drugs[["drugid", "full_inchikey", "smiles"]].drop_duplicates()
    if ctrp_drugs["drugid"].duplicated().any():
        raise ValueError("CTRP drugid maps to multiple exact InChIKeys")

    pert_info = pd.read_csv(
        paths["lincs_pert_info_path"], sep="\t", compression="gzip",
        usecols=["pert_id", "pert_type", "inchi_key", "canonical_smiles"], dtype="string",
    )
    pert_info = pert_info.loc[pert_info["pert_type"].eq("trt_cp")].copy()
    pert_info["full_inchikey"] = pert_info["inchi_key"].map(_canonical_id)
    pert_info = pert_info.dropna(subset=["pert_id", "full_inchikey", "canonical_smiles"])
    pert_info = pert_info.merge(
        ctrp_drugs[["full_inchikey", "drugid"]].drop_duplicates(),
        on="full_inchikey", how="inner", validate="many_to_many",
    )
    if pert_info.empty:
        raise ValueError("No exact full-InChIKey LINCS/CTRP structure intersection")

    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    structure_rows: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    kept_keys: set[str] = set()
    for key, group in pert_info.groupby("full_inchikey", sort=True):
        smiles_values = sorted(set(group["canonical_smiles"].astype(str)))
        mol = None
        canonical = None
        for smiles in smiles_values:
            candidate = Chem.MolFromSmiles(smiles)
            if candidate is not None and Chem.MolToInchiKey(candidate).upper() == key:
                mol = candidate
                canonical = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
                break
        if mol is None or not key or key == "NAN":
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        drug_row = len(fingerprints)
        fingerprints.append(arr)
        kept_keys.add(key)
        structure_rows.append(
            {
                "full_inchikey": key,
                "canonical_smiles": canonical,
                "drug_feature_row_index": drug_row,
                "n_lincs_pert_ids": int(group["pert_id"].nunique()),
                "ctrp_drugid": ";".join(sorted(set(group["drugid"].astype(str)))),
            }
        )
    if not kept_keys:
        raise ValueError("No LINCS canonical structure passed exact RDKit/InChIKey verification")
    structure_table = pd.DataFrame(structure_rows).drop_duplicates("full_inchikey").sort_values("full_inchikey")
    fingerprints = np.stack(fingerprints)
    # Reindex structure vectors to sorted exact-key order.
    old_row = dict(zip(structure_table["full_inchikey"], structure_table["drug_feature_row_index"].astype(int)))
    drug_keys = sorted(kept_keys)
    drug_features = np.stack([fingerprints[old_row[key]] for key in drug_keys])
    key_to_drug_row = {key: i for i, key in enumerate(drug_keys)}
    structure_table = structure_table.set_index("full_inchikey").loc[drug_keys].reset_index()
    structure_table["drug_feature_row_index"] = np.arange(len(drug_keys), dtype=int)

    pert_info = pert_info.loc[pert_info["full_inchikey"].isin(kept_keys), ["pert_id", "full_inchikey"]].drop_duplicates()
    candidate = conditions.merge(cell_map, left_on="cell_id", right_on="cell_id", how="inner", validate="many_to_one")
    candidate = candidate.merge(pert_info, on="pert_id", how="inner", validate="many_to_many")
    candidate = candidate.loc[candidate["full_inchikey"].isin(kept_keys)].copy()
    dose_unit = candidate["pert_dose_unit"].astype("string").str.strip().str.lower().str.replace("μ", "u", regex=False).str.replace("µ", "u", regex=False)
    time_unit = candidate["pert_time_unit"].astype("string").str.strip().str.lower()
    candidate["dose_uM"] = pd.to_numeric(candidate["pert_dose"], errors="coerce")
    candidate["time_hours"] = pd.to_numeric(candidate["pert_time"], errors="coerce")
    candidate = candidate.loc[
        dose_unit.eq("um") & time_unit.eq("h")
        & np.isfinite(candidate["dose_uM"]) & np.isfinite(candidate["time_hours"])
        & candidate["dose_uM"].gt(0) & candidate["time_hours"].gt(0)
        & pd.to_numeric(candidate["n_control_groups"], errors="coerce").gt(0)
    ].copy()
    if candidate.empty:
        raise ValueError("No exact, positive-dose/time conditions with matched control metadata")
    candidate["dose_uM"] = candidate["dose_uM"].astype(float)
    candidate["time_hours"] = candidate["time_hours"].astype(float)
    # Choose one exposure pair per exact structure using only non-locked identity/condition metadata.
    exposure = (
        candidate.groupby(["full_inchikey", "dose_uM", "time_hours"], as_index=False)
        .agg(n_development_objects=("source_object_id", "nunique"), n_conditions=("condition_id", "nunique"))
        .sort_values(
            ["full_inchikey", "n_development_objects", "dose_uM", "time_hours"],
            ascending=[True, False, True, True], kind="mergesort",
        )
        .drop_duplicates("full_inchikey", keep="first")
    )
    candidate = candidate.merge(
        exposure[["full_inchikey", "dose_uM", "time_hours"]],
        on=["full_inchikey", "dose_uM", "time_hours"], how="inner", validate="many_to_one",
    )
    # A structure-key/cell/exposure is a single prediction target. Pick the highest-QC aggregate
    # metadata row if two LINCS pert_id aliases share an exact key; ties are condition_id stable.
    candidate = candidate.sort_values(
        ["cell_id", "full_inchikey", "n_treatment_instances", "condition_id"],
        ascending=[True, True, False, True], kind="mergesort",
    ).drop_duplicates(["cell_id", "full_inchikey", "dose_uM", "time_hours"], keep="first")
    max_conditions = int(config["data"]["max_conditions"])
    if len(candidate) > max_conditions:
        # Deterministic response-blind balanced cap: cycle exact keys, selecting one row per key
        # per pass; within key order by stable hash of object/cell/condition identity.
        groups = {
            key: group.assign(
                _rank=group.apply(
                    lambda r: _stable_hash(f"{config['model']['seed']}|cap|{r.source_object_id}|{r.cell_id}|{r.condition_id}"), axis=1
                )
            ).sort_values("_rank", kind="mergesort")
            for key, group in candidate.groupby("full_inchikey", sort=True)
        }
        selected_indices: list[int] = []
        positions = {key: 0 for key in groups}
        keys = sorted(groups)
        while len(selected_indices) < max_conditions:
            advanced = False
            for key in keys:
                pos = positions[key]
                group = groups[key]
                if pos < len(group) and len(selected_indices) < max_conditions:
                    selected_indices.append(int(group.index[pos]))
                    positions[key] += 1
                    advanced = True
            if not advanced:
                break
        candidate = candidate.loc[selected_indices].copy()
    candidate = candidate.sort_values(["source_object_id", "full_inchikey", "cell_id", "condition_id"], kind="mergesort").reset_index(drop=True)
    if len(candidate) > max_conditions or candidate["condition_id"].duplicated().any():
        raise ValueError("Frozen response-blind condition domain exceeds cap or has duplicate condition IDs")
    candidate["drug_feature_row_index"] = candidate["full_inchikey"].map(key_to_drug_row).astype(int)
    candidate["response_row_index"] = candidate["condition_id"].astype(int)
    candidate["log10_dose_uM"] = np.log10(candidate["dose_uM"].astype(float))
    candidate["log10_time_hours"] = np.log10(candidate["time_hours"].astype(float))
    candidate["log10_dose_um"] = candidate["log10_dose_uM"]
    candidate["fold_id"] = -1
    object_load = candidate.groupby("source_object_id").size().rename("n_conditions").reset_index()
    folds = _assign_folds(object_load, int(config["model"]["folds"]), int(config["model"]["seed"]))
    candidate["fold_id"] = candidate["source_object_id"].map(folds).astype(int)
    candidate["rna_row_index"] = candidate["depmap_model_id"].astype(str).map(
        dict(zip(cell_map["depmap_model_id"].astype(str), cell_map["rna_row_index"].astype(int)))
    ).astype(int)
    treatments = candidate[["condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "fold_id", "source_object_id"]].merge(
        treatment_meta[["inst_id", "control_group_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "rna_plate", "_valid_source"]],
        on=condition_keys, how="left", validate="many_to_many",
    )
    treatments = treatments.loc[treatments["_valid_source"].fillna(False)].copy()
    controls = treatments[["condition_id", "control_group_id", "cell_id", "pert_time", "pert_time_unit", "rna_plate"]].drop_duplicates().merge(
        control_meta.loc[_true_mask(control_meta["control_finite"]), ["inst_id", "control_group_id", "cell_id", "pert_time", "pert_time_unit", "rna_plate"]],
        on=["control_group_id", "cell_id", "pert_time", "pert_time_unit", "rna_plate"], how="left", validate="many_to_many",
    )
    treatment_ids = treatments.groupby("condition_id")["inst_id"].agg(lambda x: ";".join(sorted(set(x.dropna().astype(str)))))
    control_ids = controls.groupby("condition_id")["inst_id"].agg(lambda x: ";".join(sorted(set(x.dropna().astype(str)))))
    candidate["source_treatment_ids"] = candidate["condition_id"].map(treatment_ids).fillna("")
    candidate["source_control_ids"] = candidate["condition_id"].map(control_ids).fillna("")
    if candidate["source_treatment_ids"].eq("").any() or candidate["source_control_ids"].eq("").any():
        raise ValueError("Some selected conditions lack traceable treatment/control IDs")
    candidate["lincs_cell_id"] = candidate["cell_id"].astype(str)
    candidate = candidate.merge(
        structure_table[["full_inchikey", "canonical_smiles"]],
        on="full_inchikey", how="left", validate="many_to_one",
    )
    if candidate["canonical_smiles"].isna().any():
        raise ValueError("Selected conditions lost their exact-key canonical structure")

    derangement_table, shuffled_rows, shuffled_donor_models = _object_derangements(
        candidate, int(config["model"]["folds"]), int(config["model"]["seed"])
    )
    ctrp_rows = pd.read_csv(
        paths["ctrp_response_path"], usecols=["cellid", "drugid"], dtype="string"
    ).dropna().drop_duplicates()
    allowed_pairs = ctrp_rows.merge(
        cell_map[["ctrp_cell_id", "depmap_model_id", "source_object_id"]],
        left_on="cellid", right_on="ctrp_cell_id", how="inner", validate="many_to_one",
    ).merge(ctrp_drugs[["drugid", "full_inchikey"]], on="drugid", how="inner", validate="many_to_one")
    allowed_pairs = allowed_pairs.loc[allowed_pairs["full_inchikey"].isin(set(candidate["full_inchikey"]))]
    allowed_pairs = allowed_pairs.groupby(["source_object_id", "full_inchikey", "cellid", "drugid"], as_index=False).size()
    allowed_pairs = allowed_pairs.rename(columns={"size": "n_response_key_rows"})
    sensitivity_index = allowed_pairs.groupby(["source_object_id", "full_inchikey"], as_index=False).agg(
        ctrp_cell_ids=("cellid", lambda x: ";".join(sorted(set(x.astype(str))))),
        ctrp_drugids=("drugid", lambda x: ";".join(sorted(set(x.astype(str))))),
        n_cell_drug_pairs=("cellid", "size"),
        n_response_key_rows=("n_response_key_rows", "sum"),
    )
    sensitivity_index = sensitivity_index.sort_values(["source_object_id", "full_inchikey"], kind="mergesort")
    candidate["target_axis_sha256"] = config["data"]["target_axis_sha256"]
    candidate["target_axis_hash"] = candidate["target_axis_sha256"]
    candidate["full_inchikey"] = candidate["full_inchikey"].astype(str)

    candidate.to_csv(outdir / "conditions_frozen.tsv", sep="\t", index=False, lineterminator="\n")
    structure_table.to_csv(outdir / "structure_metadata.tsv", sep="\t", index=False, lineterminator="\n")
    sensitivity_index.to_csv(outdir / "sensitivity_pair_index.tsv", sep="\t", index=False, lineterminator="\n")
    allowed_pairs.to_csv(outdir / "sensitivity_cell_drug_pairs.tsv", sep="\t", index=False, lineterminator="\n")
    cell_map.to_csv(outdir / "cell_identity_frozen.tsv", sep="\t", index=False, lineterminator="\n")
    derangement_table.to_csv(outdir / "fold_local_state_derangements.tsv", sep="\t", index=False, lineterminator="\n")
    np.save(outdir / "drug_morgan2048_by_key.npy", drug_features, allow_pickle=False)
    np.save(outdir / "shuffled_rna_row_by_fold.npy", shuffled_rows, allow_pickle=False)
    np.save(outdir / "shuffled_donor_model_id_by_fold.npy", shuffled_donor_models, allow_pickle=False)

    no_numeric_scan = {"lincs_delta_path", "depmap_expression_matrix_path", "ctrp_response_path"}
    source_hashes = {
        name: _sha256(path)
        for name, path in paths.items()
        if path.is_file() and name not in no_numeric_scan
    }
    condition_manifest = json.loads((root / rel["lincs_conditions_manifest_path"]).read_text(encoding="utf-8"))
    depmap_audit = json.loads((root / rel["depmap_audit_path"]).read_text(encoding="utf-8"))
    depmap_matrix = np.load(paths["depmap_expression_matrix_path"], mmap_mode="r", allow_pickle=False)
    if depmap_matrix.ndim != 2 or depmap_matrix.shape[1] != 978:
        raise ValueError(f"DepMap exact978 matrix header mismatch: {depmap_matrix.shape}")
    # Access shape only; no expression array values are read in metadata phase.
    if not np.array_equal(np.sort(depmap_map["exact978_row"].to_numpy()), np.arange(depmap_matrix.shape[0])):
        raise ValueError("DepMap exact978 row mapping does not cover matrix rows")
    input_target_manifest = {
        "depmap_expression_matrix_shape": list(depmap_matrix.shape),
        "depmap_expression_dtype": str(depmap_matrix.dtype),
        "depmap_ordered_gene_ids_sha256": depmap_audit["gene_harmonization"]["ordered_gene_ids_sha256"],
        "lincs_target_axis_sha256": config["data"]["target_axis_sha256"],
        "lincs_manifest_n_genes": condition_manifest.get("n_genes"),
        "lincs_manifest_condition_delta_sha256": condition_manifest.get("files_sha256", {}).get("condition_delta.npy"),
        "lincs_delta_matrix_opened": False,
        "depmap_expression_values_read": False,
        "ctrp_aac_values_read": False,
    }
    if input_target_manifest["depmap_ordered_gene_ids_sha256"] != input_target_manifest["lincs_target_axis_sha256"]:
        raise ValueError("DepMap and LINCS ordered exact978 axes do not match")
    if int(input_target_manifest["lincs_manifest_n_genes"]) != 978:
        raise ValueError("LINCS target contract is not exact978")
    elapsed = time.monotonic() - started
    cap = int(config["resources"]["shared_metadata_cpu_cap_seconds"])
    if elapsed > cap:
        raise TimeoutError(f"Metadata freeze exceeded its reserved CPU slice: {elapsed:.3f}>{cap}")
    files = [
        "conditions_frozen.tsv", "structure_metadata.tsv", "sensitivity_pair_index.tsv",
        "sensitivity_cell_drug_pairs.tsv",
        "cell_identity_frozen.tsv", "fold_local_state_derangements.tsv",
        "drug_morgan2048_by_key.npy", "shuffled_rna_row_by_fold.npy",
        "shuffled_donor_model_id_by_fold.npy",
    ]
    outputs = {name: _sha256(outdir / name) for name in files}
    manifest = {
        "experiment_ids": ["EXP-066", "EXP-067"],
        "run_id": run_id,
        "status": "METADATA_FROZEN_ROOT_READBACK_REQUIRED",
        "protocol_version": "response_benefit_shared_contract_v1",
        "condition_count": int(len(candidate)),
        "object_count": int(candidate["source_object_id"].nunique()),
        "cell_line_count": int(candidate["cell_id"].nunique()),
        "exact_inchikey_count": int(candidate["full_inchikey"].nunique()),
        "ctrp_patient_drug_pair_count": int(len(sensitivity_index)),
        "ctrp_cell_drug_pair_count": int(len(allowed_pairs)),
        "object_fold_counts": {
            str(int(fold)): int(candidate.loc[candidate["fold_id"].eq(fold), "source_object_id"].nunique())
            for fold in sorted(candidate["fold_id"].unique())
        },
        "condition_fold_counts": {
            str(int(fold)): int((candidate["fold_id"] == fold).sum())
            for fold in sorted(candidate["fold_id"].unique())
        },
        "target_gene_count": 978,
        "max_condition_cap": int(max_conditions),
        "selection_policy": "per exact key choose dose/time pair with maximum development PatientID coverage; tie by numeric dose then time; response-blind; if over cap, deterministic round-robin exact-key cap by stable object/cell/condition hash",
        "asset_hashes": source_hashes,
        "target_asset_contract": input_target_manifest,
        "outputs_sha256": outputs,
        "elapsed_cpu_seconds": elapsed,
        "cpu_cap_seconds": cap,
        "gpu_hours": 0.0,
        "metadata_only": True,
        "ctrp_labels_metadata_only_validated": True,
        "forbidden_values_read": [],
        "opened_value_arrays": [],
    }
    (outdir / "metadata_freeze.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outdir


def load_frozen_response_domain(
    shared_dir: str | Path,
    *,
    read_lincs_targets: bool = False,
    read_ctrp_labels: bool = False,
) -> dict[str, Any]:
    """Load approved values only after independent root metadata gate PASS."""
    loader_started = time.process_time()
    shared = Path(shared_dir)
    receipt_path = shared / "root_metadata_readback.json"
    if not receipt_path.is_file():
        raise PermissionError("Root metadata readback receipt is required before values can be loaded")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_ROOT_INDEPENDENT_RESPONSE_BENEFIT_METADATA_READBACK":
        raise PermissionError("Root metadata readback did not pass")
    manifest = json.loads((shared / "metadata_freeze.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "METADATA_FROZEN_ROOT_READBACK_REQUIRED":
        raise PermissionError("The shared metadata freeze is not in the expected immutable state")
    if receipt.get("run_id") != manifest.get("run_id") or receipt.get("outputs_sha256") != manifest.get("outputs_sha256"):
        raise PermissionError("Root metadata receipt does not bind this frozen domain")
    for name, expected in manifest.get("outputs_sha256", {}).items():
        if _sha256(shared / name) != expected:
            raise PermissionError(f"Frozen metadata output changed after root readback: {name}")
    frame = pd.read_csv(shared / "conditions_frozen.tsv", sep="\t")
    rna_map = pd.read_csv(shared / "cell_identity_frozen.tsv", sep="\t")
    structure = pd.read_csv(shared / "structure_metadata.tsv", sep="\t", dtype={"full_inchikey": "string"})
    if structure["full_inchikey"].duplicated().any():
        raise ValueError("Frozen structure table must contain one row per complete key")
    if "canonical_smiles" not in frame.columns:
        frame = frame.merge(
            structure[["full_inchikey", "canonical_smiles"]],
            on="full_inchikey", how="left", validate="many_to_one",
        )
    if "lincs_cell_id" not in frame.columns:
        frame["lincs_cell_id"] = frame["cell_id"].astype(str)
    target_ids = _read_target_gene_ids(shared)
    actual_axis_sha256 = hashlib.sha256(("\n".join(target_ids) + "\n").encode("utf-8")).hexdigest()
    if actual_axis_sha256 != str(manifest["target_asset_contract"]["lincs_target_axis_sha256"]):
        raise ValueError("Response target axis changed after its metadata freeze")
    root = Path(__file__).resolve().parents[3]
    matrix_path = root / "data/processed/depmap/24q2_rnaseq_exact978/ccle_24q2_exact978_log2tpm.npy"
    depmap = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    rna_rows = sorted(set(frame["rna_row_index"].astype(int)))
    rna_by_model = np.asarray(depmap[rna_rows, :], dtype=np.float32)
    model_row_lookup = {row: i for i, row in enumerate(rna_rows)}
    frame["rna_selected_row_index"] = frame["rna_row_index"].astype(int).map(model_row_lookup).astype(int)
    drug_features = np.load(shared / "drug_morgan2048_by_key.npy", allow_pickle=False)
    shuffled_source = np.load(shared / "shuffled_rna_row_by_fold.npy", allow_pickle=False)
    shuffle_rows = np.asarray(
        [[model_row_lookup[row] for row in shuffled_source[f, :]] for f in range(shuffled_source.shape[0])],
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "condition_metadata": frame,
        "rna_by_model": rna_by_model,
        "rna_row_ids": rna_rows,
        "drug_morgan_by_key": drug_features,
        "shuffled_rna_row_by_fold": shuffle_rows,
        "shuffled_donor_model_id_by_fold": np.load(shared / "shuffled_donor_model_id_by_fold.npy", allow_pickle=False),
        "target_gene_ids": target_ids,
        "target_axis_hash": actual_axis_sha256,
        "target_axis_sha256": actual_axis_sha256,
        "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": _sha256(receipt_path),
        "metadata_run_id": manifest["run_id"],
        "load_cpu_seconds": 0.0,
        "value_access_audit": {
            "depmap_rows_loaded": len(rna_rows),
            "lincs_targets_loaded": bool(read_lincs_targets),
            "ctrp_labels_loaded": bool(read_ctrp_labels),
        },
    }
    if read_lincs_targets:
        base = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"
        delta_path = base / "condition_delta.npy"
        delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
        row_ids = frame["response_row_index"].to_numpy(dtype=np.int64)
        result["delta_by_condition"] = np.asarray(delta[row_ids, :], dtype=np.float32)
    if read_ctrp_labels:
        if not manifest.get("ctrp_labels_metadata_only_validated", True):
            raise PermissionError("CTRP label metadata is not frozen")
        ctrp_path = root / "data/processed/exp013/ctrp_v2/responses.csv"
        pair_index = pd.read_csv(shared / "sensitivity_pair_index.tsv", sep="\t", dtype={"source_object_id": "string"})
        cell_pairs = pd.read_csv(shared / "sensitivity_cell_drug_pairs.tsv", sep="\t", dtype="string")
        allowed = set(zip(cell_pairs["cellid"].astype(str), cell_pairs["drugid"].astype(str)))
        pieces = []
        for chunk in pd.read_csv(ctrp_path, usecols=["cellid", "drugid", "aac_published"], dtype={"cellid": "string", "drugid": "string"}, chunksize=50000):
            mask = [
                (str(cell), str(drug)) in allowed
                for cell, drug in zip(chunk["cellid"], chunk["drugid"])
            ]
            selected = chunk.loc[mask, ["cellid", "drugid", "aac_published"]].copy()
            selected["aac_published"] = pd.to_numeric(selected["aac_published"], errors="coerce")
            selected = selected.loc[np.isfinite(selected["aac_published"])]
            if not selected.empty:
                pieces.append(selected)
        result["ctrp_aac_rows"] = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["cellid", "drugid", "aac_published"])
    result["load_cpu_seconds"] = time.process_time() - loader_started
    return result


def _read_target_gene_ids(shared_dir: Path) -> list[str]:
    root = Path(__file__).resolve().parents[3]
    path = root / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        genes = payload
    elif isinstance(payload, dict):
        genes = payload.get("genes") or payload.get("gene_ids") or payload.get("pr_gene_ids")
    else:
        genes = None
    if not isinstance(genes, list) or len(genes) != 978:
        raise ValueError("Frozen LINCS target gene list is not a 978-ID list")
    return [str(g.get("pr_gene_id", g.get("gene_id", g))) if isinstance(g, dict) else str(g) for g in genes]
