"""Independent root verification of the response-blind EXP-066/067 freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, required=True)
    args = parser.parse_args()
    shared = args.shared_dir if args.shared_dir.is_absolute() else ROOT / args.shared_dir
    manifest_path = shared / "metadata_freeze.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "METADATA_FROZEN_ROOT_READBACK_REQUIRED":
        raise ValueError("Metadata freeze status is not awaiting root readback")
    conditions = pd.read_csv(shared / "conditions_frozen.tsv", sep="\t", dtype={"source_object_id": "string", "full_inchikey": "string"})
    cells = pd.read_csv(shared / "cell_identity_frozen.tsv", sep="\t", dtype="string")
    derangements = pd.read_csv(shared / "fold_local_state_derangements.tsv", sep="\t", dtype="string")
    response_index = pd.read_csv(shared / "sensitivity_pair_index.tsv", sep="\t", dtype="string")
    response_pairs = pd.read_csv(shared / "sensitivity_cell_drug_pairs.tsv", sep="\t", dtype="string")
    structure = pd.read_csv(shared / "structure_metadata.tsv", sep="\t", dtype="string")
    morgan_path = shared / "drug_morgan2048_by_key.npy"
    shuffle_path = shared / "shuffled_rna_row_by_fold.npy"
    donor_path = shared / "shuffled_donor_model_id_by_fold.npy"
    morgan = np.load(morgan_path, mmap_mode="r", allow_pickle=False)
    shuffle = np.load(shuffle_path, mmap_mode="r", allow_pickle=False)
    donor_ids = np.load(donor_path, mmap_mode="r", allow_pickle=False)
    if not manifest.get("metadata_only") or manifest.get("opened_value_arrays") != []:
        raise ValueError("Metadata stage reports access to protected value arrays")
    if manifest.get("forbidden_values_read") != []:
        raise ValueError("Metadata stage reports forbidden numeric value reads")
    if len(conditions) == 0 or len(conditions) > int(manifest["max_condition_cap"]):
        raise ValueError("Frozen condition denominator is empty or above the cap")
    if conditions["condition_id"].duplicated().any():
        raise ValueError("condition_id is not one-to-one")
    if conditions["full_inchikey"].isna().any() or conditions["full_inchikey"].str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").fillna(False).eq(False).any():
        raise ValueError("Incomplete or malformed full InChIKey in frozen conditions")
    if conditions["target_axis_sha256"].nunique() != 1 or conditions["target_axis_sha256"].iloc[0] != manifest["target_asset_contract"]["lincs_target_axis_sha256"]:
        raise ValueError("Target axis hash differs between rows and manifest")
    if conditions["source_split_role"].isin(["locked_eval", "outer", "PDO"]).any():
        raise ValueError("Forbidden role present in metadata freeze")
    object_fold_n = conditions.groupby("source_object_id")["fold_id"].nunique()
    if not object_fold_n.eq(1).all():
        raise ValueError("PatientID crosses outer fold boundaries")
    if cells["depmap_model_id"].duplicated().any() or cells["cell_id"].duplicated().any():
        raise ValueError("Frozen cell identity table is not one-to-one by model and LINCS cell")
    model_patient_n = conditions.groupby("depmap_model_id")["source_object_id"].nunique()
    if not model_patient_n.eq(1).all():
        raise ValueError("A DepMap model maps to multiple PatientID objects")
    if set(conditions["depmap_model_id"]) - set(cells["depmap_model_id"]):
        raise ValueError("Condition domain contains a model absent from frozen cell identity table")
    if not set(conditions["cell_id"]).issubset(set(cells["cell_id"])):
        raise ValueError("Condition domain contains a LINCS cell absent from frozen identity table")
    if not conditions["response_row_index"].equals(conditions["condition_id"]):
        if not np.array_equal(conditions["response_row_index"].to_numpy(), conditions["condition_id"].to_numpy()):
            raise ValueError("Response row index no longer matches the frozen condition key")
    expected_objects = conditions[["source_object_id", "fold_id"]].drop_duplicates()
    if expected_objects["source_object_id"].duplicated().any():
        raise ValueError("PatientID was assigned to multiple folds")
    if not set(conditions["full_inchikey"]).issubset(set(structure["full_inchikey"])):
        raise ValueError("A frozen condition references a key without frozen structure metadata")
    if structure["full_inchikey"].duplicated().any() or structure["full_inchikey"].str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").fillna(False).eq(False).any():
        raise ValueError("Frozen structure metadata contains duplicate or malformed full keys")
    if morgan.ndim != 2 or morgan.shape != (len(structure), 2048):
        raise ValueError("Morgan fingerprint matrix has the wrong shape")
    if not np.isin(morgan, [0, 1]).all():
        raise ValueError("Morgan fingerprint matrix is not binary")
    from rdkit import rdBase

    fingerprint_contract = "Morgan(radius=2,fpSize=2048,binary,bit_index=0..2047)"
    fingerprint_contract_sha256 = hashlib.sha256(fingerprint_contract.encode("utf-8")).hexdigest()
    n_folds = len(manifest["condition_fold_counts"])
    if shuffle.shape != (n_folds, len(conditions)) or donor_ids.shape != (n_folds, len(conditions)):
        raise ValueError("Fold-local state derangement matrices do not match the frozen denominator")
    if len(derangements) != conditions["cell_id"].nunique() * n_folds:
        raise ValueError("Derangement map row count does not match fold×cell coverage")
    for fold in range(n_folds):
        object_to_fold = dict(zip(conditions["source_object_id"].astype(str), conditions["fold_id"].astype(int)))
        rows = derangements.loc[pd.to_numeric(derangements["fold_id"]).eq(fold)]
        if rows["target_lincs_cell_id"].duplicated().any():
            raise ValueError("Derangement mapping has duplicate target cells within fold")
        for row in rows.itertuples(index=False):
            target_fold = object_to_fold[str(row.source_object_id)]
            donor_fold = object_to_fold[str(row.donor_object_id)]
            expected_partition = "heldout" if target_fold == fold else "train"
            target_is_heldout = target_fold == fold
            donor_is_heldout = donor_fold == fold
            if row.partition != expected_partition or donor_is_heldout != target_is_heldout:
                raise ValueError("State derangement crossed the TRAIN/heldout fold boundary")
            if str(row.source_object_id) == str(row.donor_object_id):
                raise ValueError("State derangement is not a whole-object derangement")
    if response_index.empty or response_pairs.empty:
        raise ValueError("No metadata-only CTRP exact-key pair domain was frozen")
    if not set(response_index["full_inchikey"]).issubset(set(conditions["full_inchikey"])):
        raise ValueError("CTRP sensitivity pair keys exceed the frozen LINCS/structure key domain")
    output_names = [
        "conditions_frozen.tsv", "structure_metadata.tsv", "sensitivity_pair_index.tsv",
        "sensitivity_cell_drug_pairs.tsv", "cell_identity_frozen.tsv", "fold_local_state_derangements.tsv",
        "drug_morgan2048_by_key.npy", "shuffled_rna_row_by_fold.npy", "shuffled_donor_model_id_by_fold.npy",
    ]
    actual_hashes = {name: sha256(shared / name) for name in output_names}
    if actual_hashes != manifest["outputs_sha256"]:
        raise ValueError("Frozen metadata outputs do not match their recorded SHA-256 values")
    receipt = {
        "status": "PASS_ROOT_INDEPENDENT_RESPONSE_BENEFIT_METADATA_READBACK",
        "run_id": manifest["run_id"],
        "condition_count": int(len(conditions)),
        "object_count": int(conditions["source_object_id"].nunique()),
        "cell_line_count": int(conditions["cell_id"].nunique()),
        "exact_inchikey_count": int(conditions["full_inchikey"].nunique()),
        "sensitivity_patient_drug_pair_count": int(len(response_index)),
        "sensitivity_cell_drug_pair_count": int(len(response_pairs)),
        "target_genes": int(manifest["target_gene_count"]),
        "folds": n_folds,
        "rdkit_version": rdBase.rdkitVersion,
        "fingerprint_contract": fingerprint_contract,
        "fingerprint_contract_sha256": fingerprint_contract_sha256,
        "patient_id_overlap_between_folds": 0,
        "forbidden_values_read": [],
        "numeric_arrays_opened_for_values": [],
        "asset_hashes": manifest["asset_hashes"],
        "outputs_sha256": actual_hashes,
        "checker": "scripts/modeling/verify_response_benefit_metadata.py; independent metadata-only readback",
    }
    out = shared / "root_metadata_readback.json"
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing root readback receipt")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
