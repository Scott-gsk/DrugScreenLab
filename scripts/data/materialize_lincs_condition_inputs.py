"""物化 EXP-037 条件级对照表达、药物结构及剂量时间输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


CONDITION = ["cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def materialize(contract_dir: Path, condition_dir: Path, cache_path: Path,
                pert_info_path: Path, output_dir: Path) -> dict:
    treatment = pd.read_parquet(contract_dir / "treatments.parquet")
    control = pd.read_parquet(contract_dir / "controls.parquet")
    conditions = pd.read_parquet(condition_dir / "conditions.parquet")
    cache = np.load(cache_path, mmap_mode="r")
    control_means = {}
    for group, frame in control.groupby("control_group_id", sort=False):
        control_means[int(group)] = np.asarray(
            cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float64
        ).mean(axis=0).astype(np.float32)
    coded = treatment.merge(conditions[["condition_id", *CONDITION]], on=CONDITION,
                            how="inner", validate="many_to_one", sort=False)
    if len(coded) != len(treatment):
        raise ValueError("condition assignment lost treatment instances")
    members = coded.groupby("condition_id", sort=True).indices
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    temp_control = output_dir / "condition_control.npy.tmp"
    control_matrix = np.lib.format.open_memmap(
        temp_control, mode="w+", dtype=np.float32, shape=(len(conditions), 978))
    for condition_id in range(len(conditions)):
        frame = coded.iloc[members[condition_id]]
        control_matrix[condition_id] = np.stack(
            [control_means[int(group)] for group in frame["control_group_id"]]
        ).mean(axis=0, dtype=np.float64)
        if (condition_id + 1) % 10000 == 0:
            control_matrix.flush()
            print(json.dumps({"condition_controls": condition_id + 1,
                              "total": len(conditions)}), flush=True)
    if not np.isfinite(control_matrix).all():
        raise ValueError("non-finite condition control expression")
    del control_matrix
    control_path = output_dir / "condition_control.npy"
    os.replace(temp_control, control_path)

    compounds = pd.read_csv(pert_info_path, sep="\t", dtype=str, keep_default_na=False)
    if compounds["pert_id"].duplicated().any():
        raise ValueError("duplicate pert_id in compound metadata")
    compounds = compounds.set_index("pert_id")
    drug_ids = sorted(conditions["pert_id"].unique())
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = np.zeros((len(drug_ids), 2048), dtype=np.uint8)
    canonical_smiles, inchi_keys = [], []
    for index, drug_id in enumerate(drug_ids):
        if drug_id not in compounds.index:
            raise ValueError(f"drug missing from pert_info: {drug_id}")
        row = compounds.loc[drug_id]
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        if molecule is None:
            raise ValueError(f"invalid preregistered canonical SMILES: {drug_id}")
        fingerprints[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
        canonical_smiles.append(Chem.MolToSmiles(molecule, canonical=True))
        inchi_keys.append(row["inchi_key"])
    drug_path = output_dir / "drug_morgan2048.npy"
    np.save(drug_path, fingerprints)
    identity_path = output_dir / "drugs.parquet"
    pd.DataFrame({"drug_index": np.arange(len(drug_ids)), "pert_id": drug_ids,
                  "canonical_smiles": canonical_smiles,
                  "inchi_key": inchi_keys}).to_parquet(identity_path, index=False)
    exposure_path = output_dir / "condition_exposure.parquet"
    exposure = conditions[["condition_id", "pert_id", "pert_dose", "pert_time"]].copy()
    exposure["log10_dose_um"] = np.log10(pd.to_numeric(exposure["pert_dose"], errors="raise"))
    exposure["time_hours"] = pd.to_numeric(exposure["pert_time"], errors="raise")
    exposure.to_parquet(exposure_path, index=False)
    manifest = {
        "protocol": "EXP-037 condition input contract v1", "status": "PASS",
        "n_conditions": len(conditions), "n_drugs": len(drug_ids),
        "control_shape": [len(conditions), 978], "drug_feature_shape": [len(drug_ids), 2048],
        "drug_feature": "RDKit Morgan count-independent bit fingerprint radius=2, 2048 bits",
        "exposure": "log10 positive micromolar dose and hours; source filters only 6h/24h",
        "source": {"continuous_contract_sha256": sha256(contract_dir / "quality_report.json"),
                   "condition_manifest_sha256": sha256(condition_dir / "manifest.json"),
                   "cache_sha256": sha256(cache_path), "pert_info_sha256": sha256(pert_info_path)},
        "files_sha256": {"condition_control.npy": sha256(control_path),
                         "drug_morgan2048.npy": sha256(drug_path),
                         "drugs.parquet": sha256(identity_path),
                         "condition_exposure.parquet": sha256(exposure_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pert-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.contract_dir, args.condition_dir, args.cache,
                         args.pert_info, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
