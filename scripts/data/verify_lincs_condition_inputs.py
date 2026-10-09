"""独立核验 EXP-037 条件对照、结构指纹和暴露输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(contract_dir: Path, condition_dir: Path, input_dir: Path, cache_path: Path) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(input_dir / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    treatments = pd.read_parquet(contract_dir / "treatments.parquet")
    controls = pd.read_parquet(contract_dir / "controls.parquet")
    conditions = pd.read_parquet(condition_dir / "conditions.parquet")
    control_input = np.load(input_dir / "condition_control.npy", mmap_mode="r")
    drugs = pd.read_parquet(input_dir / "drugs.parquet")
    fingerprints = np.load(input_dir / "drug_morgan2048.npy", mmap_mode="r")
    exposure = pd.read_parquet(input_dir / "condition_exposure.parquet")
    if control_input.shape != (len(conditions), 978):
        raise ValueError("condition control axis mismatch")
    if fingerprints.shape != (len(drugs), 2048) or drugs["drug_index"].tolist() != list(range(len(drugs))):
        raise ValueError("drug fingerprint axis mismatch")
    if not exposure["condition_id"].equals(conditions["condition_id"]):
        raise ValueError("exposure condition identity mismatch")
    if not np.allclose(exposure["log10_dose_um"], np.log10(pd.to_numeric(exposure["pert_dose"]))):
        raise ValueError("dose transform mismatch")
    if not np.allclose(exposure["time_hours"], pd.to_numeric(exposure["pert_time"])):
        raise ValueError("time transform mismatch")
    cache = np.load(cache_path, mmap_mode="r")
    control_means = {}
    for group, frame in controls.groupby("control_group_id", sort=False):
        control_means[int(group)] = np.asarray(
            cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float64).mean(axis=0)
    samples = np.linspace(0, len(conditions) - 1, 25, dtype=int)
    maximum = 0.0
    key_columns = ["cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
    for condition_id in samples:
        key = conditions.iloc[condition_id]
        mask = np.ones(len(treatments), dtype=bool)
        for column in key_columns:
            mask &= treatments[column].astype(str).to_numpy() == str(key[column])
        source = treatments.loc[mask]
        expected = np.stack([control_means[int(x)] for x in source["control_group_id"]]).mean(axis=0)
        error = float(np.max(np.abs(expected - control_input[condition_id])))
        maximum = max(maximum, error)
        if error > 2e-6:
            raise ValueError(f"condition control mismatch at {condition_id}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprint_samples = np.linspace(0, len(drugs) - 1, 100, dtype=int)
    for index in fingerprint_samples:
        molecule = Chem.MolFromSmiles(drugs.iloc[index]["canonical_smiles"])
        expected = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
        if not np.array_equal(expected, fingerprints[index]):
            raise ValueError(f"fingerprint mismatch at drug index {index}")
    return {"status": "PASS", "n_conditions": len(conditions), "n_drugs": len(drugs),
            "sampled_controls": len(samples), "sampled_fingerprints": len(fingerprint_samples),
            "max_control_abs_difference": maximum}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract_dir, args.condition_dir, args.input_dir, args.cache)
    (args.input_dir / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
