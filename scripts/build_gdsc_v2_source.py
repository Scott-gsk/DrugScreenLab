"""Build an auditable GDSC2 source subset for EXP-014 without changing legacy assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.transfer.chemistry import standardize_smiles


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/exp014/gdsc_v2"))
    parser.add_argument("--report", type=Path, default=Path("data/artifacts/EXP-014/gdsc_v2_build.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    old = args.root / "processed/exp012"
    raw = pd.read_parquet(args.root / "processed/exp010/gdsc2_fitted_release8.5.parquet")
    legacy_samples = pd.read_parquet(old / "gdsc2_cell_line_samples.parquet")
    with np.load(old / "gdsc2_cell_line_indexed.npz", allow_pickle=True) as archive:
        expression = np.asarray(archive["expression"], dtype=np.float32)
        sample_ids = archive["sample_id"].astype(str)
        genes = archive["ordered_genes"].astype(str)
    bridge = json.loads((old / "structure_resolution_cache.json").read_text(encoding="utf-8"))
    accepted = {}
    excluded_structures = []
    for row in bridge:
        identity = standardize_smiles(row.get("canonical_smiles"))
        name = str(row["source_name"])
        if identity.accepted and identity.inchikey == row.get("inchikey"):
            accepted[name] = identity
        else:
            excluded_structures.append({"source_name": name, "reason": identity.status})
    source = raw.loc[raw.DATASET.eq("GDSC2") & raw.DRUG_NAME.astype(str).isin(accepted)].copy()
    source["sample_id"] = source.SANGER_MODEL_ID.astype(str)
    source["drug_id"] = source.DRUG_NAME.astype(str).map(lambda name: accepted[name].inchikey)
    source = source[source.sample_id.isin(set(sample_ids))].copy()
    key_counts = source.groupby(["sample_id", "drug_id"], sort=False).size()
    unique_keys = key_counts[key_counts.eq(1)].index
    unique = source.set_index(["sample_id", "drug_id"]).loc[unique_keys].reset_index()
    used_drugs = sorted(unique.drug_id.unique())
    used_samples = set(unique.sample_id)
    sample_mask = np.array([sample in used_samples for sample in sample_ids])
    kept_sample_ids = sample_ids[sample_mask]
    samples = legacy_samples.set_index("sample_id").loc[kept_sample_ids].reset_index().copy()
    samples["patient_id"] = samples["sample_id"]
    # Different source labels can deliberately resolve to one full InChIKey.
    # Retain every alias but emit exactly one feature row per chemical identity.
    by_key: dict[str, list[tuple[str, object]]] = {}
    for name, identity in accepted.items():
        if identity.inchikey in used_drugs:
            by_key.setdefault(identity.inchikey, []).append((name, identity))
    drug_rows = []
    for key, aliases in sorted(by_key.items()):
        canonicals = {identity.canonical_isomeric_smiles for _, identity in aliases}
        if len(canonicals) != 1:
            raise ValueError(f"full InChIKey {key} maps to conflicting canonical SMILES: {sorted(canonicals)}")
        statuses = {identity.status for _, identity in aliases}
        if not statuses <= {"accepted", "counterion_removed"}:
            raise ValueError(f"unexpected accepted-structure status for {key}: {sorted(statuses)}")
        drug_rows.append({
            "drug_id": key, "canonical_smiles": canonicals.pop(), "inchikey": key,
            "source_drug_aliases_json": json.dumps(sorted(name for name, _ in aliases), ensure_ascii=False),
            "standardization_statuses_json": json.dumps(sorted(statuses)),
        })
    drugs = pd.DataFrame(drug_rows).sort_values("drug_id").reset_index(drop=True)
    pairs = unique[["sample_id", "drug_id"]].copy()
    pairs["response"] = 1.0 - unique.AUC.to_numpy(float)
    pairs["response_name"] = "1_minus_auc"
    pairs["response_scale"] = "GDSC2 fitted AUC transformed to 1-AUC; only unique original SANGER_MODEL_ID×structure rows"
    pairs["higher_is_sensitive"] = True
    pairs = pairs.sort_values(["sample_id", "drug_id"]).reset_index(drop=True)
    sample_index = {value: i for i, value in enumerate(kept_sample_ids)}
    drug_index = {value: i for i, value in enumerate(drugs.drug_id)}
    pairs["pair_sample_idx"] = pairs.sample_id.map(sample_index)
    pairs["pair_drug_idx"] = pairs.drug_id.map(drug_index)
    args.output.mkdir(parents=True)
    samples.to_parquet(args.output / "samples.parquet", index=False)
    drugs.to_parquet(args.output / "drugs.parquet", index=False)
    pairs.to_parquet(args.output / "pairs.parquet", index=False)
    np.savez_compressed(args.output / "indexed.npz", expression=expression[sample_mask], sample_id=kept_sample_ids, ordered_genes=genes)
    report = {
        "cohort_id": "gdsc2_cell_line_v2", "role": "EXP014_SOURCE_TRAIN", "selection": "input_metadata_only: accepted standardized full-InChIKey structure and unique original SANGER_MODEL_ID×drug_id; all duplicate keys excluded",
        "raw_response_sha256": digest(args.root / "processed/exp010/gdsc2_fitted_release8.5.parquet"), "structure_bridge_sha256": digest(old / "structure_resolution_cache.json"),
        "counts": {"raw_gdsc2_rows": int(len(raw)), "accepted_structure_names": len(accepted), "structure_unavailable_names": len(excluded_structures), "duplicate_source_keys_excluded": int((key_counts > 1).sum()), "response_pairs": len(pairs), "samples": len(samples), "drugs": len(drugs), "genes": len(genes)},
        "drug_alias_policy": "same full InChIKey plus same canonical SMILES is one drug row with source_drug_aliases_json; conflicting canonical SMILES fail the build",
        "excluded_structure_names": excluded_structures, "output_sha256": {path.name: digest(path) for path in args.output.iterdir()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
