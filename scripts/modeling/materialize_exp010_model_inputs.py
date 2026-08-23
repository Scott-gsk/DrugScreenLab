"""Materialize leakage-safe real-data model inputs for EXP-010.

Writes only derived, reproducible artifacts below processed/exp010/model_inputs.
No raw or registered source asset is modified.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(value: object) -> str:
    return "".join(c for c in str(value).casefold() if c.isalnum())


def drug_id_norm(value: object) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return norm(value)


def top_variance(frame: pd.DataFrame, count: int) -> list[str]:
    if frame.shape[0] < 2 or frame.shape[1] < count:
        raise ValueError(f"cannot select {count} genes from {frame.shape}")
    variance = frame.var(axis=0, ddof=0)
    variance = variance.dropna()
    if len(variance) < count:
        raise ValueError(f"only {len(variance)} genes have train-defined variance")
    # Gene-symbol tie break makes selection invariant to parquet column order.
    return sorted(variance.index, key=lambda gene: (-float(variance[gene]), str(gene)))[:count]


def encode_strings(values: pd.Series | list[str]) -> np.ndarray:
    return np.asarray([str(x) for x in values], dtype="U")


def auc_to_sensitivity(values: pd.Series) -> pd.Series:
    auc = pd.to_numeric(values, errors="raise").astype(float)
    if auc.isna().any() or (auc < 0).any() or (auc > 1).any():
        raise ValueError(f"GDSC AUC must be finite in [0,1], observed [{auc.min()}, {auc.max()}]")
    return 1.0 - auc


def save_npz(path: Path, partitions: dict[str, pd.DataFrame], expression: dict[str, pd.DataFrame], genes: list[str]) -> None:
    payload: dict[str, np.ndarray] = {}
    for partition, rows in partitions.items():
        samples = rows["sample_id"].astype(str)
        payload[f"{partition}_gene"] = expression[partition].loc[samples, genes].to_numpy(np.float32)
        payload[f"{partition}_smiles"] = encode_strings(rows["canonical_smiles"])
        payload[f"{partition}_y"] = rows["response"].to_numpy(np.float32)
        payload[f"{partition}_sample_id"] = encode_strings(samples)
        payload[f"{partition}_drug_id"] = encode_strings(rows["drug_name"])
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/EXP-010.yaml")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DRUGSCREEN_DATA_ROOT", ROOT / "data")))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-row-cap", type=int, help="response-blind timing/dry-run cap; omit for complete source materialization")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed, gene_count = int(config["seed"]), int(config["models"]["architecture"]["gene_input_dim"])
    processed = args.data_root / "processed/exp010"
    output = args.output or processed / "model_inputs"
    output.mkdir(parents=True, exist_ok=True)

    cmp = pd.read_parquet(processed / "cmp_gdsc2_gene_contract.parquet")
    mcrc_expr = pd.read_parquet(processed / "mcrc_gene_contract.parquet")
    response = pd.read_parquet(processed / "mcrc_monotherapy_dss.parquet").rename(columns={"dss": "response"})
    if "higher_is_sensitive" not in response or not response["higher_is_sensitive"].fillna(False).astype(bool).all():
        raise ValueError("mCRC DSS contract must declare higher_is_sensitive=true for every row")
    gdsc = pd.read_parquet(processed / "gdsc2_fitted_release8.5.parquet")
    mcrc_structures = json.loads((args.data_root / "external/exp010/pubchem/mcrc_drug_structures.json").read_text(encoding="utf-8"))
    mcrc_smiles = {norm(x["primary_name"]): x["canonical_smiles"] for x in mcrc_structures if x.get("mapped") and x.get("canonical_smiles")}
    response["canonical_smiles"] = response["drug_name"].map(lambda x: mcrc_smiles.get(norm(x)))
    if response["canonical_smiles"].isna().any():
        raise ValueError(f"mCRC structures missing: {sorted(response.loc[response.canonical_smiles.isna(), 'drug_name'].unique())}")
    response = response[response.sample_id.astype(str).isin(mcrc_expr.index.astype(str))].copy()

    patients = sorted(response.patient_id.astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    patient_fold = {patient: i % int(config["split"]["outer_folds"]) for i, patient in enumerate(patients)}
    response["outer_fold"] = response.patient_id.astype(str).map(patient_fold)

    gdsc_structures = json.loads((args.data_root / "external/exp010/pubchem/drug_structures.json").read_text(encoding="utf-8"))
    gdsc_smiles = {norm(x["primary_name"]): x["canonical_smiles"] for x in gdsc_structures if x.get("mapped") and x.get("canonical_smiles")}
    gdsc = gdsc.dropna(subset=["SANGER_MODEL_ID", "DRUG_ID", "DRUG_NAME", "AUC"]).copy()
    gdsc["sample_id"] = gdsc.SANGER_MODEL_ID.astype(str)
    gdsc["drug_name"] = gdsc.DRUG_NAME.astype(str)
    gdsc["response"] = auc_to_sensitivity(gdsc.AUC)
    gdsc["higher_is_sensitive"] = True
    # Use the full 310-name GDSC PubChem cache. The separate 49-name mCRC cache
    # is reserved for target mapping and must not define source pretraining.
    gdsc["canonical_smiles"] = gdsc.DRUG_NAME.map(lambda x: gdsc_smiles.get(norm(x)))
    gdsc = gdsc[gdsc.sample_id.isin(cmp.index.astype(str)) & gdsc.canonical_smiles.notna()].copy()
    gdsc = gdsc.drop_duplicates(["sample_id", "drug_name"])
    full_source_supported_smiles = set(gdsc.canonical_smiles.unique())
    full_source_drug_count = int(gdsc.drug_name.nunique())
    source_models = sorted(gdsc.sample_id.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(source_models)
    validation_models = set(source_models[::5])
    gdsc["partition"] = np.where(gdsc.sample_id.isin(validation_models), "validation", "train")
    if args.source_row_cap:
        # Probe cap is response-blind and spreads rows across models so feature
        # variance and memory behavior are meaningful even at a tiny budget.
        gdsc = (gdsc.sort_values(["partition", "sample_id", "drug_name"])
                .groupby(["partition", "sample_id"], group_keys=False).head(1)
                .groupby("partition", group_keys=False).head(args.source_row_cap))
    source_train_models = sorted(gdsc.loc[gdsc.partition == "train", "sample_id"].unique())
    source_genes = top_variance(cmp.loc[source_train_models], gene_count)
    (output / "source_genes_8192.txt").write_text("\n".join(source_genes) + "\n", encoding="utf-8")

    # Upstream learns BPE over a SMILES text stream. Freeze it using source train
    # molecules only; validation/target structures are not read by learn_bpe.
    from subword_nmt.learn_bpe import learn_bpe
    train_smiles = sorted(gdsc.loc[gdsc.partition == "train", "canonical_smiles"].unique())
    source_supported_smiles = set(train_smiles)
    response["source_supported"] = response.canonical_smiles.isin(full_source_supported_smiles)
    support_counts = response[response.source_supported].groupby("sample_id").drug_name.nunique()
    if support_counts.empty or int(support_counts.min()) < int(config["data_gates"]["min_mapped_monotherapies_per_main_pdo"]):
        raise ValueError(f"source-supported mCRC PDO drug gate failed; minimum={support_counts.min() if len(support_counts) else 0}")
    (output / "source_supported_mcrc_drugs.json").write_text(
        json.dumps(sorted(response.loc[response.source_supported, "drug_name"].unique()), indent=2), encoding="utf-8")
    codes_path = output / "gdsc_train_bpe.codes"
    with io.StringIO("\n".join(train_smiles) + "\n") as source, codes_path.open("w", encoding="utf-8") as target:
        learn_bpe(source, target, num_symbols=10000)

    source_parts = {p: gdsc.loc[gdsc.partition == p].copy() for p in ("train", "validation")}
    source_parts["test"] = source_parts["validation"].copy()  # never used for selection; smoke inference contract
    source_expr = {p: cmp for p in source_parts}
    save_npz(output / "M2_source_gdsc.npz", source_parts, source_expr, source_genes)
    source_manifest = [{"dataset_domain": "source_gdsc", "sample_id": row.sample_id, "partition": row.partition}
                       for row in gdsc[["sample_id", "partition"]].drop_duplicates().itertuples(index=False)]
    (output / "M2_source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    fold_meta = []
    for fold in range(5):
        validation_fold = (fold + int(config["split"]["validation_offset"])) % 5
        labels = np.where(response.outer_fold == fold, "test", np.where(response.outer_fold == validation_fold, "validation", "train"))
        current = response.assign(partition=labels)
        parts = {p: current.loc[current.partition == p].copy() for p in ("train", "validation", "test")}
        expressions = {p: mcrc_expr for p in parts}
        m3_parts = {p: rows.loc[rows.canonical_smiles.isin(source_supported_smiles)].copy() for p, rows in parts.items()}
        if any(frame.empty for frame in m3_parts.values()):
            raise ValueError(f"fold {fold} has an empty source-supported M3 partition")
        save_npz(output / f"fold{fold}_M3_sourcegenes.npz", m3_parts, expressions, source_genes)
        target_genes = top_variance(mcrc_expr.loc[sorted(parts["train"].sample_id.unique())], gene_count)
        save_npz(output / f"fold{fold}_M4_targetgenes.npz", parts, expressions, target_genes)
        (output / f"fold{fold}_M4_genes_8192.txt").write_text("\n".join(target_genes) + "\n", encoding="utf-8")
        manifest = [{"dataset_domain": "target_mcrc", "patient_id": row.patient_id,
                     "pdo_id": row.sample_id, "sample_id": row.sample_id, "partition": row.partition}
                    for row in current[["patient_id", "sample_id", "partition"]].drop_duplicates().itertuples(index=False)]
        manifest_path = output / f"fold{fold}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fold_meta.append({"fold": fold, "patients": {p: int(parts[p].patient_id.nunique()) for p in parts},
                          "rows": {p: len(parts[p]) for p in parts}, "m4_gene_sha256": digest(output / f"fold{fold}_M4_genes_8192.txt")})
        fold_meta[-1]["m3_source_supported_rows"] = {p: len(m3_parts[p]) for p in m3_parts}
    summary = {"experiment_id": "EXP-010", "seed": seed, "gene_contract_count": cmp.shape[1],
               "source": {"models": len(source_models), "rows": len(gdsc), "drugs": gdsc.drug_name.nunique(),
                          "full_mapped_structures": len(full_source_supported_smiles),
                          "full_mapped_gdsc_drugs": full_source_drug_count,
                          "pubchem_cache_mapped_records": sum(bool(x.get("mapped") and x.get("canonical_smiles")) for x in gdsc_structures),
                          "training_models": len(source_train_models), "selected_genes": len(source_genes)},
               "label_schema": {"source": {"raw": "GDSC2 fitted AUC", "transform": "1-AUC",
                                               "range": [float(gdsc.response.min()), float(gdsc.response.max())],
                                               "higher_is_sensitive": True},
                                "target": {"field": "DSS", "range": [float(response.response.min()), float(response.response.max())],
                                           "higher_is_sensitive": bool(response.higher_is_sensitive.all())}},
               "target": {"patients": len(patients), "samples": response.sample_id.nunique(), "rows": len(response)},
               "common_test_support": {"drugs": int(response.loc[response.source_supported, 'drug_name'].nunique()),
                                       "rows": int(response.source_supported.sum()),
                                       "min_drugs_per_pdo": int(support_counts.min()),
                                       "max_drugs_per_pdo": int(support_counts.max()),
                                       "all_pdos_at_least_5": bool((support_counts >= 5).all())},
               "folds": fold_meta, "bpe_codes_sha256": digest(codes_path),
               "source_genes_sha256": digest(output / "source_genes_8192.txt"),
               "source_row_cap": args.source_row_cap,
               "source_label_provenance": "pre-registered GDSC2 release-8.5 fitted AUC transformed to 1-AUC; raw release file is complete and checksum-verified"}
    (output / "materialization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
