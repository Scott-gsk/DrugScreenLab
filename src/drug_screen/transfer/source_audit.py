"""Reconstruct and verify the public Kryeziu mCRC PDO transfer bundle."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from drug_screen.data.mcrc_source import rebuild_mcrc, sha256


def _same(left, right, columns, keys):
    def normalized(frame):
        result = frame.loc[:, columns].copy()
        for column in result.select_dtypes(include="object"):
            result[column] = result[column].map(
                lambda value: tuple(value.tolist()) if isinstance(value, np.ndarray)
                else tuple(value) if isinstance(value, list) else value
            )
        return result.sort_values(keys).reset_index(drop=True)
    return normalized(left).equals(normalized(right))


def audit_mcrc_source(root: Path) -> dict:
    raw = root / "raw/mcrc_pdo_2026/v3"
    processed = root / "processed/transfer/mcrc_v3"
    legacy = root / "external/exp010/pubchem/mcrc_drug_structures.json"
    fixed = root / "raw/transfer_public_assets/pubchem/mcrc_isomeric_20260912.json"
    meta = json.loads((raw / "dataset_metadata.json").read_text(encoding="utf-8"))
    source_hashes = {}
    for entry in meta["files"]:
        path = raw / entry["filename"]
        digest = sha256(path)
        if digest != entry["content_details"]["sha256_hash"]:
            raise ValueError(f"original source checksum mismatch: {path}")
        source_hashes[path.name] = digest
    rebuilt = rebuild_mcrc(
        pd.read_excel(raw / "Data S1.xlsx"), pd.read_excel(raw / "Data S2.xlsx"),
        pd.read_excel(raw / "Data S5.xlsx"), legacy, fixed,
    )
    samples = pd.read_parquet(processed / "samples.parquet")
    drugs = pd.read_parquet(processed / "drugs.parquet")
    pairs = pd.read_parquet(processed / "pairs.parquet")
    with np.load(processed / "indexed.npz", allow_pickle=False) as z:
        expression, sample_ids, gene_ids = z["expression"], tuple(z["sample_id"].astype(str)), tuple(z["ordered_genes"].astype(str))
    checks = {
        "expression_max_abs_error": float(np.max(np.abs(expression - rebuilt.expression))),
        "sample_table_exact_match": _same(samples, rebuilt.samples, list(samples.columns), ["sample_id"]),
        "drug_table_exact_match": _same(drugs, rebuilt.drugs, list(drugs.columns), ["drug_id"]),
        "pair_table_exact_match": _same(pairs, rebuilt.pairs, ["sample_id", "drug_id", "drug_name", "response", "higher_is_sensitive", "response_name"], ["sample_id", "drug_id"]),
        "sample_and_gene_order_exact_match": sample_ids == rebuilt.sample_ids and gene_ids == rebuilt.gene_ids,
    }
    paired = set(pairs.sample_id)
    identity = {str(k): int(v) for k, v in samples.identity_status.value_counts().items()}
    conflicts = drugs[drugs.drug_id.ne(drugs.inchikey) | drugs.drug_id.isna() | drugs.canonical_isomeric_smiles.isna()]
    ox = [x for x in rebuilt.exclusions if x.get("drug_name") == "Oxaliplatin" and x.get("reason") == "unsupported_coordination_representation"]
    passed = (
        checks["expression_max_abs_error"] == 0.0
        and all(value for key, value in checks.items() if key != "expression_max_abs_error")
        and len(samples) == 92 and len(paired) == 91 and len(drugs) == 48
        and len(pairs) == 3183 and conflicts.empty and len(ox) == 1
    )
    return {
        "cohort_id": "mcrc_pdo_v3", "source": "https://data.mendeley.com/datasets/hr94h42xdc/3", "source_license": "CC BY 4.0",
        "original_file_sha256": source_hashes, "legacy_name_cid_mapping_sha256": sha256(legacy), "fixed_isomeric_pubchem_sha256": sha256(fixed),
        "output_sha256": {p.name: sha256(p) for p in (processed / "samples.parquet", processed / "drugs.parquet", processed / "pairs.parquet", processed / "indexed.npz")},
        "rebuild_checks": checks,
        "coverage": {"expression_samples": len(samples), "response_paired_samples": len(paired), "response_paired_patients": int(samples[samples.sample_id.isin(paired)].patient_id.nunique()), "patients": int(samples.patient_id.nunique()), "genes": int(expression.shape[1]), "accepted_single_drugs": len(drugs), "response_pairs": len(pairs), "samples_without_response": sorted(set(samples.sample_id) - paired), "identity_evidence_counts": identity},
        "drug_mapping": {"accepted_full_inchikey_and_isomeric_smiles": len(drugs), "identity_conflict_count": len(conflicts), "oxaliplatin_status": "quarantined_unsupported_coordination_representation"},
        "reconstruction_status": "PASS" if passed else "REVIEW_REQUIRED",
        "scientific_admission": "PENDING_G0_IDENTITY_AND_ASSAY_REVIEW" if passed else "NOT_ADMITTED",
        "allowed_use": ["同一研究内审计同药跨 PDO 排序的可行性和覆盖规模"],
        "not_yet_supported": ["独立外部验证或临床治疗结论", "32 个仅由样本名前缀推导的身份", "未经曲线重拟合的剂量反应结论", "Oxaliplatin 直至配位化合物表示完成"],
    }
