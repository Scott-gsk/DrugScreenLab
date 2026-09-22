"""Freeze the Kryeziu (mcrc_v3) data contract: identity, drugs, admission, expression manifest, FROZEN split.

Formal script (decision-bearing). Promotes the DRAFT assets into frozen files,
following ``docs/reports/split_freeze_checklist_DRAFT.md`` (freeze-time gaps):

1. Patient identity table  -> data/processed/identity/kryeziu_patient_identity_FROZEN.json
   (92 per-sample records from the identity audit: 60 S1-verified / 32 prefix-only pending,
   with patient-level aggregation and full provenance checksums)
2. Drug identity table     -> data/processed/identity/kryeziu_drug_identity_48.json (+ .parquet)
   (48 structure-confirmed monotherapies; exclusions frozen: combination/biologic drugs,
   source metadata columns, Oxaliplatin quarantine)
3. Target admission        -> data/processed/identity/transfer/kryeziu_mcrc_target_admission.json
   (51 patients / 60 samples cohort with all conditional limitations preserved;
   status promoted to DEVELOPMENT_ADMITTED; points at the FROZEN split)
4. Expression manifest     -> data/splits/kryeziu_expression_manifest_FROZEN.json
   (pins indexed.npz sha256, genome-wide 19421 genes, and the frozen
   GDSC-Kryeziu intersection feature space: 19091 genes,
   gene_order_sha256=0fd5fdd7ab9d53d569bef5c2a91b508c62d6cb9d1000bc3aeac6f5d424424b39)
5. Frozen split            -> data/splits/kryeziu_patient_outer5fold_FROZEN.json
   (outer 5 folds + materialized inner val lists + freeze metadata and checksums)
6. Leakage re-check on the frozen folds via ``check_split_leakage`` (5/5 must pass),
   plus sample-level overlap and sample->patient consistency checks (dry-run parity).
7. Freeze receipt          -> data/splits/kryeziu_data_contract_freeze_receipt_20260919.json

All DRAFT sources are kept untouched as read-only copies. No file under
data/raw is read or written.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drug_screen.evaluation.protocols import check_split_leakage  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))
FROZEN_AT = dt.datetime.now(TZ).isoformat(timespec="seconds")

SPLIT_DRAFT = PROJECT_ROOT / "data/splits/kryeziu_patient_outer5fold_DRAFT.json"
SPLIT_FROZEN = PROJECT_ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
RECEIPT_INNER_VAL = PROJECT_ROOT / "data/splits/kryeziu_inner_val_materialization_receipt.json"
EXPR_MANIFEST_OUT = PROJECT_ROOT / "data/splits/kryeziu_expression_manifest_FROZEN.json"
FREEZE_RECEIPT_OUT = PROJECT_ROOT / "data/splits/kryeziu_data_contract_freeze_receipt_20260919.json"

IDENTITY_OUT = PROJECT_ROOT / "data/processed/identity/kryeziu_patient_identity_FROZEN.json"
DRUG_JSON_OUT = PROJECT_ROOT / "data/processed/identity/kryeziu_drug_identity_48.json"
DRUG_PARQUET_OUT = PROJECT_ROOT / "data/processed/identity/kryeziu_drug_identity_48.parquet"
DRUG_JSON_DRAFT = PROJECT_ROOT / "data/processed/identity/kryeziu_drug_identity_48_DRAFT.json"
DRUG_PARQUET_DRAFT = PROJECT_ROOT / "data/processed/identity/kryeziu_drug_identity_48_DRAFT.parquet"

ADMISSION_DRAFT = PROJECT_ROOT / "data/processed/identity/transfer/kryeziu_mcrc_target_admission_DRAFT.json"
ADMISSION_OUT = PROJECT_ROOT / "data/processed/identity/transfer/kryeziu_mcrc_target_admission.json"

AUDIT = PROJECT_ROOT / "data/artifacts/resource_acquisition_20260916/kryeziu_audit/kryeziu_identity_assay_audit.json"
MCRC_ROOT = PROJECT_ROOT / "data/processed/transfer/mcrc_v3"
MCRC_MANIFEST = MCRC_ROOT / "manifest.json"
KRYEZIU_NPZ = MCRC_ROOT / "indexed.npz"
GDSC_NPZ = PROJECT_ROOT / "data/processed/exp014/gdsc_v3/indexed.npz"

GENE_CONTRACT = PROJECT_ROOT / "data/processed/identity/transfer/gdsc_kryeziu_gene_space_contract_DRAFT.json"
GENE_LIST = PROJECT_ROOT / "data/processed/identity/transfer/gdsc_kryeziu_gene_intersection_ordered_DRAFT.txt"
GENE_INDEX_MAP = PROJECT_ROOT / "data/processed/identity/transfer/gdsc_kryeziu_gene_index_map_DRAFT.json"

TRANSFER_COHORTS = PROJECT_ROOT / "data/registry/transfer_cohorts.json"
SPLIT_FREEZE_REPORT_REL = "docs/reports/split_freeze_report_20260919.md"

EXPECTED = {
    "gene_order_sha256": "0fd5fdd7ab9d53d569bef5c2a91b508c62d6cb9d1000bc3aeac6f5d424424b39",
    "kryeziu_npz_sha256": "ddbab601e461169a7385ac0658a433a7735581e5e9b99bdfd062e9bec5f4c727",
    "gdsc_npz_sha256": "2b27c47b72f53d01d1acc609f13f3b6dac96f92542acdc128fb9d22f3a6b68c7",
    "n_intersection_genes": 19091,
    "n_genome_wide_genes": 19421,
    "n_samples": 92,
    "n_drugs": 48,
    "n_verified_patients": 51,
    "n_verified_samples": 60,
    "n_pending_samples": 32,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp_freeze")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(messages: list[str]) -> None:
    raise SystemExit("FREEZE ABORTED — verification failures:\n  " + "\n  ".join(messages))


def verify_inputs() -> dict:
    """Verify every pinned input before writing any frozen asset."""
    issues: list[str] = []
    checks: dict[str, object] = {}

    split = load_json(SPLIT_DRAFT)
    receipt = load_json(RECEIPT_INNER_VAL)
    checks["split_draft_sha256"] = sha256_file(SPLIT_DRAFT)
    checks["inner_val_receipt_sha256"] = sha256_file(RECEIPT_INNER_VAL)

    if split.get("n_folds") != 5:
        issues.append(f"split n_folds != 5: {split.get('n_folds')}")
    if not receipt.get("all_folds_ok"):
        issues.append("inner-val materialization receipt not all_folds_ok")
    if split.get("status") != "DRAFT":
        issues.append(f"unexpected split status: {split.get('status')}")
    for fold in split["folds"]:
        if not fold.get("val_patient_ids"):
            issues.append(f"fold {fold['fold']} has empty val_patient_ids (run materialize_kryeziu_inner_val.py first)")

    # expression matrix + gene intersection contract
    kryeziu_sha = sha256_file(KRYEZIU_NPZ)
    checks["kryeziu_indexed_npz_sha256"] = kryeziu_sha
    if kryeziu_sha != EXPECTED["kryeziu_npz_sha256"]:
        issues.append(f"kryeziu indexed.npz sha256 drifted: {kryeziu_sha}")
    gene_sha = sha256_file(GENE_LIST)
    checks["gene_intersection_sha256"] = gene_sha
    if gene_sha != EXPECTED["gene_order_sha256"]:
        issues.append(f"gene intersection list sha256 drifted: {gene_sha}")

    kz = np.load(KRYEZIU_NPZ, allow_pickle=False)
    kz_genes = [str(g) for g in kz["ordered_genes"]]
    kz_samples = [str(s) for s in kz["sample_id"]]
    checks["kryeziu_npz_shape"] = list(kz["expression"].shape)
    if kz["expression"].shape != (EXPECTED["n_samples"], EXPECTED["n_genome_wide_genes"]):
        issues.append(f"kryeziu expression shape: {kz['expression'].shape}")
    if len(set(kz_samples)) != EXPECTED["n_samples"]:
        issues.append("kryeziu sample_id not unique")

    intersection = GENE_LIST.read_text(encoding="utf-8").split()
    checks["n_intersection_genes"] = len(intersection)
    if len(intersection) != EXPECTED["n_intersection_genes"]:
        issues.append(f"intersection gene count: {len(intersection)}")
    if not set(intersection).issubset(set(kz_genes)):
        missing = sorted(set(intersection) - set(kz_genes))[:5]
        issues.append(f"intersection genes absent from kryeziu matrix: {missing}")

    gdsc_sha = sha256_file(GDSC_NPZ)
    checks["gdsc_indexed_npz_sha256"] = gdsc_sha
    if gdsc_sha != EXPECTED["gdsc_npz_sha256"]:
        issues.append(f"gdsc indexed.npz sha256 drifted: {gdsc_sha}")
    gd = np.load(GDSC_NPZ, allow_pickle=False)
    gd_genes = {str(g) for g in gd["ordered_genes"]}
    if not set(intersection).issubset(gd_genes):
        missing = sorted(set(intersection) - gd_genes)[:5]
        issues.append(f"intersection genes absent from gdsc matrix: {missing}")

    # audit + counts
    audit_sha = sha256_file(AUDIT)
    checks["identity_audit_sha256"] = audit_sha
    audit = load_json(AUDIT)
    per_sample = audit["identity_audit"]["identity_by_expression_sample"]
    verified = [r for r in per_sample if r["identity_status"] == "S1_PDO_patient_verified"]
    pending = [r for r in per_sample if r["identity_status"] == "pending_G0_identity_verification"]
    checks["n_verified_samples"] = len(verified)
    checks["n_pending_samples"] = len(pending)
    checks["n_verified_patients"] = len({r["patient_id"] for r in verified})
    checks["n_pending_patients"] = len({r["patient_id"] for r in pending})
    if len(verified) != EXPECTED["n_verified_samples"] or len(pending) != EXPECTED["n_pending_samples"]:
        issues.append(f"audit sample split: verified={len(verified)} pending={len(pending)}")
    if len({r["patient_id"] for r in verified}) != EXPECTED["n_verified_patients"]:
        issues.append(f"verified patients != 51: {len({r['patient_id'] for r in verified})}")
    if audit["identity_audit"].get("status") != "CONDITIONAL":
        issues.append("identity audit status changed from CONDITIONAL")

    # admission DRAFT cohort consistency vs frozen split
    admission = load_json(ADMISSION_DRAFT)
    if admission.get("n_admitted_patients") != EXPECTED["n_verified_patients"] or admission.get("n_admitted_samples") != EXPECTED["n_verified_samples"]:
        issues.append("admission DRAFT cohort counts drifted from 51/60")
    fold_test_patients = [p for f in split["folds"] for p in f["test_patient_ids"]]
    if sorted(fold_test_patients) != sorted(admission["admitted_patient_ids"]):
        issues.append("split test coverage != admitted patients (51)")
    if len(fold_test_patients) != len(set(fold_test_patients)):
        issues.append("patients repeated across outer test folds")

    # drug table draft: json vs parquet consistency
    drug_draft = load_json(DRUG_JSON_DRAFT)
    if drug_draft.get("n_drugs") != EXPECTED["n_drugs"]:
        issues.append(f"drug draft n_drugs: {drug_draft.get('n_drugs')}")
    if drug_draft.get("quarantine", {}).get("Oxaliplatin", {}).get("in_48_table") is not False:
        issues.append("Oxaliplatin quarantine policy changed in drug draft")
    import pandas as pd
    drug_parquet_draft = pd.read_parquet(DRUG_PARQUET_DRAFT)
    if len(drug_parquet_draft) != EXPECTED["n_drugs"]:
        issues.append(f"drug draft parquet rows: {len(drug_parquet_draft)}")
    json_ids = {d["drug_id"] for d in drug_draft["drugs"]}
    parquet_ids = set(drug_parquet_draft["drug_id"].astype(str))
    if json_ids != parquet_ids:
        issues.append("drug draft json/parquet drug_id sets differ")
    checks["drug_draft_json_sha256"] = sha256_file(DRUG_JSON_DRAFT)
    checks["drug_draft_parquet_sha256"] = sha256_file(DRUG_PARQUET_DRAFT)

    if issues:
        fail(issues)
    return {
        "split": split,
        "receipt": receipt,
        "audit": audit,
        "audit_sha256": audit_sha,
        "admission_draft": admission,
        "drug_draft": drug_draft,
        "per_sample": per_sample,
        "verified": verified,
        "pending": pending,
        "intersection": intersection,
        "checks": checks,
    }


def build_identity_table(ctx: dict) -> dict:
    per_sample = [
        {
            "sample_id": r["sample_id"],
            "patient_id": r["patient_id"],
            "identity_status": r["identity_status"],
            "patient_id_provenance": r["patient_id_provenance"],
            "source_sample_types": r.get("source_sample_types", ""),
        }
        for r in sorted(ctx["per_sample"], key=lambda r: r["sample_id"])
    ]
    verified_patients = sorted({r["patient_id"] for r in ctx["verified"]})
    pending_patients = sorted({r["patient_id"] for r in ctx["pending"]})
    return {
        "name": "kryeziu_patient_identity",
        "status": "FROZEN",
        "frozen_at": FROZEN_AT,
        "cohort_id": "kryeziu_mcrc_v3",
        "split_unit": "patient_id",
        "counts": {
            "n_expression_samples": len(per_sample),
            "n_verified_samples": len(ctx["verified"]),
            "n_pending_samples": len(ctx["pending"]),
            "n_verified_patients": len(verified_patients),
            "n_pending_prefix_only_patients": len(pending_patients),
        },
        "verified_patients": verified_patients,
        "pending_prefix_only_patients": pending_patients,
        "per_sample": per_sample,
        "exclusion_policy": {
            "verified_cohort": "S1_PDO patient verified (Data S1 patient column + PDO sample_type): 51 patients / 60 samples admitted to the frozen patient-grouped evaluation",
            "prefix_only_excluded": "32 samples whose patient grouping is only sample-prefix derived stay EXCLUDED from the frozen cohort until an independent PDO-patient crosswalk is checksummed (audit release condition)",
        },
        "provenance": {
            "identity_audit": str(AUDIT.relative_to(PROJECT_ROOT)),
            "identity_audit_sha256": ctx["audit_sha256"],
            "identity_audit_status": ctx["audit"]["identity_audit"]["status"],
            "audit_decision_admission": ctx["audit"]["decision"]["admission"],
            "processed_samples_parquet": "data/processed/transfer/mcrc_v3/samples.parquet",
            "processed_manifest_sha256": ctx["admission_draft"]["processed_manifest_sha256"],
            "split_frozen": str(SPLIT_FROZEN.relative_to(PROJECT_ROOT)),
            "generated_by": "scripts/data/freeze_kryeziu_data_contract.py",
            "generated_at": FROZEN_AT,
        },
        "notes": [
            "Formal promotion of the DRAFT verified/pending lists (split_freeze_checklist item: 身份表正式版).",
            "The 32 prefix-only samples keep identity_status=pending_G0_identity_verification and must not enter automated training before an independent crosswalk (audit forbidden clause).",
        ],
    }


def build_drug_table(ctx: dict) -> tuple[dict, dict]:
    draft = ctx["drug_draft"]
    mcrc_manifest = load_json(MCRC_MANIFEST)
    exclusions_raw = mcrc_manifest.get("exclusions", [])
    combos, metadata_cols, oxali = [], [], None
    for item in exclusions_raw:
        name = item.get("drug_name")
        if not name:
            continue
        reason = item.get("reason")
        if reason == "combination_or_biologic":
            combos.append(name)
        elif reason == "source_metadata_column":
            metadata_cols.append(name)
        elif name == "Oxaliplatin":
            oxali = item
    if oxali is None:
        fail(["Oxaliplatin exclusion entry not found in mcrc_v3 manifest"])
    formal = {
        "name": "kryeziu_drug_identity_48",
        "status": "FROZEN",
        "contract_version": "frozen-1.0",
        "frozen_at": FROZEN_AT,
        "n_drugs": draft["n_drugs"],
        "structure_status_default": draft["structure_status_default"],
        "scheme_ref": draft["scheme_ref"],
        "provenance": {
            **draft["provenance"],
            "promoted_from": str(DRUG_JSON_DRAFT.relative_to(PROJECT_ROOT)),
            "promoted_from_sha256": ctx["checks"]["drug_draft_json_sha256"],
            "mcrc_v3_manifest": str(MCRC_MANIFEST.relative_to(PROJECT_ROOT)),
            "mcrc_v3_manifest_sha256": sha256_file(MCRC_MANIFEST),
            "identity_audit_sha256": ctx["audit_sha256"],
            "generated_by": "scripts/data/freeze_kryeziu_data_contract.py",
            "generated_at": FROZEN_AT,
        },
        "quarantine": {
            "Oxaliplatin": {
                **draft["quarantine"]["Oxaliplatin"],
                "manifest_reasons": ["unsupported_coordination_representation (mixture)", "quarantined_fixed_structure"],
                "policy": "QUARANTINED for SMILES graph inputs per kryeziu audit; requires an identity-preserving coordination-complex representation before any release",
            }
        },
        "exclusions_frozen": {
            "combination_or_biologic": sorted(set(combos)),
            "source_metadata_columns": sorted(set(metadata_cols)),
            "oxaliplatin_quarantined": ["Oxaliplatin"],
        },
        "drugs": draft["drugs"],
        "notes": [
            "48 retained monotherapies with fixed PubChem CID + full InChIKey + canonical isomeric SMILES (unchanged from audited DRAFT content).",
            "Formal promotion closes split_freeze_checklist item: 药物表正式版（含 exclusions 冻结）.",
            "The companion .parquet carries the same 48 rows byte-identical to the audited DRAFT parquet.",
        ],
    }
    return formal, mcrc_manifest


def build_expression_manifest(ctx: dict) -> dict:
    contract = load_json(GENE_CONTRACT)
    kz = np.load(KRYEZIU_NPZ, allow_pickle=False)
    return {
        "name": "kryeziu_expression_manifest",
        "status": "FROZEN",
        "frozen_at": FROZEN_AT,
        "cohort_id": "kryeziu_mcrc_v3",
        "expression_matrix": {
            "path": str(KRYEZIU_NPZ.relative_to(PROJECT_ROOT)),
            "sha256": ctx["checks"]["kryeziu_indexed_npz_sha256"],
            "format": "npz(expression, sample_id, ordered_genes)",
            "shape": [int(kz["expression"].shape[0]), int(kz["expression"].shape[1])],
            "dtype": str(kz["expression"].dtype),
            "units": "log2(CPM + 1), full-library totals, duplicate HGNC rows summed",
            "n_samples": EXPECTED["n_samples"],
            "n_genome_wide_genes": EXPECTED["n_genome_wide_genes"],
            "mcrc_v3_manifest": str(MCRC_MANIFEST.relative_to(PROJECT_ROOT)),
        },
        "feature_space": {
            "genome_wide": {
                "n_genes": EXPECTED["n_genome_wide_genes"],
                "rule": "any top-k gene selection must be fit on training folds only; never fit on outer test",
            },
            "transfer_intersection": {
                "feature_space": contract["feature_space"],
                "n_genes": EXPECTED["n_intersection_genes"],
                "gene_order_contract_id": contract["gene_order_contract_id"],
                "gene_order_sha256": ctx["checks"]["gene_intersection_sha256"],
                "verified_against_expected": ctx["checks"]["gene_intersection_sha256"] == EXPECTED["gene_order_sha256"],
                "gene_list_path": str(GENE_LIST.relative_to(PROJECT_ROOT)),
                "index_map_path": str(GENE_INDEX_MAP.relative_to(PROJECT_ROOT)),
                "source_contract": str(GENE_CONTRACT.relative_to(PROJECT_ROOT)),
                "source_contract_sha256": sha256_file(GENE_CONTRACT),
                "gdsc_expression_path": contract["source"]["expression_path"],
                "gdsc_expression_sha256": ctx["checks"]["gdsc_indexed_npz_sha256"],
                "subset_rule": contract["target"]["subset_rule"],
                "units_note": contract["units_note"],
            },
        },
        "verification": {
            "intersection_genes_present_in_kryeziu": True,
            "intersection_genes_present_in_gdsc": True,
            "gene_order_sha256_matches_pinned_contract": ctx["checks"]["gene_intersection_sha256"] == EXPECTED["gene_order_sha256"],
            "kryeziu_npz_sha256_matches_mcrc_v3_manifest": ctx["checks"]["kryeziu_indexed_npz_sha256"] == EXPECTED["kryeziu_npz_sha256"],
            "gdsc_npz_sha256_matches_gene_space_contract": ctx["checks"]["gdsc_indexed_npz_sha256"] == EXPECTED["gdsc_npz_sha256"],
        },
        "provenance": {
            "mcrc_v3_manifest_sha256": sha256_file(MCRC_MANIFEST),
            "generated_by": "scripts/data/freeze_kryeziu_data_contract.py",
            "generated_at": FROZEN_AT,
        },
        "notes": [
            "Companion manifest to data/splits/kryeziu_patient_outer5fold_FROZEN.json (same directory; cross-referenced).",
            "Closes split_freeze_checklist item: 表达矩阵版本钉死（禁止静默换基因集）.",
            "not_978_default: LINCS exact-978 space is NOT imposed on the PDO task; the PDO contract space is genome-wide or the 19091-gene intersection above.",
        ],
    }


def run_leakage_checks(split: dict) -> list[dict]:
    results = []
    for fold in split["folds"]:
        train_p = [str(x) for x in fold["train_patient_ids"]]
        val_p = [str(x) for x in fold["val_patient_ids"]]
        test_p = [str(x) for x in fold["test_patient_ids"]]
        train_inner_p = sorted(set(train_p) - set(val_p))
        train_s = {str(x) for x in fold["train_sample_ids"]}
        val_s = {str(x) for x in fold["val_sample_ids"]}
        test_s = {str(x) for x in fold["test_sample_ids"]}

        report = check_split_leakage(
            train_patients=train_inner_p,
            test_patients=test_p,
            allow_drug_overlap=True,
            extra_notes=[f"fold {fold['fold']} frozen re-check (outer test vs inner train)"],
        )
        report_val = check_split_leakage(
            train_patients=train_inner_p,
            test_patients=val_p,
            allow_drug_overlap=True,
            extra_notes=[f"fold {fold['fold']} frozen re-check (inner val vs inner train)"],
        )

        issues: list[str] = []
        if not (set(val_p) <= set(train_p)):
            issues.append("val patients not subset of outer train")
        if set(val_p) & set(test_p):
            issues.append("val patients intersect outer test")
        if set(train_inner_p) & set(val_p):
            issues.append("inner train/val patient overlap")
        if train_s & test_s:
            issues.append("train/test sample overlap")
        if val_s & test_s:
            issues.append("val/test sample overlap")
        if not (val_s <= train_s):
            issues.append("val samples not subset of outer train samples")
        # NOTE: val samples intentionally live inside the outer train_sample_ids
        # (nested split definition); inner-train samples = train_sample_ids minus
        # val_sample_ids, to be excluded by the training side.
        for side, pids, sids in (("train", train_p, train_s), ("val", val_p, val_s), ("test", test_p, test_s)):
            owners = {s.rsplit("_t", 1)[0] for s in sids}
            if not owners <= set(pids):
                issues.append(f"{side} sample->patient mismatch")
            if len(sids) != len(fold[f"{side}_sample_ids"]):
                issues.append(f"{side} sample duplicates")

        ok = report.ok and report_val.ok and not issues
        results.append({
            "fold": fold["fold"],
            "outer": {"train_patients": len(train_p), "val_patients": len(val_p), "test_patients": len(test_p)},
            "inner": {"train_inner_patients": len(train_inner_p), "val_patients": len(val_p)},
            "samples": {
                "train_outer": len(train_s), "val": len(val_s), "test": len(test_s),
                "train_inner": len(train_s) - len(val_s),
            },
            "check_split_leakage_outer_test_vs_inner_train": report.as_dict(),
            "check_split_leakage_inner_val_vs_inner_train": report_val.as_dict(),
            "extra_sample_level_checks": {"ok": not issues, "issues": issues},
            "ok": ok,
        })
    return results


def main() -> int:
    ctx = verify_inputs()
    print("input verification: PASS")

    # 1) identity table
    identity = build_identity_table(ctx)
    atomic_write_json(IDENTITY_OUT, identity)
    identity_sha = sha256_file(IDENTITY_OUT)
    print(f"identity table frozen: {IDENTITY_OUT.name} sha256={identity_sha}")

    # 2) drug table (json + byte-identical parquet copy of the audited DRAFT)
    drug, mcrc_manifest = build_drug_table(ctx)
    atomic_write_json(DRUG_JSON_OUT, drug)
    shutil.copyfile(DRUG_PARQUET_DRAFT, DRUG_PARQUET_OUT)
    drug_json_sha = sha256_file(DRUG_JSON_OUT)
    drug_parquet_sha = sha256_file(DRUG_PARQUET_OUT)
    print(f"drug table frozen: {DRUG_JSON_OUT.name} sha256={drug_json_sha}; parquet sha256={drug_parquet_sha}")

    # 3) expression manifest
    expr_manifest = build_expression_manifest(ctx)
    atomic_write_json(EXPR_MANIFEST_OUT, expr_manifest)
    expr_manifest_sha = sha256_file(EXPR_MANIFEST_OUT)
    print(f"expression manifest frozen: {EXPR_MANIFEST_OUT.name} sha256={expr_manifest_sha}")

    # 4) FROZEN split
    split = json.loads(json.dumps(ctx["split"]))  # deep copy of the materialized DRAFT
    receipt = ctx["receipt"]
    split["status"] = "FROZEN"
    split["random_seed"] = int(receipt["random_seed"])
    split["notes"] = list(split["notes"]) + [
        f"FROZEN {FROZEN_AT}: inner val materialized per pinned algorithm (see inner_val.algorithm_ref); random_seed pinned in-file.",
        "DRAFT file retained read-only at data/splits/kryeziu_patient_outer5fold_DRAFT.json.",
    ]
    split["inner_val"] = {
        "algorithm_ref": receipt["algorithm_ref"],
        "algorithm": receipt["algorithm"],
        "random_seed": int(receipt["random_seed"]),
        "seed_source": receipt["seed_source"],
        "materialization_receipt": str(RECEIPT_INNER_VAL.relative_to(PROJECT_ROOT)),
        "materialization_receipt_sha256": ctx["checks"]["inner_val_receipt_sha256"],
        "per_fold": [
            {
                "fold": fr["fold"],
                "n_train_inner_patients": fr["n_train_inner_patients"],
                "n_val_patients": fr["n_val_patients"],
                "n_train_inner_samples": fr["n_train_inner_samples"],
                "n_val_samples": fr["n_val_samples"],
                "inner_val_skipped": fr["inner_val_skipped"],
            }
            for fr in receipt["folds"]
        ],
    }
    split["blocker_resolution"] = {
        "IDENTITY_PARTIAL": "standing constraint — 32 prefix-only samples stay excluded (frozen identity table records both cohorts); release requires checksummed independent crosswalk",
        "INNER_VAL_UNSPECIFIED": "resolved — val lists materialized in every fold per the pinned algorithm (seed 20260918)",
        "DRUG_SET_OXALIPLATIN": "standing constraint — Oxaliplatin stays quarantined in the frozen 48-drug table",
    }
    split["freeze"] = {
        "frozen_at": FROZEN_AT,
        "frozen_by": "DSL-数据管家 (scripts/data/freeze_kryeziu_data_contract.py)",
        "freeze_report": SPLIT_FREEZE_REPORT_REL,
        "expression_manifest": str(EXPR_MANIFEST_OUT.relative_to(PROJECT_ROOT)),
        "expression_manifest_sha256": expr_manifest_sha,
        "patient_identity_table": str(IDENTITY_OUT.relative_to(PROJECT_ROOT)),
        "patient_identity_table_sha256": identity_sha,
        "drug_identity_table": str(DRUG_JSON_OUT.relative_to(PROJECT_ROOT)),
        "drug_identity_table_sha256": drug_json_sha,
        "drug_identity_parquet": str(DRUG_PARQUET_OUT.relative_to(PROJECT_ROOT)),
        "drug_identity_parquet_sha256": drug_parquet_sha,
        "target_admission": str(ADMISSION_OUT.relative_to(PROJECT_ROOT)),
        "target_admission_note": "admission sha256 registered in datasets.json / freeze report (path-referenced here to avoid a circular hash dependency)",
        "draft_copy_sha256": ctx["checks"]["split_draft_sha256"],
        "identity_audit_sha256": ctx["audit_sha256"],
        "kryeziu_indexed_npz_sha256": ctx["checks"]["kryeziu_indexed_npz_sha256"],
        "gene_intersection_sha256": ctx["checks"]["gene_intersection_sha256"],
    }
    atomic_write_json(SPLIT_FROZEN, split)
    split_frozen_sha = sha256_file(SPLIT_FROZEN)
    print(f"split frozen: {SPLIT_FROZEN.name} sha256={split_frozen_sha}")

    # 5) admission formal
    admission = json.loads(json.dumps(ctx["admission_draft"]))
    admission["status"] = "DEVELOPMENT_ADMITTED"
    admission["scope"] = "Frozen target contract for GDSC->Kryeziu PDO transfer; patient-grouped development evaluation on S1-verified cohort; not clinical, not external validation"
    admission["split_path"] = str(SPLIT_FROZEN.relative_to(PROJECT_ROOT))
    admission["split_sha256"] = split_frozen_sha
    admission["split_status"] = "FROZEN"
    admission["drug_identity_path"] = str(DRUG_JSON_OUT.relative_to(PROJECT_ROOT))
    admission["drug_identity_sha256"] = drug_json_sha
    admission["patient_identity_table"] = str(IDENTITY_OUT.relative_to(PROJECT_ROOT))
    admission["patient_identity_table_sha256"] = identity_sha
    admission["expression_manifest"] = str(EXPR_MANIFEST_OUT.relative_to(PROJECT_ROOT))
    admission["expression_manifest_sha256"] = expr_manifest_sha
    admission["registry_sha256"] = sha256_file(TRANSFER_COHORTS)
    admission["frozen_at"] = FROZEN_AT
    admission["status_history"] = {
        "previous": ctx["admission_draft"].get("status", "EXPLORATORY_ADMITTED"),
        "changed_to": "DEVELOPMENT_ADMITTED",
        "reason": "data contract frozen (split/identity/drug/expression manifest); conditional limitations unchanged",
        "changed_at": FROZEN_AT,
    }
    admission["notes"] = list(admission["notes"]) + [
        f"FROZEN {FROZEN_AT}: 51 patients / 60 samples cohort with conditional limitations preserved; split is frozen (inner val materialized, seed 20260918).",
        "Conditional restrictions carried over verbatim: only S1-verified patients/samples admitted; 32 prefix-only excluded; Oxaliplatin quarantined; no clinical or external-validation claims.",
    ]
    atomic_write_json(ADMISSION_OUT, admission)
    admission_sha = sha256_file(ADMISSION_OUT)
    print(f"admission frozen: {ADMISSION_OUT.name} sha256={admission_sha}")

    # 6) leakage re-check on the frozen file as written
    frozen_split = load_json(SPLIT_FROZEN)
    leakage = run_leakage_checks(frozen_split)
    all_ok = all(r["ok"] for r in leakage)
    for r in leakage:
        print(
            f"leakage fold {r['fold']}: outer {r['outer']['train_patients']}/{r['outer']['val_patients']}/{r['outer']['test_patients']} "
            f"inner {r['inner']['train_inner_patients']}/{r['inner']['val_patients']} ok={r['ok']}"
        )
    print(f"leakage re-check 5/5: {'PASS' if all_ok and len(leakage) == 5 else 'FAIL'}")

    # 7) freeze receipt
    freeze_receipt = {
        "name": "kryeziu_data_contract_freeze",
        "frozen_at": FROZEN_AT,
        "frozen_by": "DSL-数据管家 (scripts/data/freeze_kryeziu_data_contract.py)",
        "freeze_report": SPLIT_FREEZE_REPORT_REL,
        "all_ok": all_ok,
        "input_verification": ctx["checks"],
        "frozen_assets": {
            "split_frozen": {"path": str(SPLIT_FROZEN.relative_to(PROJECT_ROOT)), "sha256": split_frozen_sha},
            "patient_identity": {"path": str(IDENTITY_OUT.relative_to(PROJECT_ROOT)), "sha256": identity_sha},
            "drug_identity_json": {"path": str(DRUG_JSON_OUT.relative_to(PROJECT_ROOT)), "sha256": drug_json_sha},
            "drug_identity_parquet": {"path": str(DRUG_PARQUET_OUT.relative_to(PROJECT_ROOT)), "sha256": drug_parquet_sha},
            "target_admission": {"path": str(ADMISSION_OUT.relative_to(PROJECT_ROOT)), "sha256": admission_sha},
            "expression_manifest": {"path": str(EXPR_MANIFEST_OUT.relative_to(PROJECT_ROOT)), "sha256": expr_manifest_sha},
        },
        "retained_draft_copies": {
            "split_draft_materialized": {"path": str(SPLIT_DRAFT.relative_to(PROJECT_ROOT)), "sha256": ctx["checks"]["split_draft_sha256"]},
            "drug_identity_draft_json": {"path": str(DRUG_JSON_DRAFT.relative_to(PROJECT_ROOT)), "sha256": ctx["checks"]["drug_draft_json_sha256"]},
            "drug_identity_draft_parquet": {"path": str(DRUG_PARQUET_DRAFT.relative_to(PROJECT_ROOT)), "sha256": ctx["checks"]["drug_draft_parquet_sha256"]},
            "admission_draft": {"path": str(ADMISSION_DRAFT.relative_to(PROJECT_ROOT))},
        },
        "inner_val": {
            "random_seed": int(receipt["random_seed"]),
            "seed_source": receipt["seed_source"],
            "per_fold": split["inner_val"]["per_fold"],
        },
        "leakage_recheck": leakage,
    }
    atomic_write_json(FREEZE_RECEIPT_OUT, freeze_receipt)
    print(f"freeze receipt: {FREEZE_RECEIPT_OUT.name}")

    if not all_ok:
        print("FREEZE INCOMPLETE: leakage re-check failed", file=sys.stderr)
        return 2
    print("FREEZE COMPLETE: all assets written, leakage re-check 5/5 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
