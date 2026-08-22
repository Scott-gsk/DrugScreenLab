"""Response-blind fail_rule audit for EXP-007 ORACLE_NEAR_NULL.

Inspects four pre-registered sources only: disease signature, dose/time,
context identity/support, and PRISM endpoint.  Does not retrain, does not
change rank metrics, and does not reselect pairs using response values.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIGNATURE = ROOT / "mvp/core_data/crc_disease_signature_exact978.tsv"
SIGNATURE_AUDIT = ROOT / "mvp/core_data/crc_disease_signature_audit.json"
GENE = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
INST = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_inst_info.txt.gz"
CELL = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz"
CONTEXT_REGISTRY = ROOT / "mvp/foundation/xpert/CONTEXT_REGISTRY.json"
PRISM = ROOT / "mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet"
PRISM_AUDIT = ROOT / "mvp/foundation/xpert/BROAD_PRISM_CRC_V1_AUDIT.json"
TREATMENT = ROOT / "mvp/core_data/_source_cache/primary-screen-replicate-collapsed-treatment-info.csv"
COVERAGE = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_COVERAGE.json"
RESULT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_RESULT.json"
PER_LINE = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_PER_LINE.json"
OUT = ROOT / "artifacts/experiments/EXP-007/ORACLE_NEAR_NULL_SOURCE_AUDIT.json"

CRC_EXACT = (
    "CL34",
    "HCT116",
    "HT29",
    "LOVO",
    "RKO",
    "SNUC4",
    "SNUC5",
    "SW480",
    "SW620",
    "SW948",
)
GENE_ORDER = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_signature() -> dict[str, Any]:
    signature = pd.read_csv(SIGNATURE, sep="\t")
    audit = json.loads(SIGNATURE_AUDIT.read_text(encoding="utf-8"))
    gene_info = pd.read_csv(GENE, sep="\t", dtype=str)
    landmarks = gene_info.loc[gene_info["pr_is_lm"].astype(str).eq("1"), "pr_gene_id"].astype(str)
    present = set(signature["pr_gene_id"].astype(str))
    missing = sorted(set(landmarks) - present)
    signed = pd.to_numeric(signature["signed_log2fc"], errors="coerce")
    return {
        "source_study": audit["source"]["study"],
        "platform": audit["source"]["platform"],
        "comparison": "tumor_vs_paired_normal_bulk_tissue",
        "not_lincs_l1000": True,
        "not_cell_line_basal": True,
        "paired_subjects": audit["sample_context"]["paired_subject_count"],
        "formula": audit["transformation"]["formula"],
        "source_semantics": audit["source"]["matrix_semantics"],
        "additional_normalization": audit["transformation"]["additional_normalization"],
        "output_sha256": audit["output"]["sha256"],
        "local_sha256": _file_sha256(SIGNATURE),
        "gene_order_sha256_registered": GENE_ORDER,
        "rows": int(len(signature)),
        "landmark_universe": 978,
        "missing_landmark_count": len(missing),
        "missing_landmark_gene_ids": missing,
        "up": int((signature["direction"].astype(str) == "up").sum()),
        "down": int((signature["direction"].astype(str) == "down").sum()),
        "zero": int((signed == 0).sum()),
        "up_down_ratio": float((signature["direction"].astype(str) == "up").sum())
        / max(int((signature["direction"].astype(str) == "down").sum()), 1),
        "abs_log2fc_median": float(np.nanmedian(np.abs(signed))),
        "abs_log2fc_q90": float(np.nanquantile(np.abs(signed), 0.90)),
        "abs_log2fc_max": float(np.nanmax(np.abs(signed))),
        "n_pairs_min": int(pd.to_numeric(signature["n_pairs"], errors="coerce").min()),
        "mismatch_notes": [
            "GSE74602 is Illumina HT-12 tissue microarray, not L1000 landmark assay",
            "tumor-vs-adjacent-normal in surgical CRC tissue is not the same state as LINCS CRC cell-line perturbation",
            "signature is global and unsigned-for-lineage; the same vector is scored on MSI and MSS lines",
            "712 up vs 235 down is a strong class imbalance for Spearman/KS reversal",
        ],
    }


def audit_dose_time() -> dict[str, Any]:
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    inst = pd.read_csv(
        INST,
        sep="\t",
        dtype=str,
        usecols=["cell_id", "pert_type", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"],
    )
    crc = inst.loc[inst["cell_id"].isin(CRC_EXACT)].copy()
    crc["dose"] = pd.to_numeric(crc["pert_dose"], errors="coerce")
    crc["time"] = pd.to_numeric(crc["pert_time"], errors="coerce")
    chem = crc.loc[crc["pert_type"].eq("trt_cp")]
    cond = (
        chem.assign(
            dose_bin=np.where(np.isclose(chem["dose"], 10.0, atol=1e-5), "10uM", "other_or_missing"),
            time_bin=np.where(
                np.isclose(chem["time"], 6.0, atol=1e-5),
                "6h",
                np.where(np.isclose(chem["time"], 24.0, atol=1e-5), "24h", "other"),
            ),
        )
        .groupby(["cell_id", "dose_bin", "time_bin"], dropna=False)
        .size()
        .reset_index(name="n_instances")
    )
    cond_table = {
        f"{row.cell_id}|{row.dose_bin}|{row.time_bin}": int(row.n_instances) for row in cond.itertuples(index=False)
    }
    return {
        "primary_condition": coverage["primary_condition"],
        "sensitivity_condition": coverage["sensitivity_condition"],
        "dose_time_not_mixed": result["evaluation_contract"]["dose_time_not_mixed"],
        "lincs_canonical": {
            "10uM_6h": coverage["canonical_conditions"]["10uM_6h"],
            "10uM_24h": coverage["canonical_conditions"]["10uM_24h"],
        },
        "crc_trt_cp_instance_counts_by_cell_dose_time": cond_table,
        "lines_with_10uM_24h": sorted(
            {
                key.split("|", 1)[0]
                for key, count in cond_table.items()
                if key.endswith("|10uM|24h") and count > 0
            }
        ),
        "mismatch_notes": [
            "PRIMARY Oracle is 10 µM / 6 h transcriptional snapshot",
            "24 h exists almost only in HT29 and was correctly kept as sensitivity, not mixed",
            "PRISM primary collapsed endpoint is 2.5 µM HTS viability, not 10 µM / 6 h transcription",
        ],
    }


def audit_context() -> dict[str, Any]:
    registry = json.loads(CONTEXT_REGISTRY.read_text(encoding="utf-8"))
    cell = pd.read_csv(CELL, sep="\t", dtype=str)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    exact = []
    for row in registry.get("contexts", []):
        if row.get("context_id") in CRC_EXACT:
            exact.append(
                {
                    "context_id": row.get("context_id"),
                    "broad_exact_context": bool(row.get("broad_exact_context")),
                    "broad_depmap_ids": row.get("broad_depmap_ids"),
                    "representative_dose_um": row.get("representative_dose_um"),
                    "representative_time_h": row.get("representative_time_h"),
                    "matched_control_record_count": row.get("matched_control_record_count"),
                }
            )
    keep = [col for col in ("cell_id", "cell_type", "primary_site", "subtype", "sample_type") if col in cell.columns]
    cell_crc = cell.loc[cell["cell_id"].isin(CRC_EXACT), keep].to_dict("records")
    cell_missing = sorted(set(CRC_EXACT) - set(cell["cell_id"].astype(str)))
    per_line = result["oracle"]["metrics_summary"]["per_line_top10"]
    shared = coverage["canonical_conditions"]["10uM_6h"]["per_context_unique_compounds"]
    non_ht29 = {cid: n for cid, n in shared.items() if cid != "HT29"}
    return {
        "crc_exact_contexts": list(CRC_EXACT),
        "registry_exact_rows": exact,
        "all_ten_marked_broad_exact": all(row["broad_exact_context"] for row in exact),
        "cell_info_missing_crc_exact": cell_missing,
        "cell_info_preview": cell_crc,
        "identity_pairs_before_prism": coverage["canonical_conditions"]["10uM_6h"]["unique_pairs_with_matched_control"],
        "pairs_after_prism_join": result["oracle"]["unique_pairs_after_prism_join"],
        "per_line_support": per_line,
        "non_ht29_unique_compounds": non_ht29,
        "ht29_unique_compounds": shared.get("HT29"),
        "support_imbalance": "HT29 holds the long-tail library; the other 9 lines are a shared core of ~76-82 compounds",
        "sw480_sw620_note": "SW480 and SW620 are the same patient primary/metastasis pair; they do not agree in sign or Top-10 lift",
        "mismatch_notes": [
            "context identity itself is exact and not the missing-line problem from the old 839-pair oracle",
            "support is not exchangeable across lines; 9/10 lines cannot be compared to HT29 Top-K as if n were equal",
            "signature is tissue-global, so line-specific MSI/CMS biology is not represented in the scoring vector",
            "SNUC4 is present in inst_info and CONTEXT_REGISTRY but absent from official GSE92742 cell_info; identity still uses inst_info cell_id + registry depmap_id",
        ],
    }


def audit_prism() -> dict[str, Any]:
    prism = pd.read_parquet(PRISM)
    audit = json.loads(PRISM_AUDIT.read_text(encoding="utf-8"))
    treatment = pd.read_csv(TREATMENT, dtype=str)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    per_line = result["oracle"]["metrics_summary"]["per_line_top10"]
    depmap_ids = {row["depmap_id"] for row in result["oracle"]["metrics"]["line_rows"]}
    crc = prism.loc[prism["depmap_id"].isin(depmap_ids)].copy()
    parsed = crc["column_name"].astype(str).str.split("::", expand=True)
    crc["column_dose"] = pd.to_numeric(parsed[1], errors="coerce") if parsed.shape[1] > 1 else np.nan
    crc["column_screen"] = parsed[2] if parsed.shape[1] > 2 else ""
    dose_counts = crc["column_dose"].value_counts(dropna=False).head(8).to_dict()
    screen_counts = crc["column_screen"].value_counts(dropna=False).to_dict()
    treat_cols = treatment["column_name"].astype(str).str.split("::", expand=True)
    treatment["column_dose"] = pd.to_numeric(treat_cols[1], errors="coerce") if treat_cols.shape[1] > 1 else np.nan
    treatment["column_screen"] = treat_cols[2] if treat_cols.shape[1] > 2 else ""
    treatment_screen = treatment["screen_id"].value_counts(dropna=False).to_dict()
    treatment_dose = treatment["dose"].value_counts(dropna=False).head(8).to_dict()
    finite = pd.to_numeric(crc["response_raw"], errors="coerce")
    return {
        "source_release": "PRISM_REPURPOSING_PRIMARY_19Q4",
        "matrix_sha256": audit["source"]["matrix_sha256"],
        "output_sha256": audit["output"]["sha256"],
        "response_unit_unique": sorted(crc["response_unit"].astype(str).unique().tolist()),
        "response_direction_unique": sorted(crc["response_direction"].astype(str).unique().tolist()),
        "sensitivity_definition": "sensitivity_score = - official PRISM log2 fold-change",
        "joined_oracle_lines": sorted(depmap_ids),
        "crc_exact_prism_rows": int(len(crc)),
        "finite_response_rows": int(np.isfinite(finite).sum()),
        "response_raw_median": float(np.nanmedian(finite)),
        "response_raw_q05": float(np.nanquantile(finite, 0.05)),
        "response_raw_q95": float(np.nanquantile(finite, 0.95)),
        "column_dose_top": {str(key): int(value) for key, value in dose_counts.items()},
        "column_screen_counts": {str(key): int(value) for key, value in screen_counts.items()},
        "treatment_info_screen_counts": {str(key): int(value) for key, value in treatment_screen.items()},
        "treatment_info_dose_top": {str(key): int(value) for key, value in treatment_dose.items()},
        "dominant_endpoint": "collapsed PRISM primary viability log2FC, mostly dose 2.5 and screen HTS/MTS",
        "not_transcriptional": True,
        "not_matched_to_10uM_6h": True,
        "per_line_oracle_top10_lift": {
            cid: {"n": row["candidate_count"], "top10_lift": row["overlap_lift"], "spearman": row["spearman"]}
            for cid, row in per_line.items()
        },
        "mismatch_notes": [
            "PRISM endpoint is multi-day barcode viability, not 6 h L1000 Delta978",
            "official primary screen dose is 2.5, not LINCS canonical 10 µM",
            "a compound that reverses a tissue tumor-vs-normal signature at 6 h need not kill the cell line at 2.5 µM",
            "HT29 n=1438 vs other lines n≈80 means Top-10 null rates are not comparable",
        ],
    }


def decide(signature: dict[str, Any], dose: dict[str, Any], context: dict[str, Any], prism: dict[str, Any]) -> dict[str, Any]:
    sources = {
        "disease_signature": {
            "status": "PLAUSIBLE_PRIMARY_MISMATCH",
            "why": (
                "GSE74602 is paired surgical CRC tissue vs adjacent normal on Illumina HT-12; "
                f"{signature['rows']}/978 landmarks, up/down={signature['up']}/{signature['down']}. "
                "It is not a LINCS cell-line disease state and is reused unchanged on all 10 lines."
            ),
        },
        "dose_time": {
            "status": "CONFIRMED_ASSAY_MISMATCH_NOT_A_MIXING_BUG",
            "why": (
                "Oracle 10 µM/6 h vs PRISM 2.5 µM multi-day viability. "
                "6 h and 24 h were not mixed; 24 h exists only in HT29 and also sat at null."
            ),
        },
        "context": {
            "status": "IDENTITY_OK_SUPPORT_NOT_EXCHANGEABLE",
            "why": (
                "All 10 CRC exact lines are present. The old 2-line/839-pair hole is closed. "
                "The remaining problem is biological/support: HT29 1438 vs shared-core ~80 on 9 lines, "
                "plus a tissue signature that ignores line subtype. "
                "SNUC4 is missing from official cell_info but is present in inst_info and the Broad registry."
            ),
        },
        "prism_endpoint": {
            "status": "PLAUSIBLE_PRIMARY_MISMATCH",
            "why": (
                "Official PRISM primary log2FC is a viability/abundance endpoint. "
                "Disease-reversal ranking of 6 h Delta978 is a different construct. "
                "Joining after identity freeze was clean; the endpoint itself is the mismatch."
            ),
        },
    }
    return {
        "overall_status": "ORACLE_NEAR_NULL_NOT_A_COVERAGE_BUG",
        "do_not_stack_models": True,
        "primary_suspects": ["disease_signature", "prism_endpoint"],
        "secondary_suspects": ["dose_time_assay_mismatch", "context_support_imbalance"],
        "ruled_out": [
            "use_of_839_pair_xpert_h5ad_oracle",
            "missing_crc_exact_lines",
            "unmatched_controls",
            "silent_6h_24h_mix",
            "response_used_to_select_pairs",
        ],
        "sources": sources,
        "allowed_next": [
            "redefine the translational claim: 6 h L1000 reversal is not licensed as a 2.5 µM PRISM-viability ranker until a matched endpoint exists",
            "keep per-line + support strata; do not macro-average",
            "if a new signature is proposed, it must be pre-registered, cell-line or CMS-aware, and response-blind",
        ],
        "forbidden_next": [
            "train a larger XPert because predicted-oracle gap is negative",
            "mix 6 h and 24 h to inflate coverage",
            "shrink to the 2 lines that previously looked eligible",
        ],
    }


def main() -> None:
    signature = audit_signature()
    dose = audit_dose_time()
    context = audit_context()
    prism = audit_prism()
    payload = {
        "format": "oracle_near_null_source_audit_v1",
        "exp_id": "EXP-007",
        "fail_rule": "Oracle near null and one CMap/weighted-KS still null → inspect signature, dose/time, context, PRISM endpoint; do not stack models",
        "result_status": "ORACLE_NEAR_NULL",
        "verdict": "CHECK_SIGNATURE_DOSE_TIME_CONTEXT_PRISM",
        "primary_metrics_unchanged": True,
        "signature": signature,
        "dose_time": dose,
        "context": context,
        "prism_endpoint": prism,
        "decision": decide(signature, dose, context, prism),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "out": str(OUT.relative_to(ROOT)),
            "overall": payload["decision"]["overall_status"],
            "primary_suspects": payload["decision"]["primary_suspects"],
            "signature_rows": signature["rows"],
            "missing_landmarks": signature["missing_landmark_count"],
            "up_down": [signature["up"], signature["down"]],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
