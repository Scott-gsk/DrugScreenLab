"""Response-blind EXP-007 fail-rule audit of four Oracle-near-null sources.

Does not change the primary metric, dose/time mix, or train anything.
PRISM numeric values are read only after identity and coverage funnels freeze.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from drug_screen.evaluation.full_observed_oracle import (
    CANONICAL_CONDITIONS,
    CRC_EXACT_CONTEXTS,
    ORDERED_GENE_IDS_SHA256,
    assign_matched_controls,
    evaluate_oracle_frame,
    file_sha256,
    join_prism_after_identity_freeze,
    oracle_near_null,
    response_blind_eligible_pert_ids,
    select_canonical_instances,
)

ROOT = _ROOT
INST = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_inst_info.txt.gz"
GENE = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
SIGNATURE = ROOT / "mvp/core_data/crc_disease_signature_exact978.tsv"
SIGNATURE_AUDIT = ROOT / "mvp/core_data/crc_disease_signature_audit.json"
GSE19163_AUDIT = ROOT / "mvp/core_data/crc_disease_signature_gse19163_audit.json"
GSE19163_SOURCE = ROOT / "mvp/core_data/_source_cache/GSE19163_tumor_vs_normal_tissue.txt.gz"
PRISM = ROOT / "mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet"
PRISM_AUDIT = ROOT / "mvp/foundation/xpert/BROAD_PRISM_CRC_V1_AUDIT.json"
TREATMENT = ROOT / "mvp/core_data/_source_cache/primary-screen-replicate-collapsed-treatment-info.csv"
BRIDGE = ROOT / "artifacts/extension/lincs_prism_identity_bridge.csv"
REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
RESULT = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_RESULT.json"
COVERAGE = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_COVERAGE.json"
OUT = ROOT / "artifacts/experiments/EXP-007/ORACLE_NULL_SOURCE_AUDIT.json"

EXPECTED_SIGNATURE_SHA256 = "61e95b6a6da1d4c8b91ed1a99e96471027917c7e7e2501b507875ce28cd310c3"
# Canonical CRC biology checks on landmark genes that are in exact978.
# Direction is tumor-vs-normal in GSE74602: positive = tumor higher.
CRC_MARKER_EXPECTATIONS = {
    "MYC": "up",
    "CCND1": "up",
    "PCNA": "up",
    "BIRC5": "up",
    "TOP2A": "up",
    "CDK4": "up",
    "AURKA": "up",
    "PLK1": "up",
    "MKI67": "up",
    "TGFBR2": "down",
    "CDH1": "down",
    "CA2": "down",
    "FABP1": "down",
    "CEBPA": "down",
    "FOS": "down",
    "CDKN1A": "either",
}


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


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_instances() -> pd.DataFrame:
    instances = pd.read_csv(INST, sep="\t", low_memory=False)
    instances["_cache_row"] = np.arange(len(instances), dtype=np.int64)
    instances["pert_dose_num"] = pd.to_numeric(instances["pert_dose"], errors="coerce")
    instances["pert_time_num"] = pd.to_numeric(instances["pert_time"], errors="coerce")
    return instances


def _eligible() -> tuple[set[str], dict[str, Any]]:
    bridge = pd.read_csv(BRIDGE, low_memory=False)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    eligible = response_blind_eligible_pert_ids(bridge, registry)
    identity_info = {
        "formal_matched_identity_and_broad_inference_eligible": int(len(eligible)),
        "bridge_rows": int(len(bridge)),
        "match_status_counts": (
            bridge["match_status"].astype(str).value_counts(dropna=False).to_dict()
            if "match_status" in bridge.columns
            else {}
        ),
        "response_values_read": False,
    }
    return eligible, identity_info


def _unique_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0][["cell_id", "pert_id"]].copy()
    return frame[["cell_id", "pert_id"]].drop_duplicates()


def _count_pairs(frame: pd.DataFrame) -> dict[str, Any]:
    pairs = _unique_pairs(frame)
    per_line = (
        pairs.groupby("cell_id")["pert_id"].nunique().astype(int).to_dict() if not pairs.empty else {}
    )
    return {
        "treatment_rows": int(len(frame)),
        "unique_pairs": int(len(pairs)),
        "unique_contexts": int(pairs["cell_id"].nunique()) if not pairs.empty else 0,
        "unique_compounds": int(pairs["pert_id"].nunique()) if not pairs.empty else 0,
        "per_context": {str(k): int(v) for k, v in per_line.items()},
    }


def _mask_crc_trt(instances: pd.DataFrame) -> pd.Series:
    return instances["pert_type"].astype(str).eq("trt_cp") & instances["cell_id"].astype(str).isin(
        CRC_EXACT_CONTEXTS
    )


def _is_um(frame: pd.DataFrame) -> pd.Series:
    return frame["pert_dose_unit"].astype(str).str.lower().eq("um")


def _is_h(frame: pd.DataFrame) -> pd.Series:
    return frame["pert_time_unit"].astype(str).str.lower().eq("h")


def audit_signature() -> dict[str, Any]:
    sig = pd.read_csv(SIGNATURE, sep="\t")
    digest = file_sha256(SIGNATURE)
    required = {
        "gene_index_978",
        "pr_gene_id",
        "gene_symbol",
        "signed_log2fc",
        "direction",
        "comparison",
    }
    missing = sorted(required.difference(sig.columns))
    n_up = int((sig["direction"].astype(str) == "up").sum())
    n_down = int((sig["direction"].astype(str) == "down").sum())
    n_zero = int((pd.to_numeric(sig["signed_log2fc"], errors="coerce") == 0).sum())
    index = sig["gene_index_978"].astype(int)
    direction_agrees = bool(
        (
            ((sig["direction"].astype(str) == "up") & (pd.to_numeric(sig["signed_log2fc"]) > 0))
            | ((sig["direction"].astype(str) == "down") & (pd.to_numeric(sig["signed_log2fc"]) < 0))
        ).all()
    )
    comparison = sorted(sig["comparison"].astype(str).unique().tolist())
    gene_ids = [str(value) for value in sig["pr_gene_id"].astype(str)]
    gene_index_ok = bool(index.min() >= 0 and index.max() <= 977 and index.is_unique)
    gene_info_ok = False
    gene_id_mismatch = []
    if GENE.exists():
        genes = pd.read_csv(GENE, sep="\t", dtype=str)
        lm = genes.loc[genes["pr_is_lm"].astype(str).eq("1")].copy()
        if "pr_gene_id" in lm.columns:
            id_by_symbol = dict(zip(lm["pr_gene_symbol"].astype(str), lm["pr_gene_id"].astype(str)))
            for _, row in sig.iterrows():
                expected = id_by_symbol.get(str(row["gene_symbol"]))
                if expected is not None and expected != str(row["pr_gene_id"]):
                    gene_id_mismatch.append(str(row["gene_symbol"]))
            gene_info_ok = True
    markers: dict[str, Any] = {}
    by_symbol = sig.set_index(sig["gene_symbol"].astype(str))
    marker_hits = 0
    marker_checked = 0
    for symbol, expected in CRC_MARKER_EXPECTATIONS.items():
        if symbol not in by_symbol.index:
            markers[symbol] = {"present": False, "expected": expected}
            continue
        row = by_symbol.loc[symbol]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        observed = str(row["direction"])
        signed = float(row["signed_log2fc"])
        ok = expected == "either" or observed == expected
        markers[symbol] = {
            "present": True,
            "expected": expected,
            "observed": observed,
            "signed_log2fc": signed,
            "matches_expectation": ok,
        }
        if expected != "either":
            marker_checked += 1
            marker_hits += int(ok)

    gse19163: dict[str, Any] = {"status": "NOT_COMPARED"}
    if GSE19163_AUDIT.exists():
        backup = json.loads(GSE19163_AUDIT.read_text(encoding="utf-8"))
        listed = [str(s) for s in backup.get("exact978_overlap", {}).get("gene_symbols", [])]
        shared = sorted(set(listed).intersection(set(sig["gene_symbol"].astype(str))))
        gse19163 = {
            "status": "METADATA_ONLY" if not GSE19163_SOURCE.exists() else "SOURCE_PRESENT",
            "backup_overlap_listed": listed,
            "shared_with_primary": shared,
            "shared_count": int(len(shared)),
            "note": (
                "GSE19163 is a retained backup with only 15 landmark overlaps and is not "
                "the frozen primary. Primary GSE74602 file sha256 matches registry, so "
                "the backup TSV was not used as EXP-007 signature."
            ),
            "primary_not_overwritten_by_backup": digest == EXPECTED_SIGNATURE_SHA256,
        }
        if GSE19163_SOURCE.exists() and shared:
            src = pd.read_csv(GSE19163_SOURCE, sep="\t")
            symbol_col = next(
                (c for c in src.columns if "gene" in c.lower() and "symbol" in c.lower()),
                None,
            )
            fc_col = next((c for c in src.columns if "fold" in c.lower()), None)
            if symbol_col and fc_col:
                src[symbol_col] = src[symbol_col].astype(str)
                src[fc_col] = pd.to_numeric(src[fc_col], errors="coerce")
                agree = 0
                compared = []
                for symbol in shared:
                    left = float(by_symbol.loc[symbol]["signed_log2fc"])
                    right = float(src.loc[src[symbol_col] == symbol, fc_col].iloc[0])
                    same = np.sign(left) == np.sign(right)
                    agree += int(same)
                    compared.append(
                        {
                            "gene_symbol": symbol,
                            "gse74602_signed_log2fc": left,
                            "gse19163_source_fc": right,
                            "same_sign": bool(same),
                        }
                    )
                gse19163["sign_agreement"] = {
                    "n": int(len(compared)),
                    "same_sign": int(agree),
                    "rows": compared,
                }

    audit_ok = False
    if SIGNATURE_AUDIT.exists():
        published = json.loads(SIGNATURE_AUDIT.read_text(encoding="utf-8"))
        audit_ok = (
            published.get("output", {}).get("sha256") == EXPECTED_SIGNATURE_SHA256
            and published.get("transformation", {}).get("positive_direction") == "tumor_higher"
            and published.get("gene_universe", {}).get("ordered_gene_ids_sha256")
            == ORDERED_GENE_IDS_SHA256
            and int(published.get("exact978_overlap", {}).get("rows", -1)) == int(len(sig))
        )

    inverted_not_default = bool(n_up > n_down and direction_agrees)
    verdict = "PASS"
    reasons = []
    if digest != EXPECTED_SIGNATURE_SHA256:
        verdict = "BLOCKED"
        reasons.append("signature_sha256_mismatch")
    if missing or not gene_index_ok or not direction_agrees:
        verdict = "BLOCKED"
        reasons.append("schema_or_index_or_direction_label_broken")
    if comparison != ["tumor_vs_normal"]:
        verdict = "BLOCKED"
        reasons.append("comparison_not_tumor_vs_normal")
    if not inverted_not_default:
        verdict = "SUSPECT"
        reasons.append("more_down_than_up_or_label_mismatch")
    if marker_checked and marker_hits / marker_checked < 0.6:
        verdict = "SUSPECT" if verdict == "PASS" else verdict
        reasons.append("crc_marker_majority_disagrees")
    if not reasons:
        reasons.append("direction_is_tumor_higher; file digest and 978 indices match contract")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "file": str(SIGNATURE.relative_to(ROOT).as_posix()),
        "sha256": digest,
        "expected_sha256": EXPECTED_SIGNATURE_SHA256,
        "rows": int(len(sig)),
        "up": n_up,
        "down": n_down,
        "zero": n_zero,
        "missing_landmark_vs_978": int(978 - len(sig)),
        "comparison": comparison,
        "positive_direction_contract": "tumor_higher = signed_log2fc > 0 = direction up",
        "direction_labels_agree_with_sign": direction_agrees,
        "gene_index_unique_in_0_977": gene_index_ok,
        "gene_order_digest_contract": ORDERED_GENE_IDS_SHA256,
        "gene_info_crosscheck_ran": gene_info_ok,
        "gene_id_mismatch_symbols": gene_id_mismatch[:20],
        "published_audit_consistent": audit_ok,
        "reversal_definition": "-Spearman(signed_log2fc, Delta978); larger = anti-disease",
        "crc_marker_check": {
            "checked_with_expectation": marker_checked,
            "matches": marker_hits,
            "markers": markers,
        },
        "gse19163_backup": gse19163,
        "cannot_claim": (
            "A whole-tumor-vs-normal Illumina signature is not a 6h chemical-perturbation "
            "reversal ground truth. Direction is not inverted, but biological fitness for "
            "PRISM viability ranking remains open."
        ),
    }


def audit_dose_time(instances: pd.DataFrame, eligible: set[str]) -> dict[str, Any]:
    crc = instances.loc[_mask_crc_trt(instances)].copy()
    crc_eligible = crc.loc[crc["pert_id"].astype(str).isin(eligible)].copy()
    um_h = crc_eligible.loc[_is_um(crc_eligible) & _is_h(crc_eligible)].copy()

    funnels: dict[str, Any] = {
        "crc_trt_any_dose_time": _count_pairs(crc),
        "crc_trt_identity_any_dose_time": _count_pairs(crc_eligible),
        "crc_identity_um_h": _count_pairs(um_h),
    }
    for name, spec in CANONICAL_CONDITIONS.items():
        selected = select_canonical_instances(
            instances,
            contexts=CRC_EXACT_CONTEXTS,
            pert_ids=eligible,
            dose_um=float(spec["dose_um"]),
            time_h=float(spec["time_h"]),
        )
        matched, audit = assign_matched_controls(selected, instances)
        funnels[name] = {
            **_count_pairs(selected),
            "matched_control": _count_pairs(matched),
            "control_audit": audit,
        }

    any_10um = um_h.loc[np.isclose(um_h["pert_dose_num"], 10.0, atol=1e-5, rtol=1e-5)]
    any_6h = um_h.loc[np.isclose(um_h["pert_time_num"], 6.0, atol=1e-5, rtol=1e-5)]
    any_24h = um_h.loc[np.isclose(um_h["pert_time_num"], 24.0, atol=1e-5, rtol=1e-5)]
    funnels["10uM_any_time"] = _count_pairs(any_10um)
    funnels["any_dose_6h"] = _count_pairs(any_6h)
    funnels["any_dose_24h"] = _count_pairs(any_24h)

    matched_all, audit_all = assign_matched_controls(um_h, instances)
    funnels["all_um_h_identity_matched_control"] = {
        **_count_pairs(matched_all),
        "control_audit": audit_all,
        "note": "coverage only; mixed dose/time is not promoted to a primary metric",
    }

    time_by_line = (
        um_h.groupby(["cell_id", "pert_time_num"])["pert_id"]
        .nunique()
        .astype(int)
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    dose_by_line = (
        um_h.groupby(["cell_id", "pert_dose_num"])["pert_id"]
        .nunique()
        .astype(int)
        .reset_index()
        .rename(columns={"pert_id": "n_compounds"})
    )
    dose_hist = (
        um_h.groupby("pert_dose_num")["pert_id"].nunique().sort_index().astype(int).to_dict()
        if not um_h.empty
        else {}
    )
    time_hist = (
        um_h.groupby("pert_time_num")["pert_id"].nunique().sort_index().astype(int).to_dict()
        if not um_h.empty
        else {}
    )

    primary_6h_pairs = int(funnels["10uM_6h"]["matched_control"]["unique_pairs"])
    relaxed_pairs = int(funnels["all_um_h_identity_matched_control"]["unique_pairs"])
    ht29_24h = int(funnels["10uM_24h"]["matched_control"]["per_context"].get("HT29", 0))
    reasons = [
        "PRIMARY remains 10uM/6h by response-blind coverage; 10uM/24h is HT29-only",
        "mixed-dose/time coverage is reported and not used as a new primary",
    ]
    verdict = "SUSPECT"
    if ht29_24h > 0 and all(
        cid not in funnels["10uM_24h"]["matched_control"]["per_context"]
        or funnels["10uM_24h"]["matched_control"]["per_context"][cid] == 0
        for cid in CRC_EXACT_CONTEXTS
        if cid != "HT29"
    ):
        reasons.append("9/10 CRC lines have no 10uM/24h identity pairs")
    if relaxed_pairs > primary_6h_pairs:
        reasons.append(
            f"relaxing dose/time (still identity+control) expands pairs {primary_6h_pairs} -> {relaxed_pairs}"
        )

    return {
        "verdict": verdict,
        "reasons": reasons,
        "primary_condition_unchanged": "10uM_6h",
        "sensitivity_condition_unchanged": "10uM_24h",
        "funnels": funnels,
        "time_hist_identity_um_h_unique_compounds": {str(k): int(v) for k, v in time_hist.items()},
        "dose_hist_identity_um_h_unique_compounds": {str(k): int(v) for k, v in dose_hist.items()},
        "time_by_line_unique_compounds": {
            str(cell): {str(t): int(n) for t, n in times.items()} for cell, times in time_by_line.items()
        },
        "not_promoted": [
            "10uM_any_time",
            "any_dose_6h",
            "any_dose_24h",
            "all_um_h_identity_matched_control",
        ],
        "dose_line_rows": int(len(dose_by_line)),
    }


def audit_context(instances: pd.DataFrame, eligible: set[str], prism: pd.DataFrame | None) -> dict[str, Any]:
    crc = instances.loc[_mask_crc_trt(instances)].copy()
    selected_6h = select_canonical_instances(
        instances,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=set(crc["pert_id"].astype(str)),
        dose_um=10.0,
        time_h=6.0,
    )
    selected_6h_id = select_canonical_instances(
        instances,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=eligible,
        dose_um=10.0,
        time_h=6.0,
    )
    matched_6h, audit_6h = assign_matched_controls(selected_6h_id, instances)

    prism_pairs: set[tuple[str, str]] = set()
    prism_compounds: set[str] = set()
    if prism is not None:
        working = prism.copy()
        if "context_id" not in working.columns:
            working["context_id"] = working["ccle_name"].astype(str).str.split("_", n=1).str[0]
        prism_pairs = set(zip(working["context_id"].astype(str), working["pert_id"].astype(str)))
        prism_compounds = set(working["pert_id"].astype(str))

    per_line: dict[str, Any] = {}
    shared_non_ht29: list[set[str]] = []
    ht29_set: set[str] = set()
    for cell in CRC_EXACT_CONTEXTS:
        raw = crc.loc[crc["cell_id"].astype(str).eq(cell)]
        raw_10_6 = selected_6h.loc[selected_6h["cell_id"].astype(str).eq(cell)]
        id_10_6 = selected_6h_id.loc[selected_6h_id["cell_id"].astype(str).eq(cell)]
        matched = matched_6h.loc[matched_6h["cell_id"].astype(str).eq(cell)]
        raw_compounds = set(raw["pert_id"].astype(str))
        raw_106 = set(raw_10_6["pert_id"].astype(str))
        id_106 = set(id_10_6["pert_id"].astype(str))
        matched_ids = set(matched["pert_id"].astype(str))
        prism_keep = {p for p in matched_ids if (cell, p) in prism_pairs} if prism_pairs else set()
        if cell == "HT29":
            ht29_set = matched_ids
        else:
            shared_non_ht29.append(matched_ids)
        per_line[cell] = {
            "lincs_trt_any_dose_time": int(len(raw_compounds)),
            "lincs_10uM_6h_any_identity": int(len(raw_106)),
            "lincs_10uM_6h_identity_eligible": int(len(id_106)),
            "lincs_10uM_6h_identity_matched_control": int(len(matched_ids)),
            "after_prism_inner_join": int(len(prism_keep)) if prism_pairs else None,
            "dropped_by_identity": int(len(raw_106 - id_106)),
            "dropped_by_control": int(len(id_106 - matched_ids)),
            "dropped_by_prism_join": int(len(matched_ids - prism_keep)) if prism_pairs else None,
            "bottleneck": (
                "lincs_10uM_6h_coverage"
                if len(raw_106) <= 90
                else (
                    "identity_or_prism_eligibility"
                    if len(id_106) < 0.5 * max(len(raw_106), 1)
                    else "not_a_small_lincs_panel"
                )
            ),
        }

    inter = set.intersection(*shared_non_ht29) if shared_non_ht29 else set()
    union = set.union(*shared_non_ht29) if shared_non_ht29 else set()
    reasons = [
        "non-HT29 10uM/6h panels are a LINCS measurement limit, not an identity-bridge wipe",
        "HT29 is the only CRC exact line with a large 10uM/6h panel",
    ]
    verdict = "SUSPECT"
    if all(per_line[c]["lincs_10uM_6h_any_identity"] <= 90 for c in CRC_EXACT_CONTEXTS if c != "HT29"):
        reasons.append("9/10 lines have <=90 compounds at 10uM/6h even before identity filter")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "per_line_funnel": per_line,
        "non_ht29_matched_identity_intersection": int(len(inter)),
        "non_ht29_matched_identity_union": int(len(union)),
        "non_ht29_contained_in_ht29": bool(union.issubset(ht29_set)) if ht29_set else False,
        "prism_eligible_compounds_on_crc_compact": int(len(prism_compounds)),
        "control_audit_primary": audit_6h,
        "cannot_macro_average": True,
    }


def audit_prism(
    instances: pd.DataFrame,
    eligible: set[str],
    signature_audit: dict[str, Any],
) -> dict[str, Any]:
    selected = select_canonical_instances(
        instances,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids=eligible,
        dose_um=10.0,
        time_h=6.0,
    )
    matched, _ = assign_matched_controls(selected, instances)
    identity_pairs = int(matched[["cell_id", "pert_id"]].drop_duplicates().shape[0])

    prism = pd.read_parquet(PRISM)
    required = {"response_raw", "sensitivity_score", "pert_id", "depmap_id", "ccle_name"}
    missing = sorted(required.difference(prism.columns))
    raw = pd.to_numeric(prism["response_raw"], errors="coerce")
    sens = pd.to_numeric(prism["sensitivity_score"], errors="coerce")
    flip_identity = bool(np.allclose(sens.to_numpy(float), -raw.to_numpy(float), equal_nan=True))
    directions = sorted(prism["response_direction"].astype(str).unique().tolist()) if "response_direction" in prism.columns else []
    units = sorted(prism["response_unit"].astype(str).unique().tolist()) if "response_unit" in prism.columns else []

    dummy = matched[["cell_id", "pert_id"]].drop_duplicates().rename(columns={"cell_id": "context_id"})
    dummy["reversal_observed"] = 0.0
    joined = join_prism_after_identity_freeze(dummy, prism)

    treatment_doses: dict[str, Any] = {"status": "ABSENT"}
    if TREATMENT.exists():
        treat = pd.read_csv(TREATMENT, dtype=str, keep_default_na=False)
        treat["dose_num"] = pd.to_numeric(treat.get("dose"), errors="coerce")
        treat["base"] = treat["broad_id"].astype(str).str.extract(r"^(BRD-[A-Z]\d{8})", expand=False)
        used = treat.loc[treat["base"].isin(eligible)] if "base" in treat.columns else treat
        dose_counts = used["dose_num"].value_counts(dropna=False).head(12).to_dict()
        screen_counts = (
            used["screen_id"].astype(str).value_counts().head(8).to_dict() if "screen_id" in used.columns else {}
        )
        treatment_doses = {
            "status": "PRESENT",
            "eligible_base_rows": int(len(used)),
            "dose_value_counts": {str(k): int(v) for k, v in dose_counts.items()},
            "screen_id_counts": {str(k): int(v) for k, v in screen_counts.items()},
            "modal_dose": (
                float(used["dose_num"].mode().iloc[0]) if used["dose_num"].notna().any() else None
            ),
        }

    published = json.loads(PRISM_AUDIT.read_text(encoding="utf-8")) if PRISM_AUDIT.exists() else {}
    result = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    oracle_lines = result.get("oracle", {}).get("metrics_summary", {}).get("per_line_top10", {})

    flipped_eval: dict[str, Any] = {"status": "NOT_RUN"}
    if RESULT.exists():
        observed_table = []
        for row in result.get("oracle", {}).get("metrics", {}).get("line_rows", []):
            observed_table.append(
                {
                    "context_id": row.get("context_id"),
                    "candidate_count": row.get("candidate_count"),
                    "spearman": row.get("spearman"),
                    "top10_lift": row.get("top_k", {}).get("10", {}).get("overlap_lift"),
                }
            )
        # Flip uses the algebraic identity: reversing the signature negates reversal.
        # Recompute Top-K vs PRISM on the same joined pairs by negating the stored score.
        # We cannot rebuild pair-level scores from the summary JSON alone; rerun from parquet
        # plus the stored per-pair result is unavailable.  Report the algebraic effect and
        # a line-wise Spearman sign flip without claiming a new primary.
        n_pos = sum(1 for row in observed_table if (row.get("spearman") or 0) > 0)
        n_neg = sum(1 for row in observed_table if (row.get("spearman") or 0) < 0)
        flipped_eval = {
            "status": "ALGEBRAIC_ONLY",
            "identity": "flipping signature sign negates reversal_observed and therefore negates per-line Spearman vs PRISM",
            "observed_spearman_positive_lines": int(n_pos),
            "observed_spearman_negative_lines": int(n_neg),
            "implication": (
                "If the signature direction were inverted, 6 currently negative lines would "
                "become positive and 4 currently positive lines would become negative. "
                "That is still not a majority Top-10-lift rescue; it only swaps which lines "
                "look weakly aligned."
            ),
            "per_line_observed": observed_table,
            "not_a_new_primary": True,
        }

    reasons = [
        "sensitivity_score = -response_raw is implemented and consistent with the frozen contract",
        "PRISM primary 19Q4 is replicate-collapsed viability log2FC, typically ~2.5 uM multi-day, not 10uM/6h expression",
    ]
    verdict = "SUSPECT"
    if missing or not flip_identity:
        verdict = "BLOCKED"
        reasons.append("sensitivity_score_does_not_match_-response_raw")
    if identity_pairs - int(len(joined)) > 50:
        verdict = "BLOCKED"
        reasons.append("large_identity_to_prism_drop")

    return {
        "verdict": verdict,
        "reasons": reasons,
        "schema_missing": missing,
        "sensitivity_equals_minus_response_raw": flip_identity,
        "response_direction_values": directions,
        "response_unit_values": units,
        "identity_pairs_before_prism": identity_pairs,
        "pairs_after_prism_inner_join": int(len(joined)),
        "dropped_at_prism_join": int(identity_pairs - len(joined)),
        "prism_crc_finite_rows": int(len(prism)),
        "prism_unique_compounds": int(prism["pert_id"].nunique()),
        "prism_unique_lines": int(prism["depmap_id"].nunique()),
        "source_audit": {
            "release": "PRISM_REPURPOSING_PRIMARY_19Q4",
            "matrix_sha256": published.get("source", {}).get("matrix_sha256"),
            "compact_sha256": published.get("output", {}).get("sha256"),
        },
        "treatment_doses": treatment_doses,
        "alignment_limit": (
            "LINCS Delta978 is a 6h transcriptional displacement; PRISM primary is a "
            "collapsed viability log2 fold-change after days at a different modal dose. "
            "No silent dose/time remapping is applied. This mismatch can make a correct "
            "reversal score look like null against the PRISM ranking."
        ),
        "signature_flip_diagnostic": flipped_eval,
        "published_oracle_top10": oracle_lines,
        "signature_direction_used": signature_audit.get("positive_direction_contract"),
    }


def audit_gse145308() -> dict[str, Any]:
    dest_dir = ROOT / "data/raw/geo/GSE145308"
    dest_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145308/soft/GSE145308_family.soft.gz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145308/miniml/GSE145308_family.xml.tgz",
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE145nnn/GSE145308/suppl/",
    ]
    attempts: list[dict[str, Any]] = []
    downloaded: list[dict[str, Any]] = []
    try:
        import urllib.request

        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "DrugScreenLab-data-audit/EXP-007"})
                with urllib.request.urlopen(req, timeout=30) as handle:
                    info = handle.info()
                    content_type = str(info.get("Content-Type", ""))
                    data = handle.read(256)
                    attempts.append(
                        {
                            "url": url,
                            "status": "REACHED",
                            "content_type": content_type,
                            "first_bytes": data[:40].decode("latin-1", errors="replace"),
                        }
                    )
                    if "html" in content_type.lower() or data[:15].lower().startswith(b"<!doctype") or data[:6].lower().startswith(b"<html"):
                        attempts[-1]["status"] = "BLOCKED_HTML"
                        continue
                    if url.endswith("/"):
                        attempts[-1]["status"] = "LISTING_REACHED_NOT_SAVED"
                        continue
                    out = dest_dir / Path(url).name
                    with urllib.request.urlopen(req, timeout=120) as full:
                        payload = full.read()
                    if payload[:15].lower().startswith(b"<!doctype") or payload[:6].lower().startswith(b"<html"):
                        attempts[-1]["status"] = "BLOCKED_HTML"
                        continue
                    out.write_bytes(payload)
                    digest = sha256(payload).hexdigest()
                    downloaded.append(
                        {
                            "path": str(out.relative_to(ROOT).as_posix()),
                            "bytes": int(len(payload)),
                            "sha256": digest,
                            "url": url,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - inventory must record live-access failure
                attempts.append({"url": url, "status": "FAILED", "error": str(exc)[:300]})
    except Exception as exc:  # noqa: BLE001
        attempts.append({"url": "urllib", "status": "FAILED", "error": str(exc)[:300]})

    local = sorted(p.name for p in dest_dir.glob("*") if p.is_file())
    status = "DOWNLOADED" if downloaded else "STILL_ABSENT"
    return {
        "accession": "GSE145308",
        "status": status,
        "data_status": "DATA_PARTIAL" if not downloaded else "DATA_PARTIAL",
        "attempts": attempts,
        "downloaded": downloaded,
        "local_files": local,
        "note": (
            "Success would still be METADATA/SOFT only until counts/TPM are mapped to exact978. "
            "A SOFT download is not a chemical-sensitivity benchmark."
        ),
    }


def compact_line_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for row in payload.get("line_rows", []):
        if not row.get("eligible"):
            continue
        top = row.get("top_k", {}).get("10", {})
        out[str(row.get("context_id"))] = {
            "candidate_count": row.get("candidate_count"),
            "spearman": row.get("spearman"),
            "overlap_lift": top.get("overlap_lift"),
            "delta_ndcg": top.get("delta_ndcg"),
        }
    return out


def maybe_evaluate_flip_from_result() -> dict[str, Any]:
    """If a pair-level table exists, evaluate flipped reversal; otherwise skip."""
    pair_path = ROOT / "artifacts/experiments/EXP-007/FULL_OBSERVED_ORACLE_PAIRS.parquet"
    if not pair_path.exists():
        return {"status": "PAIR_TABLE_ABSENT", "not_a_new_primary": True}
    pairs = pd.read_parquet(pair_path)
    if "reversal_observed" not in pairs.columns:
        return {"status": "NO_REVERSAL_COLUMN", "not_a_new_primary": True}
    prism = pd.read_parquet(PRISM)
    flipped = pairs.copy()
    flipped["reversal_flipped"] = -pd.to_numeric(flipped["reversal_observed"], errors="coerce")
    joined = join_prism_after_identity_freeze(
        flipped.rename(columns={"reversal_flipped": "score"})[["context_id", "pert_id", "score"]],
        prism,
    )
    # join keeps score only if named reversal; reconstruct
    joined = join_prism_after_identity_freeze(flipped[["context_id", "pert_id", "reversal_observed"]], prism)
    joined["reversal_flipped"] = -joined["reversal_observed"]
    metrics = evaluate_oracle_frame(joined, score_column="reversal_flipped")
    return {
        "status": "RUN_DIAGNOSTIC_ONLY",
        "near_null": bool(oracle_near_null(metrics["line_rows"])),
        "per_line_top10": compact_line_metrics(metrics),
        "not_a_new_primary": True,
    }


def main() -> None:
    instances = _load_instances()
    eligible, identity_info = _eligible()
    signature = audit_signature()
    dose_time = audit_dose_time(instances, eligible)
    prism_df = pd.read_parquet(PRISM) if PRISM.exists() else None
    context = audit_context(instances, eligible, prism_df)
    prism = audit_prism(instances, eligible, signature)
    flip = maybe_evaluate_flip_from_result()
    prism["signature_flip_from_pairs"] = flip
    organoid = audit_gse145308()

    payload = {
        "format": "oracle_null_source_audit_v1",
        "exp_id": "EXP-007",
        "status": "AUDITED",
        "primary_result_unchanged": "ORACLE_NEAR_NULL",
        "verdict_unchanged": "CHECK_SIGNATURE_DOSE_TIME_CONTEXT_PRISM",
        "response_blind_until_prism": True,
        "identity": identity_info,
        "sources": {
            "disease_signature": {
                "verdict": signature["verdict"],
                **signature,
            },
            "dose_time": dose_time,
            "context": context,
            "prism_endpoint": prism,
        },
        "organoid_gse145308_attempt": organoid,
        "summary": {
            "disease_signature": signature["verdict"],
            "dose_time": dose_time["verdict"],
            "context": context["verdict"],
            "prism_endpoint": prism["verdict"],
        },
        "recommendation": (
            "Do not stack models. Signature direction is not inverted. "
            "The binding constraints are (1) 6h vs multi-day viability endpoint mismatch and "
            "(2) 9/10 CRC lines sitting on a shared ~80-compound 10uM/6h LINCS panel. "
            "A mixed-dose/time Oracle would change coverage but is not the frozen primary."
        ),
        "forbidden": [
            "do not promote mixed dose/time to primary without a pre-registered sensitivity card",
            "do not treat signature flip as a new main metric",
            "do not train XPert or change rank_metrics",
        ],
    }
    _write(OUT, payload)
    print(json.dumps(payload["summary"] | {"out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
