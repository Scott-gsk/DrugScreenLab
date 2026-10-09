#!/usr/bin/env python3
"""Independent readback for the EXP-050 breast subtype source-domain comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_breast_subtype_pairwise_comparison_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
FULL = OUT / "full"
SMOKE = OUT / "smoke"
KEY_PATTERN = re.compile(r"^[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]$")
EXPOSURE_COLS = ["inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key"]
MODEL_IDS = ["ACH-000536", "ACH-000148", "ACH-000768", "ACH-000019"]
PAIRS = [
    "BT20_vs_HS578T", "BT20_vs_MDAMB231", "HS578T_vs_MDAMB231",
    "BT20_vs_MCF7", "HS578T_vs_MCF7", "MDAMB231_vs_MCF7",
]
N_BOOT = 10_000
BOOT_SEED = 20261006


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def exposure_tuples(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for values in frame[EXPOSURE_COLS].itertuples(index=False, name=None):
        normalized: list[str] = []
        for col, value in zip(EXPOSURE_COLS, values):
            if col in {"dose_key", "time_key"}:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                normalized.append("" if pd.isna(number) else format(float(number), ".12g"))
            elif col in {"dose_unit_key", "time_unit_key"}:
                normalized.append(str(value).casefold())
            else:
                normalized.append(str(value).strip().upper())
        rows.append(tuple(normalized))
    return rows


def safe_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_package_files() -> dict[str, Any]:
    """Make summary-listed exclusion audits available beside the full outputs and refresh run hashes."""
    if not OUT.is_dir() or not FULL.is_dir() or not SMOKE.is_dir():
        raise FileNotFoundError("Run, full, or smoke directory is missing")
    copied: list[str] = []
    for name in ["excluded_placeholder_key_summary.tsv", "excluded_placeholder_key_rows.tsv"]:
        source = SMOKE / name
        target = FULL / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing same-run smoke audit file: {source}")
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise FileExistsError(f"Refusing to replace nonmatching audit file: {target}")
        else:
            shutil.copy2(source, target)
            copied.append(name)
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"Copied same-run audit file differs from smoke: {name}")

    files = [
        p for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name not in {"sha256_manifest.tsv", "root_independent_readback_receipt.json", "console.log"}
    ]
    hashes = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    pd.DataFrame(hashes).to_csv(OUT / "sha256_manifest.tsv", sep="\t", index=False, lineterminator="\n")
    return {
        "copied_from_same_run_smoke": copied,
        "copied_files": {name: sha256_file(FULL / name) for name in ["excluded_placeholder_key_summary.tsv", "excluded_placeholder_key_rows.tsv"]},
        "manifest_rows": len(hashes),
        "manifest_sha256": sha256_file(OUT / "sha256_manifest.tsv"),
    }


def bootstrap_check(drug: pd.DataFrame, observed: pd.DataFrame, observed_pairs: pd.DataFrame) -> tuple[int, float]:
    metrics = [
        *[f"{pair}_{suffix}" for pair in PAIRS for suffix in ["gene_spearman_rho", "gene_nonzero_sign_agreement", "hallmark_spearman_rho", "hallmark_nonzero_sign_agreement"]],
        "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho",
        "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement",
        "basal_internal_mean_hallmark_spearman_rho", "basal_vs_mcf7_mean_hallmark_spearman_rho",
        "basal_internal_mean_hallmark_nonzero_sign_agreement", "basal_vs_mcf7_mean_hallmark_nonzero_sign_agreement",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_spearman_rho",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_nonzero_sign_agreement",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_mean_gene_delta_spearman_rho",
        "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_nonzero_sign_agreement",
    ]
    rng = np.random.default_rng(BOOT_SEED)
    max_diff = 0.0
    checked = 0
    for metric in metrics:
        if metric not in drug:
            continue
        values = pd.to_numeric(drug[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
        boot = values[draws].mean(axis=1)
        row = observed.loc[observed.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f"Bootstrap mean output missing/duplicated metric: {metric}")
        expected = np.asarray([values.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975)])
        actual = row[["estimate_equal_key_mean", "ci95_low", "ci95_high"]].iloc[0].to_numpy(dtype=float)
        diff = float(np.max(np.abs(expected - actual)))
        max_diff = max(max_diff, diff)
        checked += 1

    contrast_specs = [
        ("basal_internal_minus_basal_vs_mcf7", "gene_spearman_rho", "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho"),
        ("basal_internal_minus_basal_vs_mcf7", "gene_nonzero_sign_agreement", "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement"),
        ("basal_internal_minus_basal_vs_mcf7", "hallmark_spearman_rho", "basal_internal_mean_hallmark_spearman_rho", "basal_vs_mcf7_mean_hallmark_spearman_rho"),
        ("basal_internal_minus_basal_vs_mcf7", "hallmark_nonzero_sign_agreement", "basal_internal_mean_hallmark_nonzero_sign_agreement", "basal_vs_mcf7_mean_hallmark_nonzero_sign_agreement"),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "gene_spearman_rho", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_spearman_rho", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "gene_nonzero_sign_agreement", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__gene_nonzero_sign_agreement", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "hallmark_spearman_rho", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_mean_gene_delta_spearman_rho", None),
        ("HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean", "hallmark_nonzero_sign_agreement", "HS578T_MDAMB231_minus_own_B_lines_vs_MCF7_mean__hallmark_nonzero_sign_agreement", None),
    ]
    for label, metric, left_name, right_name in contrast_specs:
        left = pd.to_numeric(drug[left_name], errors="coerce")
        if right_name is None:
            vals = left.dropna().to_numpy(dtype=float)
        else:
            right = pd.to_numeric(drug[right_name], errors="coerce")
            valid = left.notna() & right.notna()
            vals = (left[valid] - right[valid]).to_numpy(dtype=float)
        if not len(vals):
            raise ValueError(f"No paired values for bootstrap contrast {label}/{metric}")
        draws = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
        boot = vals[draws].mean(axis=1)
        row = observed_pairs.loc[observed_pairs.comparison.eq(label) & observed_pairs.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f"Paired bootstrap output missing/duplicated contrast: {label}/{metric}")
        expected = np.asarray([vals.mean(), np.quantile(boot, 0.025), np.quantile(boot, 0.975)])
        actual = row[["estimate_equal_key_paired_difference", "ci95_low", "ci95_high"]].iloc[0].to_numpy(dtype=float)
        diff = float(np.max(np.abs(expected - actual)))
        max_diff = max(max_diff, diff)
        checked += 1
    return checked, max_diff


def verify() -> dict[str, Any]:
    summary = safe_json(FULL / "summary.json")
    run_manifest = safe_json(OUT / "run_manifest.json")
    smoke = safe_json(SMOKE / "smoke_receipt.json")
    if summary.get("status") != "COMPLETED_SOURCE_DOMAIN_PAIRWISE_DESCRIPTION_ROOT_READBACK_PASS" or summary.get("run_id") != RUN_ID:
        raise ValueError("Full run summary is not the expected successful strict-domain run")
    if run_manifest.get("run_id") != RUN_ID or float(run_manifest.get("gpu_hours", -1)) != 0.0:
        raise ValueError("Run manifest identity or GPU accounting mismatch")
    if summary.get("input_sha256") != run_manifest.get("input_sha256"):
        raise ValueError("Summary and run manifest input hashes differ")
    if smoke.get("status") != "SMOKE_PASS_COST_GATE_PASS" or smoke.get("expected_strict_domain", {}).get("exposures") != 1285:
        raise ValueError("Strict-domain smoke receipt is missing or inconsistent")

    inputs = {key: (Path(value) if Path(value).is_absolute() else ROOT / value) for key, value in run_manifest["inputs"].items()}
    common_raw = read_tsv(inputs["parent_exact_exposure_domain"], dtype=str, keep_default_na=False)
    if len(common_raw) != 1297 or common_raw.inchi_key.nunique() != 109:
        raise ValueError("Parent raw domain is not 1297 rows/109 distinct key strings")
    common_raw["inchi_key"] = common_raw.inchi_key.astype(str).str.strip().str.upper()
    valid_mask = common_raw.inchi_key.map(lambda key: bool(KEY_PATTERN.fullmatch(key)))
    excluded = common_raw.loc[~valid_mask].copy()
    valid_parent = common_raw.loc[valid_mask].copy()
    if len(excluded) != 12 or set(excluded.inchi_key) != {"RESTRICTED"} or len(valid_parent) != 1285 or valid_parent.inchi_key.nunique() != 108:
        raise ValueError("Independent parent strict-key correction does not reproduce 1285 rows/108 keys")
    if set(excluded.pert_iname.astype(str)) != {"THZ-2-98-01"}:
        raise ValueError("Parent's 12 invalid-key condition rows do not match the recorded compound name")

    pert = read_tsv(inputs["GSE70138_pert_info"], compression="gzip", dtype=str, usecols=["pert_id", "inchi_key"])
    pert["inchi_key"] = pert.inchi_key.astype("string").str.strip().str.upper()
    placeholder_ids = set(pert.loc[pert.inchi_key.eq("RESTRICTED"), "pert_id"].astype(str))
    inst = read_tsv(inputs["GSE70138_inst_info"], compression="gzip", dtype=str, usecols=["pert_id", "pert_iname"])
    placeholder_names = set(inst.loc[inst.pert_id.astype(str).isin(placeholder_ids), "pert_iname"].dropna().astype(str))
    if len(placeholder_ids) != 2 or placeholder_names != {"THZ-2-98-01", "WH-4-025"}:
        raise ValueError("Source perturbagen metadata no longer independently resolves restricted to the two recorded identities")

    excluded_summary = read_tsv(FULL / "excluded_placeholder_key_summary.tsv", dtype=str)
    excluded_rows = read_tsv(FULL / "excluded_placeholder_key_rows.tsv", dtype=str)
    if len(excluded_summary) != 1 or len(excluded_rows) != 12:
        raise ValueError("Run-level exclusion audit rows are incomplete")
    if set(excluded_rows.inchi_key.astype(str).str.upper()) != {"RESTRICTED"} or set(excluded_rows.pert_iname.astype(str)) != {"THZ-2-98-01"}:
        raise ValueError("Run-level excluded condition audit differs from parent rows")
    if set(excluded_summary.iloc[0].source_compound_names.split(";")) != placeholder_names:
        raise ValueError("Run-level excluded identity summary differs from source metadata")

    condition = read_tsv(FULL / "condition_metrics.tsv", dtype={"inchi_key": str})
    if len(condition) != 1285 or condition.inchi_key.nunique() != 108:
        raise ValueError("Strict condition metrics are not 1285 rows/108 full keys")
    condition["inchi_key"] = condition.inchi_key.astype("string").str.strip().str.upper()
    if not condition.inchi_key.map(lambda key: bool(KEY_PATTERN.fullmatch(str(key)))).all():
        raise ValueError("Condition table includes a malformed full InChIKey")
    parent_conditions = set(exposure_tuples(valid_parent))
    new_conditions = set(exposure_tuples(condition))
    if len(parent_conditions) != 1285 or new_conditions != parent_conditions or len(new_conditions) != len(condition):
        missing = sorted(parent_conditions - new_conditions)[:3]
        extra = sorted(new_conditions - parent_conditions)[:3]
        raise ValueError(f"Strict condition keys differ from valid parent exact set: parent={len(parent_conditions)} new={len(new_conditions)} missing={missing} extra={extra}")

    parent_metrics = read_tsv(inputs["parent_measured_metrics"], dtype={"inchi_key": str}).rename(columns={"dose": "dose_key", "dose_unit": "dose_unit_key", "time": "time_key", "time_unit": "time_unit_key"})
    parent_metrics["inchi_key"] = parent_metrics.inchi_key.astype("string").str.strip().str.upper()
    parent_metrics = parent_metrics.loc[parent_metrics.inchi_key.map(lambda key: bool(KEY_PATTERN.fullmatch(str(key))))].copy()
    if len(parent_metrics) != 1285 or set(exposure_tuples(parent_metrics)) != parent_conditions:
        raise ValueError("Parent measured-metric table does not match valid strict exposure set")
    parent_metrics = parent_metrics.sort_values(EXPOSURE_COLS, kind="mergesort").reset_index(drop=True)
    condition_sorted = condition.sort_values(EXPOSURE_COLS, kind="mergesort").reset_index(drop=True)
    baseline_diffs: dict[str, float] = {}
    for new_prefix, old_prefix in [
        ("BT20_vs_MCF7", "BT20_basal_A_vs_MCF7"),
        ("HS578T_vs_MCF7", "HS578T_basal_B_vs_MCF7"),
        ("MDAMB231_vs_MCF7", "MDAMB231_basal_B_vs_MCF7"),
    ]:
        lhs = pd.to_numeric(condition_sorted[f"{new_prefix}_gene_spearman_rho"], errors="coerce")
        rhs = pd.to_numeric(parent_metrics[f"{old_prefix}_gene_spearman_rho"], errors="coerce")
        if int((lhs.notna() & rhs.notna()).sum()) != 1285 or not np.allclose(lhs, rhs, atol=1e-10, rtol=0, equal_nan=True):
            raise ValueError(f"Valid-key parent per-condition baseline readback failed: {new_prefix}")
        baseline_diffs[new_prefix] = float(np.nanmax(np.abs(lhs.to_numpy() - rhs.to_numpy())))

    arrays = np.load(FULL / "condition_vectors.npz", allow_pickle=False)
    delta = arrays["delta"]
    genes = arrays["gene_symbols"].astype(str)
    model_ids = arrays["model_ids"].astype(str)
    vector_keys = arrays["exposure_keys"].astype(str)
    if delta.shape != (1285, 4, 978) or not np.isfinite(delta).all() or len(genes) != 978 or len(set(genes)) != 978 or model_ids.tolist() != MODEL_IDS:
        raise ValueError("Independent vector dimensions, finite values, or axes failed")
    vector_key_set = {tuple(row.tolist()) for row in vector_keys}
    if vector_keys.shape != (1285, 5) or vector_key_set != new_conditions:
        raise ValueError("Vector condition keys do not match strict metrics domain")

    drug = read_tsv(FULL / "drug_equal_weight_metrics.tsv", dtype={"inchi_key": str})
    if len(drug) != 108 or drug.inchi_key.nunique() != 108 or set(drug.inchi_key.astype(str).str.upper()) != set(condition.inchi_key):
        raise ValueError("Equal-full-key metric rows do not match strict 108-key set")
    if int(pd.to_numeric(drug.n_exact_dose_time_exposures, errors="coerce").sum()) != 1285:
        raise ValueError("Within-key exposure counts do not sum to 1285")
    hall = read_tsv(FULL / "hallmark_direction_summary.tsv")
    key_hall = read_tsv(FULL / "drug_key_hallmark_directions.tsv", dtype={"inchi_key": str})
    if len(hall) != 50 or hall.hallmark.nunique() != 50 or not (pd.to_numeric(hall.n_exact_InChIKeys_total, errors="coerce") == 108).all():
        raise ValueError("Hallmark equal-key summary does not cover the 50-set/108-key denominator")
    if len(key_hall) != 108 * 50 or key_hall.inchi_key.nunique() != 108 or key_hall.hallmark.nunique() != 50:
        raise ValueError("Per-key Hallmark direction rows do not cover 108 keys by 50 sets")

    boot = read_tsv(FULL / "bootstrap_equal_key_means.tsv")
    paired = read_tsv(FULL / "paired_bootstrap_ci.tsv")
    for frame, ncol in [(boot, "n_full_InChIKeys"), (paired, "n_paired_full_InChIKeys")]:
        if not (pd.to_numeric(frame.bootstrap_replicates, errors="coerce") == N_BOOT).all() or not (pd.to_numeric(frame.seed, errors="coerce") == BOOT_SEED).all() or not (pd.to_numeric(frame[ncol], errors="coerce") == 108).all():
            raise ValueError("Bootstrap replicate count, seed, or exact-key denominator mismatch")
    boot_checked, boot_max_diff = bootstrap_check(drug, boot, paired)
    if boot_max_diff > 5e-12:
        raise ValueError(f"Independent seeded bootstrap recomputation mismatch: max_abs={boot_max_diff}")

    manifest = read_tsv(OUT / "sha256_manifest.tsv", dtype=str)
    expected_files = [
        p for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name not in {"sha256_manifest.tsv", "root_independent_readback_receipt.json", "console.log"}
    ]
    if len(manifest) != len(expected_files):
        raise ValueError(f"SHA manifest file count mismatch: manifest={len(manifest)} actual={len(expected_files)}")
    manifest_by_path = manifest.set_index("path").to_dict("index")
    for path in expected_files:
        rel = str(path.relative_to(ROOT))
        row = manifest_by_path.get(rel)
        if row is None or int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise ValueError(f"SHA manifest mismatch for {rel}")
    runner_path = ROOT / "scripts/evaluation/analyze_exp050_breast_subtype_pairwise_comparison.py"
    if sha256_file(runner_path) == run_manifest.get("code_sha256"):
        code_hash_status = "MATCH"
    else:
        code_hash_status = "CURRENT_VERIFIER_DIFFERS_FROM_EXECUTED_RUNNER_EXPECTED"

    main_gene = paired.loc[paired.comparison.eq("basal_internal_minus_basal_vs_mcf7") & paired.metric.eq("gene_spearman_rho")].iloc[0]
    main_sign = paired.loc[paired.comparison.eq("basal_internal_minus_basal_vs_mcf7") & paired.metric.eq("gene_nonzero_sign_agreement")].iloc[0]
    key_equal_means = {}
    for metric in [
        "basal_internal_mean_gene_rho", "basal_vs_mcf7_mean_gene_rho",
        "basal_internal_mean_gene_sign_agreement", "basal_vs_mcf7_mean_gene_sign_agreement",
    ]:
        row = boot.loc[boot.metric.eq(metric)]
        if len(row) != 1:
            raise ValueError(f"Missing source-domain key-mean summary metric: {metric}")
        key_equal_means[metric] = {k: float(row.iloc[0][k]) for k in ["estimate_equal_key_mean", "ci95_low", "ci95_high", "n_full_InChIKeys"]}
    comparison_means = {pair: float(drug[f"{pair}_gene_spearman_rho"].mean()) for pair in PAIRS}
    return {
        "status": "ROOT_INDEPENDENT_READBACK_PASS",
        "run_id": RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source-domain description only; no PDO, outer/locked, or GSE216540 sample metadata accessed.",
        "identity_audit": {
            "parent_raw_rows": len(common_raw), "parent_raw_distinct_key_strings": int(common_raw.inchi_key.nunique()),
            "excluded_placeholder_rows": len(excluded), "excluded_key_string": "RESTRICTED",
            "excluded_parent_row_names": sorted(set(excluded.pert_iname.astype(str))),
            "placeholder_source_pert_ids": sorted(placeholder_ids), "placeholder_source_compound_names": sorted(placeholder_names),
            "strict_exposures": len(valid_parent), "strict_full_InChIKeys": int(valid_parent.inchi_key.nunique()),
        },
        "readback": {
            "condition_rows": len(condition), "vector_shape": list(delta.shape), "genes_unique": len(set(genes)) == 978,
            "model_axis": model_ids.tolist(), "hallmark_sets": int(hall.hallmark.nunique()),
            "key_equal_weight_rows": len(drug), "key_equal_exposure_sum": int(pd.to_numeric(drug.n_exact_dose_time_exposures).sum()),
            "parent_valid_key_metric_rows": len(parent_metrics), "baseline_max_abs_differences": baseline_diffs,
            "bootstrap_metrics_recomputed": boot_checked, "bootstrap_max_abs_difference": boot_max_diff,
            "sha_manifest_rows_verified": len(manifest), "sha_manifest_sha256": sha256_file(OUT / "sha256_manifest.tsv"),
            "runner_code_hash_status": code_hash_status,
        },
        "descriptive_summary": {
            "equal_key_mean_gene_rho": comparison_means,
            "source_domain_key_equal_means": key_equal_means,
            "basal_internal_minus_basal_vs_MCF7_gene_rho": {k: float(main_gene[k]) for k in ["estimate_equal_key_paired_difference", "ci95_low", "ci95_high", "n_paired_full_InChIKeys"]},
            "basal_internal_minus_basal_vs_MCF7_gene_sign_agreement": {k: float(main_sign[k]) for k in ["estimate_equal_key_paired_difference", "ci95_low", "ci95_high", "n_paired_full_InChIKeys"]},
        },
        "limitations": [
            "Four fixed source cell lines are not a breast-subtype population sample.",
            "Bootstrap intervals resample the observed 108-key chemical domain; they are not patient or subtype population intervals.",
            "The comparison measures transcriptional-response similarity and does not establish drug efficacy, PDO transfer, or clinical benefit.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-manifest", action="store_true", help="Copy same-run smoke exclusion audits into full/ and refresh the run hash manifest.")
    args = parser.parse_args()
    if args.refresh_manifest:
        print(json.dumps(refresh_package_files(), ensure_ascii=False, indent=2))
        return
    receipt = verify()
    target = OUT / "root_independent_readback_receipt.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
