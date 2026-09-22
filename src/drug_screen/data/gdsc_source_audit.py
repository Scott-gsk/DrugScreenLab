"""Read-only provenance audit for the indexed EXP-012 GDSC2 source cohort."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.transfer.chemistry import standardize_smiles


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} does not match the registered GDSC2 raw manifest")


def _frame_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError:
        return False
    return True


def audit_gdsc_source(root: Path) -> dict:
    """Verify actual response, expression, structure and cell-line correspondence.

    The function intentionally does not assign an admission verdict.  It reports
    verified facts and any unresolved conflict; protocol ownership decides G0.
    """
    manifest_path = root / "processed/exp012/gdsc2_cell_line_raw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["source_hashes"]
    external = root / "external/exp010"
    source_paths = {
        "gdsc2_fitted_release8.5.parquet": root / "processed/exp010/gdsc2_fitted_release8.5.parquet",
        "cmp_gdsc2_gene_contract.parquet": root / "processed/exp010/cmp_gdsc2_gene_contract.parquet",
        "drug_structures.json": external / "pubchem/drug_structures.json",
        "Cell_Lines_Details.xlsx": external / "gdsc2_release8.5/Cell_Lines_Details.xlsx",
    }
    for name, path in source_paths.items():
        _require_equal(name, sha256(path), expected[name])

    raw_response_path = external / "gdsc2_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx"
    raw_response = pd.read_excel(raw_response_path)
    fitted = pd.read_parquet(source_paths["gdsc2_fitted_release8.5.parquet"])
    required_response = {"DATASET", "SANGER_MODEL_ID", "CELL_LINE_NAME", "TCGA_DESC", "DRUG_NAME", "AUC"}
    if missing := required_response.difference(raw_response.columns):
        raise ValueError(f"fitted GDSC2 source lacks columns: {sorted(missing)}")
    if not _frame_equal(raw_response, fitted):
        raise ValueError("processed GDSC2 fitted table is not an exact copy of the registered source workbook")
    if set(raw_response.DATASET.astype(str)) != {"GDSC2"}:
        raise ValueError("fitted response workbook is not exclusively GDSC2")

    samples = pd.read_parquet(root / "processed/exp012/gdsc2_cell_line_samples.parquet")
    drugs = pd.read_parquet(root / "processed/exp012/gdsc2_cell_line_drugs.parquet")
    pairs = pd.read_parquet(root / "processed/exp012/gdsc2_cell_line_pairs.parquet")
    with np.load(root / "processed/exp012/gdsc2_cell_line_indexed.npz", allow_pickle=True) as archive:
        indexed_expression = np.asarray(archive["expression"], dtype=float)
        indexed_samples = archive["sample_id"].astype(str)
        indexed_genes = archive["ordered_genes"].astype(str)

    if not np.array_equal(indexed_samples, samples.sample_id.astype(str).to_numpy()):
        raise ValueError("indexed expression sample order differs from samples parquet")
    if len(set(indexed_genes)) != len(indexed_genes):
        raise ValueError("indexed expression has duplicate genes")

    expression = pd.read_parquet(source_paths["cmp_gdsc2_gene_contract.parquet"])
    if not set(samples.sample_id.astype(str)).issubset(set(expression.index.astype(str))):
        raise ValueError("indexed samples are absent from registered CMP expression table")
    if not set(indexed_genes).issubset(set(expression.columns.astype(str))):
        raise ValueError("indexed genes are absent from registered CMP expression table")
    rebuilt_expression = np.log2(expression.loc[indexed_samples, indexed_genes].to_numpy(dtype=float) + 1.0)
    finite = np.isfinite(rebuilt_expression) & np.isfinite(indexed_expression)
    if not np.array_equal(np.isnan(rebuilt_expression), np.isnan(indexed_expression)):
        raise ValueError("indexed expression missing-value mask differs from CMP source")
    expression_error = float(np.max(np.abs(rebuilt_expression[finite] - indexed_expression[finite]))) if finite.any() else 0.0
    if expression_error > 1e-5:
        raise ValueError(f"indexed expression is not log2(CMP TPM+1): max error {expression_error}")

    source_by_model = raw_response.groupby("SANGER_MODEL_ID", sort=False).agg(
        cell_line_name=("CELL_LINE_NAME", "first"), tcga_desc=("TCGA_DESC", "first")
    )
    if source_by_model.index.duplicated().any():
        raise ValueError("duplicate SANGER model identities in GDSC2 source")
    sample_source = source_by_model.reindex(samples.sample_id.astype(str))
    missing_source_samples = sorted(set(samples.sample_id.astype(str)).difference(source_by_model.index.astype(str)))
    tissue_available = sample_source.tcga_desc.notna()
    tissue_conflicts = int((
        sample_source.loc[tissue_available, "tcga_desc"].astype(str).to_numpy()
        != samples.sample_tissue.astype(str).to_numpy()[tissue_available.to_numpy()]
    ).sum())

    cell_details = pd.read_excel(source_paths["Cell_Lines_Details.xlsx"])
    details_name = "Sample Name"
    if details_name not in cell_details:
        raise ValueError("Cell_Lines_Details lacks Sample Name")
    names = sample_source.cell_line_name.astype(str)
    metadata_coverage = int(names.dropna().isin(set(cell_details[details_name].dropna().astype(str))).sum())

    # The PubChem response payload is registered by the legacy raw manifest,
    # while this sibling cache preserves the exact GDSC drug-name aliases used
    # to build the indexed cohort.  Both are checked; the cache is not treated
    # as a new structure source.
    structure_cache_path = root / "processed/exp012/structure_resolution_cache.json"
    structure_rows = json.loads(structure_cache_path.read_text(encoding="utf-8"))
    structures: dict[str, list[dict]] = {}
    for row in structure_rows:
        structures.setdefault(str(row.get("source_name")), []).append(row)
    source_drug_names = set(raw_response.DRUG_NAME.astype(str))
    mapped: dict[str, str] = {}
    structure_issues: list[dict] = []
    for name in sorted(source_drug_names):
        rows = structures.get(name, [])
        if len(rows) != 1:
            structure_issues.append({"drug_name": name, "reason": "missing_or_ambiguous_structure_row", "rows": len(rows)})
            continue
        row = rows[0]
        identity = standardize_smiles(row.get("canonical_smiles"))
        if not identity.accepted or identity.inchikey != row.get("inchikey"):
            structure_issues.append({"drug_name": name, "reason": "source_structure_invalid_or_key_conflict", **asdict(identity)})
            continue
        mapped[name] = identity.inchikey
    bundle_drugs = set(drugs.drug_id.astype(str))
    source_to_bundle = {name: key for name, key in mapped.items() if key in bundle_drugs}
    if len(source_to_bundle) != len(bundle_drugs):
        missing = sorted(bundle_drugs.difference(source_to_bundle.values()))
        structure_issues.extend({"drug_id": key, "reason": "bundle_drug_not_resolved_from_source_name"} for key in missing)
    bundle_structure_issues = [issue for issue in structure_issues if "drug_id" in issue]

    raw_mapped = raw_response.loc[raw_response.DRUG_NAME.astype(str).isin(source_to_bundle)].copy()
    raw_mapped["drug_id"] = raw_mapped.DRUG_NAME.astype(str).map(source_to_bundle)
    raw_mapped["sample_id"] = raw_mapped.SANGER_MODEL_ID.astype(str)
    grouped_auc = raw_mapped.groupby(["sample_id", "drug_id"], sort=False).AUC.agg(list)
    duplicate_source_pairs = int(sum(len(values) > 1 for values in grouped_auc))
    pair_index = pd.MultiIndex.from_frame(pairs[["sample_id", "drug_id"]].astype(str))
    candidates = grouped_auc.reindex(pair_index)
    missing_source_pairs = int(candidates.isna().sum())
    response_values = pairs.response.to_numpy(float)
    candidate_match = np.array([
        False if values is None or not isinstance(values, list) else bool(np.isclose(1.0 - np.asarray(values, dtype=float), value, rtol=0, atol=1e-12).any())
        for values, value in zip(candidates.to_numpy(), response_values, strict=True)
    ])
    unmatched_response_pairs = int((~candidate_match).sum())
    response_error = (float(max(
        min(abs((1.0 - np.asarray(values, dtype=float)) - value))
        for values, value in zip(candidates.to_numpy(), response_values, strict=True)
        if isinstance(values, list)
    )) if missing_source_pairs != len(candidates) else None)
    endpoint_ok = set(pairs.response_name.astype(str)) == {"1_minus_auc"} and pairs.higher_is_sensitive.eq(True).all()

    return {
        "cohort_id": "gdsc2_cell_line",
        "audit_type": "original_source_correspondence",
        "source_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in source_paths.items()} | {"GDSC2_fitted_dose_response_27Oct23.xlsx": {"path": str(raw_response_path), "sha256": sha256(raw_response_path)}, "structure_resolution_cache.json": {"path": str(structure_cache_path), "sha256": sha256(structure_cache_path), "role": "legacy exact-name bridge"}},
        "response": {"status": "PASS" if missing_source_pairs == 0 and unmatched_response_pairs == 0 and response_error is not None and response_error <= 1e-12 and endpoint_ok else "CONFLICT", "raw_rows": len(raw_response), "processed_rows": len(fitted), "indexed_pairs": len(pairs), "endpoint": "1_minus_auc", "higher_is_sensitive": True, "source_pairs_missing": missing_source_pairs, "source_key_duplicates": duplicate_source_pairs, "unmatched_indexed_response_pairs": unmatched_response_pairs, "max_abs_transform_error": response_error},
        "expression": {"status": "PASS", "source_shape": list(expression.shape), "indexed_shape": list(indexed_expression.shape), "transform": "log2(CMP TPM+1)", "max_abs_transform_error": expression_error, "missing_value_mask_matches": True},
        "cell_identity": {"status": "PASS" if not missing_source_samples and tissue_conflicts == 0 else "CONFLICT", "indexed_samples": len(samples), "fitted_response_identity_key": "SANGER_MODEL_ID", "source_identity_missing": missing_source_samples, "tcga_desc_conflicts": tissue_conflicts, "cell_details_name_coverage": metadata_coverage, "cell_details_name_total": len(cell_details)},
        "drug_identity": {"status": "PASS" if not bundle_structure_issues else "CONFLICT", "source_drug_names": len(source_drug_names), "bundle_drugs": len(bundle_drugs), "resolved_source_to_bundle_drugs": len(source_to_bundle), "bundle_issues": bundle_structure_issues, "excluded_source_name_issues": [issue for issue in structure_issues if "drug_name" in issue]},
        "unmeasured_metadata": ["batch", "culture_condition", "dose/time at individual response level", "technical-repeat QC"],
        "split_assessment": {"status": "NOT_ASSESSED", "reason": "No frozen EXP-014 GDSC2 source train/validation/cold-cell assignment was supplied to this audit; it must be checked when the concrete source configuration is frozen."},
        "scientific_admission": "PENDING_PROTOCOL_OWNER_G0_REVIEW",
    }
