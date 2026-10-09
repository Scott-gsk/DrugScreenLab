"""Freeze EXP-045 common expression and dependency gene axes from local metadata.

This audit reads registered file hashes, table schemas, expression gene names,
dependency gene names, and only the ``ordered_genes`` member from the source
index archive. Checkpoint metadata is reused from prior receipts. It does not
read expression or dependency numeric values, patient/cell identifiers, DSS,
predictions, or evaluation scores, and it does not reopen checkpoint payloads.
Training remains blocked by EXP-045 identity and functional-holdout gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
EXP036 = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
SOURCE_INDEX = ROOT / "data/processed/exp011/gdsc2_cell_line_indexed.npz"
OFFICIAL_REPO = ROOT / "data/external/exp010/pharmaformer/repo"
DEPMAP_ROOT = ROOT / "data/raw/depmap/24q2"
NATURE_ROOT = ROOT / "data/processed/transfer/biobank_functional_v1"
ADMISSION = ROOT / "data/artifacts/EXP-026/organoid_depmap_admission.json"
EXP036_ASSET_AUDIT = ROOT / "artifacts/experiments/EXP-036/20260923_pharmaformer_asset_audit/asset_audit.json"
EXP036_REAL_SMOKE = ROOT / "artifacts/experiments/EXP-036/20260923_pharmaformer_asset_audit/real_input_smoke.json"
OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2"
EXPECTED_OFFICIAL_COMMIT = "f526ddda4091151f082b38367571a94823314aba"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(axis: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(axis) + "\n").encode("utf-8"))


def _validate_unique_axis(values: Sequence[str], name: str) -> list[str]:
    axis = [str(value).strip() for value in values]
    if not axis or any(not value for value in axis):
        raise ValueError(f"{name} must be non-empty and contain no empty genes")
    if len(axis) != len(set(axis)):
        raise ValueError(f"{name} contains duplicate normalized gene symbols")
    return axis


def _assert_exact_ordered_axis_match(
    expected: Sequence[str], observed: Sequence[str]
) -> bool:
    """Fail closed unless two non-empty, unique gene axes match in exact order."""
    expected_axis = _validate_unique_axis(expected, "expected gene axis")
    observed_axis = _validate_unique_axis(observed, "observed gene axis")
    if expected_axis != observed_axis:
        raise ValueError("observed gene axis does not exactly match expected order")
    return True


def _load_npz_ordered_genes(path: Path) -> list[str]:
    """Read only the one-dimensional string ``ordered_genes`` NPZ member."""
    with np.load(path, allow_pickle=False) as source_index:
        if "ordered_genes" not in source_index.files:
            raise ValueError("source index NPZ is missing ordered_genes")
        values = source_index["ordered_genes"]
    if values.ndim != 1 or values.dtype.kind not in {"S", "U"}:
        raise ValueError("source index ordered_genes must be a one-dimensional string array")
    if values.dtype.kind == "S":
        try:
            genes = [value.decode("utf-8") for value in values.tolist()]
        except UnicodeDecodeError as exc:
            raise ValueError("source index ordered_genes is not valid UTF-8") from exc
    else:
        genes = values.tolist()
    return _validate_unique_axis(genes, "EXP-011 source ordered_genes axis")


def _depmap_gene_symbols(columns: Sequence[str], *, key_column_count: int = 1) -> list[str]:
    """Parse registered DepMap headers ``SYMBOL (EntrezID)`` exactly."""
    if key_column_count < 1 or key_column_count >= len(columns):
        raise ValueError("invalid DepMap key column count")
    symbols: list[str] = []
    for raw in columns[key_column_count:]:
        header = str(raw).strip()
        if " (" not in header or not header.endswith(")"):
            raise ValueError("DepMap feature header does not match SYMBOL (EntrezID)")
        symbol, entrez = header.rsplit(" (", 1)
        entrez = entrez[:-1]
        if not symbol or not entrez.isdigit():
            raise ValueError("DepMap feature header has an invalid symbol/Entrez suffix")
        symbols.append(symbol)
    return _validate_unique_axis(symbols, "DepMap gene axis")


def _nature_expression_gene_symbols(columns: Sequence[str]) -> list[str]:
    """Use the registered EXP-026 rule that removes the terminal ``_exp``."""
    feature_columns = [str(column) for column in columns if str(column).endswith("_exp")]
    non_feature = [str(column) for column in columns if not str(column).endswith("_exp")]
    if non_feature != ["sample_ID"]:
        raise ValueError("Nature expression schema differs from the registered sample_ID plus genes contract")
    return _validate_unique_axis(
        [column[: -len("_exp")] for column in feature_columns], "Nature expression gene axis"
    )


def _ordered_intersection(reference: Sequence[str], *other_axes: Sequence[str]) -> list[str]:
    sets = [set(axis) for axis in other_axes]
    return [gene for gene in reference if all(gene in axis for axis in sets)]


def _positions(source: Sequence[str], target: Sequence[str]) -> list[int]:
    lookup = {gene: index for index, gene in enumerate(source)}
    try:
        return [lookup[gene] for gene in target]
    except KeyError as exc:
        raise ValueError("target gene axis contains a symbol absent from its source axis") from exc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_file_sha(entry: dict[str, Any], filename: str) -> str:
    return str(entry["files"][filename]["sha256"])


def _nature_expected_sha(admission: dict[str, Any], name: str) -> str:
    check_name = f"parquet:{name}"
    matches = [item for item in admission["checks"] if item.get("check") == check_name]
    if len(matches) != 1 or not matches[0].get("passed"):
        raise ValueError(f"missing passing registered Nature checksum for {name}")
    if matches[0]["expected"] != matches[0]["actual"]:
        raise ValueError(f"Nature admission checksum mismatch for {name}")
    return str(matches[0]["expected"])


def _current_file_metadata(path: Path) -> dict[str, int]:
    return {"current_byte_size": int(path.stat().st_size)}


def _checkpoint_axis_evidence(
    project_genes: Sequence[str], asset_audit: dict[str, Any], real_smoke: dict[str, Any]
) -> list[dict[str, Any]]:
    if asset_audit.get("exp036_gene_axis_sha256") != _sha256_file(EXP036 / "genes.json"):
        raise ValueError("existing source-only audit is not bound to the current EXP-036 gene file")
    if len(asset_audit.get("checkpoints", [])) != 5:
        raise ValueError("existing source-only audit does not contain all five source folds")
    rows = []
    for item in asset_audit["checkpoints"]:
        if item.get("gene_input_size") != len(project_genes):
            raise ValueError("frozen source checkpoint input width differs from the EXP-036 axis")
        rows.append(
            {
                "fold": item["fold"],
                "checkpoint_sha256_from_prior_audit_not_recomputed": item["checkpoint_sha256"],
                "gene_input_size_from_prior_audit": item["gene_input_size"],
                "prior_state_dict_tensor_count": item["state_dict_n_tensors"],
                "current_checkpoint_payload_reopened": False,
            }
        )
    fold0 = rows[0]
    if (
        real_smoke.get("checkpoint_sha256")
        != fold0["checkpoint_sha256_from_prior_audit_not_recomputed"]
        or real_smoke.get("gene_input_size") != len(project_genes)
        or real_smoke.get("status") != "STRICT_RELOAD_AND_REAL_INPUT_SMOKE_PASS_ENGINEERING_ONLY"
    ):
        raise ValueError("existing fold-0 real-input smoke is not bound to the frozen gene axis")
    fold0["existing_real_input_smoke_status"] = real_smoke["status"]
    return rows


def audit() -> dict[str, Any]:
    datasets = _load_json(ROOT / "data/registry/datasets.json")
    transfer = _load_json(ROOT / "data/registry/datasets_transfer.json")
    dataset_entries = {item["id"]: item for item in datasets}
    transfer_entries = {item["id"]: item for item in transfer}
    depmap_expression_entry = dataset_entries["depmap_public_24q2_rnaseq_raw_v1"]
    depmap_dependency_entry = dataset_entries["depmap_public_24q2_crispr_gene_effect_raw_v1"]
    nature_raw_entry = transfer_entries["transfer_organoid_biobank_supplementary_tables"]
    nature_processed_entry = transfer_entries["transfer_organoid_biobank_paired_dependency_v1"]

    exp_manifest = _load_json(EXP036 / "manifest.json")
    project_genes = _validate_unique_axis(
        _load_json(EXP036 / "genes.json"), "EXP-036 gene axis"
    )
    contract_expected = exp_manifest["files_sha256"]["genes.json"]
    contract_actual = _sha256_file(EXP036 / "genes.json")
    if contract_actual != contract_expected or len(project_genes) != exp_manifest["n_common_genes"]:
        raise ValueError("EXP-036 gene axis differs from its registered manifest")
    source_genes = _load_npz_ordered_genes(SOURCE_INDEX)
    source_axis_matches_exp036 = _assert_exact_ordered_axis_match(project_genes, source_genes)
    existing_asset_audit = _load_json(EXP036_ASSET_AUDIT)
    real_smoke = _load_json(EXP036_REAL_SMOKE)
    checkpoints = _checkpoint_axis_evidence(project_genes, existing_asset_audit, real_smoke)

    dep_expr_path = DEPMAP_ROOT / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    dep_crispr_path = DEPMAP_ROOT / "CRISPRGeneEffect.csv"
    with dep_expr_path.open("r", encoding="utf-8-sig", newline="") as stream:
        dep_expr_columns = next(csv.reader(stream))
    with dep_crispr_path.open("r", encoding="utf-8-sig", newline="") as stream:
        dep_crispr_columns = next(csv.reader(stream))
    depmap_expression_genes = _depmap_gene_symbols(dep_expr_columns)
    depmap_dependency_genes = _depmap_gene_symbols(dep_crispr_columns)

    nature_expr_path = NATURE_ROOT / "expression_author_log2tpm.parquet"
    nature_dep_path = NATURE_ROOT / "dependency_by_library.parquet"
    nature_expression_columns = pq.ParquetFile(nature_expr_path).schema.names
    nature_expression_genes = _nature_expression_gene_symbols(nature_expression_columns)
    nature_dependency_table = pq.read_table(nature_dep_path, columns=["gene"])
    nature_dependency_values = [str(value).strip() for value in nature_dependency_table.column("gene").to_pylist()]
    if any(not value for value in nature_dependency_values):
        raise ValueError("Nature dependency gene field contains empty symbols")
    nature_dependency_genes = list(dict.fromkeys(nature_dependency_values))
    nature_dependency_genes = _validate_unique_axis(nature_dependency_genes, "Nature dependency gene axis")
    del nature_dependency_table, nature_dependency_values

    common_expression = _ordered_intersection(
        project_genes, depmap_expression_genes, nature_expression_genes
    )
    common_dependency_output = _ordered_intersection(
        project_genes, depmap_dependency_genes, nature_dependency_genes
    )
    if not common_expression or not common_dependency_output:
        raise ValueError("common D15 axes must be non-empty")
    feature_positions = {
        "exp036_project": _positions(project_genes, common_expression),
        "depmap_expression": _positions(depmap_expression_genes, common_expression),
        "nature_expression": _positions(nature_expression_genes, common_expression),
    }
    output_positions = {
        "exp036_project": _positions(project_genes, common_dependency_output),
        "depmap_chronos_header": _positions(depmap_dependency_genes, common_dependency_output),
        "nature_dependency_first_seen_gene_axis": _positions(nature_dependency_genes, common_dependency_output),
    }
    if feature_positions["exp036_project"] != sorted(feature_positions["exp036_project"]):
        raise ValueError("common expression axis must preserve EXP-036 source order")
    if output_positions["exp036_project"] != sorted(output_positions["exp036_project"]):
        raise ValueError("common dependency output axis must preserve EXP-036 source order")

    nature_admission = _load_json(ADMISSION)
    depmap_expression_metadata = _current_file_metadata(dep_expr_path)
    depmap_dependency_metadata = _current_file_metadata(dep_crispr_path)
    for entry, filename, metadata in (
        (depmap_expression_entry, dep_expr_path.name, depmap_expression_metadata),
        (depmap_dependency_entry, dep_crispr_path.name, depmap_dependency_metadata),
    ):
        if metadata["current_byte_size"] != int(entry["files"][filename]["bytes"]):
            raise ValueError(f"registered byte size differs for {filename}")
    nature_files = {}
    for name, path in (
        (nature_expr_path.name, nature_expr_path),
        (nature_dep_path.name, nature_dep_path),
        ("samples.parquet", NATURE_ROOT / "samples.parquet"),
    ):
        nature_files[name] = {
            **_current_file_metadata(path),
            "prior_admission_sha256": _nature_expected_sha(nature_admission, name),
            "current_full_payload_sha256_recomputed": False,
        }
    nature_zip_check = [
        item for item in nature_admission["checks"]
        if item.get("check") == "raw:41586_2026_10830_MOESM3_ESM.zip"
    ]
    nature_prior_admission_matches_registry = (
        len(nature_zip_check) == 1
        and bool(nature_zip_check[0].get("passed"))
        and nature_zip_check[0].get("expected") == nature_raw_entry["checksum"]["value"]
        and nature_zip_check[0].get("actual") == nature_raw_entry["checksum"]["value"]
    )
    if not nature_prior_admission_matches_registry:
        raise ValueError("Nature prior admission checksum does not match the registered raw asset")

    official_commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if official_commit != EXPECTED_OFFICIAL_COMMIT or official_commit != existing_asset_audit["official_repo_commit"]:
        raise ValueError("official PharmaFormer code revision differs from the frozen audit")
    return {
        "audit_id": "EXP-045-shared-expression-and-dependency-output-axes-v2",
        "status": "AXIS_IDENTITY_PASS_INPUT_AND_OUTPUT_LISTS_FROZEN_TRAINING_BLOCKED",
        "scope": {
            "read": [
                "registered SHA-256 values and source versions",
                "EXP-036 ordered genes.json",
                "only ordered_genes from data/processed/exp011/gdsc2_cell_line_indexed.npz",
                "prior source input-order and checkpoint metadata receipts",
                "DepMap 24Q2 expression and Chronos CSV headers only",
                "Nature expression Parquet schema only",
                "Nature dependency gene metadata column only",
                "prior source checkpoint SHA-256/dimension metadata and strict real-input smoke receipt only",
            ],
            "not_read": [
                "DepMap RNA expression values",
                "DepMap Chronos dependency values",
                "Nature expression values",
                "Nature dependency LFC values",
                "patient/cell/model identity fields",
                "DSS/drug-response labels, predictions, evaluation scores",
                "all EXP-011 NPZ members other than ordered_genes",
                "current checkpoint payloads",
                "full payloads of DepMap expression/Chronos or Nature expression/dependency assets for hash recomputation",
            ],
            "training_authorized": False,
        },
        "provenance": {
            "depmap_expression_registry": depmap_expression_entry["id"],
            "depmap_expression_version": depmap_expression_entry["version"],
            "depmap_expression_registered_sha256_not_recomputed": _registry_file_sha(depmap_expression_entry, dep_expr_path.name),
            "depmap_expression_current_file": depmap_expression_metadata,
            "depmap_expression_header_axis_sha256": _axis_sha256(depmap_expression_genes),
            "depmap_dependency_registry": depmap_dependency_entry["id"],
            "depmap_dependency_version": depmap_dependency_entry["version"],
            "depmap_dependency_registered_sha256_not_recomputed": _registry_file_sha(depmap_dependency_entry, dep_crispr_path.name),
            "depmap_dependency_current_file": depmap_dependency_metadata,
            "depmap_dependency_header_axis_sha256": _axis_sha256(depmap_dependency_genes),
            "nature_raw_registry": nature_raw_entry["id"],
            "nature_raw_version": nature_raw_entry["version"],
            "nature_raw_registered_sha256_not_recomputed": nature_raw_entry["checksum"]["value"],
            "nature_raw_prior_admission_matches_registry": bool(nature_prior_admission_matches_registry),
            "nature_processed_registry": nature_processed_entry["id"],
            "nature_processed_version": nature_processed_entry["version"],
            "nature_processed_files": nature_files,
            "exp036_manifest_sha256": _sha256_file(EXP036 / "manifest.json"),
            "exp036_gene_file_sha256": contract_actual,
            "source_index_gene_axis_sha256": _axis_sha256(source_genes),
            "source_index_axis_recheck": {
                "path": str(SOURCE_INDEX.resolve()),
                "repository_relative_path": SOURCE_INDEX.relative_to(ROOT).as_posix(),
                "npz_member": "ordered_genes",
                "ordered_genes_count": len(source_genes),
                "ordered_genes_axis_sha256": _axis_sha256(source_genes),
                "matches_exp036_genes_json_exact_order": source_axis_matches_exp036,
                "other_npz_members_read": [],
                "full_archive_sha256_recomputed": False,
            },
            "existing_source_index_sha256_prior_receipt_not_recomputed": existing_asset_audit["source_index_sha256"],
            "historical_source_index_sha_binding": {
                "asset_audit_receipt_path": EXP036_ASSET_AUDIT.relative_to(ROOT).as_posix(),
                "source_index_path_recorded_in_prior_receipt": "source_index_path" in existing_asset_audit,
                "source_index_sha256_from_prior_receipt_not_recomputed": existing_asset_audit["source_index_sha256"],
                "old_full_file_sha_to_current_path_correspondence_reprovable": False,
                "limitation": "The historical asset audit receipt did not save the source-index path, so its prior full-file SHA cannot currently be bound again to this path.",
            },
            "source_axis_binding": "Current source ordered_genes was read from the recorded EXP-011 NPZ path and exactly compared with EXP-036 genes.json. The historical asset audit omitted that path, so its prior full-file SHA correspondence to this path cannot be re-proven.",
            "official_pharmaformer_commit": official_commit,
            "exp036_asset_audit_sha256": _sha256_file(EXP036_ASSET_AUDIT),
            "source_checkpoints": checkpoints,
        },
        "normalization": {
            "exp036": "registered genes.json strings, whitespace stripped only",
            "depmap": "feature headers SYMBOL (EntrezID), retain exact SYMBOL; one leading entity-key column excluded",
            "nature_expression": "remove terminal _exp suffix only, as in registered EXP-026 helper",
            "nature_dependency": "gene field exact symbols, whitespace stripped only, first-seen order for source position map",
            "cross_resource_matching": "exact case-sensitive gene symbols only; no aliases, case folding, Entrez inference, or fuzzy matching",
            "duplicate_symbol_policy": "fail closed; no silent collapsing",
            "common_axis_order": "preserve EXP-036 source gene order",
        },
        "feature_input_axis": {
            "n_exp036_genes": len(project_genes),
            "n_depmap_expression_genes": len(depmap_expression_genes),
            "n_nature_expression_genes": len(nature_expression_genes),
            "n_depmap_expression_project_intersection": len(set(project_genes) & set(depmap_expression_genes)),
            "n_nature_expression_project_intersection": len(set(project_genes) & set(nature_expression_genes)),
            "n_common_expression_genes": len(common_expression),
            "n_exp036_input_genes_excluded_from_three_way_axis": len(project_genes) - len(common_expression),
            "n_nature_project_genes_missing_from_depmap_expression": len(
                (set(project_genes) & set(nature_expression_genes)) - set(depmap_expression_genes)
            ),
            "ordered_axis_sha256": _axis_sha256(common_expression),
            "ordered_genes": common_expression,
            "column_positions": feature_positions,
        },
        "dependency_output_axis": {
            "n_exp036_genes": len(project_genes),
            "n_depmap_chronos_genes": len(depmap_dependency_genes),
            "n_nature_dependency_genes": len(nature_dependency_genes),
            "n_depmap_dependency_project_intersection": len(set(project_genes) & set(depmap_dependency_genes)),
            "n_nature_dependency_project_intersection": len(set(project_genes) & set(nature_dependency_genes)),
            "n_common_dependency_output_genes": len(common_dependency_output),
            "n_exp036_genes_excluded_from_common_dependency_output": len(project_genes) - len(common_dependency_output),
            "ordered_axis_sha256": _axis_sha256(common_dependency_output),
            "ordered_genes": common_dependency_output,
            "column_positions": output_positions,
            "independent_domain_heads_and_missing_masks_required": True,
        },
        "next_gates": [
            "independently review this axis receipt and exact source-order evidence",
            "freeze D15 all-arm input slice and both output-head gene order/masks in EXP-045 config",
            "implement/test explicit gene_fc1 column slice and source-transform vector selection; no zero fill",
            "freeze patient/Source identity exclusions and independent functional holdout before reading dependency values",
            "fit per-domain and per-PDO fold preprocessing on allowed training entities, then verify real-input read-write alignment",
            "measure full slow-fold cost and obtain final Sol GO before any training",
        ],
    }


def write_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"EXP-045 gene-axis audit output exists: {output_dir}")
    receipt = audit()
    output_dir.mkdir(parents=True)
    (output_dir / "axis_contract.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    receipt = write_audit(args.output_dir)
    print(
        "status={}; common_expression_genes={}; common_dependency_output_genes={}; training_authorized={}".format(
            receipt["status"],
            receipt["feature_input_axis"]["n_common_expression_genes"],
            receipt["dependency_output_axis"]["n_common_dependency_output_genes"],
            receipt["scope"]["training_authorized"],
        )
    )


if __name__ == "__main__":
    main()
