"""Build a response-blind DepMap 24Q2 RNA-seq → ordered-978 context adapter.

Official source: Broad DepMap Public 24Q2 on figshare.plus
(DOI 10.25452/figshare.plus.25880521.v1).  Later 26Q1 is announced on
depmap.org but the public figshare.plus bulk RNA-seq file for 26Q1 was
not downloadable from this session (portal WAF).  24Q2 is therefore the
PRIMARY_SOURCE_VERIFICATION release actually retrieved from official
Figshare plus.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.data.lincs_landmarks import (
    CRC_EXACT_CONTEXTS,
    ORDERED_GENE_IDS_SHA256,
    gene_order_digest,
    ordered_landmark_gene_ids,
)
from drug_screen.evaluation.phase1_prism import spearman

GENE = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
GCTX = ROOT / (
    "data/interim/lincs/GSE92742/"
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
)
CONTEXT_REGISTRY = ROOT / "mvp/foundation/xpert/CONTEXT_REGISTRY.json"
DEFAULT_EXPR = ROOT / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
DEFAULT_MODEL = ROOT / "data/raw/depmap/24q2/Model.csv"
OUT_DIR = ROOT / "data/processed/depmap/24q2_rnaseq_exact978"
INTAKE = ROOT / "artifacts/experiments/EXP-007/CCLE_RNASEQ_INTAKE.json"
_ENTREZ = re.compile(r"\((\d+)\)\s*$")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gctx_row_ids() -> list[str]:
    import h5py

    with h5py.File(GCTX, "r") as handle:
        values = handle["0/META/ROW/id"][:]
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _entrez(column: str) -> str | None:
    match = _ENTREZ.search(str(column))
    return match.group(1) if match else None


def _within_sample_rank(matrix: np.ndarray) -> np.ndarray:
    ranks = np.empty(matrix.shape, dtype=np.float32)
    for i, row in enumerate(matrix):
        finite = np.isfinite(row)
        values = row.copy()
        if not finite.all():
            fill = float(np.nanmean(row)) if finite.any() else 0.0
            values[~finite] = fill
        order = np.argsort(values, kind="mergesort")
        ranked = np.empty_like(values, dtype=np.float32)
        ranked[order] = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
        ranks[i] = ranked
    return ranks


def _quantile_bin(matrix: np.ndarray, n_bins: int = 10) -> np.ndarray:
    ranks = _within_sample_rank(matrix)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    return np.floor(ranks * n_bins).clip(0, n_bins - 1).astype(np.float32)


def _relative_expression(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.nanmean(matrix, axis=1, keepdims=True)
    scale = np.nanstd(matrix, axis=1, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (centered / scale).astype(np.float32)


def _train_fitted_scaling(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = matrix[train_mask]
    if len(train) == 0:
        raise ValueError("train-fitted scaling requires a non-empty train role")
    mean = np.nanmean(train, axis=0, keepdims=True)
    std = np.nanstd(train, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return ((matrix - mean) / std).astype(np.float32)


def _pairwise_mean_spearman(left: np.ndarray, right: np.ndarray, limit: int = 48) -> float | None:
    n = min(len(left), limit)
    if n == 0:
        return None
    values = []
    for i in range(n):
        corr = spearman(left[i], right[i])
        if corr is not None:
            values.append(corr)
    return float(np.mean(values)) if values else None


def build(
    *,
    expr_path: Path = DEFAULT_EXPR,
    model_path: Path = DEFAULT_MODEL,
    output_dir: Path = OUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_info = pd.read_csv(GENE, sep="\t", dtype=str)
    landmark_ids = ordered_landmark_gene_ids(gene_info, gctx_row_ids=_gctx_row_ids())
    digest = gene_order_digest(landmark_ids)
    if digest != ORDERED_GENE_IDS_SHA256:
        raise RuntimeError(f"gene-order sha256 mismatch: {digest}")
    id_to_symbol = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_gene_symbol"].astype(str)))
    landmark_symbols = [id_to_symbol[gene_id] for gene_id in landmark_ids]

    models = pd.read_csv(model_path)
    expr = pd.read_csv(expr_path)
    first = expr.columns[0]
    if first == "" or first.lower() in {"unnamed: 0", "modelid", "depmap_id"}:
        expr = expr.rename(columns={first: "ModelID"})
    if "ModelID" not in expr.columns:
        raise RuntimeError(f"expression matrix lacks ModelID; columns start {list(expr.columns[:5])}")

    gene_cols = [col for col in expr.columns if col != "ModelID"]
    entrez_by_col = {col: _entrez(col) for col in gene_cols}
    present_entrez = {value for value in entrez_by_col.values() if value}
    missing = [gene_id for gene_id in landmark_ids if gene_id not in present_entrez]
    selected_cols: list[str | None] = []
    ambiguous: list[str] = []
    for gene_id in landmark_ids:
        matches = [col for col, entrez in entrez_by_col.items() if entrez == gene_id]
        if len(matches) == 1:
            selected_cols.append(matches[0])
        elif len(matches) > 1:
            ambiguous.append(gene_id)
            selected_cols.append(None)
        else:
            selected_cols.append(None)
    if ambiguous:
        raise RuntimeError(f"ambiguous Entrez mapping for landmark genes: {ambiguous[:8]}")

    matrix = np.full((len(expr), 978), np.nan, dtype=np.float32)
    for index, col in enumerate(selected_cols):
        if col is None:
            continue
        matrix[:, index] = pd.to_numeric(expr[col], errors="coerce").to_numpy(np.float32)
    finite_per_gene = np.isfinite(matrix).sum(axis=0)
    finite_per_line = np.isfinite(matrix).sum(axis=1)

    registry = json.loads(CONTEXT_REGISTRY.read_text(encoding="utf-8"))
    lincs_contexts = {str(row["context_id"]) for row in registry.get("contexts", [])}
    models["StrippedCellLineName"] = models["StrippedCellLineName"].astype(str).str.upper()
    models["ModelID"] = models["ModelID"].astype(str)
    expr["ModelID"] = expr["ModelID"].astype(str)
    overlap_by_stripped = set(models["StrippedCellLineName"]) & {name.upper() for name in lincs_contexts}
    overlap_crc = sorted(name for name in CRC_EXACT_CONTEXTS if name.upper() in overlap_by_stripped)
    model_to_stripped = dict(zip(models["ModelID"], models["StrippedCellLineName"]))
    expr["stripped"] = expr["ModelID"].map(model_to_stripped)
    expr_overlap = expr.loc[expr["stripped"].isin({name.upper() for name in lincs_contexts})].copy()
    crc_mask = expr["stripped"].isin({name.upper() for name in CRC_EXACT_CONTEXTS})
    train_mask = (~crc_mask).to_numpy()
    if int(train_mask.sum()) == 0:
        train_mask = np.ones(len(expr), dtype=bool)

    normalizations = {
        "raw_log2tpm": matrix,
        "within_sample_rank": _within_sample_rank(matrix),
        "quantile_bin10": _quantile_bin(matrix, n_bins=10),
        "relative_expression": _relative_expression(matrix),
        "train_fitted_zscore": _train_fitted_scaling(matrix, train_mask),
    }
    comparison = {}
    raw = normalizations["raw_log2tpm"]
    for name, values in normalizations.items():
        comparison[name] = {
            "mean_spearman_vs_raw_first48": _pairwise_mean_spearman(raw, values),
            "finite_fraction": float(np.isfinite(values).mean()),
        }

    keep = np.isfinite(matrix).all(axis=1)
    context_rows = []
    for position, row in expr.loc[keep].reset_index(drop=True).iterrows():
        context_rows.append(
            {
                "depmap_id": str(row["ModelID"]),
                "stripped_cell_line_name": str(row["stripped"]) if pd.notna(row["stripped"]) else None,
                "exact978_row": int(position),
                "lincs_overlap": bool(pd.notna(row["stripped"]) and str(row["stripped"]).upper() in {n.upper() for n in lincs_contexts}),
                "crc_exact_context": bool(pd.notna(row["stripped"]) and str(row["stripped"]).upper() in {n.upper() for n in CRC_EXACT_CONTEXTS}),
            }
        )
    kept_matrix = matrix[keep]
    adapter_path = output_dir / "ccle_24q2_exact978_log2tpm.npy"
    mapping_path = output_dir / "ccle_24q2_exact978_mapping.json"
    np.save(adapter_path, kept_matrix.astype(np.float32))
    mapping_path.write_text(json.dumps(context_rows, indent=2) + "\n", encoding="utf-8")

    audit = {
        "format": "ccle_rnaseq_intake_v1",
        "status": "DATA_PARTIAL",
        "primary_source_verification": {
            "release": "DepMap Public 24Q2",
            "official_title": "DepMap 24Q2 Public",
            "doi": "10.25452/figshare.plus.25880521.v1",
            "figshare_article": "https://figshare.com/articles/dataset/DepMap_24Q2_Public/25880521",
            "release_notes_url": "https://forum.depmap.org/t/announcing-the-26q1-release/4606",
            "current_portal_release_announced": "DepMap Public 26Q1",
            "current_portal_file": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
            "why_not_26q1": (
                "depmap.org portal download is WAF/HTML gated from this session; "
                "figshare.plus 26Q1 article was not resolvable. 24Q2 is the latest "
                "official public RNA-seq matrix successfully retrieved from Broad DepMap figshare.plus."
            ),
            "expression_file": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
            "expression_file_id": 46490878,
            "expression_url": "https://ndownloader.figshare.com/files/46490878",
            "model_file": "Model.csv",
            "model_file_id": 46489732,
            "model_url": "https://ndownloader.figshare.com/files/46489732",
            "description": "Gene expression TPM values of protein-coding genes; log2(TPM+1).",
            "pipeline": "https://github.com/broadinstitute/depmap_omics",
        },
        "local": {
            "expression_path": str(expr_path.relative_to(ROOT).as_posix()),
            "model_path": str(model_path.relative_to(ROOT).as_posix()),
            "expression_bytes": int(expr_path.stat().st_size),
            "model_bytes": int(model_path.stat().st_size),
            "expression_sha256": _sha256(expr_path),
            "model_sha256": _sha256(model_path),
            "expression_shape": [int(len(expr)), int(len(gene_cols))],
        },
        "gene_harmonization": {
            "ordered_gene_ids_sha256": digest,
            "mapping_key": "DepMap column Entrez ID in parentheses -> GSE92742 pr_gene_id",
            "landmark_genes": 978,
            "mapped_genes": int(978 - len(missing)),
            "missing_genes": missing,
            "missing_count": int(len(missing)),
            "ambiguous_count": 0,
            "duplicate_gene_rule": "require unique Entrez match; no symbol fallback",
        },
        "overlap": {
            "lincs_registry_contexts": int(len(lincs_contexts)),
            "depmap_models": int(models["ModelID"].nunique()),
            "expression_models": int(expr["ModelID"].nunique()),
            "lincs_overlap_by_stripped_name": int(len(overlap_by_stripped)),
            "crc_exact_overlap": overlap_crc,
            "crc_exact_overlap_count": int(len(overlap_crc)),
            "expression_rows_overlapping_lincs": int(len(expr_overlap)),
            "join_key": "Model.StrippedCellLineName == LINCS context_id (uppercase exact)",
        },
        "normalization_probe": {
            "compared": list(normalizations),
            "fit_population": "train_fitted_zscore uses all non-CRC-exact DepMap lines; others are within-sample",
            "not_architecture_search": True,
            "comparison": comparison,
        },
        "adapter_output": {
            "matrix": str(adapter_path.relative_to(ROOT).as_posix()),
            "mapping": str(mapping_path.relative_to(ROOT).as_posix()),
            "rows_complete_978": int(keep.sum()),
            "shape": [int(keep.sum()), 978],
            "dtype": "float32",
            "units": "official log2(TPM+1); not LINCS X_ctl",
            "cannot_replace_matched_control": True,
        },
        "response_blind": True,
    }
    INTAKE.parent.mkdir(parents=True, exist_ok=True)
    INTAKE.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "intake_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=Path, default=DEFAULT_EXPR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    audit = build(expr_path=args.expr, model_path=args.model)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "release": audit["primary_source_verification"]["release"],
                "mapped_genes": audit["gene_harmonization"]["mapped_genes"],
                "missing_genes": audit["gene_harmonization"]["missing_count"],
                "lincs_overlap": audit["overlap"]["lincs_overlap_by_stripped_name"],
                "crc_exact_overlap": audit["overlap"]["crc_exact_overlap_count"],
                "complete_rows": audit["adapter_output"]["rows_complete_978"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
