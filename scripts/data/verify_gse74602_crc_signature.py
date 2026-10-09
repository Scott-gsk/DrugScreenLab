"""从原始矩阵、注释与官方配对独立复算 EXP-038 GSE74602 开发签名。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp
from statsmodels.stats.multitest import multipletests

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_annotation(path: Path) -> pd.DataFrame:
    """Read the GPL annotation independently of the signature builder."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line.startswith("ID\tGene title\tGene symbol\t"):
                break
        else:
            raise ValueError("GPL6104 annotation table header not found")
    frame = pd.read_csv(path, sep="\t", skiprows=line_number, dtype=str, keep_default_na=False)
    frame = frame.loc[frame["ID"].str.startswith("ILMN_")].copy()
    if frame["ID"].duplicated().any():
        raise ValueError("duplicate GPL6104 probe identity")
    return frame[["ID", "Gene symbol"]]


def rank_mean_normalize(values: np.ndarray) -> np.ndarray:
    """Independent implementation of per-column ranks onto mean sorted values."""
    sorted_values = np.sort(values, axis=0)
    reference = sorted_values.mean(axis=1)
    output = np.empty(values.shape, dtype=np.float64)
    for column in range(values.shape[1]):
        ranks = rankdata(values[:, column], method="average")
        output[:, column] = np.interp(ranks, np.arange(1, len(reference) + 1), reference)
    return output


def verify(matrix_path: Path, annotation_path: Path, pair_audit_path: Path,
           lincs_gene_info_path: Path, signature_dir: Path) -> dict:
    manifest = json.loads((signature_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(pair_audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS_OFFICIAL_NUMBERED_TUMOR_NORMAL_PAIRING":
        raise ValueError("official pair audit is not PASS")
    expected_sources = {"matrix_sha256": sha256(matrix_path),
                        "annotation_sha256": sha256(annotation_path),
                        "official_pair_audit_sha256": sha256(pair_audit_path),
                        "lincs_gene_info_sha256": sha256(lincs_gene_info_path)}
    if manifest["source"] != expected_sources:
        raise ValueError("signature source checksum differs from registered inputs")
    matrix = pd.read_csv(matrix_path, sep="\t")
    source = matrix.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(source).all() or (source <= 0).any():
        raise ValueError("source signals must be finite positive")
    normalized = np.log2(rank_mean_normalize(source))
    annotation = load_annotation(annotation_path).set_index("ID")
    symbols = annotation.reindex(matrix["ID_REF"].astype(str))["Gene symbol"].fillna("").astype(str)
    keep = symbols.str.fullmatch(r"[A-Za-z][A-Za-z0-9-]*").fillna(False).to_numpy()
    gene_matrix = pd.DataFrame(
        normalized[keep], index=symbols.iloc[np.flatnonzero(keep)].str.upper(),
        columns=matrix.columns[1:]).groupby(level=0, sort=True).median()
    tumors = [pair["tumor_sample"] for pair in audit["pairs"]]
    normals = [pair["normal_sample"] for pair in audit["pairs"]]
    difference = gene_matrix[tumors].to_numpy(dtype=np.float64) - gene_matrix[normals].to_numpy(dtype=np.float64)
    test = ttest_1samp(difference, 0.0, axis=1, nan_policy="raise")
    pvalue = np.asarray(test.pvalue, dtype=np.float64)
    pvalue[~np.isfinite(pvalue)] = 1.0
    expected = pd.DataFrame({
        "gene_symbol": gene_matrix.index.astype(str),
        "mean_paired_log2fc": difference.mean(axis=1),
        "p_value": pvalue,
        "q_value": multipletests(pvalue, method="fdr_bh")[1],
        "n_pairs": 30,
    })
    actual = pd.read_parquet(signature_dir / "all_gene_statistics.parquet")
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-12)
    saved_gene_order = json.loads((signature_dir / "all_gene_order.json").read_text(encoding="utf-8"))
    if saved_gene_order != expected["gene_symbol"].tolist():
        raise ValueError("saved gene axis differs from independently aggregated symbols")
    stored_difference = np.load(signature_dir / "gene_pair_differences.npy")
    if stored_difference.shape != difference.shape:
        raise ValueError("stored paired-difference shape mismatch")
    max_difference_error = float(np.max(np.abs(stored_difference - difference)))
    if max_difference_error > 2e-6:
        raise ValueError(f"stored paired differences mismatch: {max_difference_error}")
    landmarks = pd.read_csv(lincs_gene_info_path, sep="\t", dtype=str, keep_default_na=False)
    landmarks = landmarks.loc[landmarks["pr_is_lm"].eq("1"), ["pr_gene_id", "pr_gene_symbol"]].copy()
    landmarks["gene_symbol"] = landmarks["pr_gene_symbol"].str.upper()
    expected_landmarks = landmarks.merge(expected, on="gene_symbol", how="left", validate="one_to_one", sort=False)
    actual_landmarks = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    pd.testing.assert_frame_equal(actual_landmarks, expected_landmarks, check_exact=False, rtol=0, atol=1e-12)
    eligible = expected_landmarks.loc[expected_landmarks["q_value"].lt(0.05)
                                               & expected_landmarks["mean_paired_log2fc"].abs().ge(0.5)]
    up = eligible.loc[eligible["mean_paired_log2fc"].gt(0)].sort_values(
        ["mean_paired_log2fc", "gene_symbol"], ascending=[False, True]).head(100)
    down = eligible.loc[eligible["mean_paired_log2fc"].lt(0)].assign(
        abs_effect=lambda rows: rows["mean_paired_log2fc"].abs()).sort_values(
            ["abs_effect", "gene_symbol"], ascending=[False, True]).head(100)
    stored_up = json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    stored_down = json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if stored_up != up["pr_gene_id"].astype(str).tolist() or stored_down != down["pr_gene_id"].astype(str).tolist():
        raise ValueError("directional signature selection differs from fixed threshold and ordering")
    if (manifest["status"] != "SIGNATURE_READY_FOR_DEVELOPMENT"
            or manifest["n_probes_raw"] != len(matrix)
            or manifest["n_probes_with_valid_gene_symbol"] != int(keep.sum())
            or manifest["n_genes_tested"] != len(expected)
            or manifest["n_landmark_genes_mapped"] != int(expected_landmarks["q_value"].notna().sum())
            or manifest["n_up_selected"] != len(stored_up)
            or manifest["n_down_selected"] != len(stored_down)
            or len(stored_up) < 20 or len(stored_down) < 20):
        raise ValueError("signature manifest counts or status disagree with recomputation")
    files = {name: sha256(signature_dir / name) for name in manifest["files_sha256"]}
    if files != manifest["files_sha256"]:
        raise ValueError("signature file checksum mismatch")
    return {
        "status": "PASS",
        "n_genes": len(expected), "n_landmarks": int(expected_landmarks["q_value"].notna().sum()),
        "n_up": len(stored_up), "n_down": len(stored_down),
        "max_paired_difference_error": max_difference_error,
        "source_sha256": {"matrix": sha256(matrix_path), "annotation": sha256(annotation_path),
                          "pair_audit": sha256(pair_audit_path), "lincs_gene_info": sha256(lincs_gene_info_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--lincs-gene-info", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.matrix, args.annotation, args.pair_audit,
                    args.lincs_gene_info, args.signature_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
