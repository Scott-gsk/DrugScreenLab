"""按 EXP-038 预登记协议构建 GSE74602 开发用结直肠癌方向签名。"""

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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_annotation(path: Path) -> pd.DataFrame:
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


def quantile_normalize_positive(raw: np.ndarray) -> np.ndarray:
    """Quantile-normalize arrays; tied intensities receive their rank-average value."""
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("array intensities must be finite and positive")
    target_quantiles = np.sort(values, axis=0).mean(axis=1)
    output = np.empty_like(values)
    for column in range(values.shape[1]):
        ranks = rankdata(values[:, column], method="average")
        output[:, column] = np.interp(
            ranks, np.arange(1, len(target_quantiles) + 1), target_quantiles)
    return output


def build(matrix_path: Path, annotation_path: Path, pair_audit_path: Path,
          lincs_gene_info_path: Path, output_dir: Path) -> dict:
    audit = json.loads(pair_audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS_OFFICIAL_NUMBERED_TUMOR_NORMAL_PAIRING":
        raise ValueError("GSE74602 official pair audit is not PASS")
    matrix = pd.read_csv(matrix_path, sep="\t")
    if matrix["ID_REF"].duplicated().any():
        raise ValueError("duplicate matrix probe ID")
    if set(matrix.columns[1:]) != {sample for pair in audit["pairs"]
                                    for sample in (pair["tumor_sample"], pair["normal_sample"])}:
        raise ValueError("GSE74602 matrix sample axis differs from official SOFT pairs")
    probes = matrix["ID_REF"].astype(str)
    signal = matrix.iloc[:, 1:].to_numpy(dtype=np.float64)
    normalized = np.log2(quantile_normalize_positive(signal))
    annotation = load_annotation(annotation_path).set_index("ID")
    symbols = annotation.reindex(probes)["Gene symbol"].fillna("").astype(str)
    valid_gene_symbol = symbols.str.fullmatch(r"[A-Za-z][A-Za-z0-9-]*")
    valid = valid_gene_symbol.fillna(False).to_numpy()
    if valid.sum() < 1000:
        raise ValueError("too few uniquely annotated probes")
    values = pd.DataFrame(normalized[valid], index=symbols.iloc[np.flatnonzero(valid)].str.upper(),
                          columns=matrix.columns[1:])
    # All probes for a gene are combined within each sample before paired testing.
    gene_values = values.groupby(level=0, sort=True).median()
    tumor_names = [pair["tumor_sample"] for pair in audit["pairs"]]
    normal_names = [pair["normal_sample"] for pair in audit["pairs"]]
    paired_diff = (gene_values[tumor_names].to_numpy(dtype=np.float64)
                   - gene_values[normal_names].to_numpy(dtype=np.float64))
    if paired_diff.shape[1] != 30 or not np.isfinite(paired_diff).all():
        raise ValueError("paired gene differences must be finite [genes, 30]")
    statistic = ttest_1samp(paired_diff, popmean=0.0, axis=1, nan_policy="raise")
    pvalues = np.asarray(statistic.pvalue, dtype=np.float64)
    pvalues[~np.isfinite(pvalues)] = 1.0  # constant paired differences remain non-significant
    qvalues = multipletests(pvalues, method="fdr_bh")[1]
    all_genes = pd.DataFrame({"gene_symbol": gene_values.index.astype(str),
                              "mean_paired_log2fc": paired_diff.mean(axis=1),
                              "p_value": pvalues, "q_value": qvalues,
                              "n_pairs": 30})
    landmarks = pd.read_csv(lincs_gene_info_path, sep="\t", dtype=str, keep_default_na=False)
    landmarks = landmarks.loc[landmarks["pr_is_lm"].eq("1")]
    if len(landmarks) != 978 or landmarks["pr_gene_symbol"].duplicated().any():
        raise ValueError("LINCS landmark gene symbols must be 978 unique")
    axis = landmarks[["pr_gene_id", "pr_gene_symbol"]].copy()
    axis["gene_symbol"] = axis["pr_gene_symbol"].str.upper()
    if axis["gene_symbol"].duplicated().any():
        raise ValueError("case-folded landmark gene symbols collide")
    aligned = axis.merge(all_genes, on="gene_symbol", how="left", validate="one_to_one", sort=False)
    if len(aligned) != 978:
        raise ValueError("landmark alignment changed axis length")
    eligible = aligned.loc[aligned["q_value"].lt(0.05)
                           & aligned["mean_paired_log2fc"].abs().ge(0.5)].copy()
    up = eligible.loc[eligible["mean_paired_log2fc"].gt(0)].sort_values(
        ["mean_paired_log2fc", "gene_symbol"], ascending=[False, True]).head(100)
    down = eligible.loc[eligible["mean_paired_log2fc"].lt(0)].assign(
        abs_effect=lambda x: x["mean_paired_log2fc"].abs()).sort_values(
            ["abs_effect", "gene_symbol"], ascending=[False, True]).head(100)
    if output_dir.exists():
        raise FileExistsError(f"signature output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    all_path = output_dir / "all_gene_statistics.parquet"
    axis_path = output_dir / "landmark_statistics.parquet"
    pair_path = output_dir / "gene_pair_differences.npy"
    all_genes.to_parquet(all_path, index=False)
    aligned.to_parquet(axis_path, index=False)
    np.save(pair_path, paired_diff.astype(np.float32))
    (output_dir / "all_gene_order.json").write_text(
        json.dumps(all_genes["gene_symbol"].tolist()) + "\n", encoding="utf-8")
    (output_dir / "up_genes.json").write_text(
        json.dumps(up["pr_gene_id"].astype(str).tolist()) + "\n", encoding="utf-8")
    (output_dir / "down_genes.json").write_text(
        json.dumps(down["pr_gene_id"].astype(str).tolist()) + "\n", encoding="utf-8")
    report = {
        "experiment_id": "EXP-038", "protocol": "GSE74602 CRC development signature v2",
        "status": "SIGNATURE_READY_FOR_DEVELOPMENT" if len(up) >= 20 and len(down) >= 20
                  else "SIGNATURE_MAIN_GATE_FAILED",
        "source": {"matrix_sha256": sha256(matrix_path),
                   "annotation_sha256": sha256(annotation_path),
                   "official_pair_audit_sha256": sha256(pair_audit_path),
                   "lincs_gene_info_sha256": sha256(lincs_gene_info_path)},
        "normalization": "quantile normalization of positive processed array intensities across 60 samples, then log2; no bead-level background data available",
        "test": "two-sided paired one-sample t over 30 explicitly numbered Tn-Nn sample pairs; BH across all mapped genes",
        "patient_independence": "unknown; title prefixes are not verified patient IDs",
        "n_probes_raw": len(matrix), "n_probes_with_valid_gene_symbol": int(valid.sum()),
        "n_genes_tested": len(all_genes), "n_landmark_genes_mapped": int(aligned["q_value"].notna().sum()),
        "n_up_selected": len(up), "n_down_selected": len(down),
        "threshold": {"q_less_than": 0.05, "abs_log2fc_at_least": 0.5,
                      "max_each_direction": 100, "min_each_direction": 20},
        "files_sha256": {path.name: sha256(path) for path in output_dir.iterdir() if path.is_file()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--lincs-gene-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.matrix, args.annotation, args.pair_audit,
                   args.lincs_gene_info, args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "status", "n_probes_raw", "n_probes_with_valid_gene_symbol",
        "n_genes_tested", "n_landmark_genes_mapped", "n_up_selected", "n_down_selected")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
