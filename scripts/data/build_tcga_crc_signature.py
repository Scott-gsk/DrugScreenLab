"""Build a paired CRC disease signature from frozen open GDC STAR TPM files."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tpm(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", usecols=[
        "gene_id", "gene_name", "tpm_unstranded"], dtype={"gene_id": str, "gene_name": str})
    if frame["gene_id"].duplicated().any():
        raise ValueError(f"duplicate Ensembl gene identity in {path}")
    return frame


def build(pairs_path: Path, raw_dir: Path, gene_info_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"TCGA signature output already exists: {output_dir}")
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    receipts = {item["file_id"]: item for item in manifest["files"]}
    if (len(pairs) != 50 or manifest["status"] != "RAW_GDC_FILES_VERIFIED"
            or len(receipts) != 100 or manifest["selected_pairs_sha256"] != sha256(pairs_path)):
        raise ValueError("frozen paired files are incomplete or disagree with raw manifest")
    first = read_tpm(raw_dir / receipts[pairs[0]["tumor"]["file_id"]]["relative_path"])
    valid = (first["gene_id"].str.startswith("ENSG")
             & first["gene_name"].fillna("").str.fullmatch(r"[A-Za-z][A-Za-z0-9-]*"))
    if valid.sum() < 1000:
        raise ValueError("too few valid gene-name rows in GDC STAR counts")
    gene_ids = first["gene_id"].to_numpy()
    gene_names = first["gene_name"].fillna("").to_numpy()
    symbols = first.loc[valid, "gene_name"].str.upper().to_numpy()
    symbol_axis = sorted(set(symbols))
    symbol_index = pd.Index(symbol_axis).get_indexer(symbols)
    if (symbol_index < 0).any():
        raise ValueError("gene symbol grouping failed")
    difference_columns = []
    for pair in pairs:
        tpm_by_role = {}
        for role in ("tumor", "normal"):
            part = pair[role]
            receipt = receipts[part["file_id"]]
            path = raw_dir / receipt["relative_path"]
            if (receipt["md5"] != part["md5sum"] or receipt["size"] != part["file_size"]
                    or sha256(path) != receipt["sha256"]):
                raise ValueError("GDC source file checksum or identity differs from frozen manifest")
            frame = read_tpm(path)
            if (not np.array_equal(frame["gene_id"].to_numpy(), gene_ids)
                    or not np.array_equal(frame["gene_name"].fillna("").to_numpy(), gene_names)):
                raise ValueError("GDC files have different ordered gene identity axes")
            tpm = frame["tpm_unstranded"].to_numpy(dtype=np.float64)
            if not np.isfinite(tpm[valid]).all() or (tpm[valid] < 0).any():
                raise ValueError("GDC STAR TPM contains missing, infinite, or negative values")
            tpm_by_role[role] = np.log2(1 + np.bincount(
                symbol_index, weights=tpm[valid], minlength=len(symbol_axis)))
        difference_columns.append(tpm_by_role["tumor"] - tpm_by_role["normal"])
    paired_difference = np.stack(difference_columns, axis=1)
    if paired_difference.shape != (len(symbol_axis), 50):
        raise ValueError("paired TCGA gene difference axis differs from 50 cases")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_value = np.asarray(ttest_1samp(paired_difference, 0.0, axis=1).pvalue,
                             dtype=np.float64)
    undefined_tests = int((~np.isfinite(p_value)).sum())
    p_value[~np.isfinite(p_value)] = 1.0
    q_value = multipletests(p_value, method="fdr_bh")[1]
    statistics = pd.DataFrame({"gene_symbol": symbol_axis,
                               "mean_paired_log2fc": paired_difference.mean(axis=1),
                               "p_value": p_value, "q_value": q_value, "n_pairs": 50})
    landmarks = pd.read_csv(gene_info_path, sep="\t", dtype=str, keep_default_na=False)
    landmarks = landmarks.loc[landmarks["pr_is_lm"].eq("1")]
    if len(landmarks) != 978 or landmarks["pr_gene_id"].duplicated().any():
        raise ValueError("LINCS landmark axis must contain exactly 978 unique gene IDs")
    axis = landmarks[["pr_gene_id", "pr_gene_symbol"]].copy()
    axis["gene_symbol"] = axis["pr_gene_symbol"].str.upper()
    if axis["gene_symbol"].duplicated().any():
        raise ValueError("case-folded LINCS gene symbols are duplicated")
    aligned = axis.merge(statistics, on="gene_symbol", how="left", validate="one_to_one", sort=False)
    eligible = aligned.loc[aligned["q_value"].lt(0.05)
                           & aligned["mean_paired_log2fc"].abs().ge(0.5)].copy()
    up = eligible.loc[eligible["mean_paired_log2fc"].gt(0)].sort_values(
        ["mean_paired_log2fc", "gene_symbol"], ascending=[False, True]).head(100)
    down = eligible.loc[eligible["mean_paired_log2fc"].lt(0)].assign(
        absolute_effect=lambda frame: frame["mean_paired_log2fc"].abs()).sort_values(
        ["absolute_effect", "gene_symbol"], ascending=[False, True]).head(100)
    output_dir.mkdir(parents=True)
    files = {"all_gene_statistics.parquet": statistics,
             "landmark_statistics.parquet": aligned}
    for name, frame in files.items():
        frame.to_parquet(output_dir / name, index=False)
    np.save(output_dir / "gene_pair_differences.npy", paired_difference.astype(np.float32))
    for name, values in {
        "all_gene_order.json": symbol_axis,
        "case_ids.json": [pair["case_id"] for pair in pairs],
        "up_genes.json": up["pr_gene_id"].astype(str).tolist(),
        "down_genes.json": down["pr_gene_id"].astype(str).tolist(),
    }.items():
        (output_dir / name).write_text(json.dumps(values, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    report = {"experiment_id": "EXP-038",
              "status": "SIGNATURE_READY_FOR_CROSS_SIGNATURE_VALIDATION"
                        if len(up) >= 20 and len(down) >= 20 else "SIGNATURE_DIRECTION_GATE_FAILED",
              "protocol": "independent TCGA-COAD/READ 50-case paired STAR TPM signature",
              "source_sha256": {"selected_pairs": sha256(pairs_path),
                                "raw_manifest": sha256(raw_dir / "manifest.json"),
                                "lincs_gene_info": sha256(gene_info_path)},
              "normalization": "sum TPM across Ensembl gene IDs sharing an uppercase valid symbol, then log2(TPM+1) per sample",
              "test": "two-sided one-sample paired tumor-minus-normal t test; BH over all valid genes",
              "n_cases": len(pairs), "n_files": len(receipts), "n_genes_tested": len(statistics),
              "n_undefined_constant_tests_set_to_p1": undefined_tests,
              "n_landmark_genes_mapped": int(aligned["q_value"].notna().sum()),
              "n_up_selected": len(up), "n_down_selected": len(down),
              "files_sha256": {name: sha256(output_dir / name) for name in (
                  *files, "gene_pair_differences.npy", "all_gene_order.json",
                  "case_ids.json", "up_genes.json", "down_genes.json")}}
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--lincs-gene-info", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.pairs, args.raw_dir, args.lincs_gene_info, args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "status", "n_cases", "n_genes_tested", "n_landmark_genes_mapped",
        "n_up_selected", "n_down_selected")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
