"""Independently recompute the paired TCGA signature with grouped TPM and manual BH."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_tpm(path: Path) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", comment="#", usecols=[
        "gene_id", "gene_name", "tpm_unstranded"], dtype={"gene_id": str, "gene_name": str})
    keep = frame["gene_id"].str.startswith("ENSG") & frame["gene_name"].fillna("").str.fullmatch(
        r"[A-Za-z][A-Za-z0-9-]*")
    retained = frame.loc[keep].copy()
    retained["gene_symbol"] = retained["gene_name"].str.upper()
    if not np.isfinite(retained["tpm_unstranded"]).all() or (retained["tpm_unstranded"] < 0).any():
        raise ValueError("independent GDC TPM read contains invalid values")
    return retained.groupby("gene_symbol", sort=True)["tpm_unstranded"].sum()


def bh(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues, kind="stable")
    sorted_p = pvalues[order]
    adjusted = np.minimum.accumulate((sorted_p * len(pvalues) / np.arange(1, len(pvalues) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def verify(pairs_path: Path, raw_dir: Path, signature_dir: Path) -> dict:
    report = json.loads((signature_dir / "manifest.json").read_text(encoding="utf-8"))
    if report["source_sha256"]["selected_pairs"] != sha256(pairs_path):
        raise ValueError("selected TCGA pair manifest has changed")
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    raw = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    receipts = {item["file_id"]: item for item in raw["files"]}
    order = json.loads((signature_dir / "all_gene_order.json").read_text(encoding="utf-8"))
    cases = json.loads((signature_dir / "case_ids.json").read_text(encoding="utf-8"))
    if cases != [pair["case_id"] for pair in pairs] or len(cases) != 50:
        raise ValueError("saved patient/case axis differs from frozen paired files")
    columns = []
    for pair in pairs:
        values = {}
        for role in ("tumor", "normal"):
            file_id = pair[role]["file_id"]
            receipt = receipts[file_id]
            path = raw_dir / receipt["relative_path"]
            if sha256(path) != receipt["sha256"]:
                raise ValueError("raw GDC file checksum changed")
            grouped = grouped_tpm(path)
            if set(grouped.index) != set(order):
                raise ValueError("independently read GDC gene axis differs from saved signature")
            values[role] = np.log2(1 + grouped.reindex(order).to_numpy(dtype=np.float64))
        columns.append(values["tumor"] - values["normal"])
    difference = np.stack(columns, axis=1)
    saved_difference = np.load(signature_dir / "gene_pair_differences.npy")
    if saved_difference.shape != difference.shape:
        raise ValueError("saved TCGA pair-difference matrix shape differs")
    max_difference_error = float(np.max(np.abs(saved_difference - difference)))
    if max_difference_error > 1e-6:
        raise ValueError("saved TCGA pair differences disagree with independent grouping")
    mean = difference.mean(axis=1)
    standard_error = difference.std(axis=1, ddof=1) / np.sqrt(len(cases))
    p = np.ones(len(mean), dtype=np.float64)
    variable = standard_error > 0
    p[variable] = 2 * student_t.sf(np.abs(mean[variable] / standard_error[variable]), df=len(cases) - 1)
    p[~variable & (mean != 0)] = 0.0
    q = bh(p)
    statistics = pd.read_parquet(signature_dir / "all_gene_statistics.parquet")
    if statistics["gene_symbol"].tolist() != order:
        raise ValueError("saved all-gene statistics axis differs from independent read")
    max_p_error = float(np.max(np.abs(p - statistics["p_value"].to_numpy())))
    max_q_error = float(np.max(np.abs(q - statistics["q_value"].to_numpy())))
    max_effect_error = float(np.max(np.abs(mean - statistics["mean_paired_log2fc"].to_numpy())))
    if max(max_p_error, max_q_error, max_effect_error) > 1e-7:
        raise ValueError("saved TCGA effect or multiple-testing statistics disagree")
    landmark = pd.read_parquet(signature_dir / "landmark_statistics.parquet")
    aligned = landmark[["pr_gene_id", "pr_gene_symbol", "gene_symbol"]].merge(
        statistics, on="gene_symbol", how="left", validate="one_to_one", sort=False)
    pd.testing.assert_frame_equal(aligned, landmark, check_exact=False, atol=1e-7, rtol=0)
    selected = landmark.loc[landmark["q_value"].lt(.05)
                            & landmark["mean_paired_log2fc"].abs().ge(.5)].copy()
    up = selected.loc[selected["mean_paired_log2fc"].gt(0)].sort_values(
        ["mean_paired_log2fc", "gene_symbol"], ascending=[False, True]).head(100)
    down = selected.loc[selected["mean_paired_log2fc"].lt(0)].assign(
        absolute_effect=lambda frame: frame["mean_paired_log2fc"].abs()).sort_values(
        ["absolute_effect", "gene_symbol"], ascending=[False, True]).head(100)
    if (json.loads((signature_dir / "up_genes.json").read_text(encoding="utf-8"))
            != up["pr_gene_id"].astype(str).tolist()
            or json.loads((signature_dir / "down_genes.json").read_text(encoding="utf-8"))
            != down["pr_gene_id"].astype(str).tolist()):
        raise ValueError("saved TCGA direction gene IDs differ from independent selection")
    for name, checksum in report["files_sha256"].items():
        if sha256(signature_dir / name) != checksum:
            raise ValueError(f"saved TCGA signature artifact checksum differs: {name}")
    return {"status": "PASS", "n_cases": len(cases), "n_genes": len(order),
            "max_pair_difference_error": max_difference_error,
            "max_effect_error": max_effect_error, "max_p_error": max_p_error,
            "max_q_error": max_q_error, "up_genes": len(up), "down_genes": len(down)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.pairs, args.raw_dir, args.signature_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
