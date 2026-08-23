"""Run model-independent analyses on frozen EXP-010 breast assets."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

from drug_screen.evaluation.exp010_breast import reactome_overrepresentation


def _bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        index = order[rank_index]
        running = min(running, float(values[index]) * len(values) / (rank_index + 1))
        adjusted[index] = running
    return adjusted


def _feature_map_from_raw(raw_tar: Path) -> dict[str, str]:
    """Use a frozen 10x features.tsv as an Ensembl-to-symbol mapping."""
    with tarfile.open(raw_tar, "r:") as outer:
        nested_member = next(member for member in outer if member.name.endswith(".tar.gz"))
        nested_handle = outer.extractfile(nested_member)
        assert nested_handle is not None
        with tarfile.open(fileobj=nested_handle, mode="r|gz") as nested:
            feature = next(member for member in nested if member.name.endswith("features.tsv.gz"))
            handle = nested.extractfile(feature)
            assert handle is not None
            with gzip.open(handle, "rt") as text:
                return {parts[0].split(".")[0]: parts[1] for line in text if len(parts := line.rstrip().split("\t")) >= 2}


def _read_gmt(path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        lines = archive.read(member).decode("utf-8").splitlines()
    return {parts[0]: parts[2:] for line in lines if len(parts := line.split("\t")) >= 3}


def analyze_gse287429(counts_path: Path, reactome_path: Path, feature_source: Path, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(counts_path, compression="gzip")
    feature_map = _feature_map_from_raw(feature_source)
    frame["symbol"] = frame["GeneID"].astype(str).str.split(".").str[0].map(feature_map)
    mapped = frame.dropna(subset=["symbol"]).copy()
    gene_sets = _read_gmt(reactome_path)
    contrasts = []
    for drug in ("DOX", "BTZ"):
        for hour in (23, 47):
            treated = [f"{drug}{hour}h_1", f"{drug}{hour}h_2"]
            controls = [f"CTR{hour}h_1", f"CTR{hour}h_2"]
            delta = mapped[treated].mean(axis=1) - mapped[controls].mean(axis=1)
            ranked = mapped.assign(delta=delta).sort_values("delta", key=lambda x: x.abs(), ascending=False)
            ora = reactome_overrepresentation(ranked["symbol"].tolist(), gene_sets, top_n=200,
                                               universe=mapped["symbol"].drop_duplicates().tolist())
            contrast_id = f"{drug}_{hour}h_vs_CONTROL_{hour}h"
            top_genes = [{"symbol": str(row.symbol), "mean_difference": float(row.delta)}
                         for row in ranked[["symbol", "delta"]].head(25).itertuples(index=False)]
            contrasts.append({"contrast_id": contrast_id, "method": "absolute time-matched mean-difference ranking",
                              "treated_samples": treated, "control_samples": controls, "top_genes": top_genes,
                              "reactome": {**ora, "rows": ora["rows"][:25]}})
    result = {"format": "exp010_gse287429_real_analysis_v1", "status": "COMPLETE_DESCRIPTIVE",
              "input_kind": "author-normalized continuous expression", "method": "time-matched mean-difference ranking",
              "not_used": "PyDESeq2 (integer raw counts unavailable)", "gene_rows": len(frame),
              "ensembl_symbol_mapped": len(mapped), "claim_scope": "perturbation response; not drug sensitivity",
              "contrasts": contrasts}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, contrast in zip(axes.flat, contrasts, strict=True):
        pathways = contrast["reactome"]["rows"][:10]
        if pathways:
            labels = [row["pathway"][:38] for row in pathways][::-1]
            values = [-np.log10(max(row["fdr_bh"], 1e-300)) for row in pathways][::-1]
            ax.barh(labels, values)
        ax.set_title(contrast["contrast_id"])
        ax.set_xlabel("-log10(BH FDR), descriptive")
    fig.savefig(output_dir / "gse287429_reactome.png", dpi=160)
    plt.close(fig)
    return result


def _read_gzip_lines(handle) -> list[str]:
    with gzip.open(handle, "rt") as stream:
        return [line.rstrip().split("\t")[0] for line in stream if line.strip()]


def _sum_selected_10x(nested_handle, selected: set[str], nested_name: str) -> pd.Series:
    with tempfile.TemporaryFile() as spool:
        shutil.copyfileobj(nested_handle, spool)
        spool.seek(0)
        with tarfile.open(fileobj=spool, mode="r|gz") as nested:
            compressed: dict[str, bytes] = {}
            for member in nested:
                for suffix in ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"):
                    if member.name.endswith(suffix):
                        member_handle = nested.extractfile(member)
                        assert member_handle is not None
                        compressed[suffix] = member_handle.read()
                if len(compressed) == 3:
                    break
    if len(compressed) != 3:
        raise ValueError(f"incomplete 10x member: {nested_name}")
    barcodes = _read_gzip_lines(io.BytesIO(compressed["barcodes.tsv.gz"]))
    features = _read_gzip_lines(io.BytesIO(compressed["features.tsv.gz"]))
    indices = np.asarray([i for i, barcode in enumerate(barcodes) if barcode in selected], dtype=int)
    with gzip.open(io.BytesIO(compressed["matrix.mtx.gz"]), "rb") as matrix_stream:
        matrix = mmread(matrix_stream).tocsr()
    values = np.asarray(matrix[:, indices].sum(axis=1)).ravel() if len(indices) else np.zeros(len(features))
    return pd.Series(values, index=features, name=nested_name[:-7])


def build_gse276609_pseudobulk(raw_tar: Path, output_path: Path) -> dict[str, object]:
    """Sum frozen 10x counts over author-provided CancerCellsOnly barcodes."""
    summaries: list[pd.Series] = []
    failures: list[dict[str, str]] = []
    with tarfile.open(raw_tar, "r:") as outer:
        members = {member.name: member for member in outer.getmembers()}
        nested_names = sorted(name for name in members if name.endswith(".tar.gz"))
        for nested_name in nested_names:
            stem = nested_name[:-7]
            cancer_name = stem + "_CancerCellsOnly.csv.gz"
            if cancer_name not in members:
                continue
            cancer_handle = outer.extractfile(members[cancer_name])
            nested_handle = outer.extractfile(members[nested_name])
            assert cancer_handle is not None and nested_handle is not None
            with gzip.open(cancer_handle, "rt") as text:
                selected = {row["barcode"] for row in csv.DictReader(text)}
            try:
                summaries.append(_sum_selected_10x(nested_handle, selected, nested_name))
            except (EOFError, tarfile.ReadError, ValueError) as error:
                failures.append({"member": nested_name, "error": f"{type(error).__name__}: {error}"})
    if not summaries:
        return {"format": "exp010_gse276609_malignant_pseudobulk_v1", "status": "BLOCKED_NESTED_10X_INTEGRITY",
                "sample_count": 0, "failed_member_count": len(failures), "failures": failures,
                "domain_shift_status": "BLOCKED_NO_PSEUDOBULK",
                "reason": "Author malignant barcode maps exist, but no matching nested 10x matrix passed full read integrity."}
    pseudobulk = pd.concat(summaries, axis=1).fillna(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pseudobulk.to_parquet(output_path)
    return {"format": "exp010_gse276609_malignant_pseudobulk_v1", "status": "PSEUDOBULK_COMPLETE_DOMAIN_BLOCKED",
            "sample_count": len(summaries), "gene_count": len(pseudobulk), "output": str(output_path),
            "failed_member_count": len(failures), "failures": failures,
            "malignant_label_source": "per-sample author-provided CancerCellsOnly barcode files",
            "domain_shift_status": "BLOCKED_MISSING_EXPLICIT_DOMAIN_LABELS",
            "reason": "Frozen package maps malignant cells but provides no audited sample-level source-domain field; filenames were not used to infer biology."}


def multiregional_status(access_path: Path) -> dict[str, object]:
    access = json.loads(access_path.read_text(encoding="utf-8"))
    return {"format": "exp010_multiregional_public_gate_v1", "status": "NOT_EVALUABLE_PUBLIC_MATRIX",
            "access_status": access["status"], "attempt_count": len(access["attempts"]),
            "accepted_asset_count": sum(bool(row.get("accepted_as_data")) for row in access["attempts"]),
            "reason": access["data_decision"], "expected_assets": access["expected_assets"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-pseudobulk", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root
    result = {"format": "exp010_breast_model_independent_bundle_v1",
              "gse287429": analyze_gse287429(root / "geo/gse287429/GSE287429_normalized_counts_allsample.txt.gz",
                                             root / "reactome/ReactomePathways.gmt.zip",
                                             root / "geo/gse276609/GSE276609_RAW.tar", args.output_dir),
              "multiregional": multiregional_status(root / "multiregional_breast_pdo/access_attempts.json")}
    if not args.skip_pseudobulk:
        result["gse276609"] = build_gse276609_pseudobulk(root / "geo/gse276609/GSE276609_RAW.tar",
                                                         args.output_dir / "gse276609_malignant_pseudobulk.parquet")
    (args.output_dir / "breast_model_independent_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
