#!/usr/bin/env python3
"""Join the published GSE320583 donor index to existing response summaries.

This is a metadata-only annotation. It copies already-computed expression
similarity fields and does not read, calculate, or infer drug-sensitivity
outcomes. The article donor index is an anonymized index, not a patient ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


RUN_ID = "20261006_gse320583_response_identity_annotation_v1"
TRANSFER_RUN_ID = "20261001_gse320583_birinapant_transfer_v3"
PUBLIC_EVIDENCE_RUN_ID = "20261006_pdo_public_evidence_completion_v1"
EXPECTED_MODELS = {"HUB-C2-120", "HUB-C2-123", "HUB-C2-152"}
EXPECTED_INCHIKEY = "PKWRMUKBEYJEIX-DXXQBUJASA-N"

OUTPUT_FIELDS = [
    "article_donor_index",
    "GEO_model_id",
    "identity_evidence_source",
    "condition_code",
    "condition_label",
    "drug_label",
    "full_inchi_key",
    "dose_um",
    "time_h",
    "source_cell_id",
    "source_plate_replicates",
    "source_target_instances",
    "pdo_model_condition_cells",
    "pdo_matched_control_cells",
    "n_common_landmark_genes",
    "gene_spearman_rho",
    "gene_pearson_r",
    "nonzero_gene_sign_agreement",
    "nonzero_gene_sign_denominator",
    "absolute_amplitudes_comparable",
    "inferential_p_value",
    "interpretation_scope",
]

SCOPE = (
    "已计算的转录变化向量相似度描述；不是药敏数值、疗效估计或独立重复推断。"
    "RNA为每模型每条件一个pooled library；来源与目标表达尺度不同，幅度不可直接比较。"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8", newline="")
    os.replace(temp, path)


def atomic_write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=OUTPUT_FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/EXP-050") / RUN_ID,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()

    crosswalk_path = (
        root
        / "artifacts/experiments/EXP-050"
        / PUBLIC_EVIDENCE_RUN_ID
        / "GSE320583_donor_model_crosswalk.tsv"
    )
    response_path = (
        root
        / "artifacts/experiments/EXP-050"
        / TRANSFER_RUN_ID
        / "gse320583_birinapant_cross_system_gene_agreement.tsv"
    )
    inputs = {"donor_model_crosswalk": crosswalk_path, "existing_expression_summary": response_path}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"required registered input missing: {name}: {path}")

    started = time.perf_counter()
    started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    crosswalk = read_tsv(crosswalk_path)
    response_rows = read_tsv(response_path)
    donor_by_model: dict[str, dict[str, str]] = {}
    for row in crosswalk:
        model_id = row["GEO_model_id"]
        if model_id in donor_by_model:
            raise ValueError(f"duplicate model in published crosswalk: {model_id}")
        donor_by_model[model_id] = row
    if set(donor_by_model) != EXPECTED_MODELS:
        raise ValueError(f"crosswalk model set mismatch: {sorted(donor_by_model)}")

    selected: dict[str, dict[str, str]] = {}
    for row in response_rows:
        if row["target_dataset"] != "GSE320583":
            continue
        if row["source_dataset"] != "GSE70138" or row["drug_label"].casefold() != "birinapant":
            continue
        if row["full_inchi_key"] != EXPECTED_INCHIKEY:
            raise ValueError("existing source-response summary is not on the registered full InChIKey")
        if row["dose_um"] != "10.0" or row["time_h"] != "24.0":
            raise ValueError("existing source-response summary exposure differs from the frozen comparison")
        model_id = row["model_id"]
        if model_id in selected:
            raise ValueError(f"duplicate model response row: {model_id}")
        selected[model_id] = row
    if set(selected) != EXPECTED_MODELS:
        raise ValueError(f"response model set mismatch: {sorted(selected)}")

    rows: list[dict[str, str]] = []
    for mapping in sorted(crosswalk, key=lambda row: int(row["article_donor_index"])):
        model_id = mapping["GEO_model_id"]
        response = selected[model_id]
        for metric in ("gene_spearman_rho", "gene_pearson_r", "nonzero_gene_sign_agreement"):
            value = float(response[metric])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite copied expression metric: {model_id} {metric}")
        rows.append(
            {
                "article_donor_index": mapping["article_donor_index"],
                "GEO_model_id": model_id,
                "identity_evidence_source": mapping["evidence_source"],
                "condition_code": "Bi",
                "condition_label": "Birinapant, 10 µM, 24 h",
                "drug_label": response["drug_label"],
                "full_inchi_key": response["full_inchi_key"],
                "dose_um": response["dose_um"],
                "time_h": response["time_h"],
                "source_cell_id": response["source_cell_id"],
                "source_plate_replicates": response["source_plate_replicates"],
                "source_target_instances": response["source_target_instances"],
                "pdo_model_condition_cells": response["pdo_model_condition_cells"],
                "pdo_matched_control_cells": response["pdo_matched_control_cells"],
                "n_common_landmark_genes": response["n_common_landmark_genes"],
                "gene_spearman_rho": response["gene_spearman_rho"],
                "gene_pearson_r": response["gene_pearson_r"],
                "nonzero_gene_sign_agreement": response["nonzero_gene_sign_agreement"],
                "nonzero_gene_sign_denominator": response["nonzero_gene_sign_denominator"],
                "absolute_amplitudes_comparable": response["absolute_amplitudes_comparable"],
                "inferential_p_value": response["inferential_p_value"],
                "interpretation_scope": SCOPE,
            }
        )

    output_path = output_dir / "gse320583_birinapant_response_with_published_donor_crosswalk.tsv"
    atomic_write_tsv(output_path, rows)
    readback = read_tsv(output_path)
    if len(readback) != 3 or {r["GEO_model_id"] for r in readback} != EXPECTED_MODELS:
        raise RuntimeError("output readback row/model validation failed")
    if any(r["inferential_p_value"] for r in readback):
        raise RuntimeError("unexpected inferential p-values appeared in a descriptive join")
    for row in readback:
        source = selected[row["GEO_model_id"]]
        for field in (
            "full_inchi_key",
            "n_common_landmark_genes",
            "gene_spearman_rho",
            "gene_pearson_r",
            "nonzero_gene_sign_agreement",
            "nonzero_gene_sign_denominator",
        ):
            if row[field] != source[field]:
                raise RuntimeError(f"source value changed during identity join: {row['GEO_model_id']} {field}")

    finished_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    elapsed = time.perf_counter() - started
    summary_path = output_dir / "summary.json"
    summary = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "COMPLETED_METADATA_ONLY_IDENTITY_ANNOTATION",
        "result": "Three existing GSE320583/Birinapant expression-similarity rows joined to the article donor index by exact GEO model ID.",
        "rows": [
            {
                "article_donor_index": row["article_donor_index"],
                "GEO_model_id": row["GEO_model_id"],
                "gene_spearman_rho": row["gene_spearman_rho"],
                "gene_pearson_r": row["gene_pearson_r"],
                "nonzero_gene_sign_agreement": row["nonzero_gene_sign_agreement"],
                "n_common_landmark_genes": row["n_common_landmark_genes"],
            }
            for row in readback
        ],
        "interpretation_limit": SCOPE,
        "drug_sensitivity_association": "NOT_ESTIMATED; numeric per-model drug sensitivity and replicate linkage are unavailable in this evidence package.",
        "gpu_hours": 0,
    }
    atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    outputs = {output_path.name: sha256(output_path), summary_path.name: sha256(summary_path)}
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "operation": "metadata-only donor identity annotation of an existing expression-similarity summary",
        "started_at_local": started_at.isoformat(timespec="seconds"),
        "finished_at_local": finished_at.isoformat(timespec="seconds"),
        "runtime_cpu_wall_minutes": elapsed / 60,
        "gpu_hours": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {name: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for name, path in inputs.items()},
        "script_path": "scripts/evaluation/annotate_exp050_gse320583_donor_crosswalk.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": outputs,
        "validation": {
            "crosswalk_models_equal_existing_response_models": True,
            "rows": len(readback),
            "full_inchikey_preserved": True,
            "exposure_preserved_10um_24h": True,
            "expression_metrics_copied_exactly": True,
            "readback_pass": True,
            "drug_sensitivity_values_read_or_computed": False,
            "inferential_p_values_added": False,
        },
        "interpretation_limit": SCOPE,
    }
    manifest_path = output_dir / "run_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    sha_manifest = output_dir / "sha256_manifest.tsv"
    sha_lines = ["relative_path\tsha256\tbytes"]
    for path in (output_path, summary_path, manifest_path):
        sha_lines.append(f"{path.name}\t{sha256(path)}\t{path.stat().st_size}")
    atomic_write_text(sha_manifest, "\n".join(sha_lines) + "\n")
    # Independent final readback includes the output checksums and row identity.
    manifest_readback = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_readback["validation"]["readback_pass"] is not True:
        raise RuntimeError("manifest readback validation failed")
    print(json.dumps({"run_id": RUN_ID, "output": str(output_path.relative_to(root)), "rows": len(readback), "gpu_hours": 0, "wall_seconds": elapsed, "sha256": outputs[output_path.name]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
