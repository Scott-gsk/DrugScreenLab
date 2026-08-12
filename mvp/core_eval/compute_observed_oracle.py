"""Compute the MVP-001 observed LINCS exact-978 reversal oracle.

This is an evaluation-only, label-blind artifact builder.  It reads the
registered exact-978 cache by memory map and only writes compact artifacts under
``mvp/core_eval``.  PRISM response values are intentionally not read.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


EXACT_CACHE = Path("data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy")
GCTX = Path(
    "data/interim/lincs/GSE92742/"
    "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
)
INST_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_inst_info.txt.gz")
PERT_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
GENE_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz")
PRISM_AUDIT = Path("mvp/core_data/prism_compact_audit.json")
SIGNATURE = Path("mvp/core_data/crc_disease_signature_exact978.tsv")
FORMAL_MIN_TOTAL = 20
FORMAL_MIN_EACH_DIRECTION = 5


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _average_rank(left)
    right_rank = _average_rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _extract(cache: np.ndarray, rows: Iterable[int], genes: np.ndarray) -> np.ndarray:
    rows_array = np.asarray(list(rows), dtype=np.int64)
    out = np.empty((len(rows_array), len(genes)), dtype=np.float32)
    for start in range(0, len(rows_array), 4096):
        stop = min(start + 4096, len(rows_array))
        out[start:stop] = np.asarray(cache[rows_array[start:stop]][:, genes])
    return out


def _frozen_mapping() -> tuple[pd.DataFrame, list[dict[str, object]], str]:
    """Load the response-independent four-drug identity freeze."""
    payload = json.loads(PRISM_AUDIT.read_text(encoding="utf-8"))
    candidates = payload.get("identity", {}).get("candidates", [])
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise RuntimeError("MVP-001 frozen PRISM candidate identity is missing or not four drugs")
    mapping = pd.DataFrame(candidates).rename(columns={"drug_name": "name"})
    required = {"broad_id", "name", "smiles", "pert_id", "inchi_key"}
    if not required.issubset(mapping.columns) or mapping["pert_id"].duplicated().any():
        raise RuntimeError("MVP-001 frozen candidate identity schema is invalid")
    mapping_digest = str(payload.get("identity", {}).get("candidate_mapping_sha256", ""))
    if not mapping_digest:
        raise RuntimeError("MVP-001 frozen candidate mapping digest is absent")
    return mapping[["broad_id", "name", "smiles", "pert_id", "inchi_key"]], [], mapping_digest


def build(output_dir: Path) -> dict[str, object]:
    lincs = pd.read_csv(PERT_INFO, sep="\t", dtype=str, compression="gzip", keep_default_na=False)
    inst = pd.read_csv(
        INST_INFO, sep="\t", dtype=str, compression="gzip", low_memory=False, keep_default_na=False
    )
    signature = pd.read_csv(SIGNATURE, sep="\t")
    required_signature = {"gene_index_978", "gene_symbol", "signed_log2fc", "direction"}
    if not required_signature.issubset(signature.columns):
        raise RuntimeError(f"signature missing fields: {sorted(required_signature.difference(signature.columns))}")
    genes = signature["gene_index_978"].astype(int).to_numpy()
    signature_values = signature["signed_log2fc"].astype(float).to_numpy()
    up = int((signature["direction"] == "up").sum())
    down = int((signature["direction"] == "down").sum())
    formal_gate = len(genes) >= FORMAL_MIN_TOTAL and min(up, down) >= FORMAL_MIN_EACH_DIRECTION

    mapping, ambiguous, mapping_digest = _frozen_mapping()
    candidate_pert_ids = set(mapping["pert_id"])
    treatment = inst[inst["pert_type"].eq("trt_cp") & inst["pert_id"].isin(candidate_pert_ids)].copy()
    match_fields = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]
    group_fields = [
        "pert_id",
        "cell_id",
        "pert_dose",
        "pert_dose_unit",
        "pert_time",
        "pert_time_unit",
    ]
    treatment["match_key"] = treatment[match_fields].astype(str).agg("||".join, axis=1)
    treatment["group_key"] = treatment[group_fields].astype(str).agg("||".join, axis=1)
    controls = inst[inst["pert_type"].eq("ctl_vehicle")].copy()
    controls["match_key"] = controls[match_fields].astype(str).agg("||".join, axis=1)
    control_ids_by_key = controls.groupby("match_key", sort=False)["inst_id"].apply(list).to_dict()
    treatment["control_ids"] = treatment["match_key"].map(control_ids_by_key)
    treatment = treatment[treatment["control_ids"].notna()].copy().reset_index(drop=True)

    with h5py.File(GCTX, "r") as handle:
        gctx_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["0/META/COL/id"][:]
        ]
    cache = np.load(EXACT_CACHE, mmap_mode="r")
    row_index = {value: index for index, value in enumerate(gctx_ids)}
    treatment_rows = treatment["inst_id"].map(row_index)
    if treatment_rows.isna().any():
        raise RuntimeError("treatment IDs missing from GCTX/cache row identity")
    control_ids = sorted({item for items in treatment["control_ids"] for item in items})
    if any(item not in row_index for item in control_ids):
        raise RuntimeError("control IDs missing from GCTX/cache row identity")
    treatment_rows_array = treatment_rows.astype(np.int64).to_numpy()
    control_rows_array = np.asarray([row_index[item] for item in control_ids], dtype=np.int64)
    control_values = _extract(cache, control_rows_array, genes)
    control_index = {item: index for index, item in enumerate(control_ids)}
    control_means = {
        key: control_values[[control_index[item] for item in ids]].mean(axis=0)
        for key, ids in control_ids_by_key.items()
        if all(item in control_index for item in ids)
    }
    treatment_values = _extract(cache, treatment_rows_array, genes)
    treatment["control_mean"] = treatment["match_key"].map(control_means)
    if treatment["control_mean"].isna().any():
        raise RuntimeError("a retained treatment group has no resolved control mean")
    deltas = treatment_values - np.vstack(treatment["control_mean"].to_numpy())
    if not np.isfinite(deltas).all():
        raise RuntimeError("non-finite exact-978 observed oracle values")

    group_rows: list[dict[str, object]] = []
    for group_key, positions in treatment.groupby("group_key", sort=True).indices.items():
        positions_array = np.asarray(positions, dtype=np.int64)
        first = treatment.iloc[int(positions_array[0])]
        group_rows.append(
            {
                "pert_id": str(first["pert_id"]),
                "cell_id": str(first["cell_id"]),
                "dose": f"{first['pert_dose']} {first['pert_dose_unit']}",
                "time": f"{first['pert_time']} {first['pert_time_unit']}",
                "group_key": str(group_key),
                "replicate_count": int(len(positions_array)),
                "delta": deltas[positions_array].mean(axis=0),
            }
        )

    group_table = pd.DataFrame(group_rows)
    score_rows: list[dict[str, object]] = []
    drug_vectors: dict[str, np.ndarray] = {}
    for pert_id, group_indices in group_table.groupby("pert_id", sort=True).indices.items():
        values = np.vstack(group_table.iloc[list(group_indices)]["delta"].to_numpy())
        vector = np.median(values, axis=0)
        drug_vectors[str(pert_id)] = vector
        spearman = _spearman(signature_values, vector)
        score_rows.append(
            {
                "pert_id": str(pert_id),
                "n_groups": int(len(values)),
                "n_instances": int(
                    group_table.iloc[list(group_indices)]["replicate_count"].astype(int).sum()
                ),
                "n_cells": int(group_table.iloc[list(group_indices)]["cell_id"].nunique()),
                "n_doses": int(group_table.iloc[list(group_indices)]["dose"].nunique()),
                "n_times": int(group_table.iloc[list(group_indices)]["time"].nunique()),
                # A larger score means stronger anti-correlation with the
                # disease signature (operational reversal), as frozen in the
                # evaluation contract.
                "reversal_observed": None if spearman is None else -spearman,
            }
        )
    scores = pd.DataFrame(score_rows)
    scores = mapping.merge(scores, on="pert_id", how="inner", validate="one_to_one")
    scores["rank_desc_reversal"] = _average_rank(-scores["reversal_observed"].astype(float).to_numpy())
    scores = scores.sort_values(["rank_desc_reversal", "broad_id"], kind="mergesort").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "MVP-001_observed_oracle_ranking.tsv"
    output_columns = [
        "rank_desc_reversal",
        "reversal_observed",
        "broad_id",
        "name",
        "pert_id",
        "inchi_key",
        "n_groups",
        "n_instances",
        "n_cells",
        "n_doses",
        "n_times",
    ]
    scores[output_columns].to_csv(ranking_path, sep="\t", index=False, float_format="%.10g")
    summary = {
        "format": "mvp001_observed_lincs_oracle_ranking_v1",
        "mvp_id": "MVP-001",
        "status": "INCONCLUSIVE_EXPLORATORY" if not formal_gate else "READY_FOR_PRISM_JOIN",
        "signature": {
            "path": str(SIGNATURE),
            "rows": int(len(genes)),
            "up": up,
            "down": down,
            "formal_gate": formal_gate,
            "formal_gate_rule": {"minimum_total": FORMAL_MIN_TOTAL, "minimum_each_direction": FORMAL_MIN_EACH_DIRECTION},
            "gene_indices": genes.tolist(),
        },
        "candidate": {
            "mapping_rows": int(len(mapping)),
            "ambiguous_rows_excluded": ambiguous,
            "mapping_digest": mapping_digest,
        },
        "oracle": {
            "matched_treatment_instances": int(len(treatment)),
            "treatment_groups": int(len(group_table)),
            "mapped_drugs": int(len(scores)),
            "unique_controls": int(len(control_ids)),
            "lincs_cells": int(treatment["cell_id"].nunique()),
            "cache_shape": [int(cache.shape[0]), int(cache.shape[1])],
            "gene_order_digest": "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623",
            "score": "-Spearman(signature signed_log2fc, drug median Delta978)",
            "aggregation": "mean within technical group; coordinate median across groups",
        },
        "score_summary": {
            "finite_scores": int(scores["reversal_observed"].notna().sum()),
            "score_min": float(scores["reversal_observed"].min()),
            "score_median": float(scores["reversal_observed"].median()),
            "score_max": float(scores["reversal_observed"].max()),
            "top10": scores[["rank_desc_reversal", "broad_id", "name", "reversal_observed"]].head(10).to_dict("records"),
        },
        "prism": {
            "response_asset": "mvp/core_data/compact_prism_response.parquet",
            "status": "READY_FOR_JOIN",
            "metrics": "computed by mvp/core_eval/evaluate_prism.py after ranking freeze",
        },
        "artifacts": {
            "ranking": str(ranking_path),
            "ranking_sha256": _digest(ranking_path),
            "group_vectors_not_serialized": True,
        },
        "known_deviations": [
            "The four-drug cohort is deliberately small for MVP feasibility; downstream top-2 diagnostics are not paper-level evidence.",
            "No PRISM response values are read by this label-blind oracle stage; the compact PRISM join is a separate frozen evaluation stage.",
        ],
    }
    summary_path = output_dir / "MVP-001_observed_oracle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("mvp/core_eval/observed_oracle"))
    args = parser.parse_args()
    result = build(args.output_dir)
    print(json.dumps({"status": result["status"], "artifacts": result["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
