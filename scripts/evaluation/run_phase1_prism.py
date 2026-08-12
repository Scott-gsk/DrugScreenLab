"""Evaluate a frozen Phase-1 candidate on the frozen compact PRISM cohort."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drug_screen.evaluation.phase1_prism import line_metrics, reversal_score, spearman
from drug_screen.modeling.phase1 import ContextChemicalDoseTimePredictor, load_phase1_manifest


ROOT = Path(__file__).resolve().parents[2]
SIGNATURE = ROOT / "mvp" / "core_data" / "crc_disease_signature_exact978.tsv"
PRISM = ROOT / "mvp" / "core_data" / "compact_prism_response.parquet"
AUDIT = ROOT / "mvp" / "core_data" / "prism_compact_audit.json"
OBSERVED = ROOT / "mvp" / "core_eval" / "observed_oracle" / "MVP-001_observed_oracle_ranking.tsv"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _summary(line_rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in line_rows if row["metrics"]["eligible"]]
    values = [float(row["metrics"]["spearman"]) for row in eligible if row["metrics"]["spearman"] is not None]
    top = [float(row["metrics"]["top2_overlap_rate"]) for row in eligible]
    return {
        "line_count": len(line_rows),
        "eligible_line_count": len(eligible),
        "macro_mean_spearman": float(np.mean(values)) if values else None,
        "macro_median_spearman": float(np.median(values)) if values else None,
        "fraction_positive_spearman": float(np.mean(np.asarray(values) > 0)) if values else None,
        "macro_mean_top2_overlap_rate": float(np.mean(top)) if top else None,
        "fraction_top2_overlap_at_least_half": float(np.mean(np.asarray(top) >= 0.5)) if top else None,
    }


def build(*, manifest_path: Path, checkpoint_path: Path, output_dir: Path, root: Path) -> dict[str, object]:
    manifest = load_phase1_manifest(manifest_path, root=root)
    cache, chemical_features = manifest.load_arrays()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "phase1_context_candidate_checkpoint_v1":
        raise RuntimeError("checkpoint is not a Phase-1 context candidate")
    normalization = checkpoint["normalization"]
    model = ContextChemicalDoseTimePredictor(
        chemical_dim=int(checkpoint["chemical_dim"]),
        context_dim=int(checkpoint["context_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        gene_count=int(checkpoint["gene_count"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    signature = pd.read_csv(SIGNATURE, sep="\t")
    signature_values = signature["signed_log2fc"].to_numpy(float)
    signature_genes = signature["gene_index_978"].astype(int).to_numpy()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    candidates = audit["identity"]["candidates"]
    candidate_ids = [str(row["pert_id"]) for row in candidates]
    candidate_names = {str(row["pert_id"]): str(row["drug_name"]) for row in candidates}

    feature_by_drug: dict[str, int] = {}
    for row in manifest.records:
        feature_by_drug.setdefault(row.drug_id, row.chemical_feature_row)
    missing = sorted(set(candidate_ids).difference(feature_by_drug))
    if missing:
        raise RuntimeError(f"Phase-1 manifest lacks frozen candidate structures: {missing}")

    inst_path = root / "data" / "raw" / "lincs" / "GSE92742" / "GSE92742_Broad_LINCS_inst_info.txt.gz"
    instances = pd.read_csv(inst_path, sep="\t", usecols=["pert_type", "pert_time", "pert_time_unit", "cell_id"])
    instances["cache_row"] = np.arange(len(instances), dtype=np.int64)
    controls = instances.loc[
        instances["pert_type"].eq("ctl_vehicle")
        & instances["pert_time_unit"].eq("h")
        & np.isclose(pd.to_numeric(instances["pert_time"], errors="coerce"), 6.0)
    ]
    context_vectors: dict[str, np.ndarray] = {}
    for context_id, group in controls.groupby("cell_id", sort=True):
        rows = group["cache_row"].to_numpy(dtype=np.int64)
        context_vectors[str(context_id)] = np.asarray(cache[rows].mean(axis=0), dtype=np.float32)
    if not context_vectors:
        raise RuntimeError("no canonical 6h untreated context controls are available")
    reference_context = np.mean(np.stack(list(context_vectors.values())), axis=0).astype(np.float32)

    prism = pd.read_parquet(PRISM)
    prism = prism[prism["pert_id"].isin(candidate_ids)].copy()
    line_frame = prism[["depmap_id", "ccle_name"]].drop_duplicates().sort_values("depmap_id")
    ranking_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for line in line_frame.itertuples(index=False):
            depmap_id = str(line.depmap_id)
            ccle_name = str(line.ccle_name)
            context_id = ccle_name.split("_", 1)[0]
            context_source = "exact_lincs_context" if context_id in context_vectors else "reference_context_fallback"
            context = context_vectors.get(context_id, reference_context)
            context_norm = (context - np.asarray(normalization["context_center"])) / np.asarray(normalization["context_scale"])
            dose_time = np.log1p(np.asarray([[10.0, 6.0]], dtype=np.float32))
            dose_time = (dose_time - np.asarray(normalization["dose_time_center"])) / np.asarray(normalization["dose_time_scale"])
            chemicals = np.stack([chemical_features[feature_by_drug[drug]] for drug in candidate_ids]).astype(np.float32)
            contexts = np.repeat(context_norm[None, :], len(candidate_ids), axis=0).astype(np.float32)
            dose_times = np.repeat(dose_time, len(candidate_ids), axis=0).astype(np.float32)
            predicted_norm = model(torch.from_numpy(contexts), torch.from_numpy(chemicals), torch.from_numpy(dose_times)).numpy()
            predicted = predicted_norm * np.asarray(normalization["target_scale"]) + np.asarray(normalization["target_center"])
            for candidate, delta in zip(candidate_ids, predicted, strict=True):
                ranking_rows.append({
                    "depmap_id": depmap_id,
                    "ccle_name": ccle_name,
                    "pert_id": candidate,
                    "drug_name": candidate_names[candidate],
                    "reversal_predicted": reversal_score(signature_values, delta[signature_genes]),
                    "context_id": context_id,
                    "context_source": context_source,
                })
    ranking = pd.DataFrame(ranking_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "phase1_prism_predicted_ranking.tsv"
    ranking.to_csv(ranking_path, sep="\t", index=False, float_format="%.10g")
    joined = prism.merge(ranking, on=["depmap_id", "ccle_name", "pert_id", "drug_name"], how="left", validate="one_to_one")
    line_rows: list[dict[str, object]] = []
    for (depmap_id, ccle_name), group in joined.groupby(["depmap_id", "ccle_name"], sort=True):
        metrics = line_metrics(group, "reversal_predicted")
        line_rows.append({
            "depmap_id": str(depmap_id),
            "ccle_name": str(ccle_name),
            "context_id": str(group["context_id"].iloc[0]),
            "context_source": str(group["context_source"].iloc[0]),
            "metrics": metrics,
        })
    exact_lines = [row for row in line_rows if row["context_source"] == "exact_lincs_context"]
    observed = pd.read_csv(OBSERVED, sep="\t")
    predicted_global = ranking.groupby("pert_id", as_index=False)["reversal_predicted"].mean()
    predicted_global = predicted_global.rename(columns={"reversal_predicted": "predicted_score"})
    observed = observed[["pert_id", "reversal_observed"]].rename(columns={"reversal_observed": "observed_score"})
    global_join = predicted_global.merge(observed, on="pert_id", how="inner", validate="one_to_one")
    top2_pred = set(global_join.nlargest(2, "predicted_score")["pert_id"])
    top2_obs = set(global_join.nlargest(2, "observed_score")["pert_id"])
    summary = {
        "format": "phase1_prism_ranking_evaluation_v1",
        "status": "READY_FOR_PROGRAM_REVIEW",
        "candidate_cohort": {
            "count": len(candidate_ids),
            "pert_ids": candidate_ids,
            "mapping_sha256": audit["identity"]["candidate_mapping_sha256"],
        },
        "phase1": {
            "manifest": str(manifest_path),
            "manifest_sha256": digest(manifest_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": digest(checkpoint_path),
            "condition": {"dose_um": 10.0, "time_h": 6.0},
            "context_controls": len(context_vectors),
        },
        "prism_asset": {"path": str(PRISM), "sha256": digest(PRISM), "rows": int(len(prism)), "lines": int(prism["depmap_id"].nunique())},
        "predicted_ranking": {
            "all_lines": _summary(line_rows),
            "exact_context_lines": _summary(exact_lines),
            "context_source_counts": pd.Series([row["context_source"] for row in line_rows]).value_counts().to_dict(),
        },
        "predicted_vs_observed_global": {
            "candidate_count": int(len(global_join)),
            "drug_score_spearman": spearman(global_join["predicted_score"].to_numpy(float), global_join["observed_score"].to_numpy(float)),
            "top2_overlap_count": int(len(top2_pred & top2_obs)),
            "predicted_scores": global_join.to_dict("records"),
        },
        "artifacts": {
            "ranking": str(ranking_path),
            "ranking_sha256": digest(ranking_path),
            "per_line_metrics": str(output_dir / "phase1_prism_per_line_metrics.json"),
        },
        "known_deviations": [
            "PRISM response labels are read only after the frozen four-drug candidate identity and Phase-1 predictions are generated.",
            "Lines without an exact LINCS cell_id use a reference-context fallback and are not individualized-context evidence.",
            "Predicted-vs-observed global comparison aggregates Phase-1 predictions across PRISM lines but uses the existing observed-oracle drug-level score; it is a feasibility diagnostic, not a matched-support formal test.",
            "PRISM has no frozen binary sensitive label, so AUROC/AUPRC/label-based Recall@K are not reported.",
        ],
        "line_metrics": line_rows,
    }
    per_line_path = output_dir / "phase1_prism_per_line_metrics.json"
    per_line_path.write_text(json.dumps(line_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary["artifacts"]["per_line_metrics"] = str(per_line_path)
    summary_path = output_dir / "phase1_prism_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = build(manifest_path=args.manifest, checkpoint_path=args.checkpoint, output_dir=args.output_dir, root=args.root)
    print(json.dumps({"status": summary["status"], "predicted": summary["predicted_ranking"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
