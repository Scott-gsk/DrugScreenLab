"""Compute the learned MVP-001 predicted reversal ranking.

The checkpoint is trained only on the frozen manifest's train rows.  This
evaluation predicts the manifest's held-out test treatment groups, aggregates
groups equally within each drug, and applies the same disease-signature
anti-correlation score as the observed LINCS oracle.  It never reads held-out
Delta978 values or PRISM response labels.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drug_screen.modeling.mvp001 import CompactManifest, DrugDoseTimePredictor


SIGNATURE = Path("mvp/core_data/crc_disease_signature_exact978.tsv")
PRISM = Path("mvp/core_data/compact_prism_response.parquet")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_rank(values: np.ndarray) -> np.ndarray:
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


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank, right_rank = average_rank(left), average_rank(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def build(
    *,
    manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    data_root: Path,
) -> dict[str, object]:
    manifest = CompactManifest.load(manifest_path, data_root=data_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocabularies"]
    model = DrugDoseTimePredictor(
        len(vocab["drug"]),
        len(vocab["dose"]),
        len(vocab["time"]),
        int(config["model"]["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    signature = pd.read_csv(SIGNATURE, sep="\t")
    genes = signature["gene_index_978"].astype(int).to_numpy()
    signature_values = signature["signed_log2fc"].astype(float).to_numpy()
    test_rows = [row for row in manifest.records if row.split == "test"]
    if not test_rows:
        raise RuntimeError("predicted reversal requires held-out test groups")

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for row in test_rows:
            drug = torch.tensor([vocab["drug"][row.drug_id]])
            dose = torch.tensor([vocab["dose"][row.dose_id]])
            time = torch.tensor([vocab["time"][row.time_id]])
            predictions.append(model(drug, dose, time).squeeze(0).numpy().astype(np.float32))
    predictions_array = np.stack(predictions)

    # Equal-weight technical groups first, then median across held-out groups,
    # matching the observed-oracle group aggregation contract.
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(test_rows):
        grouped.setdefault(row.treatment_group_id, []).append(index)
    group_vectors: dict[str, tuple[object, np.ndarray]] = {}
    for group_id, indices in sorted(grouped.items()):
        first = test_rows[indices[0]]
        group_vectors[group_id] = (first.drug_id, predictions_array[indices].mean(axis=0))
    by_drug: dict[str, list[np.ndarray]] = {}
    for drug_id, vector in group_vectors.values():
        by_drug.setdefault(str(drug_id), []).append(vector)

    rows: list[dict[str, object]] = []
    for drug_id, vectors in sorted(by_drug.items()):
        vector = np.median(np.stack(vectors), axis=0)
        raw_spearman = spearman(signature_values, vector[genes])
        rows.append(
            {
                "pert_id": drug_id,
                "n_test_groups": len(vectors),
                "reversal_predicted": None if raw_spearman is None else -raw_spearman,
            }
        )
    scores = pd.DataFrame(rows)
    if scores["reversal_predicted"].isna().any():
        raise RuntimeError("predicted reversal contains a non-finite drug score")
    scores["rank_desc_reversal"] = average_rank(-scores["reversal_predicted"].to_numpy(float))

    # Attach the frozen PRISM identity metadata without using response values
    # to choose conditions or candidates.
    prism = pd.read_parquet(PRISM)
    # Manifest `drug_id` is the frozen LINCS perturbagen name; resolve it to
    # the PRISM pert_id only through the compact identity table.
    scores = scores.rename(columns={"pert_id": "drug_name"})
    identity = prism[["pert_id", "broad_id", "drug_name"]].drop_duplicates("drug_name")
    scores = scores.merge(identity, on="drug_name", how="left", validate="one_to_one")
    if scores[["broad_id", "drug_name"]].isna().any().any():
        raise RuntimeError("predicted cohort is not fully represented in PRISM compact identity")
    scores = scores.sort_values(["rank_desc_reversal", "pert_id"], kind="mergesort").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "MVP-001_predicted_reversal_ranking.tsv"
    columns = ["rank_desc_reversal", "reversal_predicted", "broad_id", "drug_name", "pert_id", "n_test_groups"]
    scores[columns].to_csv(ranking_path, sep="\t", index=False, float_format="%.10g")

    observed_path = output_dir.parent / "observed_oracle" / "MVP-001_observed_oracle_ranking.tsv"
    observed = pd.read_csv(observed_path, sep="\t")
    observed = observed[observed["pert_id"].isin(scores["pert_id"])][["pert_id", "reversal_observed", "rank_desc_reversal"]]
    joined = scores[["pert_id", "reversal_predicted", "rank_desc_reversal"]].merge(observed, on="pert_id", suffixes=("_predicted", "_observed"), validate="one_to_one")
    observed_predicted_spearman = spearman(joined["reversal_observed"].to_numpy(float), joined["reversal_predicted"].to_numpy(float))
    rank_overlap_top2 = len(set(joined.nsmallest(2, "rank_desc_reversal_predicted")["pert_id"]) & set(joined.nsmallest(2, "rank_desc_reversal_observed")["pert_id"]))
    summary = {
        "format": "mvp001_predicted_reversal_ranking_v1",
        "mvp_id": "MVP-001",
        "status": "READY_FOR_PRISM_JOIN",
        "model": {
            "config_path": str(config_path),
            "config_sha256": digest(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": digest(checkpoint_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest.source_digest,
            "seed": checkpoint.get("seed"),
        },
        "signature": {
            "path": str(SIGNATURE),
            "sha256": digest(SIGNATURE),
            "rows": int(len(signature)),
            "up": int((signature["direction"] == "up").sum()),
            "down": int((signature["direction"] == "down").sum()),
        },
        "prediction": {
            "split": "held_out_test_groups",
            "test_rows": len(test_rows),
            "test_groups": len(group_vectors),
            "drug_count": len(scores),
            "aggregation": "mean within technical group; coordinate median across held-out groups",
            "score": "-Spearman(signature signed_log2fc, predicted Delta978)",
        },
        "predicted_vs_observed": {
            "same_candidate_cohort": True,
            "candidate_count": int(len(joined)),
            "drug_score_spearman": observed_predicted_spearman,
            "top2_overlap_count": rank_overlap_top2,
            "top2_overlap_rate": rank_overlap_top2 / 2.0,
            "joined_scores": joined.to_dict("records"),
        },
        "prism": {
            "response_asset": str(PRISM),
            "response_sha256": digest(PRISM),
            "metrics": None,
        },
        "artifacts": {
            "ranking": str(ranking_path),
            "ranking_sha256": digest(ranking_path),
            "predictions_serialized": False,
        },
        "known_deviations": [
            "Predicted ranking uses held-out test treatment groups, while the observed oracle aggregates all eligible LINCS groups.",
            "Only the frozen four-drug PRISM-overlap cohort is evaluated; no response-dependent cohort expansion is performed.",
        ],
    }
    summary_path = output_dir / "MVP-001_predicted_reversal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/mvp/MVP-001/compact_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/mvp/MVP-001/small.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/mvp001-small-core4/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("mvp/core_eval/predicted_reversal"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    summary = build(
        manifest_path=args.manifest,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_root=args.data_root,
    )
    print(json.dumps({"status": summary["status"], "artifacts": summary["artifacts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
