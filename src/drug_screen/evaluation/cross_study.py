"""Strict, frozen cross-study ranking evaluator for one external FAST check."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import spearman


class CrossStudyBlockedError(RuntimeError):
    """Raised when the external label asset is unavailable or violates the contract."""


REQUIRED_LABEL_COLUMNS = {"study_id", "context_id", "pert_id", "sensitivity_score"}
REQUIRED_PREDICTION_COLUMNS = {"study_id", "context_id", "pert_id", "reversal_predicted"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_frozen_cross_study(
    *,
    labels_path: Path | str,
    predictions_path: Path | str,
    output_path: Path | str,
    frozen_candidate_ids: list[str],
) -> dict[str, Any]:
    """Evaluate a precomputed external sensitivity table without tuning or dropping IDs."""
    labels_file = Path(labels_path)
    predictions_file = Path(predictions_path)
    if not labels_file.is_file():
        raise CrossStudyBlockedError(f"external labels are unavailable: {labels_file}")
    if not predictions_file.is_file():
        raise CrossStudyBlockedError(f"external predictions are unavailable: {predictions_file}")
    labels = pd.read_csv(labels_file, sep="\t")
    predictions = pd.read_csv(predictions_file, sep="\t")
    if not REQUIRED_LABEL_COLUMNS.issubset(labels.columns):
        raise CrossStudyBlockedError("external labels do not satisfy the frozen schema")
    if not REQUIRED_PREDICTION_COLUMNS.issubset(predictions.columns):
        raise CrossStudyBlockedError("external predictions do not satisfy the frozen schema")
    expected = set(frozen_candidate_ids)
    observed = set(labels["pert_id"].astype(str))
    predicted = set(predictions["pert_id"].astype(str))
    if observed != expected or predicted != expected:
        raise CrossStudyBlockedError("candidate identity set differs from the frozen cohort")
    joined = labels.merge(
        predictions,
        on=["study_id", "context_id", "pert_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise CrossStudyBlockedError("label/prediction support sets differ")
    rows: list[dict[str, Any]] = []
    for (study_id, context_id), group in joined.groupby(["study_id", "context_id"], sort=True):
        rows.append({
            "study_id": str(study_id),
            "context_id": str(context_id),
            "candidate_count": int(len(group)),
            "spearman": spearman(
                group["reversal_predicted"].to_numpy(float),
                group["sensitivity_score"].to_numpy(float),
            ),
        })
    values = [float(row["spearman"]) for row in rows if row["spearman"] is not None]
    result = {
        "format": "frozen_cross_study_evaluation_v1",
        "status": "COMPLETE",
        "labels": {"path": str(labels_file), "sha256": file_sha256(labels_file), "rows": int(len(labels))},
        "predictions": {"path": str(predictions_file), "sha256": file_sha256(predictions_file), "rows": int(len(predictions))},
        "candidate_ids": frozen_candidate_ids,
        "line_count": len(rows),
        "macro_mean_spearman": float(np.mean(values)) if values else None,
        "per_context": rows,
        "labels_used_for_tuning": False,
    }
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
