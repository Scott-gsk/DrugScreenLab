"""Evaluate frozen EXP-010 prediction tables; never trains or tunes a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from drug_screen.evaluation.exp010 import cluster_bootstrap_delta, patient_ranking_metrics, promising_fast_decision
from drug_screen.evaluation.exp010_breast import gse303509_rank_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m2-predictions", type=Path)
    parser.add_argument("--m3-predictions", type=Path)
    parser.add_argument("--gse303509-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    result: dict[str, object] = {"format": "exp010_evaluation_bundle_v1", "labels_used_for_tuning": False}
    if bool(args.m2_predictions) != bool(args.m3_predictions):
        parser.error("--m2-predictions and --m3-predictions must be supplied together")
    if args.m2_predictions:
        m2 = patient_ranking_metrics(pd.read_csv(args.m2_predictions))
        m3 = patient_ranking_metrics(pd.read_csv(args.m3_predictions))
        m2_pat = pd.DataFrame(m2["per_patient"])
        m3_pat = pd.DataFrame(m3["per_patient"])
        bootstrap = cluster_bootstrap_delta(m3_pat, m2_pat, repeats=args.bootstrap_repeats, seed=args.seed)
        result["mcrc"] = {"M2": m2, "M3": m3, "m3_minus_m2": bootstrap,
                          "decision": promising_fast_decision(m3["macro"], m2["macro"], bootstrap["ci_95"][0])}
    if args.gse303509_predictions:
        frame = pd.read_csv(args.gse303509_predictions)
        if set(frame.columns) != {"torg_id", "predicted_score"}:
            raise ValueError("GSE303509 predictions require exactly torg_id,predicted_score")
        result["gse303509"] = gse303509_rank_validation(dict(zip(frame.torg_id.astype(str), frame.predicted_score.astype(float), strict=True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
