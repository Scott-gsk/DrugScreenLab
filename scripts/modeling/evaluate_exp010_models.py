"""Unified leakage-safe evaluation of formal EXP-010 mCRC predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ndcg5(observed: np.ndarray, predicted: np.ndarray) -> float:
    k = min(5, len(observed))
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    gain = lambda value: np.power(2.0, np.maximum(value, 0.0)) - 1.0
    actual = float(np.sum(gain(observed[np.argsort(-predicted)[:k]]) * discount))
    ideal = float(np.sum(gain(observed[np.argsort(-observed)[:k]]) * discount))
    return actual / ideal if ideal > 0 else float("nan")


def patient_metrics(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    samples = []
    for (patient, sample), group in frame.groupby(["patient_id", "sample_id"]):
        rho = spearmanr(group.dss, group[prediction]).statistic if group.dss.nunique() > 1 else np.nan
        samples.append({"patient_id": patient, "sample_id": sample, "spearman": rho,
                        "ndcg5": ndcg5(group.dss.to_numpy(), group[prediction].to_numpy()),
                        "mae": float(np.mean(np.abs(group.dss - group[prediction])))})
    return pd.DataFrame(samples).groupby("patient_id", as_index=False)[["spearman", "ndcg5", "mae"]].mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    frames = []
    for fold in range(5):
        base = pd.read_parquet(args.artifact_root / f"baselines/fold{fold}.parquet")
        base = base[base.source_supported].copy()
        base["fold"] = fold
        for variant, location in (("M2", f"M2_frozen/fold{fold}/M2_fold{fold}_predictions.npz"),
                                  ("M3", f"M3/fold{fold}/M3_fold{fold}_predictions.npz"),
                                  ("M4", f"M4/fold{fold}/M4_fold{fold}_predictions.npz")):
            values = np.load(args.artifact_root / location, allow_pickle=False)
            predicted = pd.DataFrame({"sample_id": values["sample_id"].astype(str),
                                      "drug_name": values["drug_id"].astype(str),
                                      variant: values["predicted"]})
            if not np.isfinite(predicted[variant]).all():
                raise ValueError(f"non-finite {variant} fold {fold}")
            if variant != "M4" and len(predicted) != len(base):
                raise ValueError(f"{variant} fold {fold} rows {len(predicted)} != common support {len(base)}")
            base = base.merge(predicted, on=["sample_id", "drug_name"], how="left", validate="one_to_one")
        frames.append(base)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 1989 or combined.patient_id.nunique() != 52 or combined.isna().any().any():
        raise ValueError("combined common-support coverage/finiteness gate failed")
    columns = ["M0", "M1_ridge", "M1_rf", "M2", "M3", "M4"]
    by_patient = {column: patient_metrics(combined, column) for column in columns}
    metrics = {column: {metric: float(table[metric].mean()) for metric in ("spearman", "ndcg5", "mae")}
               for column, table in by_patient.items()}
    delta = by_patient["M3"].set_index("patient_id").spearman - by_patient["M2"].set_index("patient_id").spearman
    delta = delta.dropna()
    rng = np.random.default_rng(args.seed)
    samples = rng.choice(delta.to_numpy(), size=(args.bootstrap, len(delta)), replace=True).mean(axis=1)
    ci = [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]
    d_spearman = metrics["M3"]["spearman"] - metrics["M2"]["spearman"]
    d_ndcg = metrics["M3"]["ndcg5"] - metrics["M2"]["ndcg5"]
    mae_worsening = (metrics["M3"]["mae"] - metrics["M2"]["mae"]) / metrics["M2"]["mae"]
    promising = d_spearman >= .05 and ci[0] > 0 and d_ndcg >= 0 and mae_worsening <= .05
    summary = {"experiment_id": "EXP-010", "evaluation_rows": len(combined), "patients": 52,
               "support": "GDSC-supported mCRC drugs only", "metrics": metrics,
               "M3_minus_M2": {"spearman": d_spearman, "patient_bootstrap_95ci": ci,
                               "ndcg5": d_ndcg, "mae_relative_worsening": mae_worsening,
                               "bootstrap_replicates": args.bootstrap},
               "verdict": "PROMISING_FAST" if promising else "NO_GO",
               "metric_direction": "higher Spearman/NDCG is better; lower DSS MAE is better; DSS and model labels are higher-is-sensitive"}
    args.output.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output / "common_support_predictions.parquet", index=False)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
