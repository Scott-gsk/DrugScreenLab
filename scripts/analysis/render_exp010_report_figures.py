"""Render fixed EXP-010 report figures from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig, stem: Path) -> list[str]:
    outputs = []
    for suffix in ("png", "svg"):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _patient_spearman(frame: pd.DataFrame, model: str) -> pd.Series:
    per_pdo = []
    for (patient, sample), group in frame.groupby(["patient_id", "sample_id"], sort=True, observed=True):
        value = spearmanr(group[model].to_numpy(float), group["dss"].to_numpy(float)).statistic
        if np.isfinite(value):
            per_pdo.append({"patient_id": patient, "sample_id": sample, "spearman": float(value)})
    return pd.DataFrame(per_pdo).groupby("patient_id", observed=True)["spearman"].mean()


def render(*, predictions_path: Path, metrics_path: Path, m2_breast_path: Path, m3_breast_path: Path,
           gse287429_figure: Path, breast_analysis_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(predictions_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_order = [name for name in ("M0", "M1_ridge", "M1_rf", "M2", "M3", "M4") if name in metrics["metrics"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    metric_specs = (("spearman", "Patient-macro Spearman", True), ("ndcg5", "Patient-macro NDCG@5", True),
                    ("mae", "DSS MAE (lower is better)", False))
    colors = ["#777777", "#5b8ff9", "#61d9a5", "#f6bd16", "#e8684a", "#9270ca"]
    for ax, (key, title, _) in zip(axes, metric_specs, strict=True):
        values = [metrics["metrics"][model][key] for model in model_order]
        ax.bar(model_order, values, color=colors[:len(model_order)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=.2)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"EXP-010 frozen common support: {metrics['patients']} patients / {metrics['evaluation_rows']} rows")
    aggregate_outputs = _save(fig, output_dir / "m0_m4_metric_comparison")

    m2_patient = _patient_spearman(predictions, "M2")
    m3_patient = _patient_spearman(predictions, "M3")
    paired = pd.concat([m2_patient.rename("M2"), m3_patient.rename("M3")], axis=1, join="inner").dropna()
    paired["delta"] = paired["M3"] - paired["M2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for _, row in paired.iterrows():
        axes[0].plot([0, 1], [row.M2, row.M3], color="#999999", alpha=.45, linewidth=.8)
    axes[0].scatter(np.zeros(len(paired)), paired.M2, color="#f6bd16", s=18, label="M2")
    axes[0].scatter(np.ones(len(paired)), paired.M3, color="#e8684a", s=18, label="M3")
    axes[0].set_xticks([0, 1], ["M2", "M3"]); axes[0].set_ylabel("Patient-level mean within-PDO Spearman")
    axes[0].set_title(f"Paired patients (n={len(paired)})")
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].hist(paired.delta, bins=12, color="#e8684a", alpha=.8)
    axes[1].set_xlabel("M3 − M2 patient Spearman"); axes[1].set_ylabel("Patients")
    ci = metrics["M3_minus_M2"]["patient_bootstrap_95ci"]
    axes[1].set_title(f"Mean Δ={paired.delta.mean():.3f}\nFrozen bootstrap 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    patient_outputs = _save(fig, output_dir / "m3_minus_m2_patient_spearman")

    breast_results = {
        "M2": json.loads(m2_breast_path.read_text(encoding="utf-8")),
        "M3": json.loads(m3_breast_path.read_text(encoding="utf-8")),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (model, result) in zip(axes, breast_results.items(), strict=True):
        rows = pd.DataFrame(result["rows"])
        rows["observed_rank"] = rows["observed_death_area_ratio"].rank(method="average")
        rows["predicted_rank"] = rows["predicted_score"].rank(method="average")
        for row in rows.itertuples(index=False):
            ax.plot([0, 1], [row.observed_rank, row.predicted_rank], marker="o", linewidth=1, label=row.torg_id)
        ax.set_xticks([0, 1], ["Observed", model]); ax.set_ylabel("Sensitivity rank (higher = more sensitive)")
        ax.set_title(f"{model}: ρ={result['spearman']:.3f}, concordant={result['pairwise_concordant']}/21\n{result['status']}")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=7)
    breast_outputs = _save(fig, output_dir / "gse303509_m2_vs_m3_rank")

    breast_analysis = json.loads(breast_analysis_path.read_text(encoding="utf-8"))
    multi = breast_analysis["multiregional"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].imshow(plt.imread(gse287429_figure)); axes[0].axis("off"); axes[0].set_title("GSE287429 Reactome (descriptive)")
    axes[1].axis("off")
    axes[1].text(.02, .95, "Multi-regional breast PDO", fontsize=15, weight="bold", va="top")
    axes[1].text(.02, .78, multi["status"], fontsize=14, color="#b33a3a", weight="bold")
    axes[1].text(.02, .62, f"Access attempts: {multi['attempt_count']}\nAccepted assets: {multi['accepted_asset_count']}", fontsize=11)
    axes[1].text(.02, .40, "No public expression + drug-response matrix passed format gates.\nNo four-drug external ranking result is claimed.", fontsize=10, va="top")
    context_outputs = _save(fig, output_dir / "breast_context_and_multiregional_status")

    inputs = [predictions_path, metrics_path, m2_breast_path, m3_breast_path, gse287429_figure, breast_analysis_path]
    manifest = {"format": "exp010_figure_manifest_v1", "selection_or_recomputation": False,
                "inputs": [{"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size} for path in inputs],
                "figures": [
                    {"id": "m0_m4_metrics", "outputs": aggregate_outputs, "claim_scope": "Frozen common-support benchmark metrics; no biological claim."},
                    {"id": "m3_minus_m2_patient", "outputs": patient_outputs, "claim_scope": "Patient-level paired visualization; frozen bootstrap CI is displayed, not recomputed."},
                    {"id": "gse303509_rank", "outputs": breast_outputs, "claim_scope": "Seven-TORG cisplatin-containing combination directional concordance only."},
                    {"id": "breast_context", "outputs": context_outputs, "claim_scope": "GSE287429 perturbation pathways are descriptive; multi-regional matrix is not evaluable."},
                ]}
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--breast-analysis-root", type=Path, required=True)
    args = parser.parse_args()
    root, breast = args.artifact_root, args.breast_analysis_root
    render(predictions_path=root / "evaluation/common_support_predictions.parquet",
           metrics_path=root / "evaluation/metrics.json",
           m2_breast_path=root / "breast_gse303509/M2_evaluation.json",
           m3_breast_path=root / "breast_gse303509/M3_ensemble_evaluation.json",
           gse287429_figure=breast / "gse287429_reactome.png",
           breast_analysis_path=breast / "breast_model_independent_analysis.json",
           output_dir=root / "report/figures")


if __name__ == "__main__":
    main()
