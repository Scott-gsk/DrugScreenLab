"""Build exportable report figures from frozen EXP-036 and EXP-048 summaries only.

This reporting entrypoint does not load raw response values or select models.
Before EXP-048 aggregation is available, it emits a protocol schematic explicitly
marked as pending; after aggregation, the same command renders the paired result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATHS = {
    25: ROOT / "artifacts/experiments/EXP-036/20260927_25pct_learning_curve_aggregate_v1/run_summary.json",
    50: ROOT / "artifacts/experiments/EXP-036/20260926_pharmaformer_transfer_50pct/fivefold_aggregate_v1/run_summary.json",
    100: ROOT / "artifacts/experiments/EXP-036/20260927_100pct_learning_curve_aggregate_v1/run_summary.json",
}
STAGE_ACCEPTANCE = ROOT / "artifacts/experiments/EXP-041/20260923_adapter_a3_audit_v1/stage_acceptance_v1.json"
FUSION_REVIEW = ROOT / "artifacts/experiments/EXP-048/20260927_fivefold_independent_review/review.json"
FUSION_SOL_DECISION = ROOT / "artifacts/experiments/EXP-048/20260927_fivefold_stage_conclusion/sol_decision.json"
DEFAULT_OUTPUT = ROOT / "docs/reports/figures/dual_route_weekly_progress_20260927"

ARM_LABELS = {
    "target_train_drug_mean": "Training-side drug mean",
    "random_full": "Random init, full fine-tune",
    "source_head": "Source weights, head only",
    "source_full": "Source weights, full fine-tune",
    "source_adapter": "Source weights, adapter",
}
STAT_LABELS = {
    "A1_random_full_vs_target_drug_mean": ("A1 random − drug mean", "A1_random_full_minus_target_train_drug_mean"),
    "A2_source_full_vs_random_full": ("A2 source full − random", "A2_source_full_minus_random_full"),
    "A3_source_adapter_vs_source_full": ("A3 adapter − source full", "A3_source_adapter_minus_source_full"),
    "source_full_vs_target_drug_mean_descriptive_mixed_supervision": (
        "Source full − drug mean*", None),
    "source_adapter_vs_target_drug_mean": ("Adapter − drug mean", None),
    "source_head_vs_target_drug_mean": ("Head only − drug mean", None),
}
FUSION_ARMS = {
    "F-A": "PDO expression + drug structure",
    "F-B": "Drug structure + source perturbation",
    "F-F": "PDO expression + drug structure + perturbation",
    "F-C": "Capacity-matched direct model",
    "F-S": "Fusion with drug-shuffled perturbation",
}
FUSION_CONTRASTS = [
    ("F-F_vs_F-A", "F-F − F-A", "Does fusion exceed the direct model?"),
    ("F-F_vs_F-B", "F-F − F-B", "Does PDO expression add information?"),
    ("F-F_vs_F-C", "F-F − F-C", "Does gain exceed model-capacity effects?"),
    ("F-F_vs_F-S", "F-F − F-S", "Does correct perturbation alignment matter?"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_route1() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    sources: dict[int, dict[str, Any]] = {}
    for percent, path in SUMMARY_PATHS.items():
        if not path.is_file():
            continue
        summary = read_json(path)
        if int(summary.get("n_folds", 0)) != 5 or int(summary.get("n_patients", 0)) != 51:
            continue
        summaries[percent] = summary
        sources[percent] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}

    stage = read_json(STAGE_ACCEPTANCE) if STAGE_ACCEPTANCE.is_file() else {}
    holm = stage.get("route_a_holm_family", {}).get("comparisons", {})
    return summaries, sources, holm


def route1_tables(output: Path, summaries: dict[int, dict[str, Any]],
                  sources: dict[int, dict[str, Any]], holm: dict[str, Any]) -> None:
    means_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for percent, summary in sorted(summaries.items()):
        means = summary.get("mean_patient_spearman", {})
        for arm, label in ARM_LABELS.items():
            means_rows.append({"label_fraction_pct": percent, "n_patients": 51,
                               "arm": arm, "arm_label": label,
                               "mean_patient_spearman": means.get(arm),
                               "source_summary": sources[percent]["path"],
                               "source_sha256": sources[percent]["sha256"]})
        for key, (label, holm_key) in STAT_LABELS.items():
            item = summary.get("statistics", {}).get(key)
            if not item:
                continue
            holm_item = holm.get(holm_key, {}) if percent == 50 and holm_key else {}
            contrast_rows.append({
                "label_fraction_pct": percent, "contrast_key": key, "contrast": label,
                "delta_spearman": item.get("estimate"),
                "ci95_lower": item.get("ci95", [None, None])[0],
                "ci95_upper": item.get("ci95", [None, None])[1],
                "p_raw": item.get("p_two_sided_uncorrected"),
                "p_holm_route_a": holm_item.get("p_holm"),
                "n_patients": item.get("n_patients"),
                "interval_unit": "held-out patient; paired bootstrap",
                "p_method": "finite-corrected two-sided sign flip",
                "p_adjustment": "Route-A five-contrast Holm" if holm_item else "unadjusted",
                "source_summary": sources[percent]["path"],
                "source_sha256": sources[percent]["sha256"],
            })
    write_csv(output / "route1_model_means.csv", means_rows,
              ["label_fraction_pct", "n_patients", "arm", "arm_label", "mean_patient_spearman",
               "source_summary", "source_sha256"])
    write_csv(output / "route1_paired_differences.csv", contrast_rows,
              ["label_fraction_pct", "contrast_key", "contrast", "delta_spearman", "ci95_lower",
               "ci95_upper", "p_raw", "p_holm_route_a", "n_patients", "interval_unit", "p_method",
               "p_adjustment", "source_summary", "source_sha256"])


def plot_route1(output: Path, summaries: dict[int, dict[str, Any]],
                holm: dict[str, Any]) -> None:
    figure, (means_ax, diff_ax) = plt.subplots(1, 2, figsize=(15, 7),
                                              gridspec_kw={"width_ratios": [1.0, 1.35]})
    palette = {
        "target_train_drug_mean": "#606c76", "random_full": "#c47a2c",
        "source_head": "#4c83b6", "source_full": "#16846b", "source_adapter": "#8966a8",
    }
    percentages = sorted(summaries)
    for arm, label in ARM_LABELS.items():
        x = [percent for percent in percentages if arm in summaries[percent].get("mean_patient_spearman", {})]
        y = [summaries[percent]["mean_patient_spearman"][arm] for percent in x]
        if x:
            means_ax.plot(x, y, marker="o", linewidth=1.8, markersize=6,
                          color=palette[arm], label=label)
    means_ax.set_title("Patient-macro drug ranking by target-label fraction")
    means_ax.set_xlabel("Target training labels available (%)")
    means_ax.set_ylabel("Mean within-patient Spearman (higher is better)")
    means_ax.set_xticks([25, 50, 100])
    means_ax.set_xlim(18, 107)
    means_ax.set_ylim(0.68, 0.78)
    means_ax.grid(axis="y", alpha=0.25)
    if 100 not in summaries:
        means_ax.axvspan(94, 106, color="#d9d9d9", alpha=0.35, hatch="///")
        means_ax.text(100, 0.687, "100%: 2/5 folds verified;\nfull summary pending",
                      ha="center", va="bottom", fontsize=8, color="#555555")
    means_ax.legend(fontsize=8, loc="upper left", frameon=False)

    plotted: list[tuple[int, str, dict[str, Any]]] = []
    display_keys = [
        "A1_random_full_vs_target_drug_mean",
        "A2_source_full_vs_random_full",
        "A3_source_adapter_vs_source_full",
        "source_full_vs_target_drug_mean_descriptive_mixed_supervision",
    ]
    for percent in percentages:
        stats = summaries[percent].get("statistics", {})
        for key in display_keys:
            if key in stats:
                plotted.append((percent, key, stats[key]))
    y_positions = np.arange(len(plotted))[::-1]
    colors = {25: "#d9822b", 50: "#3478a8", 100: "#16846b"}
    for ypos, (percent, key, item) in zip(y_positions, plotted):
        low, high = item["ci95"]
        effect = item["estimate"]
        diff_ax.errorbar(effect, ypos, xerr=[[effect - low], [high - effect]],
                         fmt="o", color=colors.get(percent, "#16846b"), capsize=3, markersize=5)
        holm_key = STAT_LABELS[key][1]
        holm_value = holm.get(holm_key, {}).get("p_holm") if percent == 50 and holm_key else None
        p_label = f"p={item['p_two_sided_uncorrected']:.3g}"
        if holm_value is not None:
            p_label += f"; Holm={holm_value:.3g}"
        diff_ax.text(0.99, ypos, p_label,
                     transform=diff_ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    diff_ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    diff_ax.set_yticks(y_positions)
    labels = [f"{percent}%  {STAT_LABELS[key][0]}" for percent, key, _ in plotted]
    diff_ax.set_yticklabels(labels, fontsize=8)
    diff_ax.set_xlim(-0.055, 0.05)
    diff_ax.set_xlabel("Paired difference in within-patient Spearman (95% CI)")
    diff_ax.set_title("Paired patient contrasts")
    diff_ax.grid(axis="x", alpha=0.2)
    diff_ax.text(0.01, -0.13, "25% and 100% p values unadjusted; 50% A1/A2/A3 use five-contrast Holm in the report.\n"
                 "* Source-full vs drug-mean changes the supervision condition; descriptive only.",
                 transform=diff_ax.transAxes, fontsize=8, va="top")

    figure.suptitle("Direct drug-response transfer: descriptive means and paired contrasts", fontsize=14)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    for suffix in ("png", "svg"):
        figure.savefig(output / f"route1_transfer_summary.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_fusion(output: Path, fusion_summary: Path | None) -> dict[str, Any]:
    design_rows = [{"contrast_key": key, "contrast": label, "question": question,
                    "status": "pending_full_fivefold_result"}
                   for key, label, question in FUSION_CONTRASTS]
    write_csv(output / "fusion_contrast_design.csv", design_rows,
              ["contrast_key", "contrast", "question", "status"])
    if not fusion_summary or not fusion_summary.is_file():
        figure, ax = plt.subplots(figsize=(13, 7))
        targets = ["F-A", "F-B", "F-C", "F-S"]
        rows_y = [0.81, 0.63, 0.45, 0.27]
        for target, y in zip(targets, rows_y):
            ax.text(0.20, y, f"F-F\n{FUSION_ARMS['F-F']}", ha="center", va="center", fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.55", "facecolor": "#e4f1ed", "edgecolor": "#16846b"},
                    transform=ax.transAxes)
            ax.text(0.80, y, f"{target}\n{FUSION_ARMS[target]}", ha="center", va="center", fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.55", "facecolor": "#eef3f6", "edgecolor": "#546a78"},
                    transform=ax.transAxes)
            ax.annotate("", xy=(0.65, y), xytext=(0.35, y),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops={"arrowstyle": "<->", "color": "#8a6f3d", "lw": 1.3})
            ax.text(0.50, y + 0.055, f"F-F vs {target}", ha="center", va="bottom", fontsize=8,
                    transform=ax.transAxes, color="#665127")
        ax.text(0.5, 0.08,
                "The same F-F model is repeated here only to display each registered pair.\n"
                "Common folds, 50% target labels, three seeds, seven observed drug keys per patient; results pending.",
                ha="center", va="center", fontsize=9, transform=ax.transAxes, color="#555555")
        ax.set_axis_off()
        ax.set_title("Five-arm fusion design — no outer-effect score available yet", fontsize=14)
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(output / f"fusion_five_arm_contrasts.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(figure)
        return {"status": "DESIGN_ONLY_RESULTS_PENDING", "summary": None}

    report = read_json(fusion_summary)
    arm_summary = report["mean_patient_spearman_by_arm"]
    comparisons = report["registered_paired_comparisons"]
    arm_rows = [{"arm": arm, **arm_summary[arm]} for arm in FUSION_ARMS if arm in arm_summary]
    comparison_rows = []
    for key, label, question in FUSION_CONTRASTS:
        item = comparisons[key]
        comparison_rows.append({"contrast_key": key, "contrast": label, "question": question,
                                "delta_spearman": item["estimate_fusion_minus_control"],
                                "ci95_lower": item["ci95_patient_bootstrap"][0],
                                "ci95_upper": item["ci95_patient_bootstrap"][1],
                                "p_raw": item["p_two_sided_sign_flip_uncorrected"],
                                "p_holm": item["p_holm_four_comparisons"],
                                "n_patients": item["n_paired_patients"]})
    write_csv(output / "fusion_arm_means.csv", arm_rows,
              ["arm", "mean_patient_spearman", "sd_between_patients", "n_valid_patients"])
    write_csv(output / "fusion_paired_differences.csv", comparison_rows,
              ["contrast_key", "contrast", "question", "delta_spearman", "ci95_lower",
               "ci95_upper", "p_raw", "p_holm", "n_patients"])

    figure, (means_ax, diff_ax) = plt.subplots(1, 2, figsize=(13, 6.5),
                                              gridspec_kw={"width_ratios": [0.9, 1.2]})
    means = [arm_summary[arm]["mean_patient_spearman"] for arm in FUSION_ARMS]
    arm_order = list(FUSION_ARMS)
    means_ax.scatter(means, arm_order, color=["#606c76", "#4c83b6", "#16846b", "#c47a2c", "#8966a8"], s=55)
    means_ax.set_xlabel("Mean within-patient Spearman")
    means_ax.set_title("Arm-level patient means\n(descriptive; paired CIs at right)")
    means_ax.grid(axis="x", alpha=0.2)
    means_ax.set_xlim(0, 1)
    diff_y = np.arange(len(comparison_rows))[::-1]
    for ypos, row in zip(diff_y, comparison_rows):
        estimate, low, high = row["delta_spearman"], row["ci95_lower"], row["ci95_upper"]
        diff_ax.errorbar(estimate, ypos, xerr=[[estimate - low], [high - estimate]],
                         fmt="o", color="#16846b", capsize=3, markersize=5)
        diff_ax.text(0.99, ypos, f"Holm p={row['p_holm']:.3g}",
                     transform=diff_ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
    diff_ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    diff_ax.set_yticks(diff_y)
    diff_ax.set_yticklabels([row["contrast"] for row in comparison_rows])
    diff_ax.set_xlabel("F-F minus comparator (95% patient-paired CI)")
    diff_ax.set_title("Registered fusion contrasts\n(positive favors F-F)")
    diff_ax.set_xlim(-0.03, 0.105)
    diff_ax.grid(axis="x", alpha=0.2)
    figure.suptitle("Nine-drug fusion: five-fold patient-level comparison", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "svg"):
        figure.savefig(output / f"fusion_five_arm_contrasts.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    fusion_status = report.get("status", "UNKNOWN")
    result: dict[str, Any] = {"status": fusion_status,
                              "summary": fusion_summary.relative_to(ROOT).as_posix(),
                              "summary_sha256": sha256(fusion_summary)}
    if FUSION_REVIEW.is_file() and FUSION_SOL_DECISION.is_file():
        review = read_json(FUSION_REVIEW)
        sol = read_json(FUSION_SOL_DECISION)
        review_matches = (
            review.get("decision") == "PASS_SCOPED_EXP048_NINE_DRUG_FIVEFOLD_RESULT"
            and review.get("evidence", {}).get("aggregate_summary_sha256") == sha256(fusion_summary)
        )
        sol_conditional = (
            sol.get("decision") == "CONDITIONALLY_ACCEPT_SCOPED_NINE_DRUG_FIVEFOLD_RESULT_PENDING_INDEPENDENT_REVIEW"
        )
        if review_matches and sol_conditional:
            result["status"] = "SCOPED_RESULT_ACCEPTED_AFTER_SOL_AND_INDEPENDENT_REVIEW"
        result["independent_review"] = {
            "path": FUSION_REVIEW.relative_to(ROOT).as_posix(),
            "sha256": sha256(FUSION_REVIEW),
            "decision": review.get("decision"),
            "summary_hash_matches": review.get("evidence", {}).get("aggregate_summary_sha256") == sha256(fusion_summary),
        }
        result["sol_decision"] = {
            "path": FUSION_SOL_DECISION.relative_to(ROOT).as_posix(),
            "sha256": sha256(FUSION_SOL_DECISION),
            "decision": sol.get("decision"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fusion-summary", type=Path, default=None,
                        help="pass the verified five-fold EXP-048 run_summary.json after scoring/review")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summaries, sources, holm = load_route1()
    if 25 not in summaries or 50 not in summaries:
        raise FileNotFoundError("complete 25% and 50% five-fold patient summaries are required")
    route1_tables(output, summaries, sources, holm)
    plot_route1(output, summaries, holm)
    fusion_summary = args.fusion_summary
    if fusion_summary is not None and not fusion_summary.is_absolute():
        fusion_summary = ROOT / fusion_summary
    fusion_state = plot_fusion(output, fusion_summary)
    source_manifest = {str(percent): source for percent, source in sources.items()}
    if STAGE_ACCEPTANCE.is_file():
        source_manifest["route_a_stage_acceptance"] = {
            "path": STAGE_ACCEPTANCE.relative_to(ROOT).as_posix(), "sha256": sha256(STAGE_ACCEPTANCE)}
    manifest = {"status": "REPORT_FIGURES_WRITTEN", "route1_sources": source_manifest,
                "route1_complete_label_fractions": sorted(summaries), "fusion": fusion_state,
                "outer_scores_used_for_model_or_direction_selection": False,
                "files": sorted(path.name for path in output.iterdir() if path.is_file())}
    (output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
