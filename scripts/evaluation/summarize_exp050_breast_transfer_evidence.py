"""Create a stratified, cross-study synthesis of frozen EXP-050 transfer results.

This script summarizes existing authoritative outputs only. It does not reread
raw expression matrices, fit models, access drug-sensitivity labels, or combine
different drugs into a single transfer score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_breast_transfer_stratified_synthesis_v1"
OUT = ROOT / "artifacts" / "experiments" / "EXP-050" / RUN_ID
INPUTS = {
    "gse287429_gene_similarity": ROOT
    / "artifacts/experiments/EXP-050/20260930_keyed_response_v7/GSE287429_MCF7_all_exposure_similarity_56.tsv",
    "gse287429_functional_summary": ROOT
    / "artifacts/experiments/EXP-050/20261001_frozen_functional_transfer_v2/summary.json",
    "gse320583_gene_agreement": ROOT
    / "artifacts/experiments/EXP-050/20261001_gse320583_birinapant_transfer_v3/gse320583_birinapant_cross_system_gene_agreement.tsv",
    "gse320583_hallmark_directions": ROOT
    / "artifacts/experiments/EXP-050/20261001_gse320583_birinapant_transfer_v3/gse320583_hallmark_cross_system_directions.tsv",
    "gse320583_hallmark_model_consistency": ROOT
    / "artifacts/experiments/EXP-050/20261001_gse320583_birinapant_transfer_v3/gse320583_birinapant_hallmark_model_consistency.tsv",
    "gse320583_summary": ROOT
    / "artifacts/experiments/EXP-050/20261001_gse320583_birinapant_transfer_v3/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    source_gene_rows = read_tsv(INPUTS["gse287429_gene_similarity"])
    source_function = json.loads(INPUTS["gse287429_functional_summary"].read_text(encoding="utf-8"))
    pdo_gene_rows = read_tsv(INPUTS["gse320583_gene_agreement"])
    pdo_hallmark_rows = read_tsv(INPUTS["gse320583_hallmark_directions"])
    pdo_hallmark_consistency = read_tsv(INPUTS["gse320583_hallmark_model_consistency"])
    pdo_summary = json.loads(INPUTS["gse320583_summary"].read_text(encoding="utf-8"))

    functional_by_contrast = {
        item["target_contrast"]: item for item in source_function["by_target_contrast"]
    }
    rows: list[dict[str, object]] = []
    for contrast in sorted({row["target_contrast"] for row in source_gene_rows}):
        group = [row for row in source_gene_rows if row["target_contrast"] == contrast]
        rho = [num(row, "spearman_rho") for row in group]
        sign = [num(row, "sign_agreement") for row in group]
        functional = functional_by_contrast[contrast]
        rows.append(
            {
                "target_dataset": "GSE287429",
                "drug": group[0]["target_drug"],
                "target_contrast": contrast,
                "target_models_or_contrasts": 1,
                "source_exposure_rows": len(group),
                "common_gene_axis_n": int(group[0]["common_gene_axis_n"]),
                "gene_spearman_median": rounded(statistics.median(rho)),
                "gene_spearman_min": rounded(min(rho)),
                "gene_spearman_max": rounded(max(rho)),
                "nonzero_gene_sign_agreement_median": rounded(statistics.median(sign)),
                "hallmark_rank_spearman_median": rounded(
                    functional["hallmark_rank_spearman_median_across_source_conditions"]
                ),
                "hallmark_rank_spearman_min": rounded(functional["hallmark_rank_spearman_min"]),
                "hallmark_rank_spearman_max": rounded(functional["hallmark_rank_spearman_max"]),
                "hallmark_matched_random_median": rounded(
                    functional["matched_random_rank_median_across_source_conditions"]
                ),
                "hallmark_conditions_above_random_97_5pct": (
                    f"{round(functional['fraction_source_conditions_hallmark_above_random_97_5pct'] * len(group))}"
                    f"/{len(group)}"
                ),
                "target_side_full_inchi_key": "present in frozen v7 identity contract",
                "independent_target_replicates": "two treatment and two control samples per contrast; study-level unit remains one contrast",
                "interpretation": "descriptive across fixed MCF7 source exposures; not independent biological n",
            }
        )

    pdo_hallmark_by_model: dict[str, list[dict[str, str]]] = {}
    for row in pdo_hallmark_rows:
        pdo_hallmark_by_model.setdefault(row["model_id"], []).append(row)
    for row in pdo_gene_rows:
        model_id = row["model_id"]
        hallmark = pdo_hallmark_by_model[model_id]
        same = sum(item["direction_relation"] == "same_sign" for item in hallmark)
        opposite = sum(item["direction_relation"] == "opposite_sign" for item in hallmark)
        rows.append(
            {
                "target_dataset": "GSE320583",
                "drug": "Birinapant",
                "target_contrast": model_id,
                "target_models_or_contrasts": 1,
                "source_exposure_rows": int(row["source_plate_replicates"]),
                "common_gene_axis_n": int(row["n_common_landmark_genes"]),
                "gene_spearman_median": rounded(num(row, "gene_spearman_rho")),
                "gene_spearman_min": rounded(num(row, "gene_spearman_rho")),
                "gene_spearman_max": rounded(num(row, "gene_spearman_rho")),
                "nonzero_gene_sign_agreement_median": rounded(
                    num(row, "nonzero_gene_sign_agreement")
                ),
                "hallmark_rank_spearman_median": None,
                "hallmark_rank_spearman_min": None,
                "hallmark_rank_spearman_max": None,
                "hallmark_matched_random_median": None,
                "hallmark_conditions_above_random_97_5pct": None,
                "source_target_hallmark_direction_same_opposite": f"{same}/{opposite}",
                "target_side_full_inchi_key": "absent; official GEO single-agent label only",
                "independent_target_replicates": "one pooled library per model-condition; cells are subsamples",
                "target_cells_and_matched_control": (
                    f"{row['pdo_model_condition_cells']}/{row['pdo_matched_control_cells']}"
                ),
                "interpretation": "same-name, dose/time-matched descriptive comparison; not two-sided full-key closure",
            }
        )

    pdo_direction_counts = {
        value: sum(row["direction_relation"] == value for row in pdo_hallmark_rows)
        for value in sorted({row["direction_relation"] for row in pdo_hallmark_rows})
    }
    pdo_birinapant_consistent_sets = [
        row
        for row in pdo_hallmark_consistency
        if row.get("condition_code") == "Bi" and row.get("cross_model_direction_consistent") == "True"
    ]
    pdo_birinapant_mixed_sets = [
        row
        for row in pdo_hallmark_consistency
        if row.get("condition_code") == "Bi" and row.get("cross_model_direction_consistent") == "False"
    ]
    source_direction_counts = source_function[
        "hallmark_gene_set_direction_counts_over_all_56_comparisons"
    ]
    summary = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "completed_stratified_cross_study_synthesis_descriptive",
        "created_at_utc": started_utc,
        "gpu_hours": 0,
        "analysis_unit_policy": "Keep drugs, studies, doses, times and target identities stratified; do not calculate a pooled transfer score.",
        "source_cell_line": "MCF7 only in all included breast source comparisons.",
        "gse287429": {
            "source_rows_are_independent_biological_replicates": False,
            "functional_direction_counts": source_direction_counts,
            "functional_direction_total": 56 * 50,
            "functional_direction_same_fraction": rounded(
                source_direction_counts["preserved_same_sign"] / (56 * 50), 4
            ),
            "contrasts": [item for item in rows if item["target_dataset"] == "GSE287429"],
            "functional_rank": source_function["by_target_contrast"],
        },
        "gse320583": {
            "target_full_inchi_key_available": False,
            "target_pooled_library_replicates_per_model_condition": 1,
            "source_target_hallmark_direction_counts": pdo_direction_counts,
            "source_target_hallmark_direction_total": len(pdo_hallmark_rows),
            "source_target_hallmark_same_fraction": rounded(
                pdo_direction_counts.get("same_sign", 0) / len(pdo_hallmark_rows), 4
            ),
            "hallmark_consistency_across_models": {
                "same_direction": len(pdo_birinapant_consistent_sets),
                "mixed_direction": len(pdo_birinapant_mixed_sets),
            },
            "models": [item for item in rows if item["target_dataset"] == "GSE320583"],
        },
        "cross_study_synthesis": [
            "Gene-level transfer differs by drug and study: BTZ has moderate median source-exposure rank correlations; DOX is near zero; all three Birinapant PDO model correlations are near zero despite matched name, dose and time.",
            "Functional-set direction matches are a modest majority in each study, with many opposites. The denominators and target designs differ, so the fractions remain separate and are not pooled.",
            "These results support compound- and context-dependent descriptive transfer, not a general breast-cancer transfer rule, PDO efficacy prediction, or clinical claim.",
        ],
        "hard_limits": [
            "GSE287429 target dose is not stated in GEO; its 56 rows reuse four target contrasts across 28 MCF7 source exposures and are not 56 independent target observations.",
            "GSE287429 source conditions are 6/24 h and target contrasts are 23/47 h; GSE70138/GSE320583 Birinapant is annotated 10 uM x 24 h on both sides, but target full InChIKey is absent.",
            "GSE320583 has one pooled library per PDO model-condition; HTO-selected cells do not create biological replicate counts.",
            "Hallmark gene-set means and rank ordering describe predefined gene groups; they are not pathway activity measurements or biological replication.",
            "This synthesis does not use PDO drug-sensitivity labels, train a model, or estimate treatment efficacy.",
        ],
        "stratified_metrics": rows,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    summary_path = OUT / "summary.json"
    table_path = OUT / "stratified_metrics.tsv"
    note_path = OUT / "analysis_result_notes.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    same, opposite = source_direction_counts["preserved_same_sign"], source_direction_counts[
        "changed_opposite_sign"
    ]
    pdo_same = pdo_direction_counts.get("same_sign", 0)
    pdo_opposite = pdo_direction_counts.get("opposite_sign", 0)
    notes = [
        "# EXP-050 跨研究转录响应迁移综合（分层报告）",
        "",
        f"运行：`{RUN_ID}`。只汇总已冻结结果表，CPU-only，新增GPU 0小时；按药物、研究、剂量/时间及靶对象分别报告，不生成合并迁移分数。",
        "",
        "## 基因级排序比较",
        "",
        "| 目标数据/药物与对照 | 来源条件数 | 共同基因数 | 来源条件间Spearman中位数（范围） | 非零变化方向一致率中位数 | 限制 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        if item["target_dataset"] == "GSE287429":
            notes.append(
                f"| GSE287429 {item['drug']} / {item['target_contrast']} | {item['source_exposure_rows']} | {item['common_gene_axis_n']} | {item['gene_spearman_median']:.3f}（{item['gene_spearman_min']:.3f}至{item['gene_spearman_max']:.3f}） | {item['nonzero_gene_sign_agreement_median']:.1%} | 目标剂量未知；来源6/24小时、目标23/47小时；来源行不是独立生物重复 |"
            )
        else:
            notes.append(
                f"| GSE320583 Birinapant / {item['target_contrast']} | 3个MCF7板重复 | {item['common_gene_axis_n']} | {item['gene_spearman_median']:.3f} | {item['nonzero_gene_sign_agreement_median']:.1%} | 同名药、10 µM×24小时；目标全结构键缺失，每条件1份pooled library |"
            )
    notes.extend(
        [
            "",
            "BTZ在两个目标时间的来源条件中呈中等基因排序相关；DOX中位相关接近零。Birinapant在三个PDO模型的相关也接近零。该分层结果提示响应迁移依赖药物/研究条件；不能据此推断普遍迁移规律。",
            "",
            "## Hallmark功能集合描述",
            "",
            f"- GSE287429固定56个比较×50组集合：{same}/{56*50}个集合均值方向同号、{opposite}/{56*50}反号（同号比例{same/(56*50):.1%}）。Hallmark集合排序Spearman中位数：BTZ 23/47小时为0.602/0.639，DOX为0.190/0.087；匹配随机集合分布仅作构成参照。",
            f"- GSE320583 Birinapant来源—目标的3模型×50组集合方向比较：{pdo_same}/{len(pdo_hallmark_rows)}同号、{pdo_opposite}/{len(pdo_hallmark_rows)}反号（同号比例{pdo_same/len(pdo_hallmark_rows):.1%}）；目标集合方向在三个模型中{len(pdo_birinapant_consistent_sets)}/50一致、{len(pdo_birinapant_mixed_sets)}/50混合。",
            "- 两队列方向同号比例都只是轻微多数，分母、平台和样本设计不同，故分别呈现且不合并。集合均值不是通路活性测量，集合/基因/来源条件也不是患者级独立样本。",
            "",
            "## 与筛药结论的关系",
            "",
            "目前没有显示可泛化到PDO的基因响应迁移规律。固定XPert来源独立对象评价未胜过零变化；TranSiGen拟合诊断也未达到有效拟合。3.4仅有带标签暴露史的来源对象探索性转录—药敏关联；没有同一独立PDO对象同时具备处理前后转录响应和结构键对齐药敏值。因此不以本次迁移结果选药、不声称药效预测，也不启动新候选训练或融合。",
            "",
            "20药类器官设计已交付，但与当前已登记PDO处理转录药物的精确结构键覆盖为0/20；预测栏保持空白。恢复目标迁移/药效验证的最小数据要求为：处理前后匹配转录、单药结构身份与匹配对照、剂量/时间、可追溯独立PDO培养重复，并在同一独立对象上有数值药敏或预定功能终点。",
            "",
            "## 输入与复核",
            "",
            "逐条件数字见 `stratified_metrics.tsv`。输入文件路径、SHA-256、脚本哈希、CPU耗时及输出哈希见 `run_manifest.json`。本次无训练、标签读取、下载、外联、采购或湿实验。",
        ]
    )
    note_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    elapsed = time.monotonic() - started
    output_hashes = {path.name: sha256(path) for path in (summary_path, table_path, note_path)}
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_cpu_wall_seconds": round(elapsed, 3),
        "gpu_hours": 0,
        "command": "python scripts/evaluation/summarize_exp050_breast_transfer_evidence.py",
        "environment": "WSL2 Conda drugscreening-gpu",
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            label: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for label, path in INPUTS.items()
        },
        "outputs": output_hashes,
        "protocol_note": "Frozen-run summaries only; no raw target labels, direct sensitivity labels, new candidate, or model fitting.",
    }
    manifest_path = OUT / "run_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"run_id": RUN_ID, "gpu_hours": 0, "outputs": {**output_hashes, "run_manifest.json": sha256(manifest_path)}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
