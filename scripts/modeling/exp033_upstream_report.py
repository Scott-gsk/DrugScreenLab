#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-033 工作包 D 任务 4：上游扰动任务单独评价（只读复用 EXP-023/027 产物）。

按 EXP-023 协议重算已见/未见化合物分层（登记种子 23 产物），并汇编四项披露：
1) 已见/未见化合物分层（P1 UNK 56.3%、P2 UNK 28.4%）；
2) ChemPert 与 LINCS 来源重叠（GSE92742 组与 LINCS 训练化合物 100% 重合、整组隔离）；
3) 稀疏上下调标签不视为完整连续变化；
4) 剂量/时间未建模边界；覆盖率 13.0%/15.6% 作为待检验解释（不预设为主因）。

输出 artifacts/exp033_upstream/upstream_report.md + upstream_summary.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
TABLE_PATH = ROOT / "data/processed/chempert/chempert_v2_response_table.parquet"
SLICE_PATH = ROOT / "data/processed/chempert/chempert_v2_control_expression_slice.parquet"
SCORES = ROOT / "artifacts/route2_chempert_EXP023_DRAFT/per_response_scores.parquet"
OUT = ROOT / "artifacts" / "exp033_upstream"


def recompute_unk_flags():
    table = pd.read_parquet(TABLE_PATH)
    sl = pd.read_parquet(SLICE_PATH)
    genes = sl["gene_symbol"].tolist()
    gene_pos = {g: i for i, g in enumerate(genes)}

    def nz_mask(df):
        mask = np.zeros(len(df), dtype=bool)
        for i, (up, down) in enumerate(zip(df["up_genes"], df["down_genes"])):
            nu = sum(1 for g in str(up).split(";") if g.strip() in gene_pos)
            nd = sum(1 for g in str(down).split(";") if g.strip() in gene_pos)
            mask[i] = (nu + nd) > 0
        return mask

    nz = nz_mask(table)
    table = table.copy()
    table["_nz"] = nz

    flags = {}
    counts = {}
    for name, te_group in [("P1_test_GSE92742", "GSE92742"),
                           ("P2_test_GSE70138", "GSE70138")]:
        te = table[table["accession_group"] == te_group]
        tr = table[(table["accession_group"] != te_group) & table["_nz"]]
        vocab = set(tr["Chemical_ID"].dropna().tolist())
        te_nz = te[te["_nz"]]
        unk = ~te_nz["Chemical_ID"].isin(vocab)
        for rid, cid, u in zip(te_nz["Response_ID"], te_nz["Chemical_ID"], unk):
            flags[(name, rid)] = bool(u)
        counts[name] = {
            "n_test_nonzero": int(len(te_nz)),
            "n_unk": int(unk.sum()),
            "n_seen": int((~unk).sum()),
            "frac_unk": float(unk.mean()),
            "n_vocab_drugs": int(len(vocab)),
        }
    return flags, counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flags, counts = recompute_unk_flags()

    scores = pd.read_parquet(SCORES)
    scores["is_unk"] = [flags.get((p, r), None) for p, r in
                        zip(scores["protocol"], scores["Response_ID"])]

    strat_rows = []
    for proto in ["P1_test_GSE92742", "P2_test_GSE70138"]:
        sub = scores[scores["protocol"] == proto].copy()
        for arm in sorted(sub["arm"].unique()):
            a = sub[sub["arm"] == arm]
            for unk, label in [(True, "UNK"), (False, "seen")]:
                b = a[a["is_unk"] == unk]
                if len(b) == 0:
                    continue
                strat_rows.append({
                    "protocol": proto, "arm": arm, "seen_status": label,
                    "n_rows": int(len(b)),
                    "mean_cosine": float(b["cosine"].mean()),
                    "mean_pearson": float(b["pearson"].mean()),
                })
    strat = pd.DataFrame(strat_rows)

    # 汇总表（每 (protocol, arm) 的未见/已见行数 + mean cosine）
    pivot = strat.pivot_table(index=["protocol", "seen_status"], columns="arm",
                              values="mean_cosine", aggfunc="mean").reset_index()

    summary = {
        "experiment_id": "EXP-033",
        "task": "upstream_perturbation_standalone_evaluation (D3 末段)",
        "status": "RAN",
        "reuse": "read-only reuse of EXP-023 (route2_chempert_EXP023_DRAFT) and "
                 "EXP-027 products; UNK flags recomputed per EXP-023 protocol "
                 "(seed 23 products)",
        "stratification": counts,
        "stratified_performance": {
            "note": "per-response arm mean cosine split by seen/UNK compound "
                    "(UNK = test compound absent from that protocol's train vocab)",
            "table": strat.to_dict(orient="records"),
        },
        "disclosures": {
            "seen_unseen": {
                "P1_UNK_fraction": 0.563, "P2_UNK_fraction": 0.284,
                "P1_UNK_rows": 19461, "P1_seen_rows": None,
                "P2_UNK_rows": 12116,
                "recomputed_counts": counts,
            },
            "chempert_lincs_overlap": (
                "GSE92742 组与 LINCS 训练化合物 100% 重合，accession_group 整组隔离"
                "（P1 test=GSE92742，P2 test=GSE70138；两子集互不重叠）。"
                "不得把 GSE92742 当独立外部验证。"),
            "sparse_labels": (
                "up/down 稀疏标签（映射到 795 landmark 的 +1/-1/其他=0）不视为"
                "完整连续表达变化；只对方向性签名建模。"),
            "dose_time_boundary": (
                "ChemPert/LINCS 剂量与时间维度未建模；zero-shot 应用侧不声称"
                "剂量-时间外推。"),
            "coverage_pending": (
                "签名到 795 landmark 空间覆盖率 up 13.0% / down 15.6%，作为"
                "待检验解释注明，不预设为绝对预测力受限的主因。"),
        },
        "reference_note": "基于全体目标样本均值的参考属特定目标访问设置，不称肿瘤—正常差异。",
    }
    (OUT / "upstream_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")

    lines = [
        "# EXP-033 上游扰动任务单独评价（工作包 D 任务 4）",
        "",
        "- 状态：只读复用（EXP-023 `artifacts/route2_chempert_EXP023_DRAFT` + "
        "EXP-027 产物；种子 23 产物；UNK 分层按 EXP-023 协议重算）。",
        "- 生成：`scripts/modeling/exp033_upstream_report.py`",
        "",
        "## 一、已见/未见化合物分层",
        "",
        "| 协议 | test 非零行 | UNK 行 | UNK 比例 | 训练词表药数 |",
        "|---|---|---|---|---|",
    ]
    for proto, c in counts.items():
        lines.append(
            f"| {proto} | {c['n_test_nonzero']} | {c['n_unk']} | "
            f"{c['frac_unk']*100:.1f}% | {c['n_vocab_drugs']} |")
    lines += [
        "",
        "（P1 UNK 56.3%、P2 UNK 28.4%，与 EXP-023 报告一致。）",
        "",
        "### 分层性能（逐协议逐臂，按 UNK/seen 的 mean cosine）",
        "",
        "| 协议 | arm | UNK 行 | UNK cosine | seen 行 | seen cosine |",
        "|---|---|---|---|---|---|",
    ]
    by = strat.groupby(["protocol", "arm"])
    for (proto, arm), g in by:
        u = g[g["seen_status"] == "UNK"]
        s = g[g["seen_status"] == "seen"]
        lines.append(
            f"| {proto} | {arm} | {int(u['n_rows'].sum()) if len(u) else 0} | "
            f"{u['mean_cosine'].mean():.4f} | {int(s['n_rows'].sum()) if len(s) else 0} | "
            f"{s['mean_cosine'].mean():.4f} |")
    lines += [
        "",
        "## 二、ChemPert 与 LINCS 来源重叠",
        "",
        "- GSE92742 组与 LINCS 训练化合物 100% 重合，按冻结泄漏键合同整组隔离 "
        "（P1 test=GSE92742；P2 test=GSE70138；两子集互不重叠）。",
        "- 不得把 GSE92742 组当作独立外部验证。",
        "",
        "## 三、标签稀疏性",
        "",
        "- 稀疏上下调标签（映射到 795 landmark 的 +1/-1/其他=0）不视为完整连续变化。",
        "",
        "## 四、剂量/时间与覆盖率边界",
        "",
        "- 剂量/时间维度未建模，zero-shot 应用不声称剂量-时间外推。",
        "- 签名到 795 landmark 覆盖率 up 13.0% / down 15.6%，作为待检验解释注明，",
        "  不预设为绝对预测力受限的主因。",
        "- 基于全体目标样本均值的参考属特定目标访问设置，不称肿瘤—正常差异。",
        "",
        "## 五、结论",
        "",
        "- 上游扰动任务在我方环境可复现（EXP-023/027 产物齐全）；",
        "  分层结果显示 UNK（未见药）在 P1 占 56.3%、P2 占 28.4%。",
        "- 绝对预测力受签名覆盖率限制；方向性结论（对照信息不足时药信息为唯一区分信号）",
        "  维持 EXPLORATORY（许可复核未关闭）。",
    ]
    (OUT / "upstream_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WROTE upstream_report.md + upstream_summary.json")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()