"""Build a compact EXP-010 report from frozen model and independent outputs."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mcrc", type=Path, required=True)
parser.add_argument("--m2-breast", type=Path, required=True)
parser.add_argument("--m3-breast", type=Path, required=True)
parser.add_argument("--inference-audit", type=Path, required=True)
parser.add_argument("--independent", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
mcrc, m2, m3, audit, independent = map(load, (args.mcrc, args.m2_breast, args.m3_breast, args.inference_audit, args.independent))
g287 = independent["gse287429"]
pathways = [{"contrast": x["contrast_id"], "top_pathway": x["reactome"]["rows"][0]["pathway"] if x["reactome"]["rows"] else None,
             "top_fdr": x["reactome"]["rows"][0]["fdr_bh"] if x["reactome"]["rows"] else None} for x in g287["contrasts"]]
report = {"experiment_id": "EXP-010", "status": "COMPLETE_WITH_LIMITATIONS",
          "headline": "PDO fine-tuning improved source-only transfer and the frozen M3 ensemble supported cisplatin-combination rank concordance, but did not beat the simple mCRC drug-median baseline.",
          "mcrc": mcrc,
          "breast_gse303509": {"inference_was_label_blind": not audit["labels_read"], "cisplatin_source_supported": audit["source_vocabulary_supported"],
                               "M2": {k: m2[k] for k in ("status", "spearman", "pairwise_concordant", "pairwise_total", "claim_scope")},
                               "M3_ensemble": {k: m3[k] for k in ("status", "spearman", "pairwise_concordant", "pairwise_total", "claim_scope")},
                               "ensemble_rule": audit["ensemble_rule"]},
          "breast_independent": {"gse287429": {"status": g287["status"], "claim_scope": g287["claim_scope"], "method": g287["method"], "top_pathways": pathways},
                                 "gse276609": independent["gse276609"], "multiregional": independent["multiregional"]},
          "claim_limits": ["GSE303509 supports only directional concordance for a cisplatin-containing combination; it is not single-agent clinical validation.",
                           "M3 did not surpass M0/M1 on the mCRC common-support test set.",
                           "GSE287429 is descriptive perturbation biology, not sensitivity evaluation.",
                           "GSE276609 domain shift is blocked by missing audited domain labels.",
                           "Multiregional public-matrix evaluation is not evaluable because all six attachments remain access-blocked."],
          "deviations": ["The first M2 attempt was aborted before checkpoint because raw AUC direction was wrong; the formal run used 1-AUC and retained an audit record.",
                         "The upstream evaluation CLI could not serialize a numpy int64; the same frozen evaluation function was called through a serialization-safe wrapper."]}
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "EXP-010_compact_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
metrics = mcrc["metrics"]
md = f"""# EXP-010 紧凑汇报草稿

## 结论

PDO 微调相对 source-only PharmaFormer 明显有效，并在冻结的乳腺 PDO 外推中达到预注册支持标准；但它没有超过 mCRC 的简单药物中位数基线，因此不能宣称总体最优。

## mCRC 五折外部验证

共同支持集为 {mcrc['patients']} 位患者、{mcrc['evaluation_rows']} 条记录。M3 的患者宏平均 Spearman 为 {metrics['M3']['spearman']:.3f}，M2 为 {metrics['M2']['spearman']:.3f}，差值 {mcrc['M3_minus_M2']['spearman']:.3f}，患者 bootstrap 95% CI [{mcrc['M3_minus_M2']['patient_bootstrap_95ci'][0]:.3f}, {mcrc['M3_minus_M2']['patient_bootstrap_95ci'][1]:.3f}]。预注册判定为 **{mcrc['verdict']}**。

强基线限制：M0 Spearman={metrics['M0']['spearman']:.3f}、NDCG@5={metrics['M0']['ndcg5']:.3f}、MAE={metrics['M0']['mae']:.3f}；M3 分别为 {metrics['M3']['spearman']:.3f}、{metrics['M3']['ndcg5']:.3f}、{metrics['M3']['mae']:.3f}，未超过 M0。

## 乳腺癌 PDO 冻结外推

Cisplatin 存在于 source 结构与 BPE vocabulary。预测阶段未读取 7 个 TORG 观察值，也未校准、翻转方向或选择 fold。M2：rho={m2['spearman']:.3f}、pairwise={m2['pairwise_concordant']}/21、{m2['status']}；五个 M3 checkpoint 等权 ensemble：rho={m3['spearman']:.3f}、pairwise={m3['pairwise_concordant']}/21、**{m3['status']}**。

该结果仅支持 cisplatin-containing combination 的方向一致性，不是顺铂单药或临床疗效验证。

## 其他乳腺输入

- GSE287429：{g287['status']}；仅描述处理后通路响应。
- GSE276609：{independent['gse276609']['status']}；缺少可信 domain 标签，domain-shift 正式阻断。
- 多区域乳腺 PDO：{independent['multiregional']['status']}；六个真实附件仍受访问阻断。
"""
(args.output_dir / "EXP-010_compact_report.md").write_text(md, encoding="utf-8")
print(json.dumps({"json": str(args.output_dir / 'EXP-010_compact_report.json'), "markdown": str(args.output_dir / 'EXP-010_compact_report.md')}))
