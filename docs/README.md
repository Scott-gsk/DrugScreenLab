# 项目文档导航（docs/）

> 本文件是 docs 目录的唯一导航入口。当前有效方案、最近报告、数据说明与实验记录以本页链接为准。

## 当前有效文档

| 类型 | 文档 | 说明 |
|---|---|---|
| 长期规则 | `../AGENTS.md` | 工作原则、权限、协作边界（不含实验数字） |
| 项目状态 | `../PROJECT_STATE.yaml` | 当前状态、实验索引、有效结论摘要 |
| 主研究方案 | `研究实施方案.md` | 研究问题、路线、长期目标 |
| 设计稿 | `design/基于统一扰动机制、转录扰动预测与疾病表达谱逆转的类器官药物筛选研究方案.md` | 长期科学设计（与主方案互补） |
| 当前执行计划 | `exec-plans/active/复核审计与文档治理执行方案_20260921.md` | 第三波：复核审计与文档治理（阶段 0~7，2026-09-22 收口） |
| 第三波收口报告 | `reports/program_closeout_20260922.md` | 工作包 A~G 全部完成、EXP-031~035 结论汇总、F2 审计修复 |
| 最新周报 | `reports/weekly_report_20260920.md` | 2026-09-13~20 周报（2026-09-21 复核修订 v2） |
| 审计证据表 | `reports/audit_evidence_20260921.md` | 工作包 A 证据清单与更正表 |
| EXP-035 专题报告 | `reports/domain_dependency_EXP035_20260922.md` | 域差异与依赖能力（F1~F3，含 F2 parquet 行序修复归档） |
| 数据合同（B 包） | `reports/data_contracts_B_20260921_DRAFT.md` | 八资源卷入身份键/角色/准入 |
| 环境说明 | `ENVIRONMENT.md` | WSL2 + conda `drugscreening-gpu` + RTX A5000 24GB |
| 质量/复现/治理 | `QUALITY.md`、`REPRODUCIBILITY.md`、`AGENT_GOVERNANCE.md` | 长期规范（与 AGENTS.md 的冲突处理见执行方案 G2） |

## 第三波实验记录（experiments/records/）

| 记录 | 状态 |
|---|---|
| EXP-031 工程修复与评估改革 | completed（214 测试 + 注册表 PASS） |
| EXP-032 来源泛化四层矩阵（工作包 C） | completed |
| EXP-033 两条路线能力梯度（工作包 D） | completed |
| EXP-034 融合重设计（工作包 E） | completed |
| EXP-035 同疾病域差异与依赖能力（工作包 F） | completed（F2 经 09-22 审计修复重跑） |

关键新资产：统一评估模块 `../src/drug_screen/evaluation/exp031_eval.py`；准入白名单 `../data/processed/identity/kryzei_admission_audit_20260921.json`（51 患者/60 样本；Pt120 参与训练已确认）；逐折结构药 `../data/processed/identity/kryzei_structural_drugs_TRAINONLY_v1.json`；共享划分 `../data/splits/`。

## 历史方案

| 文档 | 状态 |
|---|---|
| `exec-plans/active/pathway_transfer_pdo_20260918.md` | 历史 / 已被 20260921 方案替代 |
| `.trae/documents/第二波研究方案_双路线对比与可解释融合_EXP026-030.md` | 第二波方案（EXP-026~030 已完成，结论保留在注册表） |

## 报告与实验记录

- 专题报告：`reports/`（文件名含 EXP 编号与日期；含 DRAFT 的为未定稿）
- 实验记录：`../experiments/records/EXP-###.md`
- 实验登记：`../experiments/registry.yaml`
- 数据登记：`../data/registry/datasets.json`、`datasets_transfer.json`
- 原版周报归档：`reports/history/weekly_report_20260920_original.md`