# 项目文档导航（docs/）

> 本文件是 docs 目录的唯一导航入口。当前有效方案、最近报告、数据说明与实验记录以本页链接为准。

## 当前有效文档

| 类型 | 文档 | 说明 |
|---|---|---|
| 乳腺癌数据调研交付报告 | [数据资源、覆盖核查与决策依据](reports/乳腺癌数据资源调研交付报告_20261009.md) | 2026-10-09；真实下载状态、40个表达—药敏交集、扰动与分型缺口；含覆盖清单及核查摘要 |
| 当前转录响应研究方案 | [乳腺癌转录响应迁移与筛药](exec-plans/active/乳腺癌转录响应迁移与筛药实施方案_20260930.md) | 七天执行窗口；转录响应主线；48 GPU小时；Luna/xhigh执行，GPT-6.1/medium监督；HTML审查 |
| EXP-050 进度审查页 | [乳腺癌转录响应迁移与筛药进展](reports/breast_transfer_research_progress.html) | 当前唯一 HTML 进展入口；源状态见 `reports/breast_transfer_research_status.json` |
| 长期规则 | `../AGENTS.md` | 工作原则、权限、协作边界（不含实验数字） |
| 多代理研究闭环 | `AGENT_GOVERNANCE.md` | 2026-09-24：Luna/max 成块执行、Sol/medium 关键决策；有界审查、任务卡恢复、子集到正式实验及消融；暂不启用写作和模拟审稿 |
| 算法研究重定向交接 | `research/algorithm_research_redirection_20260925.md` | 2026-09-25：算法方法与可证伪增益为首要目标；旧 EXP 作为证据工作包；保留双路线独立评价和必做融合；预算仍为单 run 20h、整包 48 GPUh |
| 项目状态 | `../PROJECT_STATE.yaml` | 当前状态、实验索引、有效结论摘要 |
| 主研究方案 | `研究实施方案.md` | 研究问题、路线、长期目标 |
| 当前双路线执行方案 | `exec-plans/active/双路线算法研究与验证方案_20260922.md` | 2026-09-23 修订：单 run 20 小时、实验包 48 GPU 小时；融合必做；Luna max 执行、Sol medium 关键监督；第 14 节执行纠偏，第 15 节 DepMap/Nature 依赖监督，第 16 节细胞系到类器官的通路迁移与解释验证（新增专项预登记待完成、尚未启动） |
| 设计稿 | `design/基于统一扰动机制、转录扰动预测与疾病表达谱逆转的类器官药物筛选研究方案.md` | 长期科学设计（与主方案互补） |
| 前轮执行计划（历史） | `exec-plans/completed/复核审计与文档治理执行方案_20260921.md` | 已运行并归档；完成度须结合 2026-09-22 逐项审计表判断 |
| 第三波运行报告（历史） | `reports/program_closeout_20260922.md` | EXP-031~035 已运行，分项复核中；报告中旧结论以日期更正和审计表为准 |
| 历史周报 | `reports/weekly_report_20260920.md` | 2026-09-13~20 的运行记录；其中旧主指标及准入规模不代表当前协议 |
| 当前审计修复表 | `reports/audit_repair_wave3_20260922.md` | 41 项逐条状态；重算与重训范围依据 |
| 审计证据表 | `reports/audit_evidence_20260921.md` | 工作包 A 证据清单与更正表 |
| EXP-035 专题报告 | `reports/domain_dependency_EXP035_20260922.md` | 域差异与依赖能力（F1~F3，含 F2 parquet 行序修复归档） |
| 数据合同（B 包） | `reports/data_contracts_B_20260921_DRAFT.md` | 八资源卷入身份键/角色/准入 |
| 环境说明 | `ENVIRONMENT.md` | WSL2 + conda `drugscreening-gpu` + RTX A5000 24GB |
| 质量/复现/治理 | `QUALITY.md`、`REPRODUCIBILITY.md`、`AGENT_GOVERNANCE.md` | 长期规范（与 AGENTS.md 的冲突处理见执行方案 G2） |

## 第三波实验记录（experiments/records/）

当前新增乳腺癌转录响应迁移实验：`../experiments/records/EXP-050.md`；状态与证据入口见上方 EXP-050 进度页。

当前结论：第三波已有运行，部分排序观察可追溯，但正式重算和受影响重训尚未验收。新方案的路线一以患者内全药物排序相对同监督条件基线的增量为主要证据；路线二分别验证连续表达扰动预测与独立实测逆转排序。两条路线已有新数据合同和可复算基线，尚无完成审查的模型结果。Kryeziu 当前准入回执为 51 位患者、60 个类器官样本；历史 52/91/947 口径不得当作当前准入。

新方案记录：`../experiments/records/EXP-036.md`（直接药敏与来源权重）、`../experiments/records/EXP-037.md`（未见细胞连续扰动）、`../experiments/records/EXP-038.md`（结直肠癌疾病签名与实测逆转排序）、`../experiments/records/EXP-039.md`（双向基因排序训练目标）、`../experiments/records/EXP-040.md`（冻结逆转排序与 PDO 药敏表型关联）、`../experiments/records/EXP-041.md`（PharmaFormer 小型适配器 A3；共享 EXP-036 50% 四臂运行）。直接药敏划分为 `../data/splits/dual_route_splits_20260922_v1.json`；LINCS 结构可解析的正式划分为 `../data/splits/lincs_cell_splits_20260922_v2.json`，v1 因纳入伪结构已被替代。EXP-036 外折 0、1 已独立复核；25%、50%、100% 学习曲线数据准入通过，25%/100% 模型待启动；EXP-037 外折 0、1 已复核、外折 2 训练中；EXP-038 开发签名与实测面板已准入，逆转排序待模型预测；EXP-039 内层面板准入通过，正式外折 0 已启动；EXP-040 PDO 药物结构与患者标签覆盖准入通过，等待上游预测复核；EXP-041 A3 协议已按总方案补入实验台账，等待 EXP-036 四臂五折与路线一校正分析。

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
| `exec-plans/completed/pathway_transfer_pdo_20260918.md` | 历史 / 已被 20260921 方案替代 |
| `.trae/documents/第二波研究方案_双路线对比与可解释融合_EXP026-030.md` | 第二波方案（EXP-026~030 已完成，结论保留在注册表） |

## 报告与实验记录

- 专题报告：`reports/`（文件名含 EXP 编号与日期；已定稿报告，DRAFT 中间稿已于 2026-09-22 清理，可从 git 历史恢复）
- 汇报 PPT：`reports/presentations/`
- 实验记录：`../experiments/records/EXP-###.md`
- 实验登记：`../experiments/registry.yaml`
- 数据登记：`../data/registry/datasets.json`、`datasets_transfer.json`
- 原版周报归档：`reports/history/weekly_report_20260920_original.md`


