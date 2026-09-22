# 评价协议草案（DRAFT）— Task A/B · 泄漏 · CI · 判断口径

> 状态：DRAFT（与 `docs/研究实施方案.md` 第三、八章对齐；未冻结前不作为正式验收签字依据）  
> 代码入口：`src/drug_screen/evaluation/`  
> 日期：2026-09-18（Asia/Shanghai）

## 0. 落盘说明

| 项 | 结果 |
|---|---|
| 用户机路径 | `D:\Code\DrugScreenLab`（`machineId 96b95440-6d33-4ea8-a3f3-6284d6770786`） |
| 代码入口 | `src/drug_screen/evaluation/`（与既有 `protocol.py` 并存；包 `__init__` 同时导出旧 API 与 Task A/B 骨架） |
| 本草案 | `docs/reports/eval_protocol_DRAFT.md` |


## 1. 任务与指标（第三章）

| 任务 | 问题 | 主指标 | 补充 |
|---|---|---|---|
| **A** 同药跨患者 | 哪些患者对该药更敏感 | 逐药跨患者 Spearman | 同终点 MAE/RMSE（可选） |
| **B** 同患者跨药物 | 优先验证哪些药 | **NDCG@5** | Spearman、Top-5 重合率 |

约定：

- 数值越高表示越敏感；未测药从候选集合剔除，**不计为错误**。
- 同一患者多 PDO：先患者内汇总，再**患者等权**统计。
- 患者内排序限定同一研究、可比较筛选条件；剂量范围不一致时解释为该筛选条件下的优先级。
- 所有方法在同一可评价药物集合上比较。

API 对照：

- `evaluate_task_a` / `summarize_task_a`
- `evaluate_task_b` / `summarize_task_b`
- `ndcg_at_k`、`spearman_rho`、`top_k_overlap`、`mae`、`rmse`

## 2. 划分无泄漏（第八章 §2）

必须通过的检查（协议层）：

1. **患者 / 细胞系 / 药物**划分无违规重叠  
   - 主任务（已知药 × 未见 PDO）：**允许**药物集合重叠；**禁止**患者（及不足时的模型组）与细胞系身份跨 train/test 重叠。  
   - 新药 / 骨架留出次级任务：`allow_drug_overlap=False`。
2. **预处理参数只来自训练折**（本草案仅文档约束；拟合态审计由数据/训练管线另报）。
3. **基因顺序、药物结构、响应方向**全库一致（合同层；评价入口假定已对齐）。
4. **LINCS / ChemPert** 重复记录不得跨集合（`lincs_chempert_duplicate_ids` + 扰动 ID 集合）。
5. 排名指标对候选数量、缺测、并列值处理一致（实现：稳定 mergesort；缺测由调用方过滤）。
6. 每次模型扩展均有删除该模块的对照（实验设计层，不在 metrics 内执行）。

API：`check_split_leakage(...) -> LeakageCheckReport`。`ok=False` 时禁止进入正式模型比较表。

## 3. 配对重采样置信区间（第八章 §3）

- 单位：患者或模型（`unit="patient"|"model"`）。
- 统计量：配对增益 \(d_i = s^{\text{model}}_i - s^{\text{baseline}}_i\) 的均值。
- 重采样：**2,000** 次有放回 bootstrap；报告 **95%** 分位区间（`alpha=0.05`）。
- API：`paired_resample_ci(scores_model, scores_baseline, n_resamples=2000)` → `PairedResampleCI`（含 `crosses_zero`）。

学习曲线（第五章 / 交付）：外层 5 折固定测试对象，训练患者比例 20/40/60/80/100%；区间宽或跨零时**必须**结合学习曲线判断是否为样本量限制（本包提供 CI 标志位，曲线绘图/调度另模块）。

## 4. 置换 / 替代对照（第三、八章）

个体信息贡献：

| 对照 | 含义 | API |
|---|---|---|
| 表达整行置换 | 打乱样本表达行，标签/药物对齐保留 | `ExpressionPermutationContrast` |
| 训练均值表达替代 | 全体换为训练集均值表达向量 | `MeanExpressionSubstituteContrast` |

输出 `PermutationContrastResult`：`baseline_score`、`disrupted_score`、`delta=baseline-disrupted`。

## 5. 结果判断口径（第八章 §3 → `judge_model_gain`）

| 证据模式 | 允许表述（claim_level） |
|---|---|
| 95% 区间**跨零**，或区间过宽（半宽 > \|点估计\|），可结合学习曲线 | **证据不足** |
| 排名改善，但置换/替代 `delta≈0` | **总体候选排序（不足个体化）** |
| Task A 改善且 Task B 未改善 | **敏感性分层（不宣称候选药物排序改善）** |
| 扰动预测改善且药敏排序未改善 | **转录预测能力（不宣称药筛收益）** |
| 区间不跨零且未触发限制条款 | **可报告模型增益**（仍需对照与学习曲线） |

禁止：将 PDO 排序输出直接解释为临床用药结论（第八章文末）。

## 6. 与阶段交付的关系（第八章 §4）

| 阶段 | 本协议最小交付 |
|---|---|
| 数据 | 泄漏检查报告草稿 + 划分清单引用 |
| 简单模型 | Task A/B 基线表 + 初步学习曲线引用 |
| 深度 / PDO / 扰动 / 联合 | 配对 CI + 置换对照 + `ResultJudgment` |

## 7. TODO（未在本骨架实现）

- [ ] 预处理“仅训练集拟合”的自动化审计钩子
- [ ] 学习曲线调度与绘图模块
- [ ] 并列值（ties）在 NDCG 中的显式 tie-aware 变体（当前：排序稳定，相关性用平均秩）
- [ ] 与冻结 `data/splits/` 的一键验收 CLI
- [ ] 将本 DRAFT 升级为 FROZEN 协议版本号

