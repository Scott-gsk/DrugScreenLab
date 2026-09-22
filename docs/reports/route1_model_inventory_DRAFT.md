# 路线一（药敏建模）模型能力盘点 DRAFT

- 状态：**DRAFT**（摸底文档；**未改训练代码**；未开训 / 未登记新 EXP）
- 生成时间：2026-09-18 14:10 CST（Asia/Shanghai）
- 本机根路径：`D:\Code\DrugScreenLab`
- 依据：`docs/研究实施方案.md` **第四章、第五章**；`PROJECT_STATE.yaml`；`experiments/registry.yaml`；`docs/exec-plans/active/pathway_transfer_pdo_20260918.md`；`docs/research/EXP-020_improve_contract_draft_20260917.md`
- 原则对齐：当前阶段 = **等数据管家冻结合同**；不以前序 EXP 结论选模；原始数据只读

---

## 1. 摘要（结论先行）

1. **`src/drug_screen/modeling/` 几乎为空**：仅占位 `__init__.py`，注明旧 runner 已移除、新范式应在此以薄适配器落地——**路线一正式模型树尚未按实施方案阶梯重建**。
2. **现存可运行药敏相关实现主要在 `src/drug_screen/transfer/`**（EXP-013 以降 PDO/冷细胞迁移原语）：B0 药均值、B1 每药 Ridge、`RidgePairModel`、`DualEncoderPairModel`、残差适配与 PDO adaptation 管线。
3. **`src/drug_screen/foundation/` 为空占位**；DiSPA / PharmaFormer / scFoundation **无当前可 import 的 adapter**。PharmaFormer 侧状态为「结果保留、实现/ runner 已移除」（见 `PROJECT_STATE.yaml`）。
4. **能力边界（状态文件核实，不作选型依据）**：冠军仅限 GDSC 同研究 cold-cell 双编码器弱信号（EXP-016）；Zhu PDO 适配（EXP-017）与 GDSC→PRISM（EXP-019）为负；下一编号 EXP-020（IMPROVE 合同草案待批）。
5. **评价侧已可用（评价审计本轮交付）**：`ndcg_at_k` / `evaluate_task_a / evaluate_task_b` / `check_split_leakage` / `paired_resample_ci` / `judge_model_gain` / 表达置换对照——模型侧合同冻结后应直接挂这些入口，不另起指标。
6. **建议**：严格按阶段 1→2→3→4→PDO 阶梯；**冻结合同前不改训练代码**；先补阶段 1 五件套与模块删除对照开关，再谈双编码器交互消融与外部方法接入。

---

## 2. 实施方案第四、五章要求对照

| 方案要求 | 仓库现状 | 证据路径 | 判定 |
|---|---|---|---|
| **阶段1** 每药均值/中位数 | 有 B0（均值；未见独立中位数 API） | `transfer/baselines.py`：`fit_b0_drug_means` / `predict_b0` / `B0Model` | **部分** |
| **阶段1** 仅药物模型（分子指纹、不用表达） | DualEncoder 可退化为药物支路，但**无独立 DrugOnly 一等公民模块**；EXP-016/019 曾训 drug-only 对照（历史产物） | `transfer/models.py`；`experiments/records/EXP-016.md` 等 | **缺口（需正式模块）** |
| **阶段1** 每药 Ridge/ElasticNet（表达或通路） | 有每药 Ridge（HVG 表达）；**未见 ElasticNet / 通路活性版正式入口** | `fit_b1_per_drug_ridge` | **部分** |
| **阶段1** 共享交互（通路×指纹低维交互） | 有 `RidgePairModel`（样本特征×药物指纹 + 截断交互） | `transfer/models.py`：`RidgePairModel` | **部分**（输入是否为通路分数取决于上游合同） |
| **阶段1** 梯度提升树 | **未找到** GBM/XGBoost/LightGBM 药敏入口 | 全库检索 | **无** |
| **阶段1** 基因表达 vs 通路；Hallmark 默认 / Reactome 敏感 | 数据合同侧有 Hallmark/Reactome 资源与 stub | `drug_screen/data/expression_contracts.py`；`scripts/build_transfer_pathways.py` | **数据侧有、建模入口未统一** |
| **阶段2** 双分支小网：拼接 / 乘性 / 通路条件化交互（一次改一种） | 有 `DualEncoderPairModel`，内部固定 `concat(h_s, h_d, h_s*h_d)`；**三种交互未做成可删除对照开关** | `transfer/models.py`：`_DualNet` / `DualEncoderPairModel` | **部分** |
| **阶段2** 回归损失 + 组内排序损失 | 有 Huber + within-PDO ranking | `DualEncoderPairModel` docstring/config | **有** |
| **阶段3** DiSPA | **无 adapter** | modeling/foundation 空 | **无** |
| **阶段3** PharmaFormer | runner/实现移除，产物保留；非当前统一协议入口 | `PROJECT_STATE.yaml` → `pharmaformer_pdo`；EXP-010/012 | **历史保留 / 现缺口** |
| **阶段3** scFoundation 冻结表示 + 统一头 | **无** | 检索无命中 | **无** |
| **阶段4** 多来源共享编码器 + 独立头；PRISM 仅固定协议外评 | 无统一多源训练器；EXP-019 为冻结源域→PRISM 零样本评测（负） | EXP-019 记录；modeling 空 | **评测有先例 / 训练模块无** |
| **第五章** 零样本→输出校准→小适配器→部分微调→全微调 | PDO adaptation / residual adapter 管线存在，**未按五档阶梯做成统一开关对照** | `transfer/pdo_adaptation.py`；`residual_adapter.py`；`transfer_experiment.py` | **部分** |
| **第五章** PDO-only Ridge / 从头小网对照 | B1/RidgePair 可作对照，但未与五档适配绑在同一协议 | baselines + models | **部分** |
| **第五章** 小样本学习曲线（20–100%） | 评价侧协议 TODO；模型侧无调度器 | `docs/reports/eval_protocol_DRAFT.md`（评价审计） | **无（待评价+模型联合）** |
| **每模块删除对照** | 评价侧有表达置换/均值表达对照；建模树无统一 ablation flag | `evaluation/permutation.py` | **评价有钩子 / 建模缺开关** |

---

## 3. 已有模型与代码入口清单

### 3.1 包布局（路线一相关）

| 路径 | 角色 |
|---|---|
| `src/drug_screen/modeling/` | **空占位**（应承接新阶梯薄适配器） |
| `src/drug_screen/foundation/` | **空占位** |
| `src/drug_screen/transfer/` | 现役 PDO/冷细胞迁移与成对模型原语 |
| `src/drug_screen/evaluation/` | 任务 A/B、泄漏、CI、置换、增益判定（本轮已增强） |
| `src/drug_screen/data/` | 表达/通路合同与资源（含 Hallmark 默认约定） |
| `scripts/run_exp013_baseline_loop.py` 等 | 历史 EXP runner（可参考，不作为新协议默认） |

### 3.2 可复用类 / 函数

| 名称 | 文件 | 对应方案位置 | 指标角色 |
|---|---|---|---|
| `fit_b0_drug_means` / `predict_b0` / `B0Model` | `transfer/baselines.py` | 阶段1 药均值；状态文件 `current_baseline` | 排序/回归下限 |
| `fit_b1_per_drug_ridge` / `predict_b1` / `B1Model` | 同上 | 阶段1 每药 Ridge | 表达增量 |
| `RidgePairModel` | `transfer/models.py` | 阶段1 共享交互 / 成对 Ridge | 成对回归+可解释贡献 |
| `DualEncoderPairModel` / `_DualNet` | 同上 | 阶段2 双编码器（交互未拆满） | Huber+组内排序；EXP-016 冠军形态近亲 |
| `ResidualAdapter` | `transfer/residual_adapter.py` | 第五章小适配器族 | 低容量适配 |
| `run_pdo_adaptation` 等 | `transfer/pdo_adaptation.py` | 第五章 PDO 适配实验壳 | 源→PDO |
| `run_transfer_experiment` | `transfer/transfer_experiment.py` | 迁移实验编排 | 加载 DualEncoder 源检查点 |
| `evaluate_task_a` / `evaluate_task_b` / `ndcg_at_k` / `judge_model_gain` / `check_split_leakage` | `evaluation/*` | 方案指标与第八章口径 | **合同冻结后强制挂载** |

### 3.3 状态与实验（核实，不作选型）

| 项 | 值 |
|---|---|
| `current_champion` | `exp016_dual_encoder_internal_cold_cell_only` |
| `current_baseline` | `target_domain_drug_mean_B0` |
| `mvp_result` | `PARTIAL_SOURCE_DOMAIN_SIGNAL_PDO_TRANSFER_NOT_ESTABLISHED` |
| EXP-016 | 同研究 cold-cell 正信号（多种子置换完成；多划分未完） |
| EXP-017 | PDO 适配负（选中策略） |
| EXP-019 | PRISM 跨研究负（Dual NDCG@5 低于 B0 与 drug-only） |
| `next_experiment_id` | EXP-020 |
| EXP-020 草案 | `docs/research/EXP-020_improve_contract_draft_20260917.md`（B0 / DrugOnly / DualEncoder；待批） |

---

## 4. 缺口（按接入优先级）

1. **阶段1 正式五件套落在 `modeling/`**：B0（均值+中位数）、DrugOnly、每药 Ridge/ElasticNet（基因与通路两套）、共享交互、GBM；统一 `fit/predict` 与删除开关。
2. **阶段2 交互消融**：concat / mul / pathway-conditioned **一次只开一种**；与评价置换钩子对齐。
3. **阶段3 薄 adapter**：DiSPA → PharmaFormer → scFoundation；先官方复现、再统一协议；禁止与通路集合替换同时变。
4. **阶段4 多源共享编码+分头**：与单源 champion 对照；PRISM 只作锁死外评。
5. **第五章五档适配 + 学习曲线**：与评价审计的学习曲线调度 TODO 联合验收。
6. **历史 PharmaFormer runner**：若需接入，按「薄适配器」重接，不复活旧选模结论。

---

## 5. 建议接入顺序（严格阶梯）

```text
等 DSL-数据管家 冻结合同与划分
  → 挂 evaluation 骨架做冒烟
  → 阶段1 五件套 + 删除对照（modeling/）
  → 阶段2 双编码器三种交互消融
  → 阶段3 DiSPA → PharmaFormer → scFoundation（验证官方后再适配）
  → 阶段4 多源
  → 第五章 PDO 五档 + 小样本曲线
```

- **不提前引入扰动模块**（路线二职责）。
- **不以前序 EXP-016/017/019 结论选模**；其代码/产物仅作接口与失败模式参考。
- EXP-020 IMPROVE 合同：仅在用户批准且数据合同就绪后登记；模型三件套复用 B0 / DrugOnly / DualEncoder 接口。

---

## 6. 本轮明确不做

- 不修改训练逻辑 / 不提交 git / 不开 GPU 训练 / 不登记 EXP-020
- 不依赖未冻结的数据接口签名
- 不把 box 空仓库或未同步 git 镜像当作本机验收证据

---

## 7. 本机只读复核命令（供协调者复查）

```powershell
# 在 DESKTOP-LEP36QM / machineId 96b95440-6d33-4ea8-a3f3-6284d6770786
Get-ChildItem -LiteralPath 'D:\Code\DrugScreenLab\src\drug_screen\modeling' -Recurse
Get-ChildItem -LiteralPath 'D:\Code\DrugScreenLab\src\drug_screen\transfer' -Filter *.py
Select-String -Path 'D:\Code\DrugScreenLab\src\drug_screen\transfer\*.py' -Pattern 'class '
Get-Content 'D:\Code\DrugScreenLab\PROJECT_STATE.yaml' -Encoding UTF8 | Select-Object -First 40
```

---

## 8. 阻塞与依赖

| 阻塞 | 说明 |
|---|---|
| 数据合同未冻结 | 训练代码禁止动；特征空间/标签方向/划分以管家交付为准 |
| Kryeziu 划分仍为 DRAFT | PDO 五档与学习曲线不能正式开跑 |
| modeling/ 空 | 新阶梯需新建模块，避免继续把协议绑死在 EXP-013 transfer API 上 |

**验收点：** 本文存在且路径正确；对照表可核对到类名；训练代码无 diff。