# DrugScreenLab 阶段收口报告（2026-09-19）

## 执行口径声明

- 本阶段（EXP-020 ~ EXP-025，通路迁移与 PDO 执行程序，spec
  `.trae/specs/advance-pdo-program-execution/` Task 0–7）由 **TRAE 主会话 Manager +
  general-purpose 子 Agent** 执行与收口，**非 Codex Sol/Terra 模型编排**；
  各执行记录中的署名如实标注。
- 运行环境：WSL2 Conda `drugscreening-gpu`（RTX A5000 24 GB），全程未使用
  Windows Python。
- 数据事实以 `PROJECT_STATE.yaml`、`experiments/registry.yaml`、
  `data/registry/datasets.json` 与 `experiments/records/EXP-0xx.md` 为准；
  本报告数字均取自上述登记与正式报告原文。

## 一、结论（先行）

**EXP-020 ~ EXP-025 六条已登记实验全部完成，本阶段按预登记闸门收口：**

1. **路线一（Kryzei 本地建模）**：B0 每药均值基线（NDCG@5 = 0.9089）极其稳健，
   最佳学习变体增益 CI 全部跨零，增益设计波次按预登记停止——**证据不足，未证明超过 B0**。
2. **官方 GDSC→Kryzei 迁移（EXP-021）**：三真种子 + 患者 2000 次 bootstrap 下
   M1/M3 全部跨零、M2 显著负——**当前协议下迁移未证明增益**，停止追加种子。
3. **路线二**：LINCS（一对一对照结构）**对照主导**；ChemPert（共享基线谱结构）
   出现**药条件增益，六组 Δ CI 全正不跨零**（EXPLORATORY，许可待复核）。
4. **联合阶段（EXP-025）**：简单联合劣于直接臂且 CI 跨零，零样本扰动排序基本
   无信号——**DO_NOT_EXPAND**，不扩展为共享通路多任务网络。
5. 收口验证：**pytest 162 通过 / 0 失败 / 0 跳过**（239.19 s）；
   **数据注册表验证 PASS**。工作树收口同步已完成（PROJECT_STATE、registry、
   本报告），未执行 git commit（用户未要求）。

**所有已登记实验均已完成，GPU 现已空闲；任何新实验需用户批准后自 EXP-026
预登记再启动。**

## 二、收口验证结果（Task 7.1 / 7.2，2026-09-19 实测）

| 验证项 | 命令（WSL2 `drugscreening-gpu`） | 结果 |
|---|---|---|
| 完整测试 | `python -m pytest --capture=no` | **162 passed, 6 warnings, 239.19s**；0 失败、0 跳过（6 条 warning 为 `cold_cell.py` 空 slice RuntimeWarning，既有已知行为，非本次引入） |
| 数据注册表 | `PYTHONPATH=src python -m drug_screen.data.registry --root data` | **PASS** |

## 三、六条 EXP 一行状态表

| EXP | status（registry / records） | 一句话结论 |
|---|---|---|
| EXP-020 | complete_best_variant_ci_crosses_zero_insufficient_evidence | Kryzei 本地建模：最佳变体 dual_mlp_genes_h64_wd02=0.9098 vs B0=0.9089，Δ+0.0010，fold-CI [−0.0021,+0.0038] 跨零 |
| EXP-021 | completed_multiseed_evidence_insufficient | 官方迁移三真种子：M1/M3 相对对照 CI 全部跨零、M2 显著负（≈−0.04），未证明增益，停止追加种子 |
| EXP-022 | complete_ctrl_only_dominant_no_drug_specific_gain_proven | LINCS 药条件消融：ctrl_only 主导（0.672/0.675），药条件头与残差头均无增益，KPGT 亦然 |
| EXP-023 | completed_drug_conditioned_gain_ci_positive_exploratory | ChemPert 正式实验：两协议三药条件臂配对 Δ 95% CI 全正不跨零（+0.039~+0.049），EXPLORATORY（许可待复核） |
| EXP-024 | completed_evidence_insufficient | 路线一增益设计波次：6 配置（2 复核 + V1–V4）全部未过双闸门，按预登记停止，不做循环调参 |
| EXP-025 | complete_negative_joint_do_not_expand | 联合切片：联合−直接 −0.0056 [−0.0136,+0.0021] 跨零，扰动排序臂 NDCG@5 仅 0.4224，不扩展 |

## 四、关键数字汇总

| 项 | 数字 | 判定 |
|---|---|---|
| B0 对照（Kryzei 患者宏平均 NDCG@5，官方口径） | **0.9089** | 基线 |
| 官方迁移 M3−B0（三种子合并，患者级 2000 bootstrap） | **+0.0016 [−0.0126, +0.0158]** | 跨零 → 证据不足 |
| 官方迁移 M2−B0（合并） | −0.0408 [−0.0674, −0.0144] | 显著负 |
| ChemPert 药条件 vs ctrl_only（六组配对 Δ CI） | **+0.039 ~ +0.049，全部为正不跨零** | 药特异增益（EXPLORATORY） |
| pathway_pca_best_reg（EXP-024 V4）患者级 Δ | **+0.0054 [+0.0006, +0.0109]** | 患者级为正；折级 +0.0011 [−0.0057,+0.0079] 跨零 → 双闸门未过 |
| pathway_pca_reg 患者级 Δ（EXP-024 复核） | +0.0060 [+0.0009, +0.0119] | 同上，折级跨零 |
| 联合 − pathway_pca_reg（EXP-025 主终点） | **−0.0056 [−0.0136, +0.0021]** | 跨零 → DO_NOT_EXPAND |
| 零样本扰动排序臂（EXP-025） | NDCG@5 0.4224 / Spearman 0.0195 | 基本无信号 |
| ChemPert 签名覆盖率（up/down → 795 空间） | 13.0% / 15.6% | 绝对预测力主要瓶颈 |

## 五、对照《研究实施方案》章节的逐项交付表

依据 `docs/研究实施方案.md`（八章）与 `docs/exec-plans/active/pathway_transfer_pdo_20260918.md`。

| 方案章节 | 交付项 | 状态 | 证据路径 |
|---|---|---|---|
| §一 目标与总体设计 | 两条路线框架、统一评价体制、不引用旧结论原则 | 已完成 | `docs/研究实施方案.md`；spec `.trae/specs/advance-pdo-program-execution/`；EXP-020~025 全程执行 |
| §二 数据组织 | Kryzei 患者级外 5 折划分/表达清单/患者身份三合同冻结 | 已完成 | `data/splits/kryeziu_patient_outer5fold_FROZEN.json`、`data/splits/kryeziu_expression_manifest_FROZEN.json`、`data/processed/identity/kryzeiu_patient_identity_FROZEN.json`、`docs/reports/split_freeze_report_20260919.md` |
| §二 数据组织 | ChemPert 主键/泄漏键合同 v1 冻结 + v2 增补（训练前登记） | 已完成 | `docs/reports/chempert_contract_freeze_20260919.md`、`data/processed/chempert/chempert_v2_manifest.json`（datasets.json 条目 `chempert_contract_v2_supplement`） |
| §二 数据组织 | CTRPv2—DepMap 来源审计 | 进行中（解析与 join 审计完成，实验合同未定） | `PROJECT_STATE.yaml` extension_status：`preferred_source_ctrp: PARSED_DEP_MAP_JOIN_AUDITED_PENDING_EXPERIMENT_CONTRACT` |
| §三 评价协议 | 任务 A/B 分离口径、患者级 2000 次 bootstrap CI、折级+患者级双闸门、EXPLORATORY 分级 | 已完成 | EXP-020~025 预登记与各报告（如 `docs/reports/official_transfer_multiseed_formal_20260919.md` 口径节）；`docs/reports/route1_gain_design_wave1_20260919.md` 判定口径节 |
| §四 路线一 阶段 1 | 统计基线（B0 每药均值等）与冻结口径复现 | 已完成 | EXP-020（B0=0.9089，官方口径复现断言 <1e-9，见 EXP-024 结果节） |
| §四 路线一 阶段 2 | 小型双分支网络及单变量变体（genes/pathway_pca、wd、残差、集成、校准） | 已完成 | EXP-020、EXP-024；`docs/reports/route1_gain_design_wave1_20260919.md` |
| §四 路线一 阶段 3 | DiSPA / PharmaFormer / scFoundation 统一协议接入 | 未启动（本波次） | —（PharmaFormer 旧结果 EXP-010/012 按方案原则不作为本方案依据） |
| §四 路线一 阶段 4 | GDSC2+CTRPv2 多来源共享编码器 | 未启动 | — |
| §五 PDO 迁移 | 零样本（M2）/输出校准与微调（M1/M3）对患者留出的正式评价 | 已完成（负/证据不足） | EXP-021；`docs/reports/official_transfer_multiseed_formal_20260919.md` |
| §五 PDO 迁移 | 小型适配器/部分微调分层消融、小样本学习曲线（20–100%） | 未启动 | — |
| §六 路线二 阶段 1 | LINCS / ChemPert 实测扰动基准（含泄漏键整组隔离） | 已完成 | EXP-022、EXP-023；`docs/reports/route2_chempert_formal_EXP023_20260919.md` |
| §六 路线二 阶段 2 | 实测扰动与药敏关系（扰动强度/通路变化/逆转） | 进行中（零样本切片已试，无信号） | EXP-025 扰动臂（reversal_score NDCG@5 0.4224） |
| §六 路线二 阶段 3 | TranSiGen / XPert 扰动预测接入 | 未启动（本波次为线性 Ridge 摸底） | — |
| §六 路线二 阶段 4 | 扰动预测转排序（三级逆转评分） | 进行中（样本差异特征口径零样本已试） | EXP-025（样本差异特征，非肿瘤—正常逆转，无正常参照） |
| §七 联合 | 简单联合（低容量第二层，内层 OOF 训练）可行性切片 | 已完成（负，DO_NOT_EXPAND） | EXP-025；`docs/reports/joint_feasibility_EXP025_20260919.md` |
| §七 联合 | 共享通路多任务网络 | 未启动（按预登记闸门不扩展） | EXP-025 判定 |
| §八 验收与工程 | CI 闸门全程执行、产物 sha256 登记、测试与注册表验证 | 已完成 | 本报告第二节；各报告产物节；`experiments/registry.yaml` |
| §八 验收与工程 | GPU 复跑挂载与环境 receipts | 已完成 | `docs/reports/gpu_relaunch_receipt_20260919.md` |

阶段交付对照方案 §八.4：数据阶段与简单模型阶段交付完成；深度模型阶段完成小网络
部分（近期大模型未启动）；PDO 阶段完成零样本/校准/微调评价（负）；扰动阶段完成
实测基准（LINCS+ChemPert）；联合阶段完成简单联合切片（负）。

## 六、诚实限制

1. **EXPLORATORY 口径贯穿关键结论**：EXP-021 迁移评价的 admission 与基因空间
   合同为 DRAFT/未冻结（Task 1 冻结的是 Kryzei 目标域合同）；所有「证据不足」
   与「未证明增益」结论限于当前协议，不外推、不宣称临床价值。
2. **ChemPert 数据许可待复核**：EXP-023 阳性结果的根因限制；许可未关闭前
   结论保持 EXPLORATORY、限研究用途、不重分发。
3. **签名覆盖率 13–16%**：ChemPert up/down 签名到 795 landmark 空间覆盖率低，
   压低所有臂的绝对 cosine（P1 模型臂未超 global_mean 基线）；增益结论仅相对
   ctrl_only 成立，不等于绝对预测力达标。
4. **样本量与分辨力限制**：51–52 患者、外层 5 折（折级闸门对 <0.008 量级差异
   分辨力有限）、每配置 3 种子（标准差仅描述性）；pathway_pca 家族患者级一致
   正向但折级跨零即属此限制的直接表现。
5. **EXP-021 口径遗留**：admission 收据 51 患者 vs runner 实际 52 患者（含
   Pt120，至多 1/52 权重）；known 药面板仅 14/48；源域单 checkpoint
   （EXP-015 seed=20260912）。合同冻结后正式重跑须修正 admission 白名单。
6. **EXP-025 扰动臂为线性 Ridge 最小化设计**：不能据此推断更强 drug-conditioned
   模型下联合的上限；LINCS 覆盖 0/48（覆盖标志零方差）。

## 七、下一波方向建议（均需用户决策，未预登记不启动）

1. **ChemPert 阳性结果后续**：①关闭数据许可复核（升级 EXPLORATORY 的前置
   条件）；②评估「更大目标基因空间」合同增补以突破 13–16% 覆盖率瓶颈（需
   重新冻结、另行登记）；③多种子复跑与未见药分层报告（本次单 seed 23）。
2. **pathway_pca 家族的验证策略**：患者级 CI 为正（+0.0054~+0.0060）但折级
   跨零，可预登记「更多种子下的预测平均部署口径」（事后参考已显示种子集成
   pathway +0.0060 不跨零）与外部验证队列（如 Zhu，须先解决患者映射口径），
   以独立数据而非循环调参来分辨。
3. **联合阶段 DO_NOT_EXPAND 后的替代路径**：上游问题先行——线性设置下 Morgan
   指纹边际贡献为负（EXP-022/023/025 一致），若继续扰动方向需先证明药物条件
   信息可用（如接入非线性 drug-conditioned 模型）再谈联合；否则主线维持
   route1，等待新输入源。
4. **程序性声明**：所有已登记实验（EXP-020~025）已完成，**GPU 现已空闲**；
   下一实验自 EXP-026 起，需用户批准假设与失败条件、预登记后方可启动；
   EXP-021 协议下不再追加种子（预登记失败条件已触发）。

## 八、收口同步记录（2026-09-19）

- `PROJECT_STATE.yaml`：active_experiment 程序状态更新为阶段收口
  （PHASE1_CLOSED，等待下一波决策）；last_completed_experiment 同步为
  EXP-025；notes 增加收口说明一行；next_experiment_id 保持 EXP-026。
- `experiments/registry.yaml`：EXP-024 status 由 `registered_pending_execution`
  同步为 `completed_evidence_insufficient`（与 `experiments/records/EXP-024.md`
  及 Task 5 完成声明一致，收口时发现 registry 滞后，本次对齐）。
- 本报告落盘 `docs/reports/program_closeout_20260919.md`；未执行 git commit
  （用户未要求）。
- 本阶段全部正式报告（2026-09-19）：`gpu_relaunch_receipt_20260919.md`、
  `split_freeze_report_20260919.md`、`official_transfer_multiseed_formal_
  20260919.md`、`chempert_contract_freeze_20260919.md`、`route2_chempert_formal_
  EXP023_20260919.md`、`route1_gain_design_wave1_20260919.md`、
  `joint_feasibility_EXP025_20260919.md`。
