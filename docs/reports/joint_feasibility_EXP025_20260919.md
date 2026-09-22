# EXP-025 联合可行性切片报告（2026-09-19）

## 一、结论（先行）

**判定：DO_NOT_EXPAND——联合阶段暂不扩展为共享通路多任务网络。**

预登记失败条件触发：简单联合模型在患者宏平均 NDCG@5 上**劣于**直接最强臂
（pathway_pca_reg），且主终点增益 CI 跨零。

| 关键数字 | 数值 |
|---|---|
| 主终点 ΔNDCG@5（联合 − pathway_pca_reg，患者级 2000 次 bootstrap 95% CI） | **−0.0056 [−0.0136, +0.0021]，跨零** |
| 三方患者宏平均 NDCG@5（Task-B 口径，51 患者） | B0 0.9087 / pathway_pca_reg **0.9147** / 扰动排序 0.4224 / 简单联合 0.9091 |
| 扰动臂（零样本 reversal_score 排序） | NDCG@5 0.4224、患者内 Spearman 0.0195——基本无信号 |

三条具体结论：

1. **零样本扰动排序在 Kryzei 上基本无效**。LINCS 线性药物条件模型派生的
   reversal_score 单独排序的 NDCG@5 仅 0.4224、Spearman≈0.02，远低于直接臂
   （≈0.91）。
2. **简单联合不但无增益，反而损害直接信号**。联合模型（第二层 Ridge，仅内层
   val OOF 训练）NDCG@5 0.9091，比 pathway_pca_reg 低 0.0056（CI 跨零），
   Spearman（0.7987）甚至低于 B0（0.8076）。扰动特征在低容量第二层中表现为
   噪声。
3. **对照主导的既有结论延续**。LINCS 留出对照上 ctrl_only Ridge
  （cosine 0.6721）优于 ctrl+Morgan（0.6392），Morgan 单独仅 0.0171——
   线性设置下药物条件指纹对 Δ 预测的边际贡献为负，这是扰动臂无信号的上游
   原因（与 EXP-023 消融一致）。

## 二、设计与训练前钉死声明

以下选择全部先于本切片任何训练确定（脚本
`scripts/modeling/exp025_joint_feasibility_DRAFT.py` 顶部常量与 EXP-025 预登记，
无任何基于结果的调整）：

- **LINCS 划分**：12,000 对照对（提议规则与 route2 linear 脚本一致），
  pair-random 80/20，seed=20260918；80% 训练、20% 仅作质量报告，不用于任何
  超参或特征选择。
- **两层 Ridge α 均固定 1.0**，全流程无搜索。
- **Morgan**：radius=2、nBits=1024（与 xpert 注册表 38/38 逐一验证一致）；
  其余 10 药用 RDKit 从 SMILES 现算。
- **top-8 Hallmark 通路**：按 |通路 ∩ landmark ∩ Kryzei 基因| 降序（y-free
  规则）。
- **reversal_score 口径**：样本差异特征（患者基线 − 全 92 样本均值基线，
  landmark z 空间，限于 landmark∩Kryzei 子空间），**非肿瘤—正常疾病逆转**
  （Kryzei 无正常参照，方案 §六.2）。
- **联合层输入（14 维）**：pathway_pca_reg OOF 预测、B0 OOF 预测、
  perturb_magnitude、reversal_score、8 个通路 Δ 均值、LINCS 覆盖标志。
- **种子**：ppr 三种子 {20260919, 20260920, 20260921} 取预测均值；
  bootstrap seed=20260918、2000 次患者单位配对重采样。
- **防泄漏**：第二层仅用各外折内层 val 的 OOF 行训练（8/9 患者）；外层
  test 预测全部来自外层训练拟合的第一层模型；外层 test 与任何训练/选择步骤
  零接触。

## 三、三方比较结果

### 主表（Task-B 官方协议：(患者,药) 去重均值 → 患者内 NDCG@5/Spearman）

| 臂 | 患者宏平均 NDCG@5 | 患者宏平均 Spearman | 折等权 NDCG@5 | stage2 raw 行口径（敏感性） | n 患者 |
|---|---|---|---|---|---|
| B0（官方 carve-fit） | 0.9087 | 0.8076 | 0.9089 | 0.9044 | 51 |
| pathway_pca_reg（3 种子均值） | **0.9147** | **0.8192** | 0.9150 | 0.9126 | 51 |
| 扰动排序（reversal_score） | 0.4224 | 0.0195 | 0.4227 | 0.3892 | 51 |
| 简单联合（第二层 Ridge） | 0.9091 | 0.7987 | 0.9094 | 0.9023 | 51 |

### CI 汇总（2000 次配对 bootstrap，seed=20260918）

| 对比 | 单位 | Δ | 95% CI | 跨零 |
|---|---|---|---|---|
| **联合 − pathway_pca_reg（主终点）** | 患者 | **−0.0056** | **[−0.0136, +0.0021]** | **是** |
| 联合 − B0 | 患者 | +0.0004 | [−0.0049, +0.0058] | 是 |
| 联合 − pathway_pca_reg | 折 | −0.0056 | [−0.0166, +0.0043] | 是 |
| 联合 − B0 | 折 | +0.0005 | [−0.0054, +0.0067] | 是 |

所有口径下联合相对直接臂均无显著差异或劣化，主终点方向为负。

### 逐折明细

| 折 | 内层 val 行数（val 患者数） | 外层 test 行数（test 患者） | B0 官方 NDCG@5 | 联合 val MAE |
|---|---|---|---|---|
| 0 | 458（8） | 646（11） | 0.9023 | 3.897 |
| 1 | 511（9） | 664（10） | 0.9108 | 3.937 |
| 2 | 401（9） | 584（10） | 0.9134 | 3.658 |
| 3 | 433（9） | 586（10） | 0.9146 | 4.004 |
| 4 | 458（9） | 664（10） | 0.9033 | 3.764 |

## 四、验证与一致性检查（全部通过）

1. **landmark 基因序锚定**：复现 route2 linear 冻结脚本 train 划分的
   ctrl mu/sd 与 `linear_ctrl_to_delta.pt` 存档比对，max|Δmu|=7.17e-04、
   max|Δsd|=1.15e-04（<1e-3）——基因序一致。
2. **B0 官方复算**：5 折 carve-fit B0 与冻结 CSV
   （`kryeziu_stage1_baseline_metrics_by_fold_DRAFT.csv`）差异 <1e-8；
   折等权宏平均 0.9089 与官方 B0=0.9089 一致。
3. **直接臂与 EXP-024 对齐**：pathway_pca_reg − B0 患者级差
   0.9147−0.9087=+0.0060，与 EXP-024 波次一患者级 +0.0060 完全一致。
4. **LINCS 留出质量（仅报告，不参与选择）**：null 0.0000 / 全局均值
   0.0895 / ctrl_only 0.6721 / morgan_only 0.0171 / ctrl+Morgan 0.6392
   （n=2,398，mean cosine）。

## 五、限制

1. **扰动臂基础模型是线性 Ridge**：本切片按最小化设计未使用非线性扰动模型。
   ctrl+Morgan 留出 cosine（0.6392）低于 ctrl_only（0.6721），说明该线性设置
   下药物指纹对 Δ 的边际贡献为负；派生特征（magnitude/reversal/通路 Δ）
   因此基本不含药物特异性信息。更强的 drug-conditioned 模型是否改变结论
   未在本切片检验。
2. **LINCS 覆盖为 0/48**：Kryzei 48 药全部不在 LINCS 12,000 训练对的
   pert_id 中，预登记的覆盖标志全 0（零方差，如实保留并报告）。
3. **reversal_score 是样本差异特征**，不是肿瘤—正常疾病逆转；Kryzei 队列
   无正常参照，该口径只反映患者相对队列均值的基线偏移方向。
4. **第二层训练样本小**（每折 8–9 个内层 val 患者），低容量 Ridge 是与
   样本量匹配的选择，但也决定了本切片只能检验「简单联合」的可行性，不能
   推断复杂联合架构的上限。
5. **判定为「暂不扩展」而非永久否定**：预登记闸门针对的是「共享通路
   多任务网络」这一扩展方向是否在当前输入源下值得投入。

## 六、下一步建议

1. 联合阶段按预登记**不扩展**为共享通路多任务网络；共享通路表征投入
   暂停，直至扰动臂出现有信号的新输入源。
2. 主线维持 route1：pathway_pca_reg 相对 B0 患者级 +0.0060 但折级跨零
   （EXP-024 结论），增益波次按预登记停止，不做循环调参。
3. 若扰动方向继续，优先解决上游问题：在线性 drug-conditioned 设置下
   Morgan 指纹无正边际贡献（本切片与 EXP-023 一致），需先证明药物条件
   信息可用，再谈联合。

## 七、产物

- 脚本：`scripts/modeling/exp025_joint_feasibility_DRAFT.py`
- 结果：`artifacts/joint_feasibility_EXP025_DRAFT/joint_feasibility_results.json`
- 联合外层 test 预测：`artifacts/joint_feasibility_EXP025_DRAFT/joint_predictions.parquet`
- 三方患者级评分：`artifacts/joint_feasibility_EXP025_DRAFT/three_way_patient_scores.parquet`
- 零样本扰动特征（92 样本 × 48 药 = 4,416 行）：
  `artifacts/joint_feasibility_EXP025_DRAFT/kryzei_perturb_features.parquet`
- LINCS Ridge 药物条件模型权重：
  `artifacts/joint_feasibility_EXP025_DRAFT/lincs_ridge_drugcond.npz`

执行环境：WSL2 Conda `drugscreening-gpu`（CPU），运行时长 18.7 秒。
