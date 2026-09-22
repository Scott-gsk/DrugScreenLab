# EXP-027 双路线统一对比报告（路线 A 直接药敏 vs 路线 B 扰动中介）

- 生成：scripts/evaluation/exp027_route_comparison.py（EXP-027）
- 终点：非结构药子集 NDCG@5（主）/ 逐药跨患者 Spearman（个性化）/ 全面板 NDCG@5（参照）
- 路线 A 全部复用已落盘预测；路线 B 零样本（不使用 Kryzei 药敏标签）

## 一、双路线总表（多种子 mean±SD；患者宏平均）

| 路线 | 臂 | 非结构 NDCG@5 | 逐药跨患者 Spearman | 全面板参照 | 种子数 |
|---|---|---|---|---|---|
| A | M1_source_scaled | 0.8604 ±0.0024 | 0.1068 ±0.0158 | 0.8916 ±0.0039 | 3 |
| A | A6_target_residual | 0.8590 ±0.0098 | 0.1221 ±0.0547 | 0.8903 ±0.0031 | 3 |
| A | M3 | 0.8563 ±0.0109 | 0.1611 ±0.0236 | 0.8940 ±0.0053 | 3 |
| A | A5_target_drugstd | 0.8539 ±0.0100 | 0.1452 ±0.0094 | 0.8882 ±0.0056 | 3 |
| A | pathway_pca_best_reg | 0.8487 ±0.0049 | 0.0239 ±0.0309 | 0.8973 ±0.0019 | 3 |
| A | A7_ref_mlp | 0.8424 ±0.0007 | -0.0071 ±0.0252 | 0.8925 ±0.0024 | 3 |
| A | A1_ablation_B0 | 0.8421 | -0.2445 | 0.8936 | 1 |
| A | M1 | 0.8414 ±0.0138 | 0.0944 ±0.0409 | 0.8863 ±0.0117 | 3 |
| A | B0 | 0.8413 ±0.0004 | -0.2595 ±0.0408 | 0.8924 ±0.0007 | 3 |
| A | M2 | 0.8008 ±0.0000 | 0.1055 ±0.0000 | 0.8517 ±0.0000 | 3 |
| B | B1_lincs_reversal | 0.7796 | -0.0772 | 0.5890 | 1 |
| B | B3_chempert_pathway_reversal | 0.7575 ±0.0127 | 0.0531 ±0.0165 | 0.5972 ±0.0277 | 3 |
| B | B2_chempert_reversal | 0.7473 ±0.0296 | 0.0939 ±0.0174 | 0.5935 ±0.0289 | 3 |

## 二、预登记配对比较（患者级 2000 次 bootstrap，非结构 NDCG@5）

| 比较 | 点估计 | 95% CI | n |
|---|---|---|---|
| A2_M1_source_scaled_vs_A1_B0 | +0.0192 | [-0.0002, +0.0387] | 52 |
| A4_pathway_pca_vs_A1_B0 | +0.0075 | [+0.0013, +0.0140] | 51 |
| A5_target_drugstd_vs_A7_ref | +0.0115 | [-0.0090, +0.0333] | 51 |
| A6_target_residual_vs_A7_ref | +0.0166 | [-0.0029, +0.0362] | 51 |
| B1_lincs_vs_A1_B0 | -0.0617 | [-0.1002, -0.0259] | 51 |
| B2_chempert_vs_A1_B0 | -0.0939 | [-0.1264, -0.0616] | 51 |
| B3_chempert_pathway_vs_A1_B0 | -0.0838 | [-0.1192, -0.0487] | 51 |
| CROSS_ROUTE_A2_vs_B2 | +0.1158 | [+0.0850, +0.1450] | 51 |

## 三、逐药跨患者 Spearman 宏平均 vs 0（患者级 bootstrap）

| 臂 | 宏平均 | 95% CI |
|---|---|---|
| B1_lincs_reversal | -0.0772 | [-0.2112, +0.0787] |
| A2_M1_source_scaled | +0.1153 | [-0.0225, +0.2389] |
| B2_chempert_reversal | +0.1149 | [-0.0234, +0.2419] |
| B3_chempert_pathway_reversal | +0.0422 | [-0.1138, +0.1821] |

## 四、按药分层（14 已知药；Spearman 为跨患者个性化度量）

| 药物 | 结构性 | 观测跨患者 SD | A2 Spearman | B1 | B2 | B3 |
|---|---|---|---|---|---|---|
| ZLHFILGSQDJULK-UHFFFAOYSA-N | nonstructural | 7.81 | 0.3060 | -0.0790 | 0.2250 | 0.1716 |
| BCFGMOOMADDAQU-UHFFFAOYSA-N | nonstructural | 6.52 | 0.3713 | -0.2660 | 0.2067 | 0.1658 |
| AAKJLRGGTJKAMG-UHFFFAOYSA-N | nonstructural | 5.57 | -0.0947 | -0.0161 | 0.1179 | -0.1556 |
| AHJRHEGDXFFMBM-UHFFFAOYSA-N | nonstructural | 5.33 | -0.1836 | 0.0596 | -0.1278 | -0.1121 |
| GHASVSINZRGABV-UHFFFAOYSA-N | nonstructural | 5.20 | 0.2458 | -0.1949 | 0.1564 | 0.0364 |
| LQBVNQSMGBZMKD-UHFFFAOYSA-N | nonstructural | 4.86 | 0.3190 | -0.0500 | 0.2862 | 0.0946 |
| ZBNZXTGUTAYRHI-UHFFFAOYSA-N | nonstructural | 4.78 | 0.0537 | -0.2554 | 0.2300 | 0.1431 |
| KBQCEQAXHPIRTF-UHFFFAOYSA-N | structural | 7.45 | 0.2144 | -0.4238 | 0.5531 | 0.5815 |
| LIRYPHYGHXZJBZ-UHFFFAOYSA-N | structural | 5.54 | -0.1364 | 0.2050 | -0.1016 | -0.2567 |
| PKCDDUHJAFVJJB-UHFFFAOYSA-N | structural | 5.23 | 0.0920 | 0.4188 | -0.2800 | -0.4762 |
| NDAZATDQFDPQBD-UHFFFAOYSA-N | structural | 4.89 | -0.1053 | -0.1378 | -0.0773 | -0.0565 |
| BKWJAKQVGHWELA-UHFFFAOYSA-N | structural | 3.87 | 0.4947 | -0.2231 | 0.4115 | 0.2646 |
| FDLYAMZZIXQODN-UHFFFAOYSA-N | structural | 2.62 | -0.0843 | -0.1063 | -0.1390 | 0.1490 |
| XXYGTCZJJLTAGH-UHFFFAOYSA-N | structural | 1.84 | 0.1218 | -0.0121 | 0.1474 | 0.0410 |

## 五、结果判读（按 EXP-027 预登记框架）

1. **跨路线主比较明确**：路线 A 代表臂 M1_source_scaled 显著优于路线 B 主臂
   B2_chempert_reversal（+0.1158，95% CI [+0.0850, +0.1450]），且路线 B 三个臂
   相对 B0 药物均值基线全部显著为负——零样本逆转分数不含药物主效应校准，
   绝对排序低于药物均值基线属预期内。
2. **A2 vs B0 复核**：点估计 +0.0192 与 EXP-026 重打分（+0.0186）一致；本次
   95% CI 下界 -0.0002 边缘触零，严格口径下降级为边缘正向，维持
   EXPLORATORY-POSITIVE 不升级。
3. **个性化信号对齐（本实验最有信息量的发现）**：逐药跨患者 Spearman 宏平均
   上，零样本 B2（+0.1149）与监督 A2（+0.1153）点估计几乎相同（CI 均跨零）。
   路线 B 不使用任何 Kryzei 药敏标签，却携带与监督路线同量级的患者间个性化
   信号；按药分层显示非结构药上 B2 有 6/7 为正（A2 为 5/7），存在 B2 优于
   A2 的具体药物（如 KBQCEQAXHPIRTF、ZBNZXTGUTAYRHI）。
4. **失败条件未触发**：路线 B 全部臂 vs B0 的 CI 均不跨零（显著为负而非
   无信号），预登记失败条件不成立。
5. **结论**：主线维持直接药敏（A2 为强基线），路线 B 定位为互补特征与解释
   模块而非独立排序器；EXP-028 融合以 A2 为强基线、路线 B 特征为第二路输入，
   按药分层支持按药自适应加权方向。响应口径候选（A5/A6）CI 跨零，与 EXP-026
   一致，不纳入后续合同。
6. ChemPert 许可复核未完成，路线 B 相关结论 EXPLORATORY 标注随行。
