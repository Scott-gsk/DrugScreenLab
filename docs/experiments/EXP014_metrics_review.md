# EXP-014 metrics v2 独立审查

审查对象为 `src/drug_screen/transfer/metrics_v2.py`，输入被定义为单一算法的预测表；`model_id` 为生物 PDO，不是算法名称。

审查测试覆盖三个不依赖实现内部结构的算例：同一患者增加 PDO 后先作 PDO 平均再进行患者宏平均；每药相关性先把同一患者的多个 PDO 值平均；固定随机种子的患者 bootstrap 保留重复抽中的患者块。还检验预测分数并列的 NDCG 行顺序不变性（已有主测试）。

统计结论：NDCG 使用 PDO 内真实标签的平均秩百分位作为线性 gain，预测并列按并列块覆盖名次的平均折扣计算，满足行顺序不影响结果。NDCG@5 仅由至少 10 药的 PDO 进入，NDCG@3 仅由 5--9 药的 PDO 进入；二者独立报告，`primary_ndcg_difference` 指向 NDCG@5，短面板不会回填为主终点。每药 Spearman 在患者×药层面先均值多个 PDO，避免 PDO 数量改变患者权重。

已修复的报告缺口：PDO 与每药行现在分别给出 `candidate_spearman_undefined_reason` 和 `baseline_spearman_undefined_reason`，取值为 `fewer_than_three_pairs`、`constant_observed` 或 `constant_predicted`。保留了原有 PDO `undefined_reason` 字段以兼容已有消费者。审查测试覆盖恒定预测的候选原因，并要求 `resamples` 与 NDCG `k` 均为有效整数。

运行时间限制：默认 2,000 次 bootstrap 对每次抽样逐药重新计算秩相关，运行量约为 bootstrap 次数 × 药物数 × 患者数，并包含 Python 循环和 SciPy `rankdata` 调用。PDO 数、药物数或患者数较大时应先做基准测试；不要因为运行变慢而减少默认重采样次数或改为按 PDO 抽样。
