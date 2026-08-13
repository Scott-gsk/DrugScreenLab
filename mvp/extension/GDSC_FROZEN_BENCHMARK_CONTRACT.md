# GDSC Frozen Cross-Study Benchmark Contract

状态：`PREPARED_SOURCE_ASSET_PENDING`。这是一次 external evaluation，不是训练集、调参集或候选药物发现集。

## 目标

完成一条固定的 `Phase-1 predicted Delta978 → disease reversal → drug ranking → GDSC` 交叉研究评估，并与现有 PRISM 结果分开报告。GDSC sensitivity 只在预测、候选身份和 ranking digest 冻结后读取。

## 冻结对象

- 候选 identity：当前 PRISM/LINCS 四药 cohort 的 `pert_id` 集合；不得根据 GDSC response 值增删药物。
- external context：GDSC/DepMap 可验证的 cell-line/model identity；不能用 PRISM fallback context 伪造 GDSC context。
- external label schema：`study_id`, `context_id`, `pert_id`, `sensitivity_score`。
- 现有 evaluator：`src/drug_screen/evaluation/cross_study.py`；要求 label 与 prediction 的候选集合和 support set 完全一致。

## 身份与响应标准化

1. drug identity 优先使用结构/InChIKey，名称只作为 alias 审计；所有多对一或一对多匹配必须进入 `AMBIGUOUS`，不能静默选择。
2. context identity 使用稳定的 cell-line/model key，并单独记录与 CCLE/DepMap 的 mapping 版本。
3. GDSC 原始 response 以 release 自带的明确 response field 为准；转换为 `sensitivity_score` 前记录单位、方向、transform 和原始字段名，禁止凭名称猜测。
4. 同一 study/context/pert_id 只能生成一条冻结标签；重复测量必须按预注册规则聚合并记录。

## 首个 Gate

`Cell-Line Gate C` 只有在以下条件同时满足时才可进入：

- 官方/作者可追溯的 GDSC release asset 已下载并记录 checksum；
- 四个冻结 `pert_id` 均有无歧义结构 mapping，且不读取 response 才完成 mapping；
- GDSC/DepMap context mapping 可审计；
- label/prediction support 完全相等；
- evaluator 返回 `COMPLETE`，`labels_used_for_tuning=false`。

若官方源仍不可获取，状态为 `NOT_RUN_EXTERNAL_ASSET_UNAVAILABLE`，不能写成 GDSC 负结果，也不能用二手/response-dependent rescue asset 替代。

## 官方入口与当前 readiness

- Sanger/DepMap drug-sensitivity documentation：<https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/>
- Cell Model Passports downloads：<https://cellmodelpassports.sanger.ac.uk/downloads>
- 官方变更记录：<https://cellmodelpassports.sanger.ac.uk/changes>
- 当前数据版本与文件字段应从 Cell Model Passports release 页面及官方 raw/fitted description 冻结；旧 CancerRxGene downloads URL 当前返回 HTTP 410，因此不把历史 URL 本身当作可复现资产。
- 当前 checkout 未登记可审计的 GDSC response archive；下一步应先锁定 release、下载入口、schema 和 checksum，再运行 frozen evaluator。
