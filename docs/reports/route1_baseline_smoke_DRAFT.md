# Route-1 基线冒烟（药物均值 + Ridge）DRAFT

- 时间：2026-09-18T15:15:56+08:00（Asia/Shanghai）
- 折：外层 fold 0（若真实数据齐）
- 模型：`DrugMeanBaseline` + `PerDrugRidgeBaseline`（`src/drug_screen/modeling/baselines_route1_DRAFT.py`）
- 约束：无 GPU 大训；不拿旧 EXP 结论选模；raw 只读；Kryeziu 仅核实身份子集

## 真实 Kryeziu fold-0

- 状态：**BLOCKED**
- fold meta：train_patients=40 test_patients=11 train_samples=47 test_samples=13
- blockers：`[{"code": "RESPONSE_OR_EXPRESSION_MISSING", "detail": "No processed DSS/response/expression parquet under data/processed/mcrc_pdo_2026 or transfer/mcrc_v3 on this host.", "expected": ["data/processed/mcrc_pdo_2026/**", "data/processed/transfer/mcrc_v3/{samples,pairs,drugs}.parquet"]}]`

### 阻塞路径（下一步）

1. 在用户机 WSL `CopyToBox` 或直接运行：确认 `data/processed/mcrc_pdo_2026` / `transfer/mcrc_v3` 中表达与 DSS/pairs。
2. 填充 `data/registry/kryeziu_structure_confirmed_drugs_DRAFT.json`（跑 `scripts/data/extract_kryeziu_structure_confirmed_drugs_DRAFT.py`）。
3. 在 loader 中实现 sample×drug 长表 join（未测响应不补 0）。
4. 再跑本脚本：`PYTHONPATH=src python scripts/modeling/run_route1_baseline_smoke_DRAFT.py`

## 合成冒烟（代码路径）

真实矩阵不齐时仍跑合成数据，验证均值/Ridge 可 fit/predict。

```json
{
  "mode": "synthetic",
  "n_train": 80,
  "n_test": 20,
  "n_drugs": 8,
  "n_ridge_models": 8,
  "drug_mean": {
    "mae": 1.5811655930208597,
    "spearman": 0.3037965966757255
  },
  "ridge": {
    "mae": 1.1129178092643852,
    "spearman": 0.7383458646616541,
    "alpha": 1.0
  }
}
```

## 总结

- 真实 fold-0：`BLOCKED`
- 合成路径：OK（drug_mean spearman=0.3038, ridge spearman=0.7383）
- 骨架已落地；待 processed 齐后替换合成段为真实 fold-0 指标（仍标 DRAFT）。
