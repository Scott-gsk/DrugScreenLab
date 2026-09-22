# 路线二：实测扰动对照加载骨架 DRAFT

- 状态：DRAFT；**未训练**；未读入 Level3/4/5 `.gctx` 全矩阵
- 时间：2026-09-18 15:20 CST
- 模块：`src/drug_screen/data/measured_perturbation_loader_DRAFT.py`
- 依据：`docs/reports/route2_perturbation_inventory_DRAFT.md` + 实施方案第六章

## 交付路径

| 路径 | 说明 |
|---|---|
| `src/drug_screen/data/measured_perturbation_loader_DRAFT.py` | 实测对照加载骨架 |
| `docs/reports/route2_measured_control_loader_DRAFT.md` | 本说明 |
| 包导出（可选） | `MeasuredPerturbationControlLoader` / `smoke_measured_control_loader` via `drug_screen.data` |

## 能力（已实现）

1. `describe_assets`：登记 LINCS raw/interim/processed、ChemPert CSV、registry 是否在盘（本机冒烟 11/11 present）。
2. `iter_lincs_inst_info` / `propose_lincs_control_pairs`：按 `(rna_plate, cell_id, pert_time)` 匹配 `ctl_vehicle|ctl_untrt|ctl_vector`；化学 / 遗传分开。
3. `pairs_to_p0_records` + `assert_matched_controls`：无对照不做 Δ 标签。
4. `load_chempert_response_metadata` + `chempert_lincs_overlap_keys`：按 `Data_Accession` 给出泄漏分组线索（正式冻结表仍归数据管家）。
5. `MeasuredPerturbationControlLoader.iter_records`：产出 `PerturbationGeneRecord` **壳**（`values=None`，`values_status=deferred`）；L5 迭代显式 `NotImplementedError`，避免与 L3 混用。

## 冒烟（本机）

```text
PYTHONPATH=src python -c "from drug_screen.data.measured_perturbation_loader_DRAFT import smoke_measured_control_loader; print(smoke_measured_control_loader(r'D:\\Code\\DrugScreenLab', pair_limit=100))"
```

结果摘要：`ok=true`；抽样 100 对对照；`assert_matched_controls=passed`；ChemPert 元数据 500 行可读。

## 验收点

- [x] raw 只读；未写 `data/raw`
- [x] 未开训 / 未调 GPU
- [x] 化/遗分离入口存在
- [x] 无对照记录不进入 p0 匹配断言成功集
- [x] L3 vs L5 不静默混用
- [ ] exact978 / gctx Δ 向量物化（合同冻结后再做）
- [ ] ChemPert∩LINCS 正式防泄漏划分表（数据管家）
- [ ] 与评价侧 `check_split_leakage` 对接干跑

## 明确不做（本轮）

Ridge / XPert / TranSiGen 训练；全量 gctx 扫描；改动路线一 Kryeziu 主路径。