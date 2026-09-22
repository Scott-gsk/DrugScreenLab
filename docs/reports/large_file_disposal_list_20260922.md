# 大文件处置清单（2026-09-22）

> 本清单为决策输入。**未经用户逐项批准，以下任何文件不会被移动或删除。**
> 依据：结构清理方案批次 9；AGENTS.md「删除前确认路径、引用和恢复点」。
> 状态口径：`data/raw/` 与第三方原始资产只读（AGENTS.md），本清单中 raw 项仅作盘点展示，默认不在处置范围。

## 一、盘点结果（按体积排序）

| # | 路径 | 体积 | 产生实验/来源 | 注册表引用 | 建议处置 | 理由 |
|---|---|---|---|---|---|---|
| 1 | `data/interim/lincs/GSE92742/` | 143.05 GB | LINCS Level-3 物化（EXP-002 基线，MCPIRE_PDO 迁移） | datasets.json 登记 `lincs_gse92742_*`（asset_type: interim, derived_expression_storage） | **待批准：项目外归档** | 可从 `data/raw/lincs/`（118 GB，保留）重建；EXP-002 已收口（accepted_negative_result） |
| 2 | `data/artifacts/EXP-012/` | 104.04 GB | EXP-012 PharmaFormer 双队列微调 | registry `artifacts: data/artifacts/EXP-012` | **已执行（2026-09-22 用户批准删除）**：1143 文件/104.04 GB 已删除，registry/PROJECT_STATE/RESULT_LOCATION 已同步注记 | 实验已回填终态 completed_exploratory_artifacts_retained；结论保留于记录，产物可移出 |
| 3 | `data/raw/lincs/` | 118.34 GB | LINCS 原始 GCTX（只读） | datasets.json 登记（raw） | 默认保留 | 原始数据只读；若路线 B 彻底关闭可另行决策 |
| 4 | `data/raw/chembl/` | 28.72 GB | ChEMBL 原始（只读） | datasets.json 登记（raw） | 默认保留 | 原始数据只读 |
| 5 | `data/external/xpert_source/` | 10.39 GB | XPert 基础模型源资产 | PROJECT_STATE：foundation IMPLEMENTATION_REMOVED_RESULTS_RETAINED | **待批准：项目外归档** | 实现已移除、结果保留；复用时可重新获取 |
| 6 | `data/external/exp010/` | 7.39 GB | EXP-010 PharmaFormer 资产 | datasets.json 登记 | **待批准：项目外归档** | EXP-010 已收口（complete_promising_fast_with_limitations） |
| 7 | `artifacts/experiments/` | 7.09 GB | EXP-002（4.9 GB）+ EXP-006（2.1 GB）运行产物 | registry 引用；EXP-007 json 已入 Git | **待批准：项目外归档** | 两者均已收口（EXP-002 accepted_negative_result；EXP-006 quantitative_complete） |
| 8 | `data/artifacts/EXP-010/` | 5.98 GB | EXP-010 运行产物 | registry `artifacts: data/artifacts/EXP-010` | **待批准：项目外归档** | 同 #6 |
| 9 | `data/external/pdo_public_candidates/` | 4.20 GB | PDO 公开候选资源获取 | datasets.json 登记 | **待批准：保留或归档** | 资源可用性待 EGA（EGAD00001015468）等申请结果明确后再定 |
| 10 | `data/artifacts/EXP-011/` | 4.38 GB | EXP-011 运行产物 | registry `artifacts: data/artifacts/EXP-011` | **待批准：项目外归档** | EXP-011 superseded_by EXP-012 |
| 11 | `data/external/xpert_assets/` | 2.85 GB | XPert 资产 | datasets.json 登记 | **待批准：项目外归档** | 同 #5 |
| 12 | `data/external/exp011/` | 1.38 GB | EXP-011 外部资产 | datasets.json 登记 | **待批准：项目外归档** | 同 #10 |

**保留（不建议处置）**：

- `artifacts/` 下各 exp026~035 与 route1/route2 DRAFT 产物目录（合计约 0.7 GB）：被 EXP-020~025/026~035 实验记录引用为证据，体积小，保留在原位。
- `data/artifacts/EXP-013~019`（合计约 0.45 GB）：已收口实验产物，体积小，保留。
- `mvp/`、`data/splits/`、`data/processed/`：数据合同与冻结产物，保留。

## 二、批准后执行口径

1. 批准项移动至 `D:\Code\DrugScreenLab_archive\cleanup_20260922\large_files\`（或用户指定位置），保持相对路径结构，登记 sha256。
2. 涉及注册表登记的资产（#1、#5~#12），移动后在 `data/registry/datasets.json` 对应条目更新路径并注明迁移日期，随后运行注册表验证。
3. 仅删除、不归档的项需用户单独明示。
4. 执行完成更新本清单批复栏。

## 三、批复栏（用户填写）

| # | 批准处置 | 备注 |
|---|---|---|
| 1 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 2 | ☑ 删除（2026-09-22 已执行） | 用户批准；1143 文件/104.04 GB；结论保留于记录 |
| 5 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 6 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 7 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 8 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 9 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 10 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 11 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
| 12 | ☐ 归档 / ☐ 保留 / ☐ 删除 | |
