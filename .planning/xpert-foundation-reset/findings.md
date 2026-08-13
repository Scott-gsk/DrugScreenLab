# Findings

## 官方 XPert

- 官方仓库：`https://github.com/GSanShui/XPert`。
- 仓库公开 `processed_data/`、`HG_data/`、`datasets/`、`configs/`、`models/`、`scripts/`、`saved_model/` 与 `evaluation_metrics/` 结构。
- README 提供 `scripts/train_example.sh` demo；L1000 训练入口为 `train_xpert.py --model XPert --config config_l1000 --drug_feat unimol --nfold split_1 --dataset l1000_sdst`。
- README 声明完整资源因文件大小限制位于 Zenodo `10.5281/zenodo.15357711` 与 Figshare `10.6084/m9.figshare.28955141`。
- 官方 h5ad contract：post-treatment 在 `adata.X`，pre-treatment 在 `adata.obsm["X_ctl"]`，metadata 在 `adata.obs`。
- 官方 README 明确支持 cold-dose/time、独立数据 fine-tuning 与 test/infer 模式。

## 论文/外部事实

- XPert 论文描述 dual-branch Transformer、pre-perturbation context、drug features、dose/time、cold-cell/cold-drug evaluation。
- 论文数据可用性指向 L1000 Level-3、Zenodo/Figshare processed resources。
- MultiDCP 官方仓库公开 Zenodo `10.5281/zenodo.5172809` 数据下载路径，并明确其 978 landmark prediction 路径，可作为 bounded fallback，但不在 XPert attempt 尚未失败时并行执行。

## 当前仓库

- Master Plan 的 science source of truth 与用户本次 reset 一致：Phase 1 应优先基于 XPert-style context-conditioned perturbation architecture；当前简化模型不应继续作为主 backbone。
- `PROJECT_STATE.yaml` 保持 EXP-004 为当前已登记研究记录；本 FOUNDATION TRACK 不修改 EXP 编号或 EXP-004 结果。EXP-005 仅保留给首个 Novel Extension。
- 当前 worktree 有未提交的 EXP-004/CRC 相关用户或本会话变更，后续必须保持不覆盖、不回退。

## Execution policy

- Foundation 由 Research Manager 自主执行，状态仅使用 `WORKING` / `BROKEN` / `XPERT_FOUNDATION_READY`。
- 官方 XPert 主体、loss、preprocessing、split 与 metrics 不改；兼容修复只能放在 foundation wrapper/环境层。
- 外部下载和运行资产均登记在 `mvp/foundation/xpert/` compact metadata；大文件保留在 ignored local asset path。
