# DrugScreenLab Program Execution Checkpoint

更新日期：2026-08-23

## 研究轨道

- `EXP-006`：遗传扰动到化学扰动迁移，量化实验已完成；正式结论与局限以
  `experiments/records/EXP-006.md` 为准。
- `EXP-007`：Full Observed Oracle 与 null-source 审计已完成；当前结果为
  `ORACLE_NEAR_NULL`，不得改写为理论链已成立。
- `EXP-008`：药物—靶点—通路残差实验；旧设计稿仅保留为历史，正式契约以
  `experiments/records/EXP-008.md` 为准。
- `EXP-009`：BindingDB soft-target 研究保留在独立恢复分支和 worktree，不并入
  主整理分支，也不与乳腺 PDO 研究混合。
- Organoid/context：CCLE、GEO 和 organoid 工作仅标记为 readiness/preparation，
  不构成完成的转化实验或 PDO 科学结论。

## 程序状态约束

本检查点只整理已有代码、证据和身份，不改变科学结论。大型数据、模型权重、预测矩阵
和缓存不进入 Git；紧凑 JSON 指标必须与对应 EXP-ID、数据版本和 provenance 绑定。

下一研究编号为 `EXP-010`，研究轨道为 `breast_pdo_transfer`。该轨道在独立 worktree
中启动；PharmaFormer、XPert 和 UniPert 通过 adapter 接入，不复制第三方源码。未经单独
批准，只允许数据契约、实验设计和测试骨架，不开始训练。

## 权威入口

- 程序状态：`PROJECT_STATE.yaml`
- 实验注册表：`experiments/registry.yaml`
- 数据注册表：`data/registry/datasets.json`
- 实验记录：`experiments/records/EXP-006.md`、`EXP-007.md`、`EXP-008.md`
- 环境与测试：`docs/ENVIRONMENT.md`

旧的方向 checkpoint 和 XPert extension 汇总报告已被本检查点及各正式实验记录取代；
其删除前版本保存在 `D:\Code\DrugScreenLab-recovery\20260823-before-reorg` 的完整恢复快照中。
