# XPert Foundation Reset

## Final status

The initial checklist is superseded by the completed execution checkpoint in `progress.md`. The Foundation track reached `XPERT_FOUNDATION_READY`; the adapter and downstream integration are complete. No new EXP-ID was created.

## 目标

将 DrugScreenLab 的主 Foundation 从历史 `Context + Chemical -> Delta978` 简化模型切换为官方 `GSanShui/XPert`，先完成 bounded FOUNDATION REPRODUCTION CHECK，再建立保持 XPert 主体不变的 DrugScreenLab adapter。

## 当前阶段

- [complete] 读取项目治理、Master Plan、当前状态与实验登记
- [complete] 核验 XPert 官方代码、数据入口与 MultiDCP fallback
- [complete] 确认本 track 不创建 EXP、不等待审批，保留 EXP-004 不变
- [in_progress] 获取/登记官方 XPert 代码与公开 demo/processed assets
- [pending] 运行 official code path 的 bounded smoke/demo
- [pending] 运行 single-fold cold-cell 与 cold-drug foundation reproduction
- [pending] 接入 DrugScreenLab exact978 → disease reversal → Broad PRISM
- [pending] Foundation verdict：XPERT_FOUNDATION_READY 或 BROKEN

## 约束

- 不修改 XPert architecture、loss、official preprocessing、split 或 evaluation definition。
- 只在 WSL2 `drugscreening-gpu` 环境中运行 Python/测试/训练。
- 不把当前自研 MLP 作为主 backbone；它只保留为 historical/simple baseline。
- 不直接依赖未冻结的大型外部资产；所有资产必须记录 source、revision、version、checksum、command。
- Foundation reproduction 采用 single seed / single fold / one cold-cell + one cold-drug，设置 bounded wall-clock。
- 本 track 是 FOUNDATION / engineering integration；不创建 EXP-ID，不覆盖或重新解释 EXP-004。
- 若 XPert 官方路径在 bounded 时间窗内不可运行，自动切换 MultiDCP fallback；不回退到当前 simple MLP 作为主 foundation。

## 成功标准

- 官方 XPert code path 可训练/推理。
- 官方数据契约、gene order、metric implementation 与 source pipeline 一致。
- cold-cell/cold-drug prediction 非 near-zero，Delta prediction 不 collapse。
- 失败时分类为 environment/data/preprocessing/gene-order/checkpoint/config/metric 问题，不改 architecture。

## Expected Decision Value / Cost

- Decision value：确认是否已有可信高质量 foundation，避免继续重复优化简化 MLP。
- Expected cost：bounded asset/download audit 与 single-fold run；不执行五折、全 seed 或完整论文复现。

## 遇到的错误

| 状态 | 说明 |
|---|---|
| `INTERRUPTED_BEFORE_EXECUTION` | 前一轮只完成只读核验与计划创建，尚未执行 XPert；本轮已获得直接执行授权。 |
