# DrugScreenLab 工作规则

## 项目状态

`AGENTS.md` 只定义长期工作规则，不写死当前研究方向、模型、数据集或 EXP 编号。
项目事实以以下文件为准：

- `PROJECT_STATE.yaml`
- `experiments/registry.yaml`
- `data/registry/datasets.json`
- 对应的 `experiments/records/EXP-###.md`

## 开始工作前

1. 先读取状态、实验记录和相关数据登记，不根据文件名猜测结论。
2. 根据状态文件确认当前主线、已归档实验和允许复用的模型；优先复用现有模型、数据管线和测试。
3. 工程整理、修复和测试不创建 EXP。改变数据、模型、训练目标或评测结论的研究工作必须登记 EXP。
4. 开始训练或正式评测前，明确数据合同、划分、主要指标、失败条件，并获得用户批准。

## 数据规则

- 只使用可合法获取并可复现的公开数据；来源、版本、路径和 checksum 必须登记。
- `data/raw/` 与第三方原始资产只读，不覆盖、不重排、不静默修正。
- 训练、验证、测试按药物、患者或生物模型分组隔离；禁止用测试结果选择样本、特征或阈值。
- 大型矩阵、原始数据、checkpoint、预测文件和日志不进入 Git。
- Git 只保留代码、数据契约、紧凑指标和足以复现的 provenance。

## 实现原则

- 先建立可运行基线，再做单一、可消融的改进；不重复造轮子。
- 外部模型通过薄 adapter 接入，项目代码只负责统一输入、输出、训练和评测接口。
- 通用逻辑放在 `src/`，命令入口放在 `scripts/`，测试放在 `tests/`。
- 临时探针、一次性计划、cache 和无决策价值的中间结果不提交。
- 负结果保留简洁结论；实现和大结果文件可归档到恢复分支，不长期占据主线。

## 运行环境

所有 Python、测试、数据处理和训练只能在 WSL2 Conda `drugscreening-gpu` 中运行。
不得使用 Windows Python，不得临时创建新环境。执行前参考 `docs/ENVIRONMENT.md`。

```bash
conda run --no-capture-output -n drugscreening-gpu python -m pytest --capture=no
PYTHONPATH=src conda run -n drugscreening-gpu python -m drug_screen.data.registry --root data
```

修改后先跑相关测试；涉及共享模块、注册表或数据合同的修改再跑完整测试和数据注册表验证。

## Git 与清理

- 使用独立分支或 worktree 隔离研究轨道。
- 不使用 `git clean`、`git reset --hard` 或覆盖式 checkout 清理用户工作。
- 删除前确认路径、引用和恢复点；数据目录不得递归移动或删除。
- 只在测试通过、状态一致且工作树 clean 后推送稳定 checkpoint。
- 不改写 master 和既有远程实验历史。

## 协作与汇报

- 复杂研究采用 `Sol Manager → Terra Executor`：主任务在 Codex 中选择 `gpt-5.6-sol`，负责研究设计、任务拆分、科学判断、证据合并和最终验收。
- 独立的数据处理、代码实现和测试任务可由 Manager 调用子 Agent，并显式选择 `gpt-5.6-terra`；简单任务直接完成，不为使用多 Agent 而拆分。
- `AGENTS.md` 只能规定编排策略，不能自行切换当前主任务模型；若当前任务不是 Sol，必须如实说明，不能宣称已按 Sol Manager 执行。
- 子 Agent 必须获得明确的目标、允许路径、输入、验收标准和禁止操作；并行写入使用独立 worktree 或互不重叠的路径。
- Manager 必须复核 Terra 的代码、测试和科学表述；不得把子 Agent 输出直接当成结论。
- 只记录实际启动的 Agent、模型和结果，不得虚构 Agent、Reviewer 或实验。
- 面向用户默认使用简体中文，先给结论，再给验证结果、限制和下一步。
- 不把准备工作写成实验完成，不把作者结论、模型输出和本项目推断混在一起。
