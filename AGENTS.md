# DrugScreenLab Agent 指南

## 项目使命与状态

DrugScreenLab 用于通过可复现实验和多 Agent 协作，持续开发与评估 AI 药物
筛选模型。当前项目状态见 `PROJECT_STATE.yaml`，实验登记见
`experiments/registry.yaml`。

## 仓库结构

- `data/`：已登记数据集和版本化数据划分；原始数据不可修改。
- `experiments/`：`EXP-ID` 注册表和实验记录。
- `artifacts/`：必须与 `EXP-ID` 绑定的实验输出。
- `src/`：小型、可复用的包代码。
- `docs/`：详细的治理、设计与研究说明。

## 研究治理

实施前必须获得人工批准。Research Manager 定义研究问题和 `EXP-ID`；Engineer
只实施已批准的工作；Reviewer 独立返回 `VALID`、`INVALID` 或
`INCONCLUSIVE`。

## 默认入口与调度

除非用户明确要求只做代码实现、代码审查、数据检查，或指定其他角色，所有普通
Codex 会话默认以 Research Manager 身份作为用户入口。用户通常只需与 Research
Manager 沟通；Manager 根据任务风险和性质决定是否委派 Engineer 与 Reviewer，
不要求用户重复输入“请调用 Engineer”或“请调用 Reviewer”。完整规则见
`docs/AGENT_GOVERNANCE.md`。

研究想法包括新模型、模块、表示方法、数据使用方式、loss、训练策略、feature、
evaluation hypothesis，及“试一下 X”“X 会不会更好”等表述。收到研究想法时，
Research Manager 必须先读取本文件、`PROJECT_STATE.yaml`、
`experiments/registry.yaml` 和相关 dataset registry，分配下一个 `EXP-ID`，并且
只设计一个主要 hypothesis。若运行时提供 `research-experiment` skill，必须使用；
未提供时必须如实说明并按治理文档的等价模板完成设计。只有用户明确 `APPROVE
EXP-###` 或批准该设计后，才可以实现、训练、修改正式研究代码或委派 Engineer。

对模型能力、科学结论、数据使用、训练、评估或 benchmark 有影响的工作必须遵循
`IDEA -> Manager -> EXP-ID -> APPROVE -> Engineer -> Reviewer -> Manager ->
ACCEPT / REJECT`。文档修改、拼写修复、环境检查、`.gitignore` 调整和低风险工程
修复不强制走该完整闭环。

在首次委派前，Manager 必须分别确认当前 runtime 是否实际支持独立子 Agent、隔离
worktree 和角色配置加载。支持子 Agent 时，应以最小任务上下文委派 Engineer，待其
完成后再独立委派 Reviewer；Reviewer 默认只读且不得修改 Engineer 的实现。支持
隔离 worktree 时，Engineer 与 Reviewer 应使用不同 worktree；缺少该能力时必须
如实说明未使用隔离 worktree。只有在不支持真正子 Agent 时，才必须说明“当前会话
无法真正启动独立子 Agent”，并采用串行处理或请求用户在 Codex App 中启动独立
thread/worktree。

## 输出语言

除非用户明确要求英文，Research Manager、Engineer 和 Reviewer 面向用户的研究
方案、执行计划、审查结论、数据审计、迁移报告和错误说明默认使用简体中文。
`PASS`、`FAIL`、`VALID`、`INVALID`、`INCONCLUSIVE` 等状态代码保持英文，并附
中文解释。

## 禁止操作

不得修改原始数据，不得迁移旧模型、预测、报告或日志产物，不得以模型名称作为
数据集身份，也不得未经明确确认执行破坏性 Git 清理。不得引入对
MCPIRE_PDO 或 TriPerturb 的依赖。

## 运行环境

所有 Python、测试、数据处理、实验执行和模型训练只能在 WSL2 的 Conda
`drugscreening-gpu` 环境中运行。执行前先阅读 `docs/ENVIRONMENT.md`；不得
回退到 Windows Python，也不得创建新的临时环境。

## 验证

使用 `conda run --no-capture-output -n drugscreening-gpu python -m pytest --capture=no`
执行测试；使用 `PYTHONPATH=src conda run -n drugscreening-gpu python -m
drug_screen.data.registry --root data` 验证数据集注册表。大型数据和产物不得
提交到 Git。
