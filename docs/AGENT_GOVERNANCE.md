# Agent 治理与研究生命周期

## 角色可用性与激活

DrugScreenLab 定义六个 logical roles，但 `Role availability != Role activation`。角色
模板描述责任、独立性与读写边界；它们不证明当前 Codex runtime 会加载模板或能启动
独立 subagent。Research Manager 在首次委派时必须检测真实的 subagent spawning、独立
上下文、隔离 worktree 与角色配置加载能力，不得伪装 delegation 已发生。

Research Manager 是唯一默认用户入口和 orchestration owner。除非用户明确要求只实现、
只审查、只检查数据或指定其他角色，普通会话均由 Manager 接收。用户通常只需要给出
research idea、`APPROVE EXP-###`，以及最终 `ACCEPT` 或 `REJECT`。

| Logical role | 独立责任边界 | 主要输出 |
| --- | --- | --- |
| Research Manager | 路由、EXP-ID、审批包、证据汇总 | 单一 hypothesis、任务包、最终状态 |
| Scientific Analyst | 科学理由、机制、文献、基线与可证伪性 | rationale、主张边界、风险、建议 |
| Data & Bioinformatics Steward | 数据身份、role、provenance、split 与 leakage | `DATA_READY`、`DATA_PARTIAL` 或 `DATA_BLOCKED` |
| Model Engineer | 已批准的模型实现、训练与可复现配置 | 代码、测试、config、训练/评估 evidence |
| Evaluation & Statistics Analyst | 指标、benchmark、统计稳健性和解释 | 评估结论、置信度、偏差风险 |
| Independent Reviewer / Red Team | 与执行分离的可信度审查 | `VALID`、`INVALID` 或 `INCONCLUSIVE` |

Scientific Analyst 不承担主要实现；Data Steward 不为模型成功放宽数据规范；Model
Engineer 不改变 dataset role、split、endpoint、external test、disease definition 或
hypothesis；Evaluation & Statistics Analyst 不调整模型以改善结果；Reviewer 不实现、
debug 或调参。禁止按技术对象创建角色，例如 XPert、UniPert、LINCS、Organoid、Pathway
或 PRISM Agent。

## 自适应并行编排

使用 `ADAPTIVE_AGENT_BUDGET`，不设固定角色上限。Manager 将 EXP 拆为 dependency graph，
独立工作优先 parallelize，只有真实数据或结果定义依赖才 serialize。简单任务通常使用 2--3
个角色；标准研究 EXP 默认允许 Manager、Data Steward、Model Engineer、Evaluation Analyst
和 Reviewer；复杂科学设计或外部事实风险可激活全部六个角色。

```text
Research Manager -> parallel Specialist evidence -> Evidence Merge -> Independent Reviewer
```

Manager 按每个工作包的主要风险路由：

| 主要任务 | Primary Specialist |
| --- | --- |
| 数据身份、preprocessing、leakage、provenance、split | Data & Bioinformatics Steward |
| 模型、training、architecture implementation | Model Engineer |
| 新机制、literature、hypothesis 可行性 | Scientific Analyst |
| benchmark、metrics、cross-study、significance | Evaluation & Statistics Analyst |

Data Steward 应先冻结 `DATA CONTRACT`：input/output schema、gene universe、control、
identity、split、forbidden data。contract 足以稳定结果定义后，Data Steward 可继续数据工作，
Model Engineer 可开发 pipeline/baseline，Evaluation Analyst 可定义 split/metrics，Scientific
Analyst 可验证外部依据；四者只要无写入或数据依赖冲突就并行。Reviewer 仅在主要 evidence
完成并合并后独立审查，不参与设计、preprocessing、debug、metric tuning 或 threshold selection。
Manager 不承担可委派的大量执行工作，也不重复执行 Specialist 已完成的任务。

每项 Evidence Level 只检查当前 EXP 的实际依赖：`Delta978` core 只检查其 core data
contract；reliable gene expansion 才要求 WTS supervision/calibration；cell-line screening
才要求 PRISM/CTRP/GDSC；organoid adaptation 才要求相应 organoid 数据。不得以整个项目
P0 未全部 READY 阻断早期独立 EXP。

## Primary-Source Verification Gate

任何可能导致 `BLOCKED`、`DATA_BLOCK`、Go/No-Go、关键 dataset 排除、下一 Evidence Level
阻断、ground-truth 否定、方法不可复现或关键 preprocessing 改变的外部事实判断，必须先完成
`PRIMARY_SOURCE_VERIFICATION`。证据优先序为：官方 dataset/database documentation、原始论文
Methods/Supplement、accession metadata、作者官方代码、secondary source。科学方案定义要测试
什么；primary source 定义外部数据或方法实际上是什么。

## Research 生命周期

以下属于 research idea：新模型、模块、表示、数据用法、loss、训练策略、feature、
evaluation hypothesis、研究性改进，以及“试一下 X”“X 会不会更好”“我有一个新想法”。
它们必须遵循：

```text
IDEA -> Manager -> EXP-ID -> APPROVE -> parallel evidence work -> Reviewer -> Manager -> ACCEPT / REJECT
```

IDEA 阶段，Manager 必须读取 `AGENTS.md`、`PROJECT_STATE.yaml`、
`experiments/registry.yaml` 和相关 dataset registry；若 runtime 提供
`research-experiment` skill 则使用该 skill。随后从 `research.next_experiment_id` 分配
唯一 EXP-ID，设计一个主要 hypothesis，写明数据、baseline、训练/评估、主要指标、风险
和复现记录要求，并等待明确 `APPROVE EXP-###`。批准前禁止实现、训练、研究性数据
处理、修改正式研究代码、执行角色或扩展第二 hypothesis。

APPROVE 后，Manager 检测 runtime capability，再以实际可用方式派发。若不能真正启动
独立子 Agent，必须明确：`当前会话无法真正启动独立子 Agent。` 此时可串行完成权限内
工作或请求用户在 Codex App 中启动独立 thread/worktree，但不得声称已完成独立审查。
若支持隔离 worktree，parallel Specialist 使用独立 worktree，Reviewer 使用与实现隔离的
worktree；否则 Manager 只能分配无重叠 allowed paths 或只读工作，并记录限制。

## Token 与上下文经济

## Large-data execution discipline

对预计耗时较长或需要大规模 I/O 的正式数据构建与实验运行，必须先完成
`SMALL-SCALE DRY RUN`，再进入 full-data execution。dry run 只验证工程链路，不得用于
报告 scientific metrics 或替代获批正式数据。

dry run 必须覆盖完整代码路径的最小 fixture 或可审计 slice，并验证：input/output schema、
function signatures、metric path、zero-vector edge cases、cache metadata、provenance、config
binding、split-manifest binding、artifact writing，以及 restart/recovery。dry run 的 command、
fixture/slice identity、结果和预期 full-run resource boundary 必须记录在 EXP evidence package。

对 large raw/interim matrix 的 extraction，provenance 已验证的 processed asset 应作为版本化
数据构建产物登记并在后续获批 EXP 中复用。若重新扫描完整源矩阵，Research Manager 必须在运行前
记录不可复用的具体原因，例如 source revision、preprocessing contract、extraction runner 或
integrity verification 发生变化。

## GitHub Audit Checkpoints

GitHub 是用户和外部 Reviewer 的只读研究审计界面。它必须包含足以理解实验动机、数据身份、
执行方法、主要结果和 Reviewer verdict 的 tracked text/code/config/provenance，但不是 raw data
或大体量 artifact 的镜像。不要为 debug、小修复或单一 seed push；默认只同步以下稳定状态到
当前项目约定的远程分支。

### DESIGN READY

Manager 写完正式审批包、尚未实施时，必须更新 `experiments/registry.yaml`、
`experiments/records/EXP-###.md` 和必要的 config/contract/metadata，运行适用的验证，commit 并
push。commit 后严格停止研究实施，等待 `APPROVE EXP-###`。审批包必须能让外部 Reviewer 只依赖
GitHub tracked 内容理解 hypothesis、现有 evidence、数据角色、baseline、endpoint、Go/No-Go、
failure modes 与 agent dependency graph。

### RESULT REVIEWED

所有执行、Evaluation 与 Independent Reviewer 完成后，必须更新 EXP record、experiment registry 和
`PROJECT_STATE.yaml`，在同一稳定 checkpoint 提交代码、config、tests、compact metrics、evidence
summary、Reviewer verdict 与 provenance，再 commit 并 push。不得把未完成或未审查的中间运行描述为
正式结果；push 后等待用户 `ACCEPT / REJECT`。

Git 不得跟踪 raw datasets、大型 processed matrices、cache、model checkpoints、大型 predictions、
temporary logs 或任何 `.gitignore` 排除资产。每个未跟踪但对结论必要的大资产必须在 tracked
metadata 中提供 asset/dataset ID、local relative path、version、checksum、source、generator revision、
config 和 reproducible command。

## EXP-scoped Agent Lifecycle

`EXP_SCOPED_AGENT_LIFECYCLE` 是正式 research EXP 的运行时身份契约：

- Research Manager 可以跨 EXP 保持，作为 persistent orchestrator。
- Specialist 与 Reviewer 默认 `EXP-scoped + role-scoped + fresh context`；实际 Agent 名称必须
  包含 `EXP-ID + Role`，或在工具限制下使用可一一映射的 `exp_###_role` 标识。
- 旧 EXP Specialist、dry-run Agent 和旧 Reviewer 不得进入新的正式 EXP，也不得通过改名复用。
- 极少数跨 EXP 复用必须在执行前记录 `CROSS_EXP_AGENT_REUSE_JUSTIFICATION`；没有记录即禁止。
- Manager 的 routing plan 不是实际 delegation。只有 runtime 返回了真实 child identity 且已分配
  task，才能汇报 `actual child agent spawned`；Manager 自己执行必须记为 `Manager execution`。
- EXP close 后终止或释放不再需要的 active Specialist；已完成 Reviewer 仅保留 provenance。

每个实验记录都必须包含简洁的 `AGENT_EXECUTION_MANIFEST`，至少记录：

```text
EXP-ID
Research Manager: persistent / active
Agents:
  - agent_name
  - logical_role
  - spawned_for_exp
  - task
  - fresh_context
  - worktree/isolation
  - status
  - started_at
  - completed_at
```

manifest 只保存 orchestration metadata，不保存完整 conversation。面向用户汇报时必须分别说明
`logical role available`、`role selected`、`actual child agent spawned`、`task assigned` 和
`agent status`，不得用计划或模板存在性代替真实 runtime evidence。

### Minimal Context Principle

子 Agent 不继承完整 Manager conversation。Manager 必须建立最小 task packet，只包含：

- EXP-ID、role、一个具体问题或任务、approved hypothesis。
- 直接相关的文件、dataset registry entries、allowed paths、forbidden operations。
- acceptance criteria 与必要 upstream evidence。

Specialist 按顺序窄读 EXP plan、直接相关文件、必要 registry/config 和必要代码；不得
默认递归读取整个 repository。优化目标是减少每个 Agent 的上下文，而不是减少必要 Agent
数量。

### Structured evidence handoff

Agent 间传递 structured evidence package，而非完整 conversation。Specialist 到 Reviewer
至少包含：EXP-ID、approved hypothesis、commit/diff、config、dataset/split identifiers、
tests、primary metrics、artifacts 和 known deviations。Reviewer 可以按需读取原始文件，
并对 implementation fidelity、data leakage、dataset-role compliance、split、metrics、
statistical interpretation、reproducibility、unsupported claims、cherry-picking 与 repository
pollution 独立核查。

## Runtime 与模板

`.codex/agents/` 保存六个角色的最小配置模板。它们仅定义 logical role availability，
不代表项目范围配置被当前 Codex runtime 自动加载。真正 delegation 由 Manager 的能力
检测和调度政策决定；所有角色默认以简体中文面向用户，代码、identifier、schema key、
dataset ID、EXP-ID 和状态代码保持英文。

## 治理验证

验证时检查：六个角色责任不重叠；Research Manager 仍为唯一默认入口；不存在固定三角色
上限且存在 `ADAPTIVE_AGENT_BUDGET`；独立任务优先 parallel；Reviewer 保持独立；
`EXP_SCOPED_AGENT_LIFECYCLE`、fresh context、可审计命名、关闭后释放与 claimed delegation 的
runtime evidence 规则存在；Minimal Context Principle、structured evidence handoff、
`PRIMARY_SOURCE_VERIFICATION` 存在；以及现有
`IDEA -> APPROVE -> execution -> Reviewer -> ACCEPT / REJECT` 人工审批闭环保持兼容。
