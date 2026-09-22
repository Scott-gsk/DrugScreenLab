# GPU 复跑挂载执行回执（2026-09-19）

- 执行者：DSL-药敏建模 Agent（TRAE general-purpose 子 Agent，非 Sol Manager 编排；按 Manager 下发的 Task 0 + Task 3.1 spec 执行）
- 对应 spec：`.trae/specs/advance-pdo-program-execution/tasks.md` Task 0（SubTask 0.1–0.5）+ Task 3.1
- 环境：WSL2 Conda `drugscreening-gpu`（未使用 Windows Python）

## 结论（先行）

1. **内层 val 核验**：Stage2 训练脚本原实现**无内层 val 划分**（4 条算法条目缺失，test 隔离条目合规）；已做最小修正补齐划分、日志与落盘，并新增 `--seed`/`--out-dir` 支撑多种子。修正后首个挂载 run 的日志已验证按钉死算法输出。
2. **环境预检**：`make env-check` 与 `make gpu-check` 均 PASS（torch 2.5.1 + CUDA 12.1，3 卡可见；仅使用 GPU0/GPU1，GPU2 未触碰）。
3. **已挂载 3 个后台任务**：官方迁移新真种子 seed=20260919（CPU, b2000）+ Stage2 两头部配置各 3 种子（GPU0/GPU1 分卡）。启动验证通过：无 traceback、fold/迭代推进、GPU0/1 有显存占用。
4. **seed2 误标已修正**：误标目录内放置 `_MISLABEL_NOTE.md`；docs/reports 经检索无引用该目录的报告，无需另行加注。

## 一、内层 val 算法一致性核验（SubTask 0.1）

核验对象：`scripts/modeling/train_stage2_dual_mlp_variants_DRAFT.py`（经 artifacts results.json 的 tag/字段结构确认，即产出 `dual_mlp_genes_h64_wd02` 与 `dual_mlp_pathway_pca_reg` 的脚本）。钉死算法出处：`docs/reports/split_freeze_checklist_DRAFT.md` §"内层 val 算法钉死说明"（2026-09-18 17:23 CST）。

逐条比对结论（修正前）：

| # | 钉死算法条目 | 修正前脚本现状 | 结论 |
|---|---|---|---|
| 1 | 患者级 4:1 train_inner:val | 无任何内层 val 划分，外层 train 全量直接训练 | 不一致 |
| 2 | 仅从外层 `train_patient_ids` 划分 | 无实现（不适用） | 补实现时遵循 |
| 3 | 先字典序排序，再 `numpy.random.RandomState(seed).permutation`，取末 ⌈n/5⌉ 名患者为 val（n<5 至少 1 名；n=1 跳过并标注） | 无实现 | 补实现时遵循 |
| 4 | seed 优先用划分 JSON `random_seed`，缺失固定 20260918 | 无实现；实测 `kryeziu_patient_outer5fold_DRAFT.json` 的 `random_seed` 为 `null` → 应固定用 **20260918** | 补实现时遵循 |
| 5 | 外层 test 禁止参与选模/早停/特征筛选/阈值 | 特征筛选（top-2000 方差基因）仅用 train 患者矩阵；固定 epochs 无早停；无阈值选择；test 仅用于最终评测 | **一致** |

最小修正内容（仅 `scripts/modeling/train_stage2_dual_mlp_variants_DRAFT.py`，未改 `src/`）：

- 新增 `inner_val_patients()`：按钉死算法实现患者级 4:1 内层 val（字典序排序 → `RandomState(seed).permutation` → 末 `max(1, ⌈n/5⌉)` 名为 val；n≤1 标注 `inner_val_skipped`）。
- 每折日志输出：`INNER_VAL fold=<id> random_seed=<seed> n_train_inner_patients=<n> n_val_patients=<n> val_patient_ids=[...]`；同步写入 results.json（顶层 + 每折 metrics）。
- 新增 `--seed`（训练种子，设置 `np.random.seed` + `torch.manual_seed`；缺省 None 保持旧未播种行为）与 `--out-dir`（支持每种子独立输出目录）。
- **未改变**训练/评测行为（无早停引入、特征筛选口径不变）。

边界说明（如实记录）：钉死算法禁止 test 参与特征筛选（已满足）；当前 top-2000 特征筛选在外层 train 全体（含内层 val 患者）上进行，钉死文本未禁止此口径，本轮保持不变；内层 val 本轮仅用于构成记录与未来选模，未参与任何训练决策。

运行时验证（修正后首个 run 实测）：fold 0 `train_p=40 → n_val=8`、fold 4 `train_p=41 → n_val=9`，与 ⌈n/5⌉ 公式一致；`random_seed=20260918`（split JSON `random_seed=null` → 回退固定值）；val 患者 ID 已逐折落日志（如 fold 3 val = [Pt126, Pt127, Pt128, Pt148, Pt40, Pt56, Pt61, Pt70, Pt86]）。

原种子识别：既有 `dual_mlp_genes_h64_wd02_gpu0_results.json` / `dual_mlp_pathway_pca_reg_gpu1_results.json` 及脚本均**无训练种子记录**（旧脚本无 `--seed` 参数）→ 原种子不可识别，多种子按 20260919/20260920/20260921 全新执行（原 run 结果保留，不与多种子混算同一序列）。

## 二、环境预检（SubTask 0.2）

- `make CONDA=/home/dell/miniconda3/bin/conda env-check` → PASS（Linux WSL2, Python 3.11.15, drugscreening-gpu）。
- `make CONDA=/home/dell/miniconda3/bin/conda gpu-check` → PASS（torch 2.5.1, cuda_available=True, CUDA 12.1, gpu_count=3：GPU0/GPU1 = RTX A5000, GPU2 = Quadro RTX 4000）。
- GPU 使用纪律：仅 GPU0/GPU1；GPU2 由其他进程占用（3588 MiB used），全程未触碰。

## 三、挂载任务清单（SubTask 0.3 / 0.4）

### 任务 A：官方迁移新真种子（SubTask 0.3，CPU 后台）

- 命令：`nohup bash scripts/modeling/_launch_official_xfer_b2000_seed20260919_DRAFT.sh`（setsid + nohup，`< /dev/null`）
- 配置：`configs/experiments/EXP-020-kryzeiu-transfer-b2000-seed20260919-DRAFT.json`（seeds=[20260919], device=cpu, bootstrap_resamples=2000, scope=...EXPLORATORY_TRANSFER_ONLY_not_frozen）
- 入口：`scripts/run_transfer_experiment.py`（机制复刻 `_write_seed13_launch.py` 产物：同 admission / source / registry / data-root / --bootstrap-resamples 2000）
- seed：**20260919**（新真种子；bootstrap=2000；EXPLORATORY_TRANSFER_ONLY 口径由 runner status 固化）
- 进程：launcher PID **13579**（bash）→ worker PID **13584**（python, CPU）
- 日志：`logs/official_xfer_b2000_seed20260919_launcher.log` + `logs/official_xfer_b2000_seed20260919_20260919_121930.log`
- 输出目录：`artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_20260919_20260919_121930`（含真实 seed）
- 启动验证：`seed=20260919 fold=0 training_patients=41 test_patients=11` 已推进，无 traceback。

### 任务 B：Stage2 `dual_mlp_genes_h64_wd02` 多种子（SubTask 0.4，GPU0）

- 配置（复刻既有头部 results.json）：feat=genes, hidden=64, depth=1, weight_decay=0.02, folds=0–4（epochs=50/batch=256/pca-dim=64 为脚本默认，与原 run 同口径）
- seeds：20260919 / 20260920 / 20260921（同卡顺序执行，bash 循环）
- 命令：`nohup bash scripts/modeling/_launch_stage2_multiseed_genes_gpu0_DRAFT.sh`
- 进程：launcher PID **13597** → 首种子 worker PID **13605**（python, CUDA_VISIBLE_DEVICES=0）
- 每种子日志：`logs/stage2_dual_mlp_genes_h64_wd02_seed<S>_gpu0.log`；launcher 日志 `logs/stage2_multiseed_genes_gpu0_launcher.log`
- 每种子输出目录：`artifacts/route1_stage2_dual_mlp_variants_DRAFT/dual_mlp_genes_h64_wd02_seed<S>/`
- 启动验证：seed20260919 已完成 FOLD 0–3、FOLD 4 训练中；device=cuda:0（GPU_NAME NVIDIA RTX A5000）；无 traceback。首折指标与原 run 同量级（ndcg@5≈0.90–0.92）。

### 任务 C：Stage2 `dual_mlp_pathway_pca_reg` 多种子（SubTask 0.4，GPU1）

- 配置：feat=pathway_pca, hidden=128, depth=1, weight_decay=0.01, pca-dim=64, folds=0–4
- seeds：20260919 / 20260920 / 20260921（同卡顺序执行）
- 命令：`nohup bash scripts/modeling/_launch_stage2_multiseed_pca_gpu1_DRAFT.sh`
- 进程：launcher PID **13611** → 首种子 worker PID **13616**（python, CUDA_VISIBLE_DEVICES=1）
- 每种子日志：`logs/stage2_dual_mlp_pathway_pca_reg_seed<S>_gpu1.log`；launcher 日志 `logs/stage2_multiseed_pca_gpu1_launcher.log`
- 每种子输出目录：`artifacts/route1_stage2_dual_mlp_variants_DRAFT/dual_mlp_pathway_pca_reg_seed<S>/`
- 启动验证：seed20260919 FOLD 0 完成（ndcg@5≈0.898）、FOLD 1 训练中；device=cuda:0（GPU_NAME NVIDIA RTX A5000）；无 traceback。

明确未启动：genes_h256 类加宽/加深配置（按 spec 禁止）。

## 四、启动后 nvidia-smi 快照（SubTask 0.5 证据）

```
index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, NVIDIA RTX A5000, 251 MiB, 24564 MiB, 0 %
1, NVIDIA RTX A5000, 288 MiB, 24564 MiB, 0 %
2, Quadro RTX 4000, 3596 MiB, 8192 MiB, 2 %
```

（挂载前 GPU1 为 6 MiB → 挂载后 288 MiB；GPU0 挂载前即有 251 MiB 基线占用，训练日志确认 device=cuda:0 + A5000。WSL2 下 `--query-compute-apps` 列表为空的已知现象；利用率 0% 为采样瞬间空隙。GPU2 未触碰。）

## 五、seed2 误标修正（Task 3.1）

- 误标目录：`artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_seed2_20260918_223331/`，其 `frozen_config.json` 记录 `config.seeds=[20260912]`，目录名 "seed2" 与实际不符。
- 处置：目录内新增 `artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_seed2_20260918_223331/_MISLABEL_NOTE.md`，写明实际 seed=20260912、与第一轮 `run_b2000_20260918_174702` 数字一致属重复轮、不得计入独立种子复现；历史目录未删除、未重命名。
- 报告检索：对 `docs/`（含 `docs/reports/`）全量检索 `run_b2000` / `b2000_seed` / `223331` / `174702` / `seed2`，未发现引用该误标目录的报告（仅 `route1_stage2_official_transfer_smoke_DRAFT.md` 引用 seed 20260912 的 bootstrap=2 SMOKE 结果，与误标目录无关）→ 无需在报告中加更正注记。

## 六、限制

- 划分/身份/登记均仍为 DRAFT：`kryeziu_patient_outer5fold_DRAFT.json` 的 `random_seed=null`（内层 val 回退 20260918）、split status DRAFT；本回执不构成冻结口径宣称。
- Stage2 多种子的原种子不可追溯（旧脚本未记录训练种子），3 个新种子与原 run 不构成同序列种子扩展；多种子结论须等全部 run 完成后按 fold-CI / 患者级 CI 口径判定。
- 官方迁移 seed20260919 为 CPU 后台运行（为 GPU0/1 让路），完成时间显著长于 GPU 轮；运行期间勿清理 `logs/` 与 `artifacts/route1_stage2_*`。
- 内层 val 本轮仅记录构成，未实际用于早停/选模（训练目标未变）；若后续引入早停须以该 val 为准并在 EXP 记录。
- WSL2 `nvidia-smi` 的 compute-apps 列表不可见，GPU 占用以显存变化 + 训练日志 device 行为证。

## 七、下一步

- 等待三任务自然结束：任务 A 完成后由 Task 3.2/3.3 收齐 20260912 / 20260913 / 20260919 ≥3 真种子并做逐算法 ΔNDCG@5 + 2000 次配对 bootstrap 汇总（官方迁移汇总归 Task 3）。
- 任务 B/C 完成后由 Task 5.1 收集多种子预测分并做患者级 CI（多种子结果收集归 Task 5）。
- 训练脚本修正仅涉 `scripts/modeling/`（最小 diff）；后续若冻结划分（Task 1.1 物化 `val_patient_ids`），须复核本脚本回退种子逻辑与 JSON 物化值一致。
