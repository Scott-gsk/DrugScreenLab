# CORE MVP FEASIBILITY REPORT

## Milestone

`MVP-001` · Cell-Line Core · `MVP LOOP` completed on 2026-08-12. This is an end-to-end
feasibility result, not a Formal Research EXP and not a biological efficacy conclusion.
The GSE117548 branch remains `DEFERRED_PDO_LEG`; no EXP-003 or new EXP-ID was created.

| Layer | Status | Evidence |
| --- | --- | --- |
| exact-978 foundation | `GREEN` | Registered cache reused; shape `1,319,138×978`, float32, cache SHA256 `04b8bb…1733`, GCTX landmark order digest `b4e2fca…b623`; all selected oracle cache IDs finite |
| observed reversal oracle | `GREEN` | GSE74602 paired CRC signature: 947 exact-978 genes (712 up/235 down); 695 matched LINCS treatment instances, 218 groups, 4 frozen candidates; 33/35 PRISM lines eligible; median per-line Spearman `0.8000`, positive fraction `0.7879` |
| learned Delta978 | `YELLOW` | Tiny/Small pipeline completes with atomic groups and both baselines. Small learned MAE `0.5381` vs constant `0.5360` and drug-mean `0.5320`; learned Spearman `0.2149` vs drug-mean `0.2377`, so no strict baseline-dominance signal |
| predicted reversal | `YELLOW` | Same signature/cohort/scoring completes; predicted-vs-observed drug-score Spearman `0.60`, top-2 overlap `2/2`; inherited limitations from bounded single-seed model |
| PRISM screening | `GREEN` | Official PRISM Repurposing 19Q4 processed response compacted to 135 finite rows across 35 colorectal DepMap lines; lower log2FC was frozen as greater sensitivity; observed and predicted ranking diagnostics both positive |
| overall core feasibility | `PROMISING` | `CORE_MVP_FEASIBILITY_PROMISING` under the predeclared MVP rule |

## Answers to the six MVP questions

1. **Oracle screening 是否成立？** 是。真实 observed LINCS exact-978 reversal 在同一冻结四药 cohort 上形成可审计 ranking，并与 PRISM sensitivity 在 33 个 eligible colorectal lines 上呈方向一致的 low-cost signal（mean Spearman `0.4788`, median `0.8000`, positive fraction `0.7879`; mean top-2 overlap `0.8030`). 这只证明当前闭环可运行并有 MVP-level signal，不证明临床疗效或机制。

2. **Learned prediction 是否成立？** 工程链成立，科学增益尚未成立。Simple one-seed drug/dose/time model 可以完成 held-out Delta978、预测 reversal 和 PRISM join，但没有同时超过 constant 与 train drug-mean diagnostic baseline；因此模型本身标为 `NO_SIGNAL_DIAGNOSTIC`，而不是把 oracle 信号归功于 learned model。

3. **Predicted vs observed gap 是什么？** 两条 ranking 使用同一 disease signature、四个 candidate、exact-978 顺序和 `-Spearman` scoring；drug-level Spearman 为 `0.60`，top-2 overlap 为 `2/2`。Predicted PRISM line-level diagnostics 与 observed 接近（mean Spearman `0.4909` vs `0.4788`），但四药 cohort 与单 seed 使该 gap 只能作为 feasibility diagnostic。

4. **当前最大瓶颈是什么？** 是 learned perturbation prediction 的信息量和覆盖，而不是 reversal/PRISM identity plumbing：Small 模型的 drug-mean baseline 仍略优，且当前只冻结四个 PRISM-overlap drugs。该判断不要求恢复复杂架构。

5. **是否值得继续投入整体 DrugScreenLab idea？** 值得继续做 Cell-Line MVP 工程化，因为 observed oracle、identity harmonization、PRISM compact benchmark 和 predicted ranking 的完整闭环已经跑通并出现方向一致 signal；但不应把本结果升级为 Formal Research 结论。

6. **下一步最高价值工作是什么？** 等用户审核本 milestone 后，优先在同一 identity/provenance contract 下扩大可审计 drug cohort，并用最简单的 bounded learned baseline定位预测瓶颈；随后再决定是否值得进入更强模型、cross-study 或 Formal EXP。不得自动恢复 EXP-003、PDO、UniPert、XPert 或复杂机制模型。

## Reproducibility and gates

- Disease signature: `mvp/core_data/crc_disease_signature_exact978.tsv`, SHA256 `61e95b6a…310c3`; source GSE74602 matrix SHA256 `381b6cb8…4b0f6`; platform annotation SHA256 `82ae57d6…6d518`; official SOFT pairing map SHA256 `9c97c8ce…ad13c1`.
- PRISM compact: `mvp/core_data/compact_prism_response.parquet`, SHA256 `5114e789…60f1`; source processed matrix was local-only and is not tracked.
- Manifest: `artifacts/mvp/MVP-001/compact_manifest.json` (tracked metadata, 230 KB), SHA256 `e6321f1b…f2acc3`; no held-out DRT lacks train support.
- Small checkpoint remains local-only at `artifacts/mvp/MVP-001/small/model.pt` (not committed, SHA256
  `c62915e5…da05`); its recipe, code/config revisions and reproducible command are tracked in
  `mvp/core_eval/MVP-001_model_provenance.json`.
- Observed ranking SHA256 `3b283ee8…f0e8`; predicted ranking SHA256 `e9d05c4a…11870`.
- Commands were executed with WSL2 Conda `drugscreening-gpu`; Tiny and Small were run before bounded evaluation. No raw datasets, full matrices, checkpoints, large predictions, or temporary logs are committed.
- `mvp/core_eval/MVP-001_core_eval_evidence.json`, `mvp/core_eval/MVP-001_model_provenance.json` and the single
  Reviewer record provide the complete tracked evidence summary; large cache, source matrix and checkpoint bodies
  remain local by policy.

## Overall decision

`PROMISING` is an MVP system-feasibility label only. After the one Independent Reviewer
checkpoint is committed and pushed, the project stops and waits for user review; it does
not automatically create or implement a Formal EXP.
