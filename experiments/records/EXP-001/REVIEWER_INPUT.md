# EXP-001 Independent Reviewer Input

## Verdict

`VALID`

## Final Readiness Assessment

`PARTIAL`

## 复审范围

本次独立复审针对 `SCIENTIFIC_CLARIFICATION` 后的 GSE92742 Level-3 premise，而非沿用
先前 verdict。核对的证据为 `P0_AUDIT_REPORT.md`、`evidence.json`、
`readiness_matrix.json`、`scripts/audit_exp001_level3.py`、`src/drug_screen/data/p0.py`
及其测试。

## 结论依据

- [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) 将 Level-3
  定义为经 invariant-set scaling 与 quantile normalization 的直接测量 landmark
  transcripts 加 inferred genes。处理层级不会改变 landmark 的 measurement provenance。
- [XPert Methods](https://www.nature.com/articles/s42256-025-01165-w) 从 Level-3 提取
  treatment/control，以 978 landmark changes 为目标，并使用同 plate DMSO control 配对。
- 本地只读审计确认 matrix `(1,319,138, 12,328)`，`pr_is_lm=1` 为精确 978，GCTX
  column-ID 集与 `inst_info.inst_id` 集相同。672,128 个 `trt_cp` 实例中，671,881 个
  有同 `rna_plate + cell_id + time` vehicle 候选；247 个无候选实例被显式排除。

## Required Boundaries

- inferred non-landmark genes 仍不得作为可靠 extended-gene experimental ground truth；
  reliable expansion 仍需要真实 WTS supervision/calibration。
- 只有通过 exact landmark mapping、metadata join、matched-control pairing 与 split
  leakage contract 的实例可进入后续 `Delta978` 派生。
- Level-2 是可选的 lower-level preprocessing/provenance/normalization reproduction
  resource，不是本地官方 Level-3 landmark-core 的自动 blocker。

`PARTIAL` 而非 `READY`：Delta 表尚未 materialize、247 个实例须排除、大矩阵未重散列，
且其他 P0 dataset/provenance blockers 尚未解除。
