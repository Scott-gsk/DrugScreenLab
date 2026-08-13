# EXP-005 Independent Reviewer / Red Team

## Verdict

`INCONCLUSIVE`

## 审查范围与证据

- 审查了 `experiments/records/EXP-005.md`、`experiments/registry.yaml`、当前 ATTEMPT-2 代码差异（`src/drug_screen/foundation/xpert_extension.py`、`scripts/foundation/run_xpert_extension_fast.py`、`src/drug_screen/evaluation/xpert_broad.py`）以及 `mvp/foundation/xpert/EXP005_FAST/` 下的 compact JSON。
- 当前记录仍为 `INCONCLUSIVE_PROTOCOL_MISMATCH · ATTEMPT-2_STRONG_FOUNDATION_EXTENSION_IN_PROGRESS`；Model Engineer 的交接明确说明只完成实现，未执行 full training，亦未生成 ATTEMPT-2 evidence package。

## 关键发现

1. **ATTEMPT-2 checkpoint 继承未被结果证据证明。** 所有 `split_*_[A|B|C].json` 均没有 `checkpoint_inheritance`、`official_parameters_frozen` 或 `additive_gate_init` 字段；所有 Broad JSON 也没有新 null baseline 或 Oracle `coverage` 字段。这些产物对应旧的随机初始化 FAST 运行，不能作为 strong-foundation extension 的结果。
2. **现有 compact 指标与 Foundation champion 不一致。** 例如 cold-cell A 的 row Spearman 约 `0.086`、cold-drug A 约 `0.053`，而官方 Foundation checkpoint 记录的 cold split Spearman 约 `0.956`/`0.954`。这支持“旧 ATTEMPT-1/协议不匹配”解释，不能据此判定 KPGT/UniPert 的增量。
3. **runner 存在 A 基线继承缺口。** `_model()` 仅在非 A 分支构造 overlay；`variant == "A"` 直接使用 `official["XPertNet"]`，即使传入 `--checkpoint` 也不会调用 `load_xpert_checkpoint`。若按当前代码执行 ATTEMPT-2，B/C 会继承 Foundation checkpoint，而 A 仍随机初始化，A/B/C 不再是批准的同一 Foundation 基线比较。
4. **gate=0 等价性仅有代码声称，没有数值审计。** 新 hook 在全零 gate 时返回带 straight-through 梯度的 residual；理论上前向 residual 为零，但没有执行前后的 A≈B(gate=0)≈C(gate=0) 数值、容差、seed 或 artifact。该检查不能表述为已通过。
5. **split 证据含有 FAST 选择限制。** runner 在无 `valid` 标签时复制 `test` 为 `valid`，且记录中的 valid/test digest 相同；虽然代码不以该副本选择 checkpoint，但它不是独立 validation，需在正式证据中显式标记为 protocol deviation。现有 compact 记录也显示仅截取各 partition 前 `4096` 行，不能外推官方 full-split 指标。
6. **Oracle 与 predicted support 不匹配。** Oracle 仅有 2 条 eligible lines（每 line 候选数约 `816`/`23`），predicted Broad 使用 10 条 lines、最多 `1836` drugs；因此 Oracle Spearman/NDCG 不能与 Broad predicted macro 指标直接作 degradation 或 efficacy 结论。新 coverage/null 代码虽有单元测试，但没有对应本 EXP 的运行产物。

## 结论边界

当前证据足以保留旧 FAST 的 `NO_MATERIAL_FAST_INCREMENT` 工程记录，但不足以验证 ATTEMPT-2 strong-foundation hypothesis，也不足以判定 KPGT 或 UniPert 的科学增益/失败。应先修正 A checkpoint 继承路径，并在同一 checkpoint、同一 split、固定 seed 下重新运行，提交 gate-equivalence、继承审计、null/Oracle coverage 和完整 structured evidence 后再审查。
