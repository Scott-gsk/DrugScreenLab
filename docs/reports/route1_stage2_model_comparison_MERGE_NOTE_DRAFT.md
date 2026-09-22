# route1 stage2 model comparison merge note (DRAFT)

- Killed redundant `dual_mlp_pathway_pca` PID 1090 mid-run to free A5000 (incomplete; not in macro).
- Kept `dual_mlp_deep_mine` for audit only; overlaps scott `dual_mlp_deeper` (both worse than shallow base).
- Do **not** widen/deepen further; next GPU work: stronger reg / smaller net or finish pathway-PCA table if still needed vs Hallmark Ridge.
- Tables: `route1_stage2_model_comparison_by_fold_DRAFT.csv`, `route1_stage2_model_comparison_macro_DRAFT.csv`
