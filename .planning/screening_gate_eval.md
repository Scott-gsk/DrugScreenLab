# EXP-005 Screening Gate Evaluation (fresh Evaluation Analyst context)

## Scope and assets

- EXP-ID: `EXP-005`; evaluation only, no model/registry/project-state edits.
- Prediction profile: `data/external/xpert_source/experiment/infer/20260813_151613/l1000_sdst_broad_crc_global_adapter_split_1_test_samples_prediction_profile.npy`
- Adapter: `data/external/xpert_source/processed_data/l1000_sdst_broad_crc_global_adapter.h5ad`
- Observed LINCS source: `data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad`
- PRISM response: `mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet`
- CRC signature: `mvp/core_data/crc_disease_signature_exact978.tsv`

## Reproducible commands

Executed in WSL2 Conda `drugscreening-gpu`:

```bash
cd /mnt/d/Code/DrugScreenLab
PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu \
  python -m drug_screen.evaluation.xpert_broad \
  --profile data/external/xpert_source/experiment/infer/20260813_151613/l1000_sdst_broad_crc_global_adapter_split_1_test_samples_prediction_profile.npy \
  --adapter data/external/xpert_source/processed_data/l1000_sdst_broad_crc_global_adapter.h5ad \
  --signature mvp/core_data/crc_disease_signature_exact978.tsv \
  --prism mvp/foundation/xpert/BROAD_PRISM_CRC_V1.parquet \
  --observed-lincs data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad \
  --output /tmp/exp005_eval.json
```

## Quantitative evidence

### Predicted reversal → Broad PRISM

- 18,179 exact joined rows; 10/10 lines eligible; 1,836 drugs.
- Macro Spearman: **−0.02736** (random null −0.00099; Δ −0.02637); 3/10 lines positive.
- Top-10 overlap (equivalent HitRate@10/Recall@10 against observed top-10): **0.0800** vs random 0.0053906 (16.07× lift).
- NDCG@10: **0.72320**, Δ vs random **+0.22240**.

### Observed LINCS Oracle → Broad PRISM

- 839 observed treatment pairs, only 2 lines (HT29 n=816; HCT116 n=23).
- Macro Spearman: **+0.15389** (random null −0.00647; Δ +0.16036); 1/2 lines positive.
- Top-10 overlap/HitRate@10/Recall@10: **0.3000** vs random 0.22012 (4.58× lift).
- NDCG@10: **0.65940**, Δ vs random **+0.09792**.

### Paired Oracle-vs-predicted reversal (same line × drug)

On the 839 exact shared pairs: mean(pred−oracle) **+0.07649**, median **+0.07647**, MAE **0.07659**, RMSE **0.08319**; Pearson **0.11382**, Spearman **0.05287**. HT29: n=816, ρ=.01879, mean gap=.07628; HCT116: n=23, ρ=.10474, mean gap=.08392.

### Single fallback CMap-style weighted-KS comparison

Because predicted reversal Spearman is negative, one exploratory weighted-KS ranking was run (no tuning/selection). Across 10 Broad lines and 18,179 rows, macro Spearman = **−0.01169** (per-line values: −.0500, .0043, −.0642, .0025, −.0296, −.0453, .0311, .0566, .0246, −.0469). This does not rescue the negative reversal→efficacy signal.

## Gate interpretation / support caveat

The Oracle support is response-blind but sparse: 839/62,248 PRISM pair identities (~1.35%) and 2/10 lines (20%). Therefore Oracle pooled metrics cannot be generalized to the full cohort. Conversely, pooled predicted NDCG/overlap masks per-context failure (7/10 lines have non-positive Spearman). Evidence is **INCONCLUSIVE/NO-GO for a biological improvement claim**; retain the frozen XPert foundation and do not treat the statistical Top-K lift as biological efficacy.

