#!/usr/bin/env bash
# Stage2 head config multi-seed: dual_mlp_genes_h64_wd02 on GPU0, seeds 20260919/20260920/20260921 sequential.
# Head config from artifacts/route1_stage2_dual_mlp_variants_DRAFT/dual_mlp_genes_h64_wd02_gpu0_results.json:
#   feat=genes hidden=64 depth=1 weight_decay=0.02 folds=0-4 (epochs/batch/pca-dim at script defaults).
# No widening/deepening variants (genes_h256-class) are launched here.
set -euo pipefail
ROOT=/mnt/d/Code/DrugScreenLab
PY=/home/dell/miniconda3/envs/drugscreening-gpu/bin/python
cd "$ROOT"
for SEED in 20260919 20260920 20260921; do
  TAG="dual_mlp_genes_h64_wd02_seed${SEED}"
  OUT="$ROOT/artifacts/route1_stage2_dual_mlp_variants_DRAFT/${TAG}"
  mkdir -p "$OUT" "$ROOT/logs"
  LOG="$ROOT/logs/stage2_${TAG}_gpu0.log"
  echo "=== START $TAG gpu=0 seed=$SEED $(date '+%F %T') ===" | tee "$LOG"
  "$PY" -u scripts/modeling/train_stage2_dual_mlp_variants_DRAFT.py \
    --gpu 0 \
    --folds 0,1,2,3,4 \
    --feat genes \
    --hidden 64 \
    --depth 1 \
    --weight-decay 0.02 \
    --seed "$SEED" \
    --tag "$TAG" \
    --out-dir "$OUT" \
    2>&1 | tee -a "$LOG"
  echo "=== DONE $TAG $(date '+%F %T') ===" | tee -a "$LOG"
done
echo ALL_SEEDS_DONE_GPU0
