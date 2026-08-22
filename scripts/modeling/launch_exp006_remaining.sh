#!/usr/bin/env bash
# Sequence remaining EXP-006 jobs after coverage freeze.
# A 10% and UniPert may already be running; this script is idempotent.
set -euo pipefail
export PATH="/home/dell/miniconda3/bin:$PATH"
source /home/dell/miniconda3/etc/profile.d/conda.sh
cd /mnt/d/Code/DrugScreenLab
export PYTHONPATH=/mnt/d/Code/DrugScreenLab/src:/mnt/d/Code/DrugScreenLab/data/external/xpert_source
ROOT=artifacts/experiments/EXP-006
LOG=$ROOT/logs
mkdir -p "$LOG" "$ROOT/runs"

run() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name $*" | tee -a "$LOG/orchestrator.log"
  conda run -n drugscreening-gpu "$@"
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/orchestrator.log"
}

if [ ! -f "$ROOT/unipert_genetic_features.npy" ]; then
  run unipert_features python scripts/modeling/build_unipert_genetic_features.py \
    --pert-info data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz \
    --unipert-source data/external/unipert_source \
    --model-dir data/external/unipert_source/current_model \
    --gene-list "$ROOT/selected_genes.txt" \
    --max-genes 0 \
    --output-features "$ROOT/unipert_genetic_features.npy" \
    --output-mapping "$ROOT/unipert_genetic_mapping.json" \
    --audit "$ROOT/unipert_genetic_audit.json"
fi

if [ ! -f "$ROOT/genetic_paired_adapter.h5ad" ]; then
  run genetic_adapter python scripts/data/build_exp006_genetic_adapter.py
fi

if [ ! -f "$ROOT/runs/A_frac0.1_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run A_0.1 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model A --fraction 0.1 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 0 --downstream --save-checkpoint
fi

if [ ! -f "$ROOT/runs/A_frac0.2_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run A_0.2 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model A --fraction 0.2 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 0 --downstream --save-checkpoint
fi

if [ ! -f "$ROOT/runs/B_frac0.1_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run B_0.1 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model B --fraction 0.1 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 3 --downstream --save-checkpoint
fi

if [ ! -f "$ROOT/runs/B_frac0.2_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run B_0.2 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model B --fraction 0.2 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 0 --skip-genetic \
    --init-checkpoint "$ROOT/runs/B_frac0.1_seed20260813/model.pt" \
    --downstream --save-checkpoint
fi

if [ ! -f "$ROOT/runs/A_frac1.0_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run A_1.0 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model A --fraction 1.0 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 0 --downstream --save-checkpoint
fi

if [ ! -f "$ROOT/runs/B_frac1.0_seed20260813/metrics.json" ]; then
  CUDA_VISIBLE_DEVICES=0 run B_1.0 python scripts/modeling/run_exp006_xpert_transfer.py \
    --model B --fraction 1.0 --device cuda:0 --seed 20260813 \
    --batch-size 32 --chemical-epochs 3 --genetic-epochs 0 --skip-genetic \
    --init-checkpoint "$ROOT/runs/B_frac0.1_seed20260813/model.pt" \
    --downstream --save-checkpoint
fi

run merge python scripts/modeling/run_exp006_xpert_transfer.py --model A --fraction 0.1 --merge-only
echo "[$(date -Is)] EXP-006 remaining jobs finished" | tee -a "$LOG/orchestrator.log"
