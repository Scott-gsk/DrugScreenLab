#!/usr/bin/env bash
# Seed 20260919 official transfer b2000 (CPU background; EXPLORATORY_TRANSFER_ONLY scope).
# Mechanism mirrors _launch_official_xfer_b2000_seed20260913_DRAFT.sh; device=cpu to keep GPU0/1 free for Stage2 multi-seed.
set -euo pipefail
ROOT=/mnt/d/Code/DrugScreenLab
PY=/home/dell/miniconda3/envs/drugscreening-gpu/bin/python
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_20260919_${TS}"
mkdir -p "$OUT" "$ROOT/logs"
CFG="$ROOT/configs/experiments/EXP-020-kryeziu-transfer-b2000-seed20260919-DRAFT.json"
ADM="$ROOT/data/processed/identity/transfer/kryeziu_mcrc_target_admission_DRAFT.json"
SRC="$ROOT/data/artifacts/EXP-015/source_seed20260912"
REG="$ROOT/data/registry/transfer_cohorts.json"
LOG="$ROOT/logs/official_xfer_b2000_seed20260919_${TS}.log"
echo OUT=$OUT | tee "$LOG"
cd "$ROOT"
"$PY" -u scripts/run_transfer_experiment.py \
  --config "$CFG" \
  --target-admission "$ADM" \
  --source-output "$SRC" \
  --data-root "$ROOT" \
  --output-dir "$OUT" \
  --registry "$REG" \
  --bootstrap-resamples 2000 \
  2>&1 | tee -a "$LOG"
echo DONE_B2000_SEED20260919 | tee -a "$LOG"
