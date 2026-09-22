# GDSC → Kryeziu/mCRC PDO transfer skeleton (DRAFT)

Status: **SKELETON_READY_GAPS**

Entry points: `scripts/train_transfer_source.py`, `scripts/run_transfer_experiment.py`.

Related prior: EXP-015 GDSC source, EXP-016 cold-cell champion, EXP-017 Zhu PDO (negative).

## Gaps before formal transfer
- NEED: Kryeziu/mCRC target admission receipt + gene-space contract aligned to GDSC source (hash-frozen) before formal run_transfer_experiment
- NEED: drug identity intersection (InChIKey) GDSC ∩ Kryeziu 48-drug DRAFT panel

Stub contract: `artifacts/route1_stage2_gdsc_pdo_transfer_skeleton_DRAFT/EXP020_gdsc_to_kryeziu_transfer_contract_STUB.json`
Machine status JSON: `artifacts/route1_stage2_gdsc_pdo_transfer_skeleton_DRAFT/skeleton_status_DRAFT.json`

Do **not** deepen DualMLP further. This skeleton is the next knife.
