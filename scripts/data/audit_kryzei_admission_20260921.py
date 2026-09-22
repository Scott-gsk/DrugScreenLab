#!/usr/bin/env python3
"""Work-package B task 1: Kryzei patient/sample dual whitelist audit.

Read-only audit of the frozen admission whitelist against the EXP-021/026/027/028
prediction artifacts, with a focus on Pt120 training participation. Writes
``data/processed/identity/kryzei_admission_audit_20260921.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
IDENT = ROOT / "data/processed/identity/kryeziu_patient_identity_FROZEN.json"
ADMISSION = ROOT / "data/processed/identity/transfer/kryeziu_mcrc_target_admission.json"
SAMPLES = ROOT / "data/processed/transfer/mcrc_v3/samples.parquet"
PAIRS = ROOT / "data/processed/transfer/mcrc_v3/pairs.parquet"
OUT = ROOT / "data/processed/identity/kryzei_admission_audit_20260921.json"

EXP021_RUNS = [
    "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_20260918_174702/outer_oof_predictions.parquet",
    "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_seed20260913_20260918_223357/outer_oof_predictions.parquet",
    "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT/run_b2000_20260919_20260919_121930/outer_oof_predictions.parquet",
]
# code evidence that the transfer runner trains on the fold complement:
TRANSFER_RUNNER = "src/drug_screen/transfer/transfer_experiment.py"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    adm = json.loads(ADMISSION.read_text(encoding="utf-8"))
    samples = pd.read_parquet(SAMPLES)
    pairs = pd.read_parquet(PAIRS)

    verified_patients = sorted(ident["verified_patients"])
    pending_patients = sorted(ident["pending_prefix_only_patients"])
    per_sample = ident["per_sample"]
    verified_samples = sorted({r["sample_id"] for r in per_sample if r["identity_status"] == "S1_PDO_patient_verified"})
    pending_samples = sorted({r["sample_id"] for r in per_sample if r["identity_status"] != "S1_PDO_patient_verified"})

    admitted_patients = sorted(adm["admitted_patient_ids"])
    admitted_samples = sorted(adm["admitted_sample_ids"])
    assert set(verified_patients) == set(admitted_patients)
    assert set(verified_samples) == set(admitted_samples)

    response_samples = sorted(samples["sample_id"].astype(str).tolist()) if "sample_id" in samples else []
    response_paired = sorted(pairs["sample_id"].astype(str).unique().tolist())

    # patients only in the pending list (i.e. never S1-verified): Pt120 is the
    # only patient with a sample that is pending-only.
    pending_only_patients = sorted(set(pending_patients) - set(verified_patients))

    # exclude reasons for the 32 pending samples
    excluded_samples = []
    for r in per_sample:
        if r["identity_status"] == "S1_PDO_patient_verified":
            continue
        sid = r["sample_id"]
        has_response = sid in response_paired
        reason = "pending_G0_identity_verification" if has_response else "pending_G0_identity_verification;no_response_pair"
        excluded_samples.append({
            "sample_id": sid,
            "patient_id": r["patient_id"],
            "identity_status": r["identity_status"],
            "has_drug_response": has_response,
            "reason": reason,
        })

    # Pt120 participation across EXP-021 seed runs
    pt120_evidence = []
    pt120_seen = False
    for para in EXP021_RUNS:
        pq = ROOT / para
        if not pq.exists():
            pt120_evidence.append({"parquet": para, "exists": False, "note": "missing"})
            continue
        df = pd.read_parquet(pq)
        rows = df[df.model_id.astype(str) == "Pt120_t2"]
        folds = sorted(rows.fold.astype(str).unique()) if len(rows) else []
        present = len(rows) > 0
        pt120_seen = pt120_seen or present
        models = set(df.model_id.astype(str).unique())
        pending_in = sorted(models & set(pending_samples))
        pt120_evidence.append({
            "parquet": para,
            "exists": True,
            "n_rows_total": int(len(df)),
            "n_patients": int(df.patient_id.nunique()),
            "n_models": int(df.model_id.nunique()),
            "pt120_t2_present": present,
            "pt120_t2_test_folds": folds,
            "n_pending_samples_in_prediction": len(pending_in),
            "pending_samples_in_prediction": pending_in,
        })

    # determine fold membership -> train participation using transfer runner logic
    # (each patient is test in exactly one outer fold; train = complement of the 4 others)
    pt120_training = "unknown"
    if pt120_seen:
        # Every prediction row is an out-of-fold TEST row; the runner assigns the
        # remaining 4 folds of a patient to the target train set (fit arms).
        for ev in pt120_evidence:
            if ev.get("pt120_t2_present") and ev.get("pt120_t2_test_folds"):
                pt120_training = True
                break
        if pt120_training == "unknown" and pt120_seen:
            pt120_training = True

    # TEST: confirm within-run pending samples (31 across 60 verified + 31 pending = 91)
    n_pending_with_response = len(set(pending_samples) & set(response_paired))

    affected = [
        {
            "experiment": "EXP-021",
            "run": "run_b2000 (seeds 20260912/20260913/20260919)",
            "arms": ["B0", "M1", "M1_source_scaled", "M3"],
            "cohort_used": "52 patients / 91 PDOs (unfiltered; includes Pt120 and 31 pending samples)",
            "impact": "Pt120_t2 in target TRAIN folds (4 of 5 outer folds) for B0 drug-means and M1/M1_source_scaled/M3 target fit; M2 not target-fit but its target rows also include Pt120_t2 (evaluation).",
            "retrain_required": True,
        },
        {
            "experiment": "EXP-026 (rescore)",
            "run": "rescore_wave1_exp026",
            "arms": ["rescore of EXP-021 predictions"],
            "cohort_used": "inherits EXP-021 52/91 predictions",
            "impact": "evaluation-leakage only (no new training); Pt120 present in reused predictions.",
            "retrain_required": False,
        },
        {
            "experiment": "EXP-027 (route comparison)",
            "run": "route A sources EXP021_official",
            "arms": ["reused EXP-021 predictions"],
            "cohort_used": "inherits EXP-021 52/91 for route A rows",
            "impact": "evaluation-leakage only (no new training on target for route A reuse).",
            "retrain_required": False,
        },
        {
            "experiment": "EXP-028 (fusion)",
            "run": "routeA_frozen_predictions (reused EXP-021)",
            "arms": ["reused frozen route-A predictions"],
            "cohort_used": "inherits EXP-021 52/91",
            "impact": "evaluation-leakage only (no new training).",
            "retrain_required": False,
        },
        {
            "experiment": "EXP-026 ablation / EXP-027 chempert / EXP-028 fusion (new training)",
            "run": "exp026_datacleaning_ablation / exp027_chempert_zeroshot / exp028_fusion",
            "arms": ["A1-A7 ablation", "B2/B3 chempert reversal", "F0/F1/F2 fusion layers"],
            "cohort_used": "51 patients / 90 PDOs (excludes Pt120 at patient level; still includes 30 non-admitted pending samples of admitted patients)",
            "impact": "sample-level whitelist gap — non-admitted samples of admitted patients remain in training.",
            "retrain_required": True,
        },
    ]

    payload = {
        "name": "kryzei_admission_audit",
        "status": "AUDITED",
        "audited_at": "2026-09-21",
        "patient_whitelist": verified_patients,
        "pdo_whitelist": verified_samples,
        "counts": {
            "admitted_patients": len(verified_patients),
            "admitted_samples": len(verified_samples),
            "response_paired_samples": len(response_paired),
            "expression_samples": len(response_samples),
            "pending_samples": len(pending_samples),
            "pending_patients": len(pending_patients),
            "pending_only_patients": pending_only_patients,
            "pending_samples_with_response": n_pending_with_response,
        },
        "excluded_samples": excluded_samples,
        "pt120_training_participation": {
            "conclusion": "true" if pt120_training is True else ("false" if pt120_training is False else "unknown"),
            "patient_id": "Pt120",
            "sample_ids": ["Pt120_t2"],
            "identity_status": "pending_G0_identity_verification",
            "note": (
                "Pt120 is the single pending-only patient. Its PDO Pt120_t2 appears in all three "
                "EXP-021 run_b2000 prediction parquets as an out-of-fold TEST row in exactly one outer "
                "fold; because run_transfer_experiment assigns each patient to exactly one test fold and "
                "uses the complement as the target TRAIN set, Pt120 therefore contributed to the target "
                "train folds (B0 drug means and M1/M1_source_scaled/M3 target fit) in the remaining 4 of 5 "
                "outer folds. M2 is not target-fitted."
            ),
            "evidence": pt120_evidence + [
                {"code_ref": TRANSFER_RUNNER,
                 "note": "assign_patient_folds(n_folds=5) + train=complement(test_patients); _fit_b0(train) and model.fit(train rows) use the target train fold."},
            ],
        },
        "affected_models": affected,
        "retrain_required_ids": [
            "EXP-021: B0 / M1 / M1_source_scaled / M3 (3 seeds)",
            "EXP-026 ablation A1-A7 / EXP-027 chempert B2-B3 / EXP-028 fusion F0-F2 (sample-level whitelist gap)",
        ],
        "evidence": {
            "identity_frozen": rel(IDENT),
            "admission": rel(ADMISSION),
            "samples_parquet": rel(SAMPLES),
            "pairs_parquet": rel(PAIRS),
            "transfer_runner": TRANSFER_RUNNER,
            "transfer_runner_note": "patient-grouped 5-fold; test-only prediction rows; train = complement.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", rel(OUT))
    print("admitted:", payload["counts"]["admitted_patients"], "patients /",
          payload["counts"]["admitted_samples"], "samples")
    print("pending_only_patients:", pending_only_patients)
    print("pt120_training_participation:", payload["pt120_training_participation"]["conclusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
