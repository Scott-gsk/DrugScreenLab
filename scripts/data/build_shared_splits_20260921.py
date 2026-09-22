#!/usr/bin/env python3
"""Work-package B task 3: build shared outer-fold split contract.

Target domain (Kryzei PDO): outer 5-fold GroupKFold on patient_id (whitelist
patients, 60 S1-verified samples), inner 3-fold GroupKFold on patient covering
only each outer train fold's patients. Cell-line side (GDSC2): identity-based
GroupKFold leave-out under the same split system id.

Split seed and model-initialization seed are recorded separately (the split is
deterministic: lexicographic sort -> seeded shuffle -> GroupKFold).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path("/mnt/d/Code/DrugScreenLab")
IDENT = ROOT / "data/processed/identity/kryeziu_patient_identity_FROZEN.json"
SAMPLES = ROOT / "data/processed/transfer/mcrc_v3/samples.parquet"
PAIRS = ROOT / "data/processed/transfer/mcrc_v3/pairs.parquet"
GDSC_SAMPLES = ROOT / "data/processed/exp012/gdsc2_cell_line_samples.parquet"
OUT_KRY = ROOT / "data/splits/kryzei_shared_splits_v1.json"
OUT_GDSC = ROOT / "data/splits/gdsc_cellline_splits_v1.json"

SPLIT_SYSTEM_ID = "bprogram_shared_20260921"
SPLIT_SEED = 20260921
MODEL_INIT_SEED = 20260921
MIN_DRUGS_PER_PATIENT = 3


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def seeded_patients(patients, seed):
    ordered = sorted(p for p in patients)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ordered))
    return [ordered[int(i)] for i in perm]


def sample_of_patient(samples_df):
    out = {}
    for sid, pid in zip(samples_df["sample_id"].astype(str), samples_df["patient_id"].astype(str)):
        out.setdefault(pid, []).append(sid)
    return out


def main() -> int:
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    whitelist_patients = sorted(ident["verified_patients"])
    whitelist_samples = set(
        r["sample_id"] for r in ident["per_sample"] if r["identity_status"] == "S1_PDO_patient_verified"
    )
    samples = pd.read_parquet(SAMPLES)
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pd.read_parquet(PAIRS)
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    pairs = pairs[pairs.sample_id.isin(whitelist_samples)].copy()
    samples = samples[samples.sample_id.isin(whitelist_samples)].copy()

    patient_of_sample = dict(zip(samples.sample_id, samples.patient_id))
    sample_of_pat = sample_of_patient(samples)

    # all 48-strong drug coverage + the 14 GDSC-overlap "known" drugs
    known14 = sorted(json.loads((ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json").read_text(encoding="utf-8"))["per_drug_stats"].keys())

    outer_patients = seeded_patients(whitelist_patients, SPLIT_SEED)
    gk = GroupKFold(n_splits=5)
    outer_folds = []
    outer_issues = []
    assignments = {}  # test-only patient -> fold, for isolation sanity

    for fold_idx, (train_idx, test_idx) in enumerate(gk.split(outer_patients, groups=outer_patients)):
        train_p = [outer_patients[i] for i in train_idx]
        test_p = [outer_patients[i] for i in test_idx]
        for p in test_p:
            assert p not in assignments, "patient assigned to two test folds"
            assignments[p] = fold_idx
        train_s = sorted(s for p in train_p for s in sample_of_pat.get(p, []))
        test_s = sorted(s for p in test_p for s in sample_of_pat.get(p, []))

        train_rows = pairs[pairs.sample_id.isin(set(train_s))]
        test_rows = pairs[pairs.sample_id.isin(set(test_s))]
        drugs_train = sorted(train_rows.drug_id.unique())
        known14_missing = [d for d in known14 if d not in set(drugs_train)]

        per_patient_counts = train_rows.groupby("sample_id").drug_id.nunique()
        # group drug sets by patient (each patient may have >1 PDO)
        patient_drugs = {}
        for sid in per_patient_counts.index:
            patient_drugs.setdefault(patient_of_sample[sid], set()).update(
                train_rows[train_rows.sample_id == sid].drug_id.tolist())

        evaluable_patients = sorted(p for p, dset in patient_drugs.items() if len(dset) >= MIN_DRUGS_PER_PATIENT)
        not_evaluable = sorted(p for p, dset in patient_drugs.items() if len(dset) < MIN_DRUGS_PER_PATIENT)

        issues = []
        if len(set(train_p) & set(test_p)) != 0:
            issues.append("leakage: train/test patient overlap")
        if known14_missing:
            issues.append(f"known14 missing from train: {known14_missing}")
        if not_evaluable:
            issues.append(f"patients with <{MIN_DRUGS_PER_PATIENT} drugs: {not_evaluable}")

        # inner 3-fold on outer train patients only
        inner_patients = seeded_patients(train_p, SPLIT_SEED)
        inner_folds = []
        gk_inner = GroupKFold(n_splits=3)
        for itr_idx, (itr_train, itr_val) in enumerate(gk_inner.split(inner_patients, groups=inner_patients)):
            inner_folds.append({
                "inner_fold": int(itr_idx),
                "inner_train_patient_ids": [inner_patients[i] for i in itr_train],
                "inner_val_patient_ids": [inner_patients[i] for i in itr_val],
            })

        outer_folds.append({
            "fold": int(fold_idx),
            "train_patient_ids": sorted(train_p),
            "test_patient_ids": sorted(test_p),
            "train_sample_ids": train_s,
            "test_sample_ids": test_s,
            "stats": {
                "n_train_patients": len(train_p),
                "n_test_patients": len(test_p),
                "n_train_samples": len(train_s),
                "n_test_samples": len(test_s),
                "n_drugs_covered_train": len(drugs_train),
                "known14_missing_in_train": known14_missing,
                "n_evaluable_patients_train": len(evaluable_patients),
                "n_not_evaluable_patients_train": len(not_evaluable),
                "n_evaluable_rows_train": int(len(train_rows[train_rows.sample_id.isin({s for p in evaluable_patients for s in sample_of_pat.get(p, [])})])),
                "min_drugs_per_patient": MIN_DRUGS_PER_PATIENT,
            },
            "inner_folds": inner_folds,
            "issues": issues,
        })
        if issues:
            outer_issues.append({f"fold {fold_idx}": issues})

    kry_payload = {
        "name": "kryzei_shared_splits",
        "version": "v1",
        "status": "FROZEN",
        "split_system_id": SPLIT_SYSTEM_ID,
        "domain": "kryzei_pdo",
        "split_id": "kryzei_shared_v1",
        "split_unit": "patient_id",
        "split_seed": SPLIT_SEED,
        "model_init_seed": MODEL_INIT_SEED,
        "splitter": "sklearn.model_selection.GroupKFold (lexicographic sort -> RandomState(split_seed) shuffle -> GroupKFold)",
        "outer_n_folds": 5,
        "inner_n_folds": 3,
        "n_patients": len(whitelist_patients),
        "n_samples": len(whitelist_samples),
        "whitelist": "S1_PDO_patient_verified (51 patients / 60 samples)",
        "outer_folds": outer_folds,
        "precheck": {
            "per_fold_independent_patients": "no patient assigned to more than one outer test fold (asserted during build)",
            "drug_coverage": "per-fold stats.known14_missing_in_train records any of the 14 GDSC-overlap drugs absent from a train fold",
            "evaluable_rows": "train rows of patients with >=3 distinct drugs",
            "non_evaluable_reasons": outer_issues if outer_issues else "none",
        },
        "provenance": {
            "generated_by": "scripts/data/build_shared_splits_20260921.py",
            "samples_parquet": rel(SAMPLES),
            "pairs_parquet": rel(PAIRS),
            "identity": rel(IDENT),
            "generated_at": "2026-09-21",
        },
    }
    OUT_KRY.parent.mkdir(parents=True, exist_ok=True)
    OUT_KRY.write_text(json.dumps(kry_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # GDSC cell-line side
    gdsc = pd.read_parquet(GDSC_SAMPLES)
    gdsc["cell_line_id"] = gdsc["cell_line_id"].astype(str)
    lines = gdsc["cell_line_id"].tolist()
    lines = seeded_patients(lines, SPLIT_SEED)
    gk_g = GroupKFold(n_splits=5)
    gdsc_folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(gk_g.split(lines, groups=lines)):
        gdsc_folds.append({
            "fold": int(fold_idx),
            "train_cell_line_ids": [lines[i] for i in train_idx],
            "test_cell_line_ids": [lines[i] for i in test_idx],
            "stats": {
                "n_train_lines": int(len(train_idx)),
                "n_test_lines": int(len(test_idx)),
            },
        })
    gdsc_payload = {
        "name": "gdsc_cellline_splits",
        "version": "v1",
        "status": "FROZEN",
        "split_system_id": SPLIT_SYSTEM_ID,
        "domain": "gdsc_cellline",
        "split_id": "gdsc_cellline_v1",
        "split_unit": "cell_line_id",
        "split_seed": SPLIT_SEED,
        "model_init_seed": MODEL_INIT_SEED,
        "splitter": "sklearn.model_selection.GroupKFold (seeded shuffle -> GroupKFold) by cell-line identity",
        "outer_n_folds": 5,
        "n_cell_lines": len(lines),
        "role": "C-package four-layer matrix leave-out (source domain / cold-cell identity isolation)",
        "outer_folds": gdsc_folds,
        "provenance": {
            "generated_by": "scripts/data/build_shared_splits_20260921.py",
            "samples_parquet": rel(GDSC_SAMPLES),
            "generated_at": "2026-09-21",
        },
    }
    OUT_GDSC.write_text(json.dumps(gdsc_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("wrote", rel(OUT_KRY))
    for f in outer_folds:
        s = f["stats"]
        print(f"kryzei fold {f['fold']}: train={s['n_train_patients']}p/{s['n_train_samples']}s "
              f"test={s['n_test_patients']}p/{s['n_test_samples']}s drugs={s['n_drugs_covered_train']} "
              f"evaluable_p={s['n_evaluable_patients_train']} issues={f['issues']}")
    print("wrote", rel(OUT_GDSC))
    for f in gdsc_folds:
        print(f"gdsc fold {f['fold']}: train={f['stats']['n_train_lines']} test={f['stats']['n_test_lines']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

