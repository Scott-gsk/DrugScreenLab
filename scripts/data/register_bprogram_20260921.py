#!/usr/bin/env python3
"""Append work-package B registry entries to datasets.json (idempotent).

Does not modify existing entries; only appends the four new B-program artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/d/Code/DrugScreenLab")
REG = ROOT / "data/registry/datasets.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry_for(rel_path: str, id_: str, role: str, status: str, schema: dict, source: dict, provenance: dict) -> dict:
    path = ROOT / "data" / rel_path
    return {
        "id": id_,
        "version": 1,
        "asset_type": "derived_metadata",
        "source": source,
        "path": {"relative": rel_path},
        "immutable": True,
        "intended_role": role,
        "status": status,
        "checksum": {"algorithm": "sha256", "value": sha256(path)},
        "schema": schema,
        "provenance": provenance,
    }


def main() -> int:
    entries = json.loads(REG.read_text(encoding="utf-8"))
    existing = {e["id"] for e in entries}

    new_entries = [
        entry_for(
            "processed/identity/kryzei_admission_audit_20260921.json",
            "kryzei_admission_audit_v1",
            "FROZEN_ADMISSION_WHITELIST_AUDIT",
            "AUDITED",
            {
                "admitted_patients": 51,
                "admitted_samples": 60,
                "pending_samples": 32,
                "pt120_training_participation": "true (EXP-021 target train folds, 4 of 5 outer folds)",
            },
            {"provider": "DrugScreenLab", "parent_asset": "kryeziu_patient_identity_frozen_v1", "accession": "workpackage_B_20260921"},
            {"generator": "scripts/data/audit_kryzei_admission_20260921.py", "created_at": "2026-09-21"},
        ),
        entry_for(
            "processed/identity/kryzei_structural_drugs_TRAINONLY_v1.json",
            "kryzei_structural_drugs_trainonly_v1",
            "FROZEN_STRUCTURAL_DRUG_LABELS_PER_TRAIN_FOLD",
            "FROZEN",
            {
                "derived_from": "train_folds_only",
                "rule": ">=80% of TRAIN-fold PDOs rank-fixed in true top-5 or bottom-5 (14 GDSC-overlap drugs)",
                "n_known_drugs": 14,
                "n_folds": 5,
            },
            {"provider": "DrugScreenLab", "parent_asset": "kryzei_structural_drugs_FROZEN.json", "accession": "workpackage_B_20260921"},
            {"generator": "scripts/data/rebuild_structural_drugs_trainonly.py", "created_at": "2026-09-21"},
        ),
        entry_for(
            "splits/kryzei_shared_splits_v1.json",
            "kryzei_shared_splits_v1",
            "FROZEN_SHARED_OUTER5_INNER3_GROUPKFOLD_PATIENT",
            "FROZEN",
            {
                "split_system_id": "bprogram_shared_20260921",
                "split_unit": "patient_id",
                "outer_n_folds": 5,
                "inner_n_folds": 3,
                "n_patients": 51,
                "n_samples": 60,
                "split_seed": 20260921,
                "model_init_seed": 20260921,
            },
            {"provider": "DrugScreenLab", "parent_asset": "kryeziu_patient_outer5fold_frozen_v1", "accession": "workpackage_B_20260921"},
            {"generator": "scripts/data/build_shared_splits_20260921.py", "created_at": "2026-09-21"},
        ),
        entry_for(
            "splits/gdsc_cellline_splits_v1.json",
            "gdsc_cellline_splits_v1",
            "FROZEN_SHARED_OUTER5_GROUPKFOLD_CELLLINE",
            "FROZEN",
            {
                "split_system_id": "bprogram_shared_20260921",
                "split_unit": "cell_line_id",
                "outer_n_folds": 5,
                "n_cell_lines": 963,
                "split_seed": 20260921,
                "model_init_seed": 20260921,
            },
            {"provider": "DrugScreenLab", "parent_asset": "gdsc2_cell_line (transfer_cohorts.json)", "accession": "workpackage_B_20260921"},
            {"generator": "scripts/data/build_shared_splits_20260921.py", "created_at": "2026-09-21"},
        ),
    ]

    added = 0
    for e in new_entries:
        if e["id"] in existing:
            print("skip existing", e["id"])
            continue
        entries.append(e)
        added += 1

    REG.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"appended {added} entries (total {len(entries)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

