"""Materialize inner-val patient lists into the Kryeziu outer 5-fold split.

Formal script (decision-bearing): fills ``val_patient_ids`` / ``val_sample_ids``
of each outer fold in ``data/splits/kryeziu_patient_outer5fold_DRAFT.json``
in place, following the pinned algorithm from
``docs/reports/split_freeze_checklist_DRAFT.md`` (2026-09-18 17:23 CST):

- patient-level 4:1 (train_inner : val) drawn ONLY from the fold's
  ``train_patient_ids``; outer ``test_patient_ids`` never participates in
  model selection, early stopping, feature selection, or thresholds.
- lexicographically sort ``train_patient_ids``, then apply
  ``numpy.random.RandomState(seed).permutation``; the LAST ceil(n/5) patients
  of the permutation form the val set.
- seed: the split file's ``random_seed`` when present, else the fixed
  fallback ``20260918`` (recorded verbatim in the receipt).
- n < 5 -> at least 1 val patient; n == 1 -> skip inner val for that fold and
  flag ``inner_val_skipped``.
- sample mapping: val samples = all of that patient's samples inside the
  fold's ``train_sample_ids``.

The update is incremental and atomic: only the per-fold ``val_patient_ids`` /
``val_sample_ids`` fields are written (train/test fields are never touched),
and the file is replaced atomically so concurrent readers never observe a
partial write. Re-running is idempotent.

Outputs a receipt at ``data/splits/kryeziu_inner_val_materialization_receipt.json``
with the seed, per-fold statistics, and validation results.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DRAFT_REL = Path("data/splits/kryeziu_patient_outer5fold_DRAFT.json")
RECEIPT_REL = Path("data/splits/kryeziu_inner_val_materialization_receipt.json")
ALGORITHM_REF = "docs/reports/split_freeze_checklist_DRAFT.md#内层-val-算法钉死说明（2026-09-18-17:23-cst）"
FALLBACK_SEED = 20260918


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp_materialize")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def pick_inner_val_patients(train_patient_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """Return (val_patient_ids_sorted, train_inner_patient_ids_sorted).

    Pinned algorithm: lexicographic sort -> RandomState(seed).permutation ->
    last ceil(n/5) patients are val. n == 1 -> ([], train) with skip flag
    handled by the caller.
    """
    ordered = sorted(str(p) for p in train_patient_ids)
    n = len(ordered)
    if n == 0:
        return [], []
    if n == 1:
        return [], ordered
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(1, math.ceil(n / 5))
    val_idx = set(int(i) for i in perm[-n_val:])
    val = sorted(ordered[i] for i in val_idx)
    train_inner = sorted(p for i, p in enumerate(ordered) if i not in val_idx)
    return val, train_inner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / SPLIT_DRAFT_REL,
                        help="path to the outer 5-fold DRAFT split JSON (updated in place)")
    parser.add_argument("--receipt", type=Path, default=PROJECT_ROOT / RECEIPT_REL,
                        help="where to write the materialization receipt JSON")
    args = parser.parse_args(argv)

    split_path: Path = args.split
    if not split_path.is_file():
        print(f"FAIL: missing split file {split_path}", file=sys.stderr)
        return 1

    before_sha = sha256_file(split_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))

    seed = split.get("random_seed")
    seed_source = "split_json.random_seed"
    if seed is None:
        seed = FALLBACK_SEED
        seed_source = f"fallback_fixed_{FALLBACK_SEED}"

    fold_reports: list[dict] = []
    all_ok = True
    changed = False
    for fold in split.get("folds", []):
        fold_id = fold["fold"]
        train_pids = [str(p) for p in fold["train_patient_ids"]]
        test_pids = {str(p) for p in fold["test_patient_ids"]}
        train_sids = [str(s) for s in fold["train_sample_ids"]]

        val_pids, train_inner_pids = pick_inner_val_patients(train_pids, int(seed))
        skipped = (len(train_pids) == 1)
        val_set = set(val_pids)
        val_sids = sorted(s for s in train_sids if s.split("_", 1)[0] in val_set)

        issues: list[str] = []
        if not val_set.issubset(set(train_pids)):
            issues.append("val patients not a subset of outer train patients")
        if val_set & test_pids:
            issues.append("val patients intersect outer test patients")
        # sample->patient consistency inside the fold (dry-run parity)
        train_sid_patients = {s.split("_", 1)[0] for s in train_sids}
        if not train_sid_patients.issubset(set(train_pids)):
            issues.append("train_sample_ids contain patients outside train_patient_ids")
        if len(set(train_pids)) != len(train_pids):
            issues.append("duplicate train_patient_ids")
        if len(set(train_sids)) != len(train_sids):
            issues.append("duplicate train_sample_ids")
        # idempotency check against any pre-existing materialization
        prev_val_pids = [str(p) for p in fold.get("val_patient_ids") or []]
        prev_val_sids = [str(s) for s in fold.get("val_sample_ids") or []]
        if prev_val_pids and sorted(prev_val_pids) != val_pids:
            issues.append("existing val_patient_ids differ from pinned-algorithm recomputation")
        if prev_val_sids and sorted(prev_val_sids) != val_sids:
            issues.append("existing val_sample_ids differ from pinned-algorithm recomputation")

        ok = not issues and not skipped
        if issues:
            all_ok = False
        if sorted(prev_val_pids) != val_pids or sorted(prev_val_sids) != val_sids:
            fold["val_patient_ids"] = val_pids
            fold["val_sample_ids"] = val_sids
            changed = True

        fold_reports.append({
            "fold": fold_id,
            "n_outer_train_patients": len(train_pids),
            "n_train_inner_patients": len(train_inner_pids),
            "n_val_patients": len(val_pids),
            "n_train_inner_samples": len(train_sids) - len(val_sids),
            "n_val_samples": len(val_sids),
            "val_patient_ids": val_pids,
            "inner_val_skipped": skipped,
            "ok": ok,
            "issues": issues,
        })

    if changed:
        atomic_write_json(split_path, split)
    after_sha = sha256_file(split_path)

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    receipt = {
        "name": "kryeziu_inner_val_materialization",
        "algorithm_ref": ALGORITHM_REF,
        "algorithm": {
            "unit": "patient_id",
            "ratio": "4:1 train_inner:val (last ceil(n/5) of permutation are val)",
            "source_pool": "outer train_patient_ids only; outer test never used for selection",
            "sort": "lexicographic patient_id sort before permutation",
            "rng": f"numpy.random.RandomState({seed}).permutation",
        },
        "random_seed": int(seed),
        "seed_source": seed_source,
        "split_file": str(SPLIT_DRAFT_REL),
        "split_sha256_before": before_sha,
        "split_sha256_after": after_sha,
        "file_modified": changed,
        "n_folds": len(fold_reports),
        "all_folds_ok": all_ok,
        "folds": fold_reports,
        "materialized_at": now.isoformat(timespec="seconds"),
        "materialized_by": "scripts/data/materialize_kryeziu_inner_val.py (DSL-数据管家)",
    }
    atomic_write_json(args.receipt, receipt)

    print(f"random_seed={seed} (source: {seed_source})")
    for fr in fold_reports:
        status = "OK" if fr["ok"] else "FAIL"
        if fr["inner_val_skipped"]:
            status = "SKIPPED(n==1)"
        print(
            f"fold {fr['fold']}: outer_train={fr['n_outer_train_patients']} "
            f"train_inner={fr['n_train_inner_patients']} val={fr['n_val_patients']} "
            f"train_inner_samples={fr['n_train_inner_samples']} val_samples={fr['n_val_samples']} "
            f"val_patients={','.join(fr['val_patient_ids'])} [{status}]"
        )
        for issue in fr["issues"]:
            print(f"  ISSUE fold {fr['fold']}: {issue}")
    print(f"split file modified: {changed} ({before_sha[:12]} -> {after_sha[:12]})")
    print(f"all folds ok: {all_ok}")
    print(f"receipt: {args.receipt}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
