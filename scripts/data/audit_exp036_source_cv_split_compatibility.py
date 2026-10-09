"""Check whether archived EXP-011 source checkpoints fit the frozen EXP-036 folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_compatibility(split_path: Path, legacy_root: Path) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    current_folds = split.get("source_folds")
    if not isinstance(current_folds, list) or len(current_folds) != 5:
        raise ValueError("frozen source split must contain exactly five folds")

    current_universe: set[str] = set()
    current_test_counts: dict[int, int] = {}
    for fold in current_folds:
        train = set(map(str, fold["outer_train_ids"]))
        test = set(map(str, fold["outer_test_ids"]))
        if train & test:
            raise ValueError(f"current fold {fold['fold']} overlaps train and test identities")
        current_universe.update(train | test)
        current_test_counts[int(fold["fold"])] = len(test)
    tests = [set(map(str, fold["outer_test_ids"])) for fold in current_folds]
    if set().union(*tests) != current_universe or any(
        tests[i] & tests[j] for i in range(5) for j in range(i + 1, 5)
    ):
        raise ValueError("current outer test folds do not partition source identities")

    rows: list[dict[str, Any]] = []
    for current_fold, current_test in zip(current_folds, tests, strict=True):
        current_fold_id = int(current_fold["fold"])
        for old_fold_id in range(5):
            old_path = legacy_root / f"fold{old_fold_id}" / f"fold{old_fold_id}_source_indices.json"
            old = json.loads(old_path.read_text(encoding="utf-8"))
            assignments = {str(key): str(value) for key, value in old["group_assignments"].items()}
            if set(assignments) != current_universe:
                raise ValueError(f"legacy fold {old_fold_id} identity universe differs from frozen source universe")
            if set(assignments.values()) - {"train", "validation", "test"}:
                raise ValueError(f"legacy fold {old_fold_id} has an unexpected group role")
            counts = {
                role: sum(assignments[identity] == role for identity in current_test)
                for role in ("train", "validation", "test")
            }
            rows.append({
                "current_outer_fold": current_fold_id,
                "legacy_checkpoint_fold": old_fold_id,
                "current_outer_test_identities": len(current_test),
                "overlap_with_legacy_train": counts["train"],
                "overlap_with_legacy_validation": counts["validation"],
                "overlap_with_legacy_test": counts["test"],
                "strictly_untouched_for_legacy_checkpoint": counts["test"] == len(current_test),
                "legacy_indices_sha256": sha256(old_path),
            })

    compatible_by_current_fold = {
        str(fold_id): [
            row["legacy_checkpoint_fold"]
            for row in rows
            if row["current_outer_fold"] == fold_id
            and row["strictly_untouched_for_legacy_checkpoint"]
        ]
        for fold_id in range(5)
    }
    return {
        "schema": "drugscreenlab.exp036-source-cv-compatibility.v1",
        "current_split": str(split_path),
        "current_split_sha256": sha256(split_path),
        "legacy_checkpoint_index_root": str(legacy_root),
        "n_source_identities": len(current_universe),
        "current_outer_test_counts": {str(key): value for key, value in current_test_counts.items()},
        "fold_comparisons": rows,
        "strictly_compatible_legacy_checkpoints_by_current_outer_fold": compatible_by_current_fold,
        "all_folds_have_strict_legacy_checkpoint": all(compatible_by_current_fold.values()),
        "interpretation": (
            "A legacy checkpoint is eligible for strict current-fold evaluation only when every current outer-test "
            "identity was held out from both its legacy training and validation sets."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    result = audit_compatibility(args.split, args.legacy_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "current_split_sha256": result["current_split_sha256"],
        "strict_compatible_checkpoint_counts": {
            fold: len(checkpoints)
            for fold, checkpoints in result["strictly_compatible_legacy_checkpoints_by_current_outer_fold"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
