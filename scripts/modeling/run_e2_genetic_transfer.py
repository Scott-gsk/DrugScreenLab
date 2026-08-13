"""Run the bounded E2 genetic-pretraining versus chemical-only probe."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from drug_screen.modeling.genetic_transfer import (
    UnifiedResponseRecord,
    fit_transfer_probe,
    group_atomic_subset,
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_records(payload: dict, *, genetic: bool) -> tuple[UnifiedResponseRecord, ...]:
    result = []
    for row in payload["records"]:
        if genetic:
            result.append(UnifiedResponseRecord.from_mapping(row))
        else:
            result.append(
                UnifiedResponseRecord.from_mapping(
                    {
                        "sample_id": row["sample_id"],
                        "treatment_group_id": row["treatment_group_id"],
                        "perturbagen_id": row["drug_id"],
                        "modality": "chemical",
                        "perturbation_direction": "small_molecule",
                        "context_id": row["context_id"],
                        "dose_um": row["dose_um"],
                        "time_h": row["time_h"],
                        "split": row["split"],
                        "treatment_cache_row": row["treatment_cache_row"],
                        "control_cache_row": row["control_cache_row"],
                        "perturbagen_feature_row": row["chemical_feature_row"],
                    }
                )
            )
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genetic-manifest", type=Path, required=True)
    parser.add_argument("--chemical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.2, 0.1])
    parser.add_argument("--max-chemical-train-groups", type=int, default=4000)
    parser.add_argument("--max-genetic-train-groups", type=int, default=2647)
    parser.add_argument("--genetic-epochs", type=int, default=5)
    parser.add_argument("--chemical-epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.fractions or any(value <= 0 or value > 1 for value in args.fractions):
        raise ValueError("fractions must be in (0, 1]")
    if args.max_chemical_train_groups < 1 or args.max_genetic_train_groups < 1:
        raise ValueError("group caps must be positive")

    repo_root = args.repo_root.resolve()
    genetic_manifest_path = args.genetic_manifest.resolve()
    chemical_manifest_path = args.chemical_manifest.resolve()
    genetic_payload = json.loads(genetic_manifest_path.read_text(encoding="utf-8"))
    chemical_payload = json.loads(chemical_manifest_path.read_text(encoding="utf-8"))
    cache_path = repo_root / genetic_payload["cache"]["relative_path"]
    features_path = repo_root / genetic_payload["perturbagen_features"]["relative_path"]
    cache = np.load(cache_path, mmap_mode="r")
    perturbagen_features = np.load(features_path, mmap_mode="r")

    genetic_records = tuple(
        row for row in load_records(genetic_payload, genetic=True) if row.split == "train"
    )
    chemical_records = load_records(chemical_payload, genetic=False)
    chemical_train_pool = tuple(row for row in chemical_records if row.split == "train")
    chemical_test = tuple(row for row in chemical_records if row.split == "test")
    if len({row.treatment_group_id for row in chemical_test}) < 1:
        raise ValueError("frozen chemical test set is empty")

    genetic_records = group_atomic_subset(
        genetic_records,
        fraction=min(1.0, args.max_genetic_train_groups / len({row.treatment_group_id for row in genetic_records})),
        seed=args.seed + 1,
    )
    chemical_train_pool = group_atomic_subset(
        chemical_train_pool,
        fraction=min(1.0, args.max_chemical_train_groups / len({row.treatment_group_id for row in chemical_train_pool})),
        seed=args.seed + 2,
    )

    results = {}
    for offset, fraction in enumerate(args.fractions):
        result = fit_transfer_probe(
            genetic_records=genetic_records,
            chemical_train_records=chemical_train_pool,
            chemical_test_records=chemical_test,
            cache=cache,
            perturbagen_features=perturbagen_features,
            chemical_fraction=fraction,
            genetic_epochs=args.genetic_epochs,
            chemical_epochs=args.chemical_epochs,
            hidden_dim=args.hidden_dim,
            seed=args.seed + offset,
            device=args.device,
        )
        baseline = result["models"]["chemical_only"]["test_metrics"]
        transfer = result["models"]["genetic_pretrain_then_chemical"]["test_metrics"]
        result["primary_endpoint"] = {
            "name": "chemical_test_spearman_gain_transfer_minus_chemical_only",
            "value": (
                transfer["spearman"] - baseline["spearman"]
                if transfer["spearman"] is not None and baseline["spearman"] is not None
                else None
            ),
            "interpretation": "FAST exploratory; positive means genetic pretraining improved frozen chemical test ranking",
        }
        results[str(fraction)] = result

    output = {
        "format": "e2_genetic_chemical_transfer_fast_result_v1",
        "status": "FAST_RESULT_READY",
        "hypothesis": "genetic supervision reduces chemical supervision required for Delta978 prediction",
        "chemical_test_frozen": True,
        "chemical_test_manifest": str(chemical_manifest_path),
        "chemical_test_manifest_sha256": digest(chemical_manifest_path),
        "chemical_train_pool": {
            "source_manifest": str(chemical_manifest_path),
            "bounded_group_cap": args.max_chemical_train_groups,
            "groups": len({row.treatment_group_id for row in chemical_train_pool}),
            "records": len(chemical_train_pool),
        },
        "genetic_pretraining_pool": {
            "source_manifest": str(genetic_manifest_path),
            "source_manifest_sha256": digest(genetic_manifest_path),
            "bounded_group_cap": args.max_genetic_train_groups,
            "groups": len({row.treatment_group_id for row in genetic_records}),
            "records": len(genetic_records),
        },
        "cache": {"path": str(cache_path), "sha256": digest(cache_path), "shape": list(cache.shape)},
        "perturbagen_features": {
            "path": str(features_path),
            "sha256": digest(features_path),
            "shape": list(perturbagen_features.shape),
        },
        "config": {
            "fractions": [float(value) for value in args.fractions],
            "genetic_epochs": args.genetic_epochs,
            "chemical_epochs": args.chemical_epochs,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
            "device": args.device,
            "normalization_fit_scope": "chemical_train_subset_only",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
