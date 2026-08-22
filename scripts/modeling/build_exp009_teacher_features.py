#!/usr/bin/env python3
"""Generate read-only EXP-009 Morgan soft-target features for XPert/SDST drugs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.exp009_teacher_adapter import (
    build_teacher_soft_target_features,
    write_teacher_soft_target_features,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts/experiments/EXP-009/teacher_morgan_probe_100k/bindingdb_teacher_morgan_probe.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-009/teacher_morgan_probe_100k/xpert_sdst_features"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    payload = build_teacher_soft_target_features(args.registry, args.checkpoint, batch_size=args.batch_size)
    audit = write_teacher_soft_target_features(payload, args.output_dir)
    print(
        json.dumps(
            {
                "artifact": audit["artifact"],
                "artifact_sha256": audit["artifact_sha256"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "eligible_drug_count": audit["eligible_drug_count"],
                "valid_feature_count": audit["valid_feature_count"],
                "invalid_structure_count": audit["invalid_structure_count"],
                "feature_shape": audit["feature_shape"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
