"""Build EXP-009 Morgan teacher soft-target features for XPert identities."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from drug_screen.foundation.exp009_teacher_adapter import (
    build_teacher_soft_target_features,
    write_teacher_soft_target_features,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
DEFAULT_CHECKPOINT = ROOT / "artifacts/experiments/EXP-009/teacher_morgan_probe_100k/bindingdb_teacher_morgan_probe.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-009/sdst_soft_target_features"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-drugs", type=int, default=None)
    args = parser.parse_args()
    payload = build_teacher_soft_target_features(
        args.registry, args.checkpoint, batch_size=args.batch_size, max_drugs=args.max_drugs
    )
    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        audit = write_teacher_soft_target_features(payload, tmp)
        audit["publication"] = "partial_directory_then_atomic_rename"
        if output.exists():
            backup = output.with_name(output.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(output, backup)
        os.replace(tmp, output)
        previous = output.with_name(output.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        artifact = output / "xpert_sdst_soft_target_features.npz"
        audit["artifact"] = str(artifact)
        audit["artifact_sha256"] = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
        (output / "identity_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
