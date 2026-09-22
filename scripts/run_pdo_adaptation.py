"""EXP-017 commands: common-space source preparation and PDO adaptation.

``prepare-common-source`` accepts target metadata only to freeze the labelled
gene universe.  It never opens target response labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.transfer.cohorts import load_registered_cohort, sha256
from drug_screen.transfer.pdo_adaptation import run_pdo_adaptation, train_common_space_source_checkpoint
from drug_screen.transfer.transfer_experiment import _target_receipt


def _code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-adaptation", help="run the approved nested PDO protocol")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--source-output", required=True, type=Path)
    run.add_argument("--data-root", required=True, type=Path)
    run.add_argument("--registry", required=True, type=Path)
    run.add_argument("--target-admission", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    prepare = commands.add_parser("prepare-common-source", help="retrain source model in source∩target unlabelled gene IDs")
    prepare.add_argument("--source-cohort", required=True)
    prepare.add_argument("--source-registry", required=True, type=Path)
    prepare.add_argument("--target-cohort", required=True)
    prepare.add_argument("--target-registry", required=True, type=Path)
    prepare.add_argument("--target-admission", required=True, type=Path)
    prepare.add_argument("--data-root", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--seed", required=True, type=int)
    prepare.add_argument("--epochs", type=int, default=50)
    prepare.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser


def main(argv: list[str] | None = None, *, loader=load_registered_cohort) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-adaptation":
        config=json.loads(args.config.read_text(encoding="utf-8"))
        admission=_target_receipt(args.target_admission, config["cohort_id"], args.registry)
        result = run_pdo_adaptation(json.loads(args.config.read_text(encoding="utf-8")), source_output=args.source_output,
                                    data_root=args.data_root, output_dir=args.output_dir, registry_path=args.registry)
        frozen=args.output_dir/"frozen_config.json"; data=json.loads(frozen.read_text()); data["target_admission"]=admission; data["target_admission_sha256"]=sha256(args.target_admission); data["runner_code_sha256"]=_code_hash(); frozen.write_text(json.dumps(data,indent=2))
        print(json.dumps(result, indent=2)); return 0
    # Admission binds the target registry before we read its gene identifiers.
    admission = _target_receipt(args.target_admission, args.target_cohort, args.target_registry)
    source = loader(args.source_cohort, root=args.data_root, registry=args.source_registry)
    target = loader(args.target_cohort, root=args.data_root, registry=args.target_registry)
    receipt = train_common_space_source_checkpoint(source, target.gene_ids, output_dir=args.output_dir,
                                                    seed=args.seed, epochs=args.epochs, device=args.device)
    provenance = {"command": "prepare-common-source", "source_cohort": args.source_cohort,
                  "target_cohort": args.target_cohort, "source_registry_sha256": sha256(args.source_registry),
                  "target_registry_sha256": sha256(args.target_registry), "target_admission": admission,
                  "target_admission_sha256": sha256(args.target_admission), "code_sha256": _code_hash(),
                  "target_response_labels_consumed": False, "source_receipt": receipt}
    (args.output_dir / "cli_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
