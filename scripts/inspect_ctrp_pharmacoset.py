"""Materialize and audit the locally registered CTRPv2 PharmacoSet RDS."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from drug_screen.data.ctrp_pharmacoset import audit_ctrp_pharmacoset
from drug_screen.data.root import data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else data_root()
    output = args.output or root / "artifacts/EXP-014/ctrp_audit.json"
    audit = audit_ctrp_pharmacoset(root / "raw/exp013/ctrp_v2/CTRPv2.rds", args.export_dir or root / "processed/transfer/ctrp_v2")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "checksum_ok": audit["registry_checksum_matches"], "export": audit["processed_export"]}, ensure_ascii=False))
    return 0 if audit["registry_checksum_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
