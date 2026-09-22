"""Fetch bounded public transfer-learning assets with an auditable receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.data.public_assets import AssetError, load_manifest, run_manifest, write_receipt
from drug_screen.data.root import data_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/data/transfer_public_assets.json"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("dry-run", "fetch", "validate"), default="dry-run")
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--keys", nargs='+', help='Fetch/validate only these manifest keys')
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        if args.keys:
            unknown = set(args.keys) - {row['key'] for row in manifest['assets']}
            if unknown: raise AssetError(f'unknown asset keys: {sorted(unknown)}')
            manifest = dict(manifest, assets=[row for row in manifest['assets'] if row['key'] in args.keys])
        root = args.data_root.resolve() if args.data_root else data_root()
        receipt = args.receipt or root / "artifacts" / "EXP-014" / f"transfer_public_assets_{args.mode}.json"
        results = run_manifest(manifest, root, args.mode)
        write_receipt(receipt, args.manifest, args.mode, results)
    except AssetError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps({"receipt": str(receipt), "results": results}, indent=2, ensure_ascii=False))
    return 1 if any(row["status"] == "error" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
