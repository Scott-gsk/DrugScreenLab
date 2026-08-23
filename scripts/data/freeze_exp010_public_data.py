#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from drug_screen.data.exp010_public import ASSETS, download, freeze_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("DRUGSCREEN_DATA_ROOT", "data")) / "external" / "exp010")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/experiments/EXP-010/data_freeze_manifest.json"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        for asset in ASSETS:
            print(download(asset, args.root))
    payload = freeze_manifest(args.root, args.manifest)
    missing = [a["filename"] for a in payload["assets"] if a["required"] and not a["present"]]
    invalid = [a["filename"] for a in payload["assets"] if a["present"] and a.get("archive_ok") is False]
    print(f"status={payload['data_status']} assets={len(payload['assets'])} missing={len(missing)} invalid={len(invalid)}")
    if missing:
        print("missing=" + ",".join(missing))
    if invalid:
        print("invalid=" + ",".join(invalid))


if __name__ == "__main__":
    main()
