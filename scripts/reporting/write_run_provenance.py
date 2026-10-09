"""Record environment, input, and dirty-source identity for a formal local run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--command", action="append", required=True)
    parser.add_argument("--code", type=Path, action="append", required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"run provenance already exists: {args.output}")
    files = args.code + args.input
    if any(not file.is_file() for file in files):
        raise ValueError("every declared code and input path must exist as a regular file")
    receipt = {
        "experiment_id": args.experiment_id,
        "status": "RUN_PROVENANCE_CAPTURED",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": args.command,
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain=v1"),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--binary"])).hexdigest(),
        "code_sha256": {str(file): sha256(file) for file in args.code},
        "input_sha256": {str(file): sha256(file) for file in args.input},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": receipt["status"], "experiment_id": args.experiment_id,
                      "output": str(args.output), "code_files": len(args.code),
                      "input_files": len(args.input)}))


if __name__ == "__main__":
    main()
