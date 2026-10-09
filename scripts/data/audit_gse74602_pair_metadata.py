"""核对 GSE74602 官方 SOFT 样本元数据与旧 30 对肿瘤/正常映射。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_samples(soft_path: Path) -> list[dict]:
    samples = []
    current = None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                if current is not None:
                    samples.append(current)
                current = {"gsm": line.split(" = ", 1)[1], "characteristics": []}
            elif current is not None and line.startswith("!Sample_title = "):
                current["title"] = line.split(" = ", 1)[1]
            elif current is not None and line.startswith("!Sample_source_name_ch1 = "):
                current["source_name"] = line.split(" = ", 1)[1]
            elif current is not None and line.startswith("!Sample_characteristics_ch1 = "):
                current["characteristics"].append(line.split(" = ", 1)[1])
    if current is not None:
        samples.append(current)
    return samples


def audit(soft_path: Path, legacy_pairs_path: Path) -> dict:
    samples = parse_samples(soft_path)
    if len(samples) != 60:
        raise ValueError(f"official SOFT has {len(samples)} samples; expected 60")
    if len({row["title"] for row in samples}) != 60:
        raise ValueError("duplicate SOFT sample titles")
    numbered: dict[tuple[str, int], dict] = {}
    for row in samples:
        name = row["source_name"]
        match = re.fullmatch(r"(?:Primary Tumor T|Normal N)(\d+)", name)
        if not match:
            raise ValueError(f"unrecognized official sample source name: {name}")
        role = "tumor" if name.startswith("Primary Tumor") else "normal"
        number = int(match.group(1))
        expected_tissue = "Tumor tissue" if role == "tumor" else "Normal non-cancerous tissue"
        if f"tissue type: {expected_tissue}" not in row["characteristics"]:
            raise ValueError(f"SOFT role and tissue type disagree: {row['title']}")
        key = (role, number)
        if key in numbered:
            raise ValueError(f"duplicate official {key}")
        numbered[key] = row
    tumor_numbers = {number for role, number in numbered if role == "tumor"}
    normal_numbers = {number for role, number in numbered if role == "normal"}
    if tumor_numbers != normal_numbers or len(tumor_numbers) != 30:
        raise ValueError("official T-number/N-number pairing is incomplete")
    pairs = []
    for number in sorted(tumor_numbers):
        tumor = numbered[("tumor", number)]
        normal = numbered[("normal", number)]
        pairs.append({"pair_number": number, "tumor_gsm": tumor["gsm"],
                      "normal_gsm": normal["gsm"],
                      "tumor_sample": tumor["title"],
                      "normal_sample": normal["title"]})
    legacy = json.loads(legacy_pairs_path.read_text(encoding="utf-8"))
    legacy_keys = {(row["tumor_sample"], row["normal_sample"]) for row in legacy}
    official_keys = {(row["tumor_sample"], row["normal_sample"]) for row in pairs}
    if legacy_keys != official_keys:
        raise ValueError("legacy pair map differs from official numbered source names")
    prefixes = {row["title"].rsplit("_", 1)[0] for row in samples}
    return {
        "status": "PASS_OFFICIAL_NUMBERED_TUMOR_NORMAL_PAIRING",
        "soft_sha256": sha256(soft_path),
        "legacy_pair_map_sha256": sha256(legacy_pairs_path),
        "n_samples": len(samples), "n_pairs": len(pairs),
        "n_shared_title_prefixes": len(prefixes),
        "identity_limit": "sample title prefix is a slide/accession prefix, not verified patient identity; Tn/Nn is an explicit sample pairing label",
        "source_limitation": "normal means adjacent non-cancerous tissue from CRC subjects; not healthy epithelium or normal PDO",
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--legacy-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.soft, args.legacy_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "n_samples", "n_pairs",
                                                "n_shared_title_prefixes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
