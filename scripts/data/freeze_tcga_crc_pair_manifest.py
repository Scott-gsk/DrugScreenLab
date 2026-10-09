"""Freeze one public STAR-count tumor/normal aliquot pair per TCGA CRC case."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(inventory_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"pair manifest output already exists: {output_dir}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    options = defaultdict(lambda: defaultdict(list))
    for item in inventory:
        if item["identity_status"] != "LINKED" or item["access"] != "open":
            continue
        if item["sample_type"] in {"Primary Tumor", "Solid Tissue Normal"}:
            options[item["case_id"]][item["sample_type"]].append(item)
    pairs = []
    for case, types in sorted(options.items()):
        if not {"Primary Tumor", "Solid Tissue Normal"} <= set(types):
            continue
        selected = {}
        for role in ("Primary Tumor", "Solid Tissue Normal"):
            candidates = sorted(types[role], key=lambda item: (
                item["sample_id"], item["aliquot_id"], item["file_id"]))
            selected[role] = candidates[0]
        if selected["Primary Tumor"]["sample_id"] == selected["Solid Tissue Normal"]["sample_id"]:
            raise ValueError("tumor and normal share a biospecimen identity")
        pairs.append({"case_id": case, "project_id": selected["Primary Tumor"]["project_id"],
                      "tumor": selected["Primary Tumor"], "normal": selected["Solid Tissue Normal"],
                      "n_tumor_file_options": len(types["Primary Tumor"]),
                      "n_normal_file_options": len(types["Solid Tissue Normal"])})
    if len(pairs) < 20:
        raise ValueError("too few public and unambiguously linked paired cases")
    file_ids = [part["file_id"] for pair in pairs for part in (pair["tumor"], pair["normal"])]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("same GDC file selected for multiple case roles")
    output_dir.mkdir(parents=True)
    pairs_path = output_dir / "selected_pairs.json"
    pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"experiment_id": "EXP-038", "status": "PAIRED_GDC_FILES_FROZEN_BEFORE_EXPRESSION_DOWNLOAD",
              "source_inventory_sha256": sha256(inventory_path),
              "selection_rule": "For each case with both Primary Tumor and Solid Tissue Normal, choose the lexicographically first sample ID, then aliquot ID, then file ID within each role; no expression or outcome used.",
              "n_cases": len(pairs), "n_files": len(file_ids),
              "download_bytes_expected": sum(int(part["file_size"]) for pair in pairs
                                             for part in (pair["tumor"], pair["normal"])),
              "selected_pairs_sha256": sha256(pairs_path)}
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.inventory, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
