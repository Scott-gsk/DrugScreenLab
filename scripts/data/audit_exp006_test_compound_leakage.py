"""Response-blind audit: EXP-006 frozen test compounds vs official XPert train.

Compares pert_id identities only. Does not read PRISM, Δ978 labels, or
response values. The champion is the official warm-split checkpoint, so
the relevant official train partition is ``split_1 == train``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.foundation.exp006_transfer import write_json  # noqa: E402


XPERT_H5AD = ROOT / "data" / "external" / "xpert_source" / "processed_data" / "l1000_sdst_78453.h5ad"
SPLITS = ROOT / "artifacts" / "experiments" / "EXP-006" / "compound_splits.json"
OUTPUT = ROOT / "artifacts" / "experiments" / "EXP-006" / "TEST_COMPOUND_LEAKAGE_AUDIT.json"


def _sha(values: list[str]) -> str:
    return sha256(("\n".join(sorted(values)) + "\n").encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpert-h5ad", type=Path, default=XPERT_H5AD)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--official-split-column", default="split_1")
    args = parser.parse_args()

    import anndata as ad

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    frozen_test = sorted({str(value) for value in splits["split"]["test"]})
    frozen_train = sorted({str(value) for value in splits["split"]["train"]})
    frozen_val = sorted({str(value) for value in splits["split"]["validation"]})
    if set(frozen_test) & set(frozen_train):
        raise ValueError("internal EXP-006 compound split is already leaking")

    data = ad.read_h5ad(args.xpert_h5ad, backed="r")
    obs = data.obs
    if args.official_split_column not in obs.columns:
        raise ValueError(f"official split column missing: {args.official_split_column}")
    official_labels = obs[args.official_split_column].astype(str)
    official_pert = obs["pert_id"].astype(str)
    by_role: dict[str, set[str]] = {}
    for label in sorted(set(official_labels)):
        by_role[str(label)] = set(official_pert.loc[official_labels.eq(label)].tolist())
    data.file.close()

    official_train = sorted(by_role.get("train", set()))
    official_valid = sorted(by_role.get("valid", set()) | by_role.get("validation", set()))
    official_test = sorted(by_role.get("test", set()))
    official_any = sorted(set().union(*by_role.values()) if by_role else set())

    overlap_train = sorted(set(frozen_test) & set(official_train))
    overlap_valid = sorted(set(frozen_test) & set(official_valid))
    overlap_test = sorted(set(frozen_test) & set(official_test))
    unseen = sorted(set(frozen_test) - set(official_any))
    overlap_rate = float(len(overlap_train) / len(frozen_test)) if frozen_test else 0.0

    if overlap_rate >= 0.5:
        claim_status = "FULL_CHEMICAL_CHAMPION_PLUS_FRACTIONAL_FT"
        interpretation = (
            "Most frozen EXP-006 test compounds already appear in the official "
            "XPert L1000 warm-split train partition. Model A is therefore not "
            "10%/20% chemical supervision from scratch; it is the full-chemical "
            "champion plus a small fine-tune on a response-blind compound subset. "
            "A vs B remains a valid comparison of additional genetic pretraining "
            "under the same FT budget, but must not be claimed as low-data "
            "chemical training from random initialization."
        )
    elif overlap_rate > 0:
        claim_status = "PARTIAL_OFFICIAL_TRAIN_OVERLAP"
        interpretation = (
            "Some frozen test compounds appear in official XPert train. Report "
            "the overlap rate and do not describe the chemical-only arm as a "
            "clean from-scratch low-data regime."
        )
    else:
        claim_status = "NO_OFFICIAL_TRAIN_OVERLAP"
        interpretation = (
            "Frozen EXP-006 test compounds are absent from official XPert train. "
            "A still inherits the champion weights, so chemical knowledge may "
            "transfer via the pretrained backbone even without these test IDs."
        )

    payload = {
        "format": "exp006_test_compound_leakage_audit_v1",
        "status": claim_status,
        "response_values_read": False,
        "prism_values_read": False,
        "selection_used_official_split": False,
        "official_asset": str(args.xpert_h5ad),
        "official_split_column": args.official_split_column,
        "official_split_semantics": "warm split used by champion l1000_sdst_warm_split.pth",
        "frozen_exp006_split": {
            "train_unique_compounds": len(frozen_train),
            "validation_unique_compounds": len(frozen_val),
            "test_unique_compounds": len(frozen_test),
            "test_sha256": _sha(frozen_test),
        },
        "official_unique_compounds": {
            role: len(values) for role, values in {
                "train": official_train,
                "valid": official_valid,
                "test": official_test,
                "any": official_any,
            }.items()
        },
        "overlap": {
            "frozen_test_in_official_train": len(overlap_train),
            "frozen_test_in_official_valid": len(overlap_valid),
            "frozen_test_in_official_test": len(overlap_test),
            "frozen_test_unseen_in_official_h5ad": len(unseen),
            "frozen_test_in_official_train_rate": overlap_rate,
            "frozen_test_in_official_train_sha256": _sha(overlap_train),
        },
        "champion_initialization": "both A and B load official warm-split champion before FT",
        "interpretation": interpretation,
        "claim_allowed": (
            "compare genetic pretraining vs chemical-only FT of the same champion"
        ),
        "claim_forbidden": (
            "A is 10%/20% chemical supervision trained from scratch on a cold compound split"
            if overlap_rate > 0
            else None
        ),
    }
    write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "overlap_rate": overlap_rate,
        "overlap_count": len(overlap_train),
        "frozen_test": len(frozen_test),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
