#!/usr/bin/env python3
"""Work-package B task 2: rebuild structural-drug labels per TRAIN fold only.

The old ``kryzei_structural_drugs_FROZEN.json`` was computed over the GLOBAL
(52-patient / 91-PDO, unfiltered) B0 rows. This script recomputes the same rule
(≥80% of PDOs with in-PDO rank fixed in the true top-5 or bottom-5, ranking over
the 14 GDSC-overlap "known" drugs only) but restricted, for each outer fold, to
that fold's *train* PDO rows (whitelist 60 samples). The FROZEN global file is
left untouched.

Outputs:
- data/processed/identity/kryzei_structural_drugs_TRAINONLY_v1.json
- data/processed/identity/kryzei_structural_drugs_TRAINONLY_v1_detail.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
GLOBAL = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
IDENTITY = ROOT / "data/processed/identity/kryeziu_patient_identity_FROZEN.json"
DRUG48 = ROOT / "data/processed/identity/kryeziu_drug_identity_48.json"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
SAMPLES = ROOT / "data/processed/transfer/mcrc_v3/samples.parquet"
PAIRS = ROOT / "data/processed/transfer/mcrc_v3/pairs.parquet"
OUT = ROOT / "data/processed/identity/kryzei_structural_drugs_TRAINONLY_v1.json"
OUT_DETAIL = ROOT / "data/processed/identity/kryzei_structural_drugs_TRAINONLY_v1_detail.parquet"

THRESH = 0.8


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def classify(known: pd.DataFrame) -> dict[str, dict]:
    """Replicate the frozen rule on the given B0-known rows."""
    known = known.copy()
    known["rank_in_pdo"] = known.groupby("sample_id")["response"].rank(ascending=False, method="min")
    known["n_in_pdo"] = known.groupby("sample_id")["drug_id"].transform("size")
    known["is_top5"] = known.rank_in_pdo <= 5
    known["is_bot5"] = known.rank_in_pdo > (known.n_in_pdo - 5)
    stats = known.groupby("drug_id").agg(
        n=("response", "size"),
        mean_rank=("rank_in_pdo", "mean"),
    )
    stats["f_top5"] = known.groupby("drug_id")["is_top5"].mean()
    stats["f_bot5"] = known.groupby("drug_id")["is_bot5"].mean()
    pan_eff = sorted(stats.index[stats.f_top5 >= THRESH].tolist())
    pan_res = sorted(stats.index[stats.f_bot5 >= THRESH].tolist())
    nonstructural = sorted(stats.index[(stats.f_top5 < THRESH) & (stats.f_bot5 < THRESH)].tolist())
    per_drug = {
        d: {
            "n": int(row["n"]),
            "mean_rank": round(float(row["mean_rank"]), 4),
            "f_top5": round(float(row["f_top5"]), 4),
            "f_bot5": round(float(row["f_bot5"]), 4),
        }
        for d, row in stats.round(4).to_dict("index").items()
    }
    return {"pan_effective": pan_eff, "pan_resistant": pan_res, "nonstructural": nonstructural, "per_drug": per_drug}


def main() -> int:
    glob = json.loads(GLOBAL.read_text(encoding="utf-8"))
    known_drugs = sorted(glob["per_drug_stats"].keys())
    g_eff = set(glob["pan_effective"])
    g_res = set(glob["pan_resistant"])
    g_non = set(glob["nonstructural"])

    ident = json.loads(IDENTITY.read_text(encoding="utf-8"))
    whitelist_samples = set(
        r["sample_id"] for r in ident["per_sample"]
        if r["identity_status"] == "S1_PDO_patient_verified"
    )

    drug48 = json.loads(DRUG48.read_text(encoding="utf-8"))
    name_by_id = {d["drug_id"]: d["original_name"] for d in drug48["drugs"]}

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    samples = pd.read_parquet(SAMPLES)
    pairs = pd.read_parquet(PAIRS)
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    pairs = pairs[pairs.sample_id.isin(whitelist_samples)]
    pairs = pairs[pairs.drug_id.isin(set(known_drugs))].copy()

    folds_out = {}
    detail_rows = []
    all_eff = set()
    all_res = set()
    for fold in split["folds"]:
        fid = fold["fold"]
        train_samples = set(map(str, fold["train_sample_ids"]))
        rows = pairs[pairs.sample_id.isin(train_samples)]
        res = classify(rows)
        folds_out[fid] = {
            "pan_effective": res["pan_effective"],
            "pan_resistant": res["pan_resistant"],
            "nonstructural": res["nonstructural"],
            "n_train_samples": len(train_samples),
            "n_known_rows": int(len(rows)),
            "n_pdos": int(rows.sample_id.nunique()),
            "n_drugs": int(rows.drug_id.nunique()),
            "per_drug": res["per_drug"],
        }
        all_eff.update(res["pan_effective"])
        all_res.update(res["pan_resistant"])

        eff = set(res["pan_effective"])
        resset = set(res["pan_resistant"])
        non = set(res["nonstructural"])
        for drug in known_drugs:
            if drug in eff:
                label = "pan_effective"
            elif drug in resset:
                label = "pan_resistant"
            else:
                label = "nonstructural"
            global_label = (
                "pan_effective" if drug in g_eff else
                "pan_resistant" if drug in g_res else "nonstructural"
            )
            stats = res["per_drug"].get(drug, {})
            detail_rows.append({
                "fold": int(fid),
                "drug_id": drug,
                "drug_name": name_by_id.get(drug, drug),
                "label_trainonly": label,
                "label_global": global_label,
                "changed_vs_global": label != global_label,
                "n_pdos": stats.get("n"),
                "mean_rank": stats.get("mean_rank"),
                "f_top5": stats.get("f_top5"),
                "f_bot5": stats.get("f_bot5"),
            })

    # comparison per fold vs old global lists
    comparison = {}
    for fid, block in folds_out.items():
        comparison[fid] = {
            "pan_effective_added": sorted(set(block["pan_effective"]) - g_eff),
            "pan_effective_removed": sorted(g_eff - set(block["pan_effective"])),
            "pan_resistant_added": sorted(set(block["pan_resistant"]) - g_res),
            "pan_resistant_removed": sorted(g_res - set(block["pan_resistant"])),
            "nonstructural_added": sorted(set(block["nonstructural"]) - g_non),
            "nonstructural_removed": sorted(g_non - set(block["nonstructural"])),
        }

    payload = {
        "name": "kryzei_structural_drugs_TRAINONLY",
        "status": "FROZEN",
        "version": "v1",
        "frozen_at": "2026-09-21",
        "derived_from": "train_folds_only",
        "rule": {
            "pan_effective": "fraction of TRAIN-fold PDOs with observed rank <= 5 >= 0.8",
            "pan_resistant": "fraction of TRAIN-fold PDOs with observed rank > n_in_pdo - 5 >= 0.8",
            "rank_method": "min (descending observed within PDO, 14 known drugs only)",
            "nonstructural": "neither condition met",
        },
        "source": {
            "global_frozen": rel(GLOBAL),
            "split": rel(SPLIT),
            "samples_parquet": rel(SAMPLES),
            "pairs_parquet": rel(PAIRS),
            "whitelist": "S1_PDO_patient_verified (60 samples)",
            "n_known_drugs": len(known_drugs),
            "known_drugs": known_drugs,
        },
        "folds": folds_out,
        "comparison_with_global": comparison,
        "provenance": {
            "generated_by": "scripts/data/rebuild_structural_drugs_trainonly.py",
            "note": "Per outer-fold TRAIN adjudication; global FROZEN file retained unchanged.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail = pd.DataFrame(detail_rows)
    detail.to_parquet(OUT_DETAIL, index=False)

    print("wrote", rel(OUT))
    print("wrote", rel(OUT_DETAIL))
    for fid, block in folds_out.items():
        print(f"fold {fid}: eff={len(block['pan_effective'])} res={len(block['pan_resistant'])} "
              f"non={len(block['nonstructural'])} n_pdos={block['n_pdos']} n_known_rows={block['n_known_rows']}")
    print("global eff:", g_eff)
    print("global res:", g_res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
