# EXP-026: 固化 Kryzei 结构性极端药物清单（FROZEN）
# 规则（预登记于 experiments/records/EXP-026.md）：
#   f_top5 >= 0.8  -> structural pan-effective（≥80% PDO 进入真 top-5）
#   f_bot5 >= 0.8  -> structural pan-resistant（≥80% PDO 处于末 5 位）
# 输入：官方迁移 run 的 B0 预测 parquet（观测值与第一波完全一致）
# 输出：data/processed/identity/kryzei_structural_drugs_FROZEN.json
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
RUN = (ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
       / "run_b2000_20260919_20260919_121930")
OUT = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
THRESH = 0.8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parquet = RUN / "outer_oof_predictions.parquet"
    df = pd.read_parquet(parquet)
    b0 = df[df.algorithm.str.startswith("B0")]
    known = b0[b0.source_drug_status == "known"].copy()

    known["rank_in_pdo"] = known.groupby("model_id")["observed"].rank(
        ascending=False, method="min")
    known["n_in_pdo"] = known.groupby("model_id")["drug_id"].transform("size")
    known["is_top5"] = known.rank_in_pdo <= 5
    known["is_bot5"] = known.rank_in_pdo > (known.n_in_pdo - 5)

    stats = known.groupby("drug_id").agg(
        n=("observed", "size"),
        mean_obs=("observed", "mean"),
        std_obs=("observed", "std"),
        mean_rank=("rank_in_pdo", "mean"),
        std_rank=("rank_in_pdo", "std"),
    )
    stats["f_top5"] = known.groupby("drug_id")["is_top5"].mean()
    stats["f_bot5"] = known.groupby("drug_id")["is_bot5"].mean()

    pan_eff = stats.index[stats.f_top5 >= THRESH].tolist()
    pan_res = stats.index[stats.f_bot5 >= THRESH].tolist()
    nonstructural = stats.index[
        (stats.f_top5 < THRESH) & (stats.f_bot5 < THRESH)].tolist()

    per_drug = {
        d: {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating))
                else v)
            for k, v in row.items()}
        for d, row in stats.round(4).to_dict("index").items()
    }

    obj = {
        "name": "kryzei_structural_drugs",
        "status": "FROZEN",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "EXP-026",
        "rule": {
            "pan_effective": "fraction of PDOs with observed rank <= 5 >= 0.8",
            "pan_resistant": "fraction of PDOs with observed rank > n_in_pdo - 5 >= 0.8",
            "rank_method": "min (descending observed within PDO, known drugs only)",
            "nonstructural": "neither condition met",
        },
        "data_source": {
            "parquet": str(parquet.relative_to(ROOT)),
            "sha256": sha256(parquet),
            "known_rows": int(len(known)),
            "n_patients": int(known.patient_id.nunique()),
            "n_pdos": int(known.model_id.nunique()),
            "n_drugs": int(known.drug_id.nunique()),
        },
        "counts": {
            "pan_effective": len(pan_eff),
            "pan_resistant": len(pan_res),
            "nonstructural": len(nonstructural),
        },
        "pan_effective": pan_eff,
        "pan_resistant": pan_res,
        "nonstructural": nonstructural,
        "per_drug_stats": per_drug,
        "usage_note": (
            "EXP-026 预登记评价改革：非结构药子集 = nonstructural 清单；"
            "残差化口径 = observed - 训练折药物均值。全面板 NDCG@5 仅作参照列。"
        ),
        "provenance": {
            "generated_by": "scripts/data/freeze_kryzei_structural_drugs.py",
            "audit_basis": "tmp/audit_b0_structure_20260919.py (2026-09-19 三轮数据审计)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    print("FROZEN ->", OUT)
    print("pan_effective:", pan_eff)
    print("pan_resistant:", pan_res)
    print("nonstructural:", nonstructural)
    print("per-drug n rows:", {d: per_drug[d]["n"] for d in per_drug})


if __name__ == "__main__":
    main()
