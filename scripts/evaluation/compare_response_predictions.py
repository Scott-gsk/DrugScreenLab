"""从正式逐键预测文件重算两臂的患者内药物排序增量。

输入为只读 parquet；每次比较必须显式指定两臂及目标标签档位。
输出保留数据哈希、逐种子与患者级分数，供另一实现独立复算。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.exp031_eval import (
    aggregate_seed_patient_scores,
    assert_comparable,
    patient_score_paired_bootstrap,
    run_formal_comparison,
)
from drug_screen.evaluation.permutation import permutation_p_value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _selection(frame: pd.DataFrame, arm: str, fraction: float) -> pd.DataFrame:
    selected = frame.loc[
        frame["arm"].eq(arm) & frame["frac"].eq(fraction)
        & frame["evaluation_role"].eq("test")
    ].copy()
    if selected.empty:
        raise ValueError(f"no test predictions for arm={arm!r}, fraction={fraction}")
    if selected[["patient_id", "sample_id", "drug_id", "seed", "fold"]].isna().any().any():
        raise ValueError("identity has missing values")
    key = ["fold", "seed", "sample_id", "drug_id"]
    if selected.duplicated(key).any():
        raise ValueError(f"duplicate prediction key: {key}")
    selected["observed"] = selected["y_true"]
    selected["predicted"] = selected["y_pred"]
    return selected


def compare(
    predictions: Path, arm_a: str, arm_b: str, fraction: float,
    *, n_bootstrap: int = 2000, n_permutation: int = 10000,
    random_seed: int = 20260922,
) -> dict:
    if arm_a == arm_b:
        raise ValueError("comparison arms must differ")
    frame = pd.read_parquet(predictions)
    required = {"arm", "frac", "evaluation_role", "seed", "fold", "patient_id",
                "sample_id", "drug_id", "y_true", "y_pred", "experiment_id",
                "contract_id", "split_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction file lacks columns: {sorted(missing)}")
    a = _selection(frame, arm_a, fraction)
    b = _selection(frame, arm_b, fraction)
    seeds_a, seeds_b = set(a["seed"]), set(b["seed"])
    if seeds_a != seeds_b:
        raise ValueError(f"seed sets differ: {sorted(seeds_a)} vs {sorted(seeds_b)}")
    seeds = sorted(seeds_a)
    per_seed: list[dict] = []
    for seed in seeds:
        left = a.loc[a["seed"].eq(seed)]
        right = b.loc[b["seed"].eq(seed)]
        assert_comparable(left, right)
        result = run_formal_comparison(
            left, right, comparison=f"{arm_a}-minus-{arm_b}",
            n_bootstrap=n_bootstrap, seed=random_seed,
        )
        per_seed.append({"seed": int(seed), "gain": result.point_estimate,
                         "ci95": [result.low, result.high],
                         "n_patients": result.n_patients})

    agg_a = aggregate_seed_patient_scores(a)
    agg_b = aggregate_seed_patient_scores(b)
    eligible = (
        agg_a.seed_mean_patient_scores.notna()
        & agg_b.seed_mean_patient_scores.notna()
        & agg_a.n_seeds_used.eq(len(seeds))
        & agg_b.n_seeds_used.eq(len(seeds))
    )
    included = sorted(eligible.index[eligible])
    if not included:
        raise ValueError("no patient has a defined paired score in every seed")
    scores_a = agg_a.seed_mean_patient_scores.reindex(included)
    scores_b = agg_b.seed_mean_patient_scores.reindex(included)
    ci = patient_score_paired_bootstrap(
        scores_a, scores_b, n=n_bootstrap, seed=random_seed,
    )
    differences = scores_a.to_numpy(float) - scores_b.to_numpy(float)
    rng = np.random.default_rng(random_seed)
    null = (rng.choice([-1, 1], size=(n_permutation, len(differences)))
            * differences).mean(axis=1)
    p_two_sided = permutation_p_value(abs(ci.point_estimate), np.abs(null))
    patient_scores = [
        {"patient_id": patient, arm_a: float(scores_a[patient]),
         arm_b: float(scores_b[patient]), "gain": float(scores_a[patient] - scores_b[patient])}
        for patient in included
    ]
    return {
        "source": str(predictions.resolve()), "source_sha256": _sha256(predictions),
        "experiment_id": sorted(frame["experiment_id"].dropna().astype(str).unique().tolist()),
        "contract_id": sorted(a["contract_id"].dropna().astype(str).unique().tolist()),
        "split_id": sorted(a["split_id"].dropna().astype(str).unique().tolist()),
        "comparison": {"arm_a": arm_a, "arm_b": arm_b, "fraction": fraction,
                       "metric": "患者内全药 Spearman；逐种子患者分数平均后患者等权配对增量",
                       "seeds": [int(s) for s in seeds]},
        "main": {"gain": ci.point_estimate, "ci95": [ci.low, ci.high],
                 "paired_signflip_p_two_sided": p_two_sided,
                 "n_valid_patients": len(included),
                 "n_excluded_patients": int(len(eligible) - len(included)),
                 "bootstrap_repeats": n_bootstrap,
                 "permutation_repeats": n_permutation},
        "per_seed": per_seed,
        "patient_scores": patient_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.predictions, args.arm_a, args.arm_b, args.fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "main": result["main"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
