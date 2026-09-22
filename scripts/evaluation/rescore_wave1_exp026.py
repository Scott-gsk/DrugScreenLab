# EXP-026: 第一波已落盘预测在新分层终点下的重打分（不训练任何模型）
# 输入（只读）：
#   - EXP-021 官方迁移 run_b2000_*（3 真种子 + 1 误标重复轮，共 4 个 parquet）
#   - EXP-024 增益波1 变体（multiseed/b0_residual/pathway_pca_best_reg/
#     inner_val_ensemble/transfer_calib_b0，各 3 种子）
#   - EXP-025 joint_predictions.parquet（联合臂，单轮）
# 验证（预登记失败条件）：
#   官方 B0 全面板患者宏平均 NDCG@5 必须复现第一波官方数字 0.9089（容差 1e-6），
#   否则中止并不发布对比表。
# 输出：
#   - artifacts/exp026_rescore/rescore_summary.json
#   - docs/reports/exp026_rescore_under_new_endpoints.md
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.stratified_metrics import (
    evaluate_stratified,
    load_structural_drugs,
)

ROOT = Path("/mnt/d/Code/DrugScreenLab")
STRUCTURAL_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
OUT_DIR = ROOT / "artifacts/exp026_rescore"
REPORT = ROOT / "docs/reports/exp026_rescore_under_new_endpoints.md"
# 官方全面板 B0 参照取自每个 run 自身的 metrics.json（b2000 管线），
# 而非 EXP-020 Stage1 管线的 0.9089（不同管线版本，见报告口径说明）。
TOL = 1e-9
EXP020_STAGE1_B0 = 0.9089  # 仅作报告注记，不用于验证

RUNS = {
    "20260912": "run_b2000_20260918_174702",
    "20260913": "run_b2000_seed20260913_20260918_223357",
    "20260919": "run_b2000_20260919_20260919_121930",
}
MISLABELED = "run_b2000_seed2_20260918_223331"  # 实为 seed20260912 重复轮
GAINWAVE = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT"
JOINT = ROOT / "artifacts/joint_feasibility_EXP025_DRAFT/joint_predictions.parquet"


def normalize_official(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["algorithm"] = out["algorithm"].str.replace(r"_seed_\d+", "", regex=True)
    return out


def normalize_gainwave(df: pd.DataFrame, arm: str, seed: str,
                       known_drugs: frozenset[str]) -> pd.DataFrame:
    """增益波/联合 parquet → 标准长表；过滤到已知 14 药子集。

    注：这些 parquet 为 51 患者口径（与官方 run 52 患者为第一波已知的
    admission 口径遗留），故 B0 配对一律在同 parquet 内重算（见 add_b0_rows）。
    """
    sub = df[df["drug_id"].isin(known_drugs)].copy()
    out = pd.DataFrame({
        "patient_id": sub["patient_id"],
        "model_id": sub["sample_id"],
        "drug_id": sub["drug_id"],
        "fold": sub["fold"],
        "algorithm": arm,
        "observed": sub["y"],
        "predicted": sub["pred"],
        "source_drug_status": "known",
    })
    out.attrs["seed"] = seed
    return out


def add_b0_rows(df: pd.DataFrame) -> pd.DataFrame:
    """在同版本 cohort 内按折重算 B0（训练患者逐药均值）并追加为算法行。

    保证与该 parquet 相同的患者集合 / 折划分，消除跨版本（51 vs 52 患者）
    的配对偏移。
    """
    base = df[df["algorithm"] == df["algorithm"].iloc[0]]
    rows = []
    for fold in sorted(base["fold"].unique()):
        train = base[base["fold"] != fold]
        means = train.groupby("drug_id")["observed"].mean()
        test = base[base["fold"] == fold].copy()
        test["predicted"] = test["drug_id"].map(means)
        test["algorithm"] = "B0"
        rows.append(test)
    return pd.concat([df] + rows, ignore_index=True)


def patient_bootstrap(diff: dict[str, float], seed: int = 0,
                      resamples: int = 2000) -> dict | None:
    ids = sorted(diff)
    vals = [diff[i] for i in ids]
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    point = float(np.mean(finite))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        draw = rng.integers(0, len(finite), len(finite))
        draws.append(np.mean([finite[i] for i in draw]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high),
            "n": len(finite), "seed": seed, "resamples": resamples}


def per_patient_scores(known: pd.DataFrame, structural) -> dict[str, dict]:
    """逐患者三终点值（用于配对 bootstrap）。"""
    from drug_screen.evaluation.stratified_metrics import (
        full_panel_patient_values, nonstructural_patient_values)
    ns = nonstructural_patient_values(known, structural)
    fp = full_panel_patient_values(known)
    ns_p = ns.rename("v").reset_index().groupby("level_0")["v"].mean()
    fp_p = fp.rename("v").reset_index().groupby("level_0")["v"].mean()
    return {"nonstructural": ns_p.to_dict(), "full_panel": fp_p.to_dict()}


def main() -> None:
    structural = load_structural_drugs(STRUCTURAL_PATH)
    known_set = structural.structural | structural.nonstructural
    results = []  # rows: source, arm, seed, ns_ndcg, pd_spearman, fp_ndcg
    b0_patient = None
    official_check = {}

    # --- EXP-021 官方迁移 runs ---
    official_reference = {}
    for seed, run in RUNS.items():
        run_dir = (ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
                   / run)
        mj = json.load(open(run_dir / "metrics.json"))
        b0_key = next(k for k in mj["metrics"] if k.startswith("B0"))
        official_reference[seed] = mj["metrics"][b0_key]["known_primary"][
            "patient_average"]["ndcg_at_5"]
        df = pd.read_parquet(run_dir / "outer_oof_predictions.parquet")
        df = normalize_official(df)
        for alg in sorted(df["algorithm"].unique()):
            summary, _, _ = evaluate_stratified(
                df, structural, algorithm=alg)
            results.append({
                "source": "EXP021_official", "arm": alg, "seed": seed,
                **{k: v for k, v in summary.as_dict().items()
                   if k != "algorithm"}})
            if alg == "B0":
                official_check[seed] = summary.full_panel_ndcg5_macro
                if b0_patient is None:
                    b0_patient = per_patient_scores(
                        df[(df.source_drug_status == "known")
                           & (df.algorithm == "B0")], structural)

    # 误标重复轮（不计入种子聚合，仅交叉验证）
    mdir = (ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
            / MISLABELED)
    mj = json.load(open(mdir / "metrics.json"))
    b0_key = next(k for k in mj["metrics"] if k.startswith("B0"))
    official_reference["mislabeled_repeat"] = mj["metrics"][b0_key][
        "known_primary"]["patient_average"]["ndcg_at_5"]
    df_m = pd.read_parquet(mdir / "outer_oof_predictions.parquet")
    df_m = normalize_official(df_m)
    s, _, _ = evaluate_stratified(df_m, structural, algorithm="B0")
    official_check["mislabeled_repeat"] = s.full_panel_ndcg5_macro

    # --- 验证：全面板 B0 必须逐 run 精确复现该 run 官方 metrics.json ---
    failed = {k: (official_check[k], official_reference[k])
              for k in official_reference
              if abs(official_check[k] - official_reference[k]) > TOL}
    if failed:
        raise SystemExit(
            f"[ABORT] full-panel B0 mismatch vs run metrics.json "
            f"(pre-registered failure): {failed}")

    # --- EXP-024 增益波1（过滤到已知 14 药；51 患者版本口径）---
    gain_arms = {}
    for sub in sorted(GAINWAVE.iterdir()):
        if not sub.is_dir():
            continue
        for pq in sorted(sub.glob("*_predictions.parquet")):
            name = pq.name.replace("_predictions.parquet", "")
            if "_seed" in name:
                parts = name.split("_seed")
                arm = parts[0].rsplit("_gpu", 1)[0]
                seed = parts[1].split("_")[0]
            else:  # 单轮臂（inner_val_ensemble / transfer_calib_b0）
                arm = name.rsplit("_gpu", 1)[0]
                seed = "single"
            df = pd.read_parquet(pq)
            gain_arms.setdefault(arm, []).append((seed, df))
    for arm, items in gain_arms.items():
        for seed, df in items:
            norm = normalize_gainwave(df, arm, seed, structural.structural
                                      | structural.nonstructural)
            summary, _, _ = evaluate_stratified(norm, structural, algorithm=arm)
            results.append({
                "source": "EXP024_gainwave", "arm": arm, "seed": seed,
                **{k: v for k, v in summary.as_dict().items()
                   if k != "algorithm"}})

    # --- EXP-025 联合臂（同上过滤）---
    jdf = pd.read_parquet(JOINT)
    jnorm = normalize_gainwave(jdf, "joint_EXP025", "single",
                               structural.structural | structural.nonstructural)
    summary, _, _ = evaluate_stratified(jnorm, structural, algorithm="joint_EXP025")
    results.append({
        "source": "EXP025_joint", "arm": "joint", "seed": "single",
        **{k: v for k, v in summary.as_dict().items() if k != "algorithm"}})

    frame = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_json(OUT_DIR / "rescore_summary.json", orient="records",
                  indent=2, force_ascii=False)

    # --- 聚合（多种子 mean±sd）---
    agg = (frame.groupby(["source", "arm"])
           .agg(ns_mean=("nonstructural_ndcg5_macro", "mean"),
                ns_sd=("nonstructural_ndcg5_macro", "std"),
                pd_mean=("per_drug_patient_spearman_macro", "mean"),
                pd_sd=("per_drug_patient_spearman_macro", "std"),
                fp_mean=("full_panel_ndcg5_macro", "mean"),
                fp_sd=("full_panel_ndcg5_macro", "std"),
                n_seeds=("seed", "nunique"))
           .reset_index().sort_values(["source", "ns_mean"],
                                      ascending=[True, False]))

    # --- 新口径下的关键配对 bootstrap（vs B0，患者级 2000 次）---
    boot = {}
    comparisons = {}
    for seed, run in RUNS.items():
        df = pd.read_parquet(
            ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
            / run / "outer_oof_predictions.parquet")
        df = normalize_official(df)
        for alg in ["M1", "M1_source_scaled", "M2", "M3"]:
            pat = per_patient_scores(
                df[(df.source_drug_status == "known") & (df.algorithm == alg)],
                structural)
            comparisons[(alg, seed)] = pat
    for alg in ["M1", "M1_source_scaled", "M2", "M3"]:
        # 三种子合并：患者级先种子平均
        merged = {}
        for pid in b0_patient["nonstructural"]:
            vals = [comparisons[(alg, s)]["nonstructural"].get(pid)
                    for s in RUNS]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                merged[pid] = float(np.mean(vals))
        diff = {pid: (merged.get(pid) - b0_patient["nonstructural"].get(pid))
                for pid in merged
                if b0_patient["nonstructural"].get(pid) is not None}
        boot[f"{alg}_vs_B0_nonstructural"] = patient_bootstrap(diff)

    # pathway_pca_best_reg vs B0（三种子；B0 在同 parquet 版本内重算，51 患者口径）
    pcr = {}
    pcr_b0 = {}
    for seed, df in gain_arms.get("pathway_pca_wd02", []):
        norm = normalize_gainwave(df, "pathway_pca_best_reg", seed,
                                   structural.structural | structural.nonstructural)
        with_b0 = add_b0_rows(norm)
        pat = per_patient_scores(
            with_b0[with_b0.algorithm == "pathway_pca_best_reg"], structural)
        pat_b0 = per_patient_scores(
            with_b0[with_b0.algorithm == "B0"], structural)
        for pid, v in pat["nonstructural"].items():
            pcr.setdefault(pid, []).append(v)
        for pid, v in pat_b0["nonstructural"].items():
            pcr_b0.setdefault(pid, []).append(v)
    if pcr:
        merged = {pid: float(np.mean(v)) for pid, v in pcr.items()}
        b0m = {pid: float(np.mean(v)) for pid, v in pcr_b0.items()}
        diff = {pid: merged[pid] - b0m[pid] for pid in merged
                if pid in b0m and b0m[pid] is not None}
        boot["pathway_pca_best_reg_vs_B0_nonstructural"] = patient_bootstrap(diff)

    (OUT_DIR / "bootstrap_new_endpoints.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False))

    # --- 报告 ---
    def fmt(x, sd=None):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        s = f"{x:.4f}"
        if sd is not None and not np.isnan(sd):
            s += f" ±{sd:.4f}"
        return s

    lines = [
        "# EXP-026 重打分：第一波预测在新分层终点下的对比（未训练任何新模型）",
        "",
        "- 生成：scripts/evaluation/rescore_wave1_exp026.py（EXP-026，只读重打分）",
        f"- 结构药清单：{STRUCTURAL_PATH.name}（7 结构 / 7 非结构）",
        "- 官方全面板 B0 验证：逐 run 精确复现该 run metrics.json 的"
        " known_primary.patient_average.ndcg_at_5（容差 1e-9）：",
        "  " + ", ".join(f"{k}={v:.6f}" for k, v in official_check.items()),
        f"- 口径注记：EXP-020 Stage1 管线的 B0={EXP020_STAGE1_B0} 与 b2000 "
        "官方迁移管线的 B0（0.8917~0.8932，逐 run 略异）来自不同 split 版本"
        "与聚合管线；本表参照列以后者（与重打分同管线）为准。",
        "",
        "## 一、三分层终点汇总（多种子 mean±SD；患者宏平均）",
        "",
        "| 来源 | 臂 | 非结构 NDCG@5 | 逐药跨患者 Spearman | 全面板 NDCG@5（参照） | 种子数 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['source']} | {r['arm']} | {fmt(r.ns_mean, r.ns_sd)} | "
            f"{fmt(r.pd_mean, r.pd_sd)} | {fmt(r.fp_mean, r.fp_sd)} | {int(r.n_seeds)} |")
    lines += [
        "",
        "## 二、新口径下的关键配对比较（患者级 2000 次 bootstrap，vs B0）",
        "",
        "| 比较（非结构 NDCG@5） | 点估计 | 95% CI | n 患者 |",
        "|---|---|---|---|",
    ]
    for key, b in boot.items():
        if b is None:
            lines.append(f"| {key} | 不可算 | — | — |")
        else:
            lines.append(
                f"| {key} | {b['point']:+.4f} | "
                f"[{b['low']:+.4f}, {b['high']:+.4f}] | {b['n']} |")
    lines += [
        "",
        "## 三、口径说明",
        "",
        "1. 非结构 NDCG@5：剔除 7 个结构药后每 PDO 5-7 药排序（资格 ≥5），",
        "   B0 在此口径下仅剩药序记忆成分。",
        "2. 逐药跨患者 Spearman：主方案 §三.3 患者间差异度量；B0 折常数预测",
        "   为零假设有效零对照（设计更正见 EXP-026 记录注记）。",
        "3. 全面板 NDCG@5：官方资格（已知药 ≥10/PDO），应与第一波一致",
        "（已验证 B0=0.9089）。",
        "4. 误标重复轮 run_b2000_seed2 仅用于交叉验证，不计入种子聚合",
        "（沿用 EXP-021 修正口径）。",
        "",
        "## 四、结果判读（待 EXP-026 报告正文补充）",
        "",
        "（本文件为重打分数据表；正式判读见 exp026_closeout.md。）",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("REPORT ->", REPORT)
    print("\n" + "\n".join(lines[7:40]))
    print("\nBOOTSTRAP:")
    for k, v in boot.items():
        print(" ", k, v if v else "n/a")


if __name__ == "__main__":
    main()
