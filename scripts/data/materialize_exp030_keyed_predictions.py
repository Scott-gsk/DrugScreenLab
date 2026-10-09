#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-030 predictions.parquet 键值化重建（W3-37 工程修复，纯工程任务）。

背景（docs/reports/audit_repair_wave3_20260922.md W3-37；
docs/reports/domain_dependency_EXP035_20260922.md 第 7 节；
experiments/records/EXP-035.md 关键审计修复节）：

EXP-030 ``artifacts/exp030_dependency_transfer/predictions.parquet`` 写入时
存在 (sample, gene) 标签与数值行序错位缺陷：

- 标签（sample/gene 列）按 sample-major 排列（第 k 行标签属于
  (te_ids[k // n_eval], sorted_genes[k % n_eval])）；
- 数值（predicted/observed 列）按 gene-major 排列，第 k 行数值属于
  (sorted_genes[k // n_te], te_ids[k % n_te])（k = g*n_te + s）。

已验证的正确重建方式（EXP-035 复核探针，与 arm_summary.json 内存口径
逐折一致；HD 0.0321 / ZS 0.0294）：以 (arm, fold) 为块按**书写行序**把
数值列 reshape(n_genes, n_te)——行 = 排序基因清单、列 = 该折测试样本
首现顺序（= EXP-030 写入时的 te_ids 顺序）。

本脚本：
1. 按上述已验证方式从原文件重建显式键值长表
   (fold, arm, sample_id, gene, predicted, observed)，写出
   ``predictions_keyed_v2.parquet``（经 ``drug_screen.data.prediction_io``
   统一出口：身份列非空、身份键唯一校验，行序无隐含语义）；
2. 校验（三重，全部写 ``predictions_keyed_v2_validation.json``）：
   a. 键值重建 == 行序重建（逐位一致，含 NaN 位置）；
   b. 与 arm_summary.json 逐折一致性：按 EXP-030 内存口径重算每臂每折
      macro Spearman（>=30 行/基因、paired 基因集 = B0 非空 ∩ ZS 非空，
      从预测值本身恢复）与逐折 paired 基因数、5 折均值比对；
   c. 独立外部核对：observed 值与 biobank 依赖矩阵（minLib 规则解析后
      的 LFC）按 (sample, gene) 键比对（值容差 = float32 舍入；NaN 位置
      必须一致）——该检查独立验证键映射本身。

原 predictions.parquet 保持只读不动。
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from drug_screen.data.gene_alignment import assert_gene_alignment
from drug_screen.data.organoid_depmap import resolve_dual_library
from drug_screen.data.prediction_io import read_predictions, write_predictions

DEFAULT_ROOT = Path("/mnt/d/Code/DrugScreenLab")
ARMS = ["B0", "ZS", "CAL", "DA", "HD"]
N_FOLDS = 5
MIN_TEST_ROWS = 30  # EXP-030 主口径：每基因 >= 30 个测试行
VALUE_TOL = {"rtol": 1e-6, "atol": 1e-6}  # float32 舍入容差
# 逐折 macro 与 arm_summary 的允许绝对差：原文件以 float32 存储（EXP-030
# 内存口径为 float64），近并列值的秩在舍入后偶发互换，使 macro 受
# O(1e-6) 量级扰动（2026-09-22 实测最大 1.08e-6，CAL 臂）。键映射本身的
# 正确性由 observed 外部核对（0 不匹配）与 paired 基因数精确一致保证。
MACRO_TOL = 5e-6


# --------------------------------------------- EXP-030 内存口径（复刻） ----

def _rank_pair(pred: np.ndarray, obs: np.ndarray) -> tuple:
    """逐基因秩对（NaN 置 +inf 排秩末尾、平均并列秩）。

    与 scripts/modeling/exp030_dependency_transfer.py 的 ``_rank_pair``
    逐行等价（该函数属于正式实验脚本，不导入，按同一实现复刻）。
    """
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    r_p = rankdata(np.where(mask, pred, np.inf), axis=1, method="average")
    r_o = rankdata(np.where(mask, obs, np.inf), axis=1, method="average")
    return mask, r_p, r_o


def _macro_from_ranks(mask: np.ndarray, r_p: np.ndarray, r_o: np.ndarray,
                      min_rows: int) -> np.ndarray:
    """主口径：逐基因 Spearman（秩的 Pearson），行数 >= min_rows 才有效。"""
    k = mask.sum(axis=1)
    n = np.maximum(k, 1)
    mp = np.where(mask, r_p, 0.0).sum(axis=1) / n
    mo = np.where(mask, r_o, 0.0).sum(axis=1) / n
    dp = np.where(mask, r_p - mp[:, None], 0.0)
    do = np.where(mask, r_o - mo[:, None], 0.0)
    cov = (dp * do).sum(axis=1)
    var = np.sqrt((dp * dp).sum(axis=1) * (do * do).sum(axis=1))
    rho = np.full(mask.shape[0], np.nan)
    ok = (k >= min_rows) & (var > 0)
    rho[ok] = cov[ok] / var[ok]
    return rho


def fold_macro_spearman(pred: np.ndarray, obs: np.ndarray,
                        paired_genes: np.ndarray) -> tuple[float, int]:
    """重算 EXP-030 逐折 macro（paired 基因集上取均值）。"""
    mask, r_p, r_o = _rank_pair(pred, obs)
    rho = _macro_from_ranks(mask, r_p, r_o, MIN_TEST_ROWS)
    sel = paired_genes & np.isfinite(rho)
    macro = float(np.mean(rho[sel])) if sel.any() else float("nan")
    return macro, int(sel.sum())


# ------------------------------------------------------------ 重建与校验 --

def rebuild_keyed(pred: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, dict]:
    """按已验证行序方式重建显式键值长表。

    对每个 (arm, fold) 块（保持文件书写顺序）：
    - 列（样本）= 标签 sample 首现顺序（= 写入时 te_ids 顺序）；
    - 数值 reshape(n_genes, n_te)：行 = 排序基因清单、列 = 样本首现顺序。
    """
    keyed_pieces: list[pd.DataFrame] = []
    structure: dict[tuple[str, int], dict] = {}
    gene_arr = np.array(genes, dtype=object)
    for arm in ARMS:
        sub_arm = pred[pred["arm"] == arm]
        for fold in range(1, N_FOLDS + 1):
            sub = sub_arm[sub_arm["fold"] == fold]
            col_ids = list(dict.fromkeys(sub["sample"].tolist()))
            n_te = len(col_ids)
            if n_te == 0:
                raise ValueError(f"empty fold block: arm={arm} fold={fold}")
            if len(sub) != len(genes) * n_te:
                raise ValueError(
                    f"block size mismatch: arm={arm} fold={fold} rows="
                    f"{len(sub)} != {len(genes)}*{n_te}")
            sample_col = np.tile(np.array(col_ids, dtype=object), len(genes))
            gene_col = np.repeat(gene_arr, n_te)
            keyed_pieces.append(pd.DataFrame({
                "fold": np.full(len(sub), fold, dtype="int8"),
                "arm": np.full(len(sub), arm, dtype=object),
                "sample_id": sample_col,
                "gene": gene_col,
                "predicted": sub["predicted"].to_numpy(np.float32),
                "observed": sub["observed"].to_numpy(np.float32),
            }))
            # 行序重建矩阵（校验 a 的参照）。
            structure[(arm, fold)] = {
                "col_ids": col_ids,
                "P": sub["predicted"].to_numpy(np.float32).reshape(
                    len(genes), n_te),
                "O": sub["observed"].to_numpy(np.float32).reshape(
                    len(genes), n_te),
            }
    keyed = pd.concat(keyed_pieces, ignore_index=True)
    return keyed, structure


def pivot_from_keyed(keyed: pd.DataFrame, arm: str, fold: int,
                      genes: list[str], col_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """从键值长表按显式 (sample_id, gene) 键透视重建矩阵（行序无关）。"""
    sub = keyed[(keyed["arm"] == arm) & (keyed["fold"] == fold)]
    piv_p = sub.pivot(index="gene", columns="sample_id", values="predicted")
    piv_o = sub.pivot(index="gene", columns="sample_id", values="observed")
    # 透视的基因行轴按显式清单断言（集合、顺序、长度完全一致），
    # 样本列按该折首现顺序重排（按名 reindex，不依赖书写行序）。
    assert_gene_alignment(piv_p.index.to_numpy().astype(str).tolist(), genes)
    piv_p = piv_p.reindex(columns=col_ids)
    piv_o = piv_o.reindex(columns=col_ids)
    if list(piv_p.columns) != list(col_ids) or list(piv_o.columns) != list(col_ids):
        raise ValueError(f"pivot columns do not cover fold samples: arm={arm} fold={fold}")
    return piv_p.to_numpy(np.float32), piv_o.to_numpy(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="项目根目录（默认 /mnt/d/Code/DrugScreenLab）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已存在的 keyed_v2 / 校验报告")
    args = parser.parse_args()
    root = args.root
    src_dir = root / "artifacts/exp030_dependency_transfer"
    src = src_dir / "predictions.parquet"
    out = src_dir / "predictions_keyed_v2.parquet"
    report_path = src_dir / "predictions_keyed_v2_validation.json"
    t0 = time.time()

    for target in (out, report_path):
        if target.exists() and not args.overwrite:
            raise ValueError(f"refusing to overwrite existing file: {target} "
                             "(pass --overwrite to regenerate)")

    # ---------- 1. 读原文件（只读），按已验证行序方式重建 ----------
    pred = pd.read_parquet(src, columns=["fold", "arm", "sample", "gene",
                                         "predicted", "observed"])
    genes = sorted({str(g) for g in pred["gene"].unique()})
    n_genes = len(genes)
    keyed, structure = rebuild_keyed(pred, genes)
    del pred

    # ---------- 2. 写出键值版本（统一出口：schema + 身份键唯一校验） ----------
    write_predictions(keyed, out, identity_columns=("fold", "arm", "sample_id", "gene"))
    n_rows = len(keyed)
    del keyed

    # ---------- 3. 读回并三重校验 ----------
    keyed = read_predictions(out, identity_columns=("fold", "arm", "sample_id", "gene"))
    n_samples = int(keyed["sample_id"].nunique())
    arm_summary = json.loads((src_dir / "arm_summary.json").read_text(encoding="utf-8"))
    audit_log = json.loads((src_dir / "audit_log.json").read_text(encoding="utf-8"))
    if arm_summary["n_evaluable_genes"] != n_genes:
        raise ValueError(
            f"n_evaluable_genes mismatch: arm_summary="
            f"{arm_summary['n_evaluable_genes']} rebuilt={n_genes}")

    # (a) 键值重建 == 行序重建（逐位、含 NaN 位置）
    bit_exact = True
    for (arm, fold), info in structure.items():
        P, O = pivot_from_keyed(keyed, arm, fold, genes, info["col_ids"])
        if not (np.array_equal(P, info["P"], equal_nan=True)
                and np.array_equal(O, info["O"], equal_nan=True)):
            bit_exact = False
            break

    # paired 基因集恢复：B0 非空 ∩ ZS 非空（fold_trainable ∩ src_trainable）
    b0_nonan: dict[int, np.ndarray] = {}
    zs_nonan: dict[int, np.ndarray] = {}
    for fold in range(1, N_FOLDS + 1):
        b0 = structure[("B0", fold)]["P"]
        zs = structure[("ZS", fold)]["P"]
        b0_nonan[fold] = ~np.isnan(b0).all(axis=1)
        zs_nonan[fold] = ~np.isnan(zs).all(axis=1)
    for fold in range(2, N_FOLDS + 1):
        if not np.array_equal(zs_nonan[fold], zs_nonan[1]):
            raise ValueError("ZS predictable gene set differs across folds")

    # (b) 逐折 macro 与 arm_summary 一致性
    fold_check: dict[str, dict] = {}
    max_abs_diff = 0.0
    paired_counts_match = True
    for arm in ARMS:
        fold_check[arm] = {}
        for fold in range(1, N_FOLDS + 1):
            info = structure[(arm, fold)]
            paired = b0_nonan[fold] & zs_nonan[fold]
            recomputed, n_paired = fold_macro_spearman(
                info["P"].astype(np.float64), info["O"].astype(np.float64), paired)
            reference = arm_summary["primary_endpoint"]["per_fold_macro"][arm][fold - 1]
            ref_count = arm_summary["primary_endpoint"]["paired_genes_per_fold"][arm][fold - 1]
            if ref_count != n_paired:
                paired_counts_match = False
            diff = (abs(recomputed - reference)
                    if np.isfinite(recomputed) and np.isfinite(reference)
                    else (0.0 if (np.isnan(recomputed) and np.isnan(reference)) else np.inf))
            max_abs_diff = max(max_abs_diff, diff)
            fold_check[arm][fold] = {
                "arm_summary": reference,
                "recomputed_from_keyed": recomputed,
                "abs_diff": diff,
                "paired_genes": {"arm_summary": ref_count, "recomputed": n_paired},
            }
    five_fold_check = {}
    for arm in ARMS:
        with warnings.catch_warnings():
            # B0 各折 macro 全为 NaN（常数预测），nanmean 空切片警告属预期。
            warnings.simplefilter("ignore", RuntimeWarning)
            recomputed = float(np.nanmean(
                [fold_check[arm][f]["recomputed_from_keyed"] for f in range(1, N_FOLDS + 1)]))
        reference = arm_summary["primary_endpoint"]["five_fold_mean"][arm]
        five_fold_check[arm] = {"arm_summary": reference,
                                "recomputed_from_keyed": recomputed}

    # (c) observed 与 biobank 依赖矩阵独立核对（键映射外部验证）
    bio_dep = pd.read_parquet(root / "data/processed/transfer/"
                              "biobank_functional_v1/dependency_by_library.parquet")
    dep_res, _ = resolve_dual_library(bio_dep)
    del bio_dep
    dep_wide = dep_res.pivot(index="sample_ID", columns="gene", values="LFC")
    del dep_res
    gene_arr = np.array(genes, dtype=object)
    n_obs_compared = 0
    n_value_mismatch = 0
    n_nan_mask_mismatch = 0
    max_obs_diff = 0.0
    fold_sizes: dict[int, int] = {}
    for fold in range(1, N_FOLDS + 1):
        col_ids = structure[("B0", fold)]["col_ids"]
        fold_sizes[fold] = len(col_ids)
        # 参照矩阵转置为 (n_genes, n_te)，与键值重建矩阵同构。
        ref = dep_wide.reindex(index=col_ids, columns=genes).to_numpy(np.float64).T
        for arm in ARMS:
            obs = structure[(arm, fold)]["O"]
            nan_mask_match = np.array_equal(np.isnan(obs), np.isnan(ref))
            if not nan_mask_match:
                n_nan_mask_mismatch += int((np.isnan(obs) != np.isnan(ref)).sum())
            both = ~np.isnan(obs) & ~np.isnan(ref)
            n_obs_compared += int(both.sum())
            diffs = np.abs(obs[both].astype(np.float64) - ref[both])
            n_value_mismatch += int((~np.isclose(
                obs[both].astype(np.float64), ref[both], **VALUE_TOL)).sum())
            if diffs.size:
                max_obs_diff = max(max_obs_diff, float(diffs.max()))
    del dep_wide
    test_sizes_match = fold_sizes == {
        f: int(size) for f, size in enumerate(
            audit_log["folds"]["test_sizes"], start=1)}
    del structure, keyed

    macro_match = (max_abs_diff <= MACRO_TOL and paired_counts_match)
    status = "PASS" if (bit_exact and macro_match and n_value_mismatch == 0
                       and n_nan_mask_mismatch == 0
                       and test_sizes_match) else "FAIL"
    report = {
        "task": "EXP-030 predictions 键值化（W3-37 工程修复）",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "source_file": str(src),
        "output_file": str(out),
        "reconstruction": {
            "method": (
                "已验证行序方式（domain_dependency_EXP035 第 7 节）：每 "
                "(arm, fold) 块按书写行序 reshape(n_genes, n_te)；"
                "行 = 排序基因清单、列 = 该折测试样本首现顺序"),
            "n_rows": int(n_rows),
            "n_genes": n_genes,
            "n_samples": int(n_samples),
            "arms": ARMS,
            "n_folds": N_FOLDS,
            "bit_exact_vs_row_order_rebuild": bool(bit_exact),
        },
        "fold_macro_check": {
            "metric": arm_summary["primary_endpoint"]["metric"],
            "tolerance_abs": MACRO_TOL,
            "max_abs_diff": float(max_abs_diff),
            "paired_genes_per_fold_all_match": bool(paired_counts_match),
            "five_fold_mean": five_fold_check,
            "per_fold": fold_check,
        },
        "observed_vs_biobank_dependency": {
            "description": "observed 与 biobank LFC（minLib 解析）按键独立核对",
            "tolerance": VALUE_TOL,
            "n_compared": int(n_obs_compared),
            "n_value_mismatch": int(n_value_mismatch),
            "n_nan_mask_mismatch": int(n_nan_mask_mismatch),
            "max_abs_diff": float(max_obs_diff),
        },
        "consistency": {
            "n_evaluable_genes_match": True,
            "fold_test_sizes": fold_sizes,
            "fold_test_sizes_match_audit_log": bool(test_sizes_match),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      default=str), encoding="utf-8")
    print(f"[materialize_exp030_keyed] status={status} rows={n_rows} "
          f"genes={n_genes} max_fold_macro_diff={max_abs_diff:.3e} "
          f"obs_mismatch={n_value_mismatch}/{n_obs_compared} "
          f"elapsed={time.time() - t0:.1f}s", flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
