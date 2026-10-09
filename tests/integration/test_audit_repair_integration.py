"""审计修复八项集成测试与正式入口最小运行记录（2026-09-22，纯工程）。

对应审计修复方案要求的"八项必备集成测试"与"正式运行入口最小运行"
（SubTask 4.2）。本套件不创建 EXP、不运行 GPU 训练、对 artifacts 只读。

八项测试与方案条目对应
----------------------
1. ``test_01_formal_prediction_files_format_and_pairing``：第三波正式预测
   文件（真实 artifacts）格式与身份检查——必需身份列存在、身份单元格
   非空、身份键唯一、预测数值列可数值化且无 inf、``model_id`` 生物语义
   守卫。schema 不合规文件以 **strict xfail** 暴露并注明缺陷（见下方
   DEFECTS）；``test_01b_cross_file_patient_drug_pairing`` 对文档记载的
   成对比较做患者×药物(×样本×折)键与标签配对检查。
2. ``test_02_multi_pdo_patient_equal_weight_matches_manual``：≥3 患者、
   ≥1 患者含 2 个 PDO 的患者等权宏平均（``intra_patient_spearman_
   patient_macro``）与手算逐步（PDO 内等权 → 患者内 PDO 等权 → 患者
   等权）逐位一致；``target_drug_mean_baseline`` 的多 PDO 逐药常数同法
   手算核对。
3. ``test_03_shuffled_gene_columns_downstream_invariance``：任意打乱表达
   矩阵基因列后，经 ``align_matrix_by_genes`` 按显式基因清单对齐，下游
   计算（逐基因 masked_spearman + 最小二乘药敏预测）逐位不变。
4. ``test_04_prediction_roundtrip_full_schema_bit_identical``：
   ``prediction_io`` 统一出口 write→read 往返，多模型/多算法/多种子完整
   schema 逐样本—药物数值逐位不变。
5. ``test_05_masked_correlations_vs_scipy_ties_constant_missing``：并列
   值、常数、缺失下 ``masked_spearman``/``masked_pearson`` 与 scipy
   参考实现（成对完整子集）一致；常数 → NaN。
6. ``test_06_joint_vs_single_task_drug_update_counts_equal``：同一小模型
   单任务（仅药敏损失）与联合（药敏+辅助损失，含辅助权重=0）两种模式
   的药敏损失参数更新次数相等；辅助任务更新存在但不计入药敏更新；
   辅助权重=0 时参数轨迹逐位一致。
7. ``test_07_preprocessing_fits_on_train_rows_only``：内层验证/外层测试
   不参与训练预处理拟合（极端异常 val/test 行不影响拟合参数；train-only
   拟合 ≠ train+val+test 拟合且等于前者；下游模型不受 val/test 篡改影响）。
8. ``test_08_direct_drug_control_valid_per_drug_predictions``：直接药敏
   对照（fold 内患者等权逐药均值基线，同监督条件）对同一患者不同药物
   输出互不相同、数值有效的预测，与观测面板 1:1 配对（含多 PDO 测试
   患者），且患者内排序主指标有定义（无 constant_predicted）。

第三波正式预测文件盘点（2026-09-22 实测，test_01 参数化来源）
--------------------------------------------------------------

==========================================  =======  ===========================
文件                                          行数    检查结论
==========================================  =======  ===========================
exp032_l1l2/l1_predictions_rebased.parquet   18.5M   xfail：model_id 被用作算
                                                      法身份（source_drug_mean_
                                                      baseline / drug_only /
                                                      dual_encoder 各跨 963 个
                                                      患者），违反 identity_
                                                      contracts（model_id 恒为
                                                      生物模型身份；算法身份应
                                                      另列 algorithm/arm）
exp032_l3l4/predictions.parquet              90988   PASS（身份键含 run_id/fold/
                                                      seed/arm 唯一；frac 为条
                                                      件列，source_zero_shot 臂
                                                      frac=NaN、fold=-1、
                                                      seed=20260912 不违约）
exp033_routeA/predictions.parquet            21160   PASS
exp033_routeB/c4_predictions.parquet         2116    xfail：缺 model_id 身份列
                                                      （生物模型身份仅由
                                                      sample_id 承担）；其余检
                                                      查（键唯一/数值/配对）通过
exp034_serial/predictions.parquet            38088   PASS（shuffled 臂身份键需
                                                      含 shuffle 列，非 shuffled
                                                      臂 shuffle=NaN）
exp034_parallel/predictions.parquet          65596   PASS（dual_pert_shuffle 臂
                                                      的 shuffle 变体身份仅由
                                                      run_id 区分——schema 弱
                                                      点，记录于盘点，不判失败）
exp034_joint/predictions.parquet             12696   PASS
------------------------------------------  -------  ---------------------------
exp033_routeB/xpert_reversal.parquet         2280    非预测文件（扰动反转打分
                                                      表），不适用预测格式检查
exp033_routeB/whitelist_chempert_            8640    非预测文件（扰动特征表），
          features.parquet                           不适用预测格式检查
==========================================  =======  ===========================

正式入口最小运行记录（SubTask 4.2）
------------------------------------
- ``scripts/data/materialize_exp030_keyed_predictions.py``：已于
  2026-09-22 完整运行，validation PASS
  （``artifacts/exp030_dependency_transfer/predictions_keyed_v2_validation.json``：
  status=PASS；键值重建与行序重建逐位一致；10966590 行 / 13539 基因 /
  162 样本 / 5 臂 5 折；逐折 macro 与 arm_summary 最大绝对差 1.08e-6 <
  5e-6 容差；observed 与 biobank LFC 独立核对 0 不匹配）。
  ``test_09a_materialize_exp030_validation_record`` 引用该留痕。
- 统一评估入口 ``aggregate_seed_patient_scores`` + ``run_formal_comparison``
  （+ ``patient_score_paired_bootstrap``）：``test_09b_unified_eval_entry_
  minimal_run`` 用最小合成数据（2 种子 × 6 患者 × 5 药物，模型臂 vs 逐药
  均值基线臂）端到端执行 canonical 链路并输出数字（``--capture=no``
  运行留痕；数据无随机源，数字确定）。
- 后续阶段（重算/重训）入口建成时，在各自任务中补最小运行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.stats import pearsonr, spearmanr

from drug_screen.data.gene_alignment import (
    align_matrix_by_genes,
    assert_gene_alignment,
)
from drug_screen.data.identity_contracts import assert_model_id_is_biological
from drug_screen.data.prediction_io import read_predictions, write_predictions
from drug_screen.data.trainfitted import FitOnTrainTransformer
from drug_screen.evaluation import (
    aggregate_seed_patient_scores,
    patient_score_paired_bootstrap,
    run_formal_comparison,
)
from drug_screen.evaluation.exp031_eval import (
    assert_comparable,
    intra_patient_spearman_patient_macro,
    target_drug_mean_baseline,
)
from drug_screen.evaluation.metrics import masked_pearson, masked_spearman

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"


# ---------------------------------------------------------------------------
# 第 1 项：正式预测文件格式 / 身份 / 配对检查
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PredFileSpec:
    """单个正式预测文件的检查规格。

    ``required_identity``：必需且非空的身份列（缺失 / 空值即缺陷）；
    ``key_extra``：仅参与身份键唯一性检查、允许取 NaN 的条件列
    （如 frac / shuffle）；``value_columns``：预测与标签数值列；
    ``defect`` 非 None 时该文件以 strict xfail 暴露缺陷。
    """

    rel_path: str
    required_identity: tuple[str, ...]
    key_extra: tuple[str, ...] = ()
    value_columns: tuple[str, ...] = ("y_pred", "y_true")
    note: str = ""
    defect: str | None = None


_PRED_FILE_SPECS: list[_PredFileSpec] = [
    _PredFileSpec(
        rel_path="exp032_l1l2/l1_predictions_rebased.parquet",
        required_identity=(
            "run_id", "fold", "seed", "condition", "permutation_id",
            "patient_id", "model_id", "sample_id", "drug_id"),
        note="药敏级长表（gene_id 为空占位列）；condition×permutation_id "
             "为标签置换审计条件轴",
        defect="身份契约缺陷：model_id 被用作算法身份（source_drug_mean_"
               "baseline / drug_only / dual_encoder 各跨 963 个患者），违反 "
               "identity_contracts——model_id 恒为生物模型身份（PDO/细胞"
               "系），算法身份应另列 algorithm/arm；assert_model_id_is_"
               "biological 守卫失败",
    ),
    _PredFileSpec(
        rel_path="exp032_l3l4/predictions.parquet",
        required_identity=(
            "run_id", "fold", "seed", "arm", "patient_id", "model_id",
            "sample_id", "drug_id"),
        key_extra=("frac",),
        note="source_zero_shot 臂 frac=NaN / fold=-1 / seed=20260912；"
             "frac 为允许空值的条件列",
    ),
    _PredFileSpec(
        rel_path="exp033_routeA/predictions.parquet",
        required_identity=(
            "run_id", "fold", "seed", "arm", "patient_id", "model_id",
            "sample_id", "drug_id"),
        key_extra=("frac",),
    ),
    _PredFileSpec(
        rel_path="exp033_routeB/c4_predictions.parquet",
        required_identity=(
            "fold", "arm", "frac", "patient_id", "sample_id",
            "drug_id", "model_id"),
        value_columns=("predicted", "observed"),
        note="键唯一 / 数值 / 与 EXP-032 C1 配对均通过；仅缺 model_id 列",
        defect="schema 缺陷：缺 model_id 身份列——按 identity_contracts，"
               "model_id 为生物模型身份列（本文件该身份仅由 sample_id 承"
               "担，未按统一契约命名）",
    ),
    _PredFileSpec(
        rel_path="exp034_serial/predictions.parquet",
        required_identity=(
            "run_id", "fold", "seed", "arm", "patient_id", "model_id",
            "sample_id", "drug_id"),
        key_extra=("shuffle",),
        note="shuffled 臂 shuffle 列区分置换变体；非 shuffled 臂 "
             "shuffle=NaN（允许空值的条件列）",
    ),
    _PredFileSpec(
        rel_path="exp034_parallel/predictions.parquet",
        required_identity=(
            "run_id", "fold", "seed", "arm", "patient_id", "model_id",
            "sample_id", "drug_id"),
        key_extra=("frac",),
        note="dual_pert_shuffle 臂的 shuffle 变体身份仅由 run_id 区分"
             "（无专用列）——schema 弱点，记录不判失败",
    ),
    _PredFileSpec(
        rel_path="exp034_joint/predictions.parquet",
        required_identity=(
            "run_id", "fold", "seed", "arm", "patient_id", "model_id",
            "sample_id", "drug_id"),
    ),
]


def _formal_params() -> list:
    params: list = []
    for spec in _PRED_FILE_SPECS:
        if spec.defect is not None:
            params.append(pytest.param(
                spec, marks=pytest.mark.xfail(reason=spec.defect, strict=True)))
        else:
            params.append(spec)
    return params


def _spec_id(spec: _PredFileSpec) -> str:
    return "/".join(Path(spec.rel_path).parts[-2:])


_FRAME_CACHE: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}


def _load_pred_frame(rel_path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """只读加载预测文件的指定列（模块级缓存，artifacts 只读）。"""
    key = (rel_path, tuple(columns))
    if key not in _FRAME_CACHE:
        path = ARTIFACTS / rel_path
        _FRAME_CACHE[key] = pd.read_parquet(path, columns=list(columns))
    return _FRAME_CACHE[key]


def _spec_columns(spec: _PredFileSpec) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        spec.required_identity + spec.key_extra + spec.value_columns))


def _check_formal_prediction_file(spec: _PredFileSpec) -> list[str]:
    """返回缺陷清单（空列表 = 通过）。

    大文件（l1l2，18.5M 行）只做 schema / 身份唯一性 / 数值有效性检查，
    不做指标重计算。
    """
    errors: list[str] = []
    path = ARTIFACTS / spec.rel_path
    if not path.exists():
        return [f"file missing: {spec.rel_path}"]

    available = set(pq.read_schema(path).names)
    missing = [c for c in spec.required_identity if c not in available]
    missing += [c for c in spec.value_columns if c not in available]
    if missing:
        return [f"{spec.rel_path}: missing required columns: {missing}"]

    frame = _load_pred_frame(spec.rel_path, _spec_columns(spec))

    # 身份单元格非空非空白
    for col in spec.required_identity:
        null_mask = frame[col].isna()
        if null_mask.any():
            errors.append(
                f"identity column {col!r} has {int(null_mask.sum())} null cells")
            continue
        if frame[col].dtype == object or str(frame[col].dtype) == "string":
            empty = frame[col].astype(str).str.strip().eq("")
            if empty.any():
                errors.append(
                    f"identity column {col!r} has {int(empty.sum())} empty cells")

    # 身份键唯一（required + 允许空值的条件列）
    key_columns = list(spec.required_identity) + list(spec.key_extra)
    dup_mask = frame.duplicated(subset=key_columns, keep=False)
    if dup_mask.any():
        n_unique = int(frame.loc[~dup_mask, key_columns]
                       .drop_duplicates().shape[0])
        errors.append(
            f"duplicate identity rows on {key_columns}: {n_unique} unique keys "
            f"but {int(dup_mask.sum())} rows involved in duplicates")

    # 预测数值列可数值化且无 inf（NaN 预测按项目约定保留，不算错误）
    for col in spec.value_columns:
        coerced = pd.to_numeric(frame[col], errors="coerce")
        bad = coerced.isna() & frame[col].notna()
        if bad.any():
            errors.append(
                f"value column {col!r} has {int(bad.sum())} non-numeric cells")
        if bool((np.abs(coerced.to_numpy(dtype=float)) == np.inf).any()):
            errors.append(f"value column {col!r} contains +/- infinity")

    # model_id 生物语义守卫（model_id + patient_id 同时存在时）
    if "model_id" in frame.columns and "patient_id" in frame.columns:
        try:
            assert_model_id_is_biological(frame)
        except ValueError as error:
            errors.append(f"{spec.rel_path}: {error}")
    return errors


@pytest.mark.parametrize("spec", _formal_params(), ids=_spec_id)
def test_01_formal_prediction_files_format_and_pairing(spec: _PredFileSpec):
    """审计条目 1：第三波正式预测文件通过格式及身份检查。

    schema 不合规文件（exp032_l1l2 的 model_id 语义违约、exp033_routeB
    c4 缺 model_id 列）以 strict xfail 暴露——XPASS 即缺陷已修复、需更新
    缺陷清单。大文件（l1l2，18.5M 行）只做 schema/身份唯一性/数值检查。
    """
    errors = _check_formal_prediction_file(spec)
    assert not errors, "\n".join(errors)


def test_01b_cross_file_patient_drug_pairing():
    """审计条目 1（配对部分）：文档记载的成对比较键与标签可配对。

    - EXP-032 L3/L4 各臂（seed=20260921、frac=1.0）共享同一患者×药物×
      样本×折面板且标签一致；
    - EXP-033 routeB C4 与 EXP-032 C1（frac=1.0）键集一致、observed 与
      y_true 逐位一致（c4_summary.json 记载的 C4 vs C1 配对比较前提）；
    - EXP-034 serial / parallel / joint 三文件面板与标签一致；
    - EXP-033 routeA 与 EXP-034 joint 面板一致（同一 whitelist）；
    - EXP-032 L1/L2 original 条件下三个 run_id（算法臂）面板与标签一致，
      置换条件面板与同臂 original 面板一致。
    """
    key_cols = ["patient_id", "sample_id", "drug_id", "fold"]

    # --- EXP-032 L3/L4 臂内配对 ---
    l3 = _load_pred_frame(
        "exp032_l3l4/predictions.parquet",
        ("run_id", "fold", "seed", "arm", "frac", "patient_id", "model_id",
         "sample_id", "drug_id", "y_pred", "y_true"))
    sub = l3[(l3["seed"] == 20260921) & (l3["frac"] == 1.0)]
    arms = sorted(sub["arm"].unique())
    assert len(arms) >= 2
    panels = {
        arm: set(map(tuple, group[key_cols].to_numpy()))
        for arm, group in sub.groupby("arm")
    }
    reference = panels[arms[0]]
    for arm, panel in panels.items():
        assert panel == reference, f"EXP-032 L3/L4 arm {arm} panel differs"
    # 标签一致：同键 y_true 跨臂唯一
    labels = sub[key_cols + ["y_true"]].drop_duplicates(
        subset=key_cols, keep="first")
    assert labels.duplicated(subset=key_cols, keep=False).sum() == 0

    # --- EXP-033 routeB C4 vs EXP-032 C1（frac=1.0, seed=20260921）---
    c4 = _load_pred_frame(
        "exp033_routeB/c4_predictions.parquet",
        ("fold", "arm", "frac", "patient_id", "sample_id", "drug_id",
         "observed", "predicted"))
    c1 = l3[(l3["arm"] == "C1") & (l3["frac"] == 1.0) & (l3["seed"] == 20260921)]
    c4_keys = set(map(tuple, c4[key_cols].to_numpy()))
    c1_keys = set(map(tuple, c1[key_cols].to_numpy()))
    assert c4_keys == c1_keys
    merged = c4.merge(c1, on=key_cols, suffixes=("_c4", "_c1"))
    assert len(merged) == len(c4) == len(c1)
    assert bool((merged["observed"] == merged["y_true"]).all())

    # --- EXP-034 serial / parallel / joint 配对 ---
    exp34_frames = {
        name: _load_pred_frame(
            f"{name}/predictions.parquet",
            ("run_id", "fold", "seed", "arm", "patient_id", "model_id",
             "sample_id", "drug_id", "y_pred", "y_true"))
        for name in ("exp034_serial", "exp034_parallel", "exp034_joint")
    }
    seed_rows = {
        name: frame[frame["seed"] == 20260921]
        for name, frame in exp34_frames.items()
    }
    panels = {
        name: set(map(tuple, group[key_cols].to_numpy()))
        for name, group in seed_rows.items()
    }
    joint_panel = panels["exp034_joint"]
    assert panels["exp034_serial"] == joint_panel
    assert panels["exp034_parallel"] == joint_panel
    # 标签跨文件一致
    joint_labels = (
        seed_rows["exp034_joint"][key_cols + ["y_true"]]
        .drop_duplicates(subset=key_cols, keep="first")
        .set_index(key_cols)["y_true"])
    for name in ("exp034_serial", "exp034_parallel"):
        other = (
            seed_rows[name][key_cols + ["y_true"]]
            .drop_duplicates(subset=key_cols, keep="first")
            .set_index(key_cols)["y_true"])
        aligned = other.reindex(joint_labels.index)
        assert aligned.notna().all()
        assert bool((aligned.to_numpy() == joint_labels.to_numpy()).all())

    # --- EXP-033 routeA 与 EXP-034 joint 面板一致 ---
    ra = _load_pred_frame(
        "exp033_routeA/predictions.parquet",
        ("run_id", "fold", "seed", "arm", "frac", "patient_id", "model_id",
         "sample_id", "drug_id", "y_pred", "y_true"))
    ra_keys = set(map(
        tuple, ra[ra["seed"] == 20260921][key_cols].to_numpy()))
    assert ra_keys == joint_panel

    # --- EXP-032 L1/L2 original 条件下跨臂配对 ---
    l1 = _load_pred_frame(
        "exp032_l1l2/l1_predictions_rebased.parquet",
        ("run_id", "fold", "seed", "condition", "permutation_id",
         "patient_id", "model_id", "sample_id", "drug_id", "y_pred", "y_true"))
    orig = l1[l1["condition"] == "original"]
    l1_panels = {
        run: set(map(tuple, group[key_cols].to_numpy()))
        for run, group in orig.groupby("run_id")
    }
    l1_ref = next(iter(l1_panels.values()))
    assert all(panel == l1_ref for panel in l1_panels.values())
    perm = l1[(l1["condition"] == "within_cancer_row_permutation")
              & (l1["permutation_id"] == 0)]
    perm_run = perm["run_id"].unique()
    assert len(perm_run) == 1
    perm_panel = set(map(tuple, perm[key_cols].to_numpy()))
    assert perm_panel == l1_panels[perm_run[0]]


# ---------------------------------------------------------------------------
# 第 2 项：多 PDO 患者等权与手算一致
# ---------------------------------------------------------------------------


def _build_multi_pdo_frame() -> pd.DataFrame:
    """4 个患者（p0/p3 各含 2 个 PDO）、5 种药物；每 (PDO, drug) 单行。

    观测与预测均为二进有理数（1/16 的整数倍）：两级均值（PDO 等权 /
    患者等权）在浮点下精确无舍入，手算与实现的逐位一致不受求和顺序
    影响；与真实 Kryeziu 面板"每 (PDO, drug) 单次观测"语义一致。
    """
    rows = []
    for drug_j, drug in enumerate(["d0", "d1", "d2", "d3", "d4"]):
        j = float(drug_j)
        # p0：双 PDO（m0a / m0b）
        rows.append({"patient_id": "p0", "model_id": "m0a", "drug_id": drug,
                     "observed": 2.0 + 0.5 * j + 0.0625,
                     "predicted": 1.0 + 0.25 * j})
        rows.append({"patient_id": "p0", "model_id": "m0b", "drug_id": drug,
                     "observed": 2.0 + 0.5 * j - 0.0625,
                     "predicted": 1.0 + 0.25 * j + 0.03125})
        # p1：单 PDO（预测与观测负相关）
        rows.append({"patient_id": "p1", "model_id": "m1a", "drug_id": drug,
                     "observed": 5.0 + 0.25 * j - (j % 3.0) * 0.125,
                     "predicted": 4.0 - 0.125 * j + (j % 2.0) * 0.0625})
        # p2：单 PDO（预测与观测反序 → 负 rho）
        rows.append({"patient_id": "p2", "model_id": "m2a", "drug_id": drug,
                     "observed": 3.0 + 0.5 * j,
                     "predicted": 9.0 - 0.25 * j})
        # p3：双 PDO（m3a / m3b）
        rows.append({"patient_id": "p3", "model_id": "m3a", "drug_id": drug,
                     "observed": 1.0 + 0.25 * j + 0.03125,
                     "predicted": 2.0 + 0.125 * j - 0.03125})
        rows.append({"patient_id": "p3", "model_id": "m3b", "drug_id": drug,
                     "observed": 1.0 + 0.25 * j - 0.0625,
                     "predicted": 2.0 + 0.125 * j + 0.0625})
    return pd.DataFrame(rows)


def _hand_patient_scores(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """手算：PDO 内等权 → 患者内 PDO 等权 → 患者内 Spearman → 患者等权。

    每 (PDO, drug) 单行，因此第一级为恒等；第二级 = 各 PDO 行值的等权
    均值（顺序求和 / 计数相除，与实现同位）；第四级 = 患者分数的算术
    平均（患者按字典序）。
    """
    per_patient_drug: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in frame.itertuples(index=False):
        per_patient_drug.setdefault(row.patient_id, {}).setdefault(
            row.drug_id, []).append((row.observed, row.predicted))

    scores: dict[str, float] = {}
    for patient in sorted(per_patient_drug):
        obs_vals, pred_vals = [], []
        for drug in sorted(per_patient_drug[patient]):
            pdo_obs = [v[0] for v in per_patient_drug[patient][drug]]
            pdo_pred = [v[1] for v in per_patient_drug[patient][drug]]
            # 患者内 PDO 等权：顺序求和 / 计数（与 pandas groupby mean 同位）
            obs_vals.append(sum(pdo_obs) / len(pdo_obs))
            pred_vals.append(sum(pdo_pred) / len(pdo_pred))
        scores[patient] = float(spearmanr(obs_vals, pred_vals).statistic)
    macro = float(np.mean([scores[p] for p in sorted(scores)]))
    return macro, scores


def _build_baseline_frame_with_folds() -> pd.DataFrame:
    """2 折患者划分：fold 0 测试患者 p3（双 PDO）；fold 1 测试患者 p0（双 PDO）。

    每折训练集恰为 4 个患者（2 的幂）且全部观测为 1/16 的整数倍，因此
    逐药常数 =（患者内 PDO 等权均值 → 患者等权均值）在浮点下精确无舍
    入，手算逐位一致不受求和顺序影响。训练折含多 PDO 患者（fold 0 的
    p0、fold 1 的 p3）。
    """
    drugs = ["d0", "d1", "d2", "d3", "d4"]
    all_patients = ("p0", "p1", "p2", "p3", "p4")
    rows = []
    drug_base = {d: 1.5 + 0.5 * j for j, d in enumerate(drugs)}
    patient_shift = {"p0": 0.0, "p1": 0.75, "p2": -0.25, "p3": 1.25,
                     "p4": -0.5}
    pdo_shift = {"m0a": 0.0625, "m0b": -0.0625, "m3a": 0.03125,
                 "m3b": -0.09375}
    for fold, test_patient in ((0, "p3"), (1, "p0")):
        for patient in all_patients:
            if patient == "p0":
                models = ("m0a", "m0b")
            elif patient == "p3":
                models = ("m3a", "m3b")
            else:
                models = (f"m_{patient}",)
            for model in models:
                for drug in drugs:
                    observed = (
                        drug_base[drug] + patient_shift[patient]
                        + pdo_shift.get(model, 0.0))
                    rows.append({
                        "patient_id": patient, "model_id": model,
                        "sample_id": model, "drug_id": drug, "fold": fold,
                        "observed": observed,
                        "evaluation_role": "test" if patient == test_patient
                                           else "train",
                    })
    return pd.DataFrame(rows)


def _hand_target_drug_mean_constants(frame: pd.DataFrame) -> dict:
    """手算 fold 内逐药常数：患者内同药均值（多 PDO 等权）→ 患者等权。"""
    train = frame[frame["evaluation_role"] == "train"]
    constants: dict[tuple[int, str], float] = {}
    for fold, sub in train.groupby("fold", sort=True):
        for drug in sorted(sub["drug_id"].unique()):
            patient_means = []
            for patient in sorted(sub["patient_id"].unique()):
                values = sub[(sub["patient_id"] == patient)
                             & (sub["drug_id"] == drug)]["observed"].tolist()
                patient_means.append(sum(values) / len(values))
            constants[(fold, drug)] = sum(patient_means) / len(patient_means)
    return constants


def test_02_multi_pdo_patient_equal_weight_matches_manual():
    """审计条目 2：多 PDO 患者的基线与手算患者等权结果逐位一致。"""
    frame = _build_multi_pdo_frame()
    assert frame["patient_id"].nunique() >= 3
    multi_pdo = frame.groupby("patient_id")["model_id"].nunique()
    assert int((multi_pdo > 1).sum()) >= 1  # p0 / p3 各 2 个 PDO

    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(
        frame[["patient_id", "drug_id", "observed", "predicted"]])
    hand_macro, hand_scores = _hand_patient_scores(frame)

    assert n_valid == 4
    assert macro == hand_macro  # 患者等权宏平均逐位一致
    got_scores = {
        row.patient_id: row.intra_patient_spearman
        for row in per_patient.itertuples(index=False)}
    for patient, hand_rho in hand_scores.items():
        assert got_scores[patient] == hand_rho  # 逐患者逐位一致
    assert bool(per_patient["reason"].isna().all())

    # --- target_drug_mean_baseline：多 PDO 患者等权逐药常数手算核对 ---
    baseline_frame = _build_baseline_frame_with_folds()
    out = target_drug_mean_baseline(baseline_frame)
    hand_constants = _hand_target_drug_mean_constants(baseline_frame)
    assert len(hand_constants) == 10  # 2 折 × 5 药
    for (fold, drug), constant in hand_constants.items():
        rows = out[(out["fold"] == fold) & (out["drug_id"] == drug)]
        assert len(rows) > 0
        assert bool((rows["predicted"] == constant).all()), (
            f"fold={fold} drug={drug}: {rows['predicted'].unique()} != "
            f"hand {constant}")


# ---------------------------------------------------------------------------
# 第 3 项：打乱基因列后按身份对齐的结果一致（端到端）
# ---------------------------------------------------------------------------


def test_03_shuffled_gene_columns_downstream_invariance():
    """审计条目 3：任意打乱输入基因列后，按身份对齐的下游结果逐位不变。

    区别于 gene_alignment 单元测试的纯对齐层：此处对齐结果进入实际下游
    计算——逐基因 masked_spearman 与最小二乘药敏预测及其排序指标。
    """
    rng = np.random.default_rng(20260922)
    n_samples, n_genes = 40, 12
    genes = [f"G{i:02d}" for i in range(n_genes)]
    mixing = rng.normal(size=(n_genes, n_genes))
    X = rng.normal(size=(n_samples, n_genes)) @ mixing
    y = X @ rng.normal(size=n_genes) + rng.normal(size=n_samples) * 0.3

    canonical = pd.DataFrame(X, columns=genes)

    # 多种打乱序（随机置换与完全逆序）逐一验证
    for perm in (rng.permutation(n_genes), np.arange(n_genes)[::-1]):
        shuffled = canonical.iloc[:, list(perm)]
        aligned = align_matrix_by_genes(shuffled, genes)
        assert_gene_alignment(aligned.columns, genes)

        # 下游 1：逐基因与药敏响应的 masked Spearman
        rho_canonical = np.array([
            masked_spearman(canonical[g].to_numpy(float), y)[0] for g in genes])
        rho_aligned = np.array([
            masked_spearman(aligned[g].to_numpy(float), y)[0] for g in genes])
        assert np.array_equal(rho_canonical, rho_aligned)

        # 下游 2：最小二乘药敏预测 + 预测/观测排序指标
        def _fit_predict(frame: pd.DataFrame) -> np.ndarray:
            design = np.column_stack(
                [np.ones(len(frame)), frame.to_numpy(dtype=float)])
            weights, *_ = np.linalg.lstsq(design, y, rcond=None)
            return design @ weights

        pred_canonical = _fit_predict(canonical)
        pred_aligned = _fit_predict(aligned)
        assert np.array_equal(pred_canonical, pred_aligned)
        assert masked_spearman(pred_canonical, y)[0] == \
            masked_spearman(pred_aligned, y)[0]

        # ndarray 输入路径（值 + 显式列名）同样对齐到清单
        aligned_array = align_matrix_by_genes(
            shuffled.to_numpy(dtype=float), genes,
            columns=list(shuffled.columns))
        assert np.array_equal(aligned_array, canonical.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# 第 4 项：预测写出读回逐位一致（完整 schema）
# ---------------------------------------------------------------------------


def test_04_prediction_roundtrip_full_schema_bit_identical(tmp_path):
    """审计条目 4：write_predictions→read_predictions 往返逐位一致。

    较 tests/data/test_prediction_io.py 的单元覆盖，这里用更完整 schema
    （2 折 × 2 种子 × 3 算法 × 3 患者（1 患者双模型）× 5 药物 = 240 行）
    再跑一遍同一统一出口 API，验证每个身份键的数值逐位不变。
    """
    identity_columns = ("fold", "seed", "run_id", "patient_id", "model_id",
                        "sample_id", "drug_id")
    algorithm_column = "algorithm"
    rows = []
    for fold in (0, 1):
        for seed in (11, 12):
            for algorithm in ("ridge_a", "ridge_b", "gbm_c"):
                for patient, models in (("p0", ("m0a", "m0b")),
                                        ("p1", ("m1a",)),
                                        ("p2", ("m2a",))):
                    for model in models:
                        for drug_j, drug in enumerate(
                                ("d0", "d1", "d2", "d3", "d4")):
                            tag = (fold * 1000 + seed + drug_j
                                   + len(algorithm) + len(model))
                            rows.append({
                                "fold": fold, "seed": seed,
                                "run_id": f"run_s{seed}", "patient_id": patient,
                                "model_id": model, "sample_id": model,
                                "drug_id": drug, algorithm_column: algorithm,
                                "predicted": float(tag) * 0.37 - 1.5,
                                "observed": float(tag) * 0.11 + 0.5,
                            })
    frame = pd.DataFrame(rows)
    assert len(frame) == 240
    sort_keys = list(identity_columns) + [algorithm_column]

    path = tmp_path / "roundtrip_full_schema.parquet"
    write_predictions(
        frame, path,
        identity_columns=identity_columns,
        value_columns=("predicted",),
        observed_columns=("observed",),
        algorithm_column=algorithm_column)
    back = read_predictions(
        path,
        identity_columns=identity_columns,
        value_columns=("predicted",),
        observed_columns=("observed",),
        algorithm_column=algorithm_column)

    assert len(back) == len(frame)
    # 键与数值逐位一致（读回按身份键规范排序，原表同序比较）
    expected = frame.sort_values(sort_keys, kind="mergesort").reset_index(
        drop=True)
    assert back[sort_keys].equals(expected[sort_keys])
    assert bool((back["predicted"].to_numpy()
                 == expected["predicted"].to_numpy()).all())
    assert bool((back["observed"].to_numpy()
                 == expected["observed"].to_numpy()).all())

    # 读回顺序由身份键决定（与书写序无关的规范化保证）
    assert back[sort_keys].equals(
        back[sort_keys].sort_values(sort_keys, kind="mergesort")
        .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 第 5 项：并列 / 常数 / 缺失相关性与参考实现一致
# ---------------------------------------------------------------------------


def test_05_masked_correlations_vs_scipy_ties_constant_missing():
    """审计条目 5：masked_spearman / masked_pearson 与 scipy 一致。

    显式案例（并列、常数、NaN、显式 mask）+ 20 组随机注入并列/缺失的
    向量对；参考实现 = scipy.stats 对成对完整子集计算；常数 → NaN。
    """
    # --- 显式案例：并列值 ---
    a_ties = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0])
    b_ties = np.array([2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 4.0, 5.0])
    rho, n_used = masked_spearman(a_ties, b_ties)
    assert n_used == 8
    assert rho == pytest.approx(spearmanr(a_ties, b_ties).statistic, abs=1e-12)
    r, _ = masked_pearson(a_ties, b_ties)
    assert r == pytest.approx(pearsonr(a_ties, b_ties).statistic, abs=1e-12)

    # --- 显式案例：常数 → (NaN, n_used) ---
    for a, b in ((np.full(7, 2.0), np.arange(7.0)),
                 (np.arange(7.0), np.full(7, 2.0)),
                 (np.full(7, 2.0), np.full(7, 3.0))):
        rho_const, n_const = masked_spearman(a, b)
        assert np.isnan(rho_const) and n_const == 7
        r_const, _ = masked_pearson(a, b)
        assert np.isnan(r_const)

    # --- 显式案例：NaN 成对完整 + 显式 mask 子集 ---
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = np.array([2.0, 1.0, 3.0, np.nan, 5.0, 6.0, 4.0, 8.0])
    valid = np.isfinite(a) & np.isfinite(b)
    rho, n_used = masked_spearman(a, b)
    assert n_used == int(valid.sum())
    assert rho == pytest.approx(
        spearmanr(a[valid], b[valid]).statistic, abs=1e-12)
    r, _ = masked_pearson(a, b)
    assert r == pytest.approx(pearsonr(a[valid], b[valid]).statistic, abs=1e-12)
    # 显式 mask 进一步子集
    mask = np.array([True, True, False, True, True, True, True, False])
    rho_masked, n_masked = masked_spearman(a, b, mask=mask)
    sub = valid & mask
    assert n_masked == int(sub.sum())
    assert rho_masked == pytest.approx(
        spearmanr(a[sub], b[sub]).statistic, abs=1e-12)

    # --- 随机案例：注入并列 + 缺失 ---
    rng = np.random.default_rng(20260922)
    for _ in range(20):
        n = int(rng.integers(10, 60))
        x = rng.normal(size=n)
        y = x * 0.7 + rng.normal(size=n) * 0.5
        # 注入并列块
        x[rng.integers(0, n, size=5)] = 1.5
        y[rng.integers(0, n, size=4)] = -0.5
        # 注入缺失
        x[rng.integers(0, n, size=3)] = np.nan
        y[rng.integers(0, n, size=3)] = np.nan
        valid = np.isfinite(x) & np.isfinite(y)
        assert valid.sum() >= 4
        assert np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0

        rho, n_used = masked_spearman(x, y)
        assert n_used == int(valid.sum())
        assert rho == pytest.approx(
            spearmanr(x[valid], y[valid]).statistic, abs=1e-12)
        r, _ = masked_pearson(x, y)
        assert r == pytest.approx(
            pearsonr(x[valid], y[valid]).statistic, abs=1e-12)


# ---------------------------------------------------------------------------
# 第 6 项：单任务与联合任务药敏更新次数一致
# ---------------------------------------------------------------------------


def _train_minimal(
    X_drug: np.ndarray,
    y_drug: np.ndarray,
    X_aux: np.ndarray,
    y_aux: np.ndarray,
    *,
    mode: str,
    aux_weight: float,
    epochs: int = 4,
    lr: float = 0.05,
    batch: int = 8,
) -> tuple[np.ndarray, int, int]:
    """最小训练循环（numpy 手写梯度下降），返回 (参数, 药敏更新次数, 总步数)。

    - ``single``：仅药敏损失，每药敏批一步（计为药敏更新）。
    - ``joint``：每药敏批一步（药敏损失 + aux_weight×辅助损失，计为药敏
      更步），并按拍插入仅辅助损失的更新步（**不**计入药敏更新次数）。
      辅助权重统一作用于全部辅助项，因此 aux_weight=0 时辅助贡献恒为
      精确 0.0，参数轨迹与 single 逐位一致。
    """
    rng = np.random.default_rng(0)
    w = rng.normal(scale=0.01, size=X_drug.shape[1])
    n_drug = X_drug.shape[0]
    n_aux = X_aux.shape[0]
    drug_updates = 0
    total_steps = 0

    def _grad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        residual = x @ w - y
        return x.T @ residual / len(y)

    for _ in range(epochs):
        aux_index = 0
        for start in range(0, n_drug, batch):
            xb = X_drug[start:start + batch]
            yb = y_drug[start:start + batch]
            drug_grad = _grad(xb, yb)
            if mode == "joint":
                # 同拍辅助批：联合损失 = 药敏 + aux_weight×辅助
                aux_end = min(aux_index + batch, n_aux)
                aux_grad = _grad(X_aux[aux_index:aux_end],
                                 y_aux[aux_index:aux_end])
                w = w - lr * (drug_grad + aux_weight * aux_grad)
                drug_updates += 1
                total_steps += 1
                # 仅辅助损失的更新步：辅助任务自身的参数更新，
                # 不包含药敏损失 → 不计入药敏更新次数
                aux_index = aux_end % n_aux
                aux_end2 = min(aux_index + batch, n_aux)
                aux_only_grad = _grad(X_aux[aux_index:aux_end2],
                                      y_aux[aux_index:aux_end2])
                w = w - lr * (aux_weight * aux_only_grad)
                total_steps += 1
            else:
                w = w - lr * drug_grad
                drug_updates += 1
                total_steps += 1
    return w, drug_updates, total_steps


def test_06_joint_vs_single_task_drug_update_counts_equal():
    """审计条目 6：联合训练的辅助任务不增加药敏损失更新步数。"""
    rng = np.random.default_rng(20260923)
    X_drug = rng.normal(size=(32, 6))
    y_drug = X_drug @ rng.normal(size=6) + rng.normal(size=32) * 0.1
    X_aux = rng.normal(size=(24, 6))
    y_aux = X_aux @ rng.normal(size=6) + rng.normal(size=24) * 0.1

    n_drug_batches = 4  # ceil(32/8)
    expected_drug_updates = n_drug_batches * 4  # epochs

    w_single, updates_single, steps_single = _train_minimal(
        X_drug, y_drug, X_aux, y_aux, mode="single", aux_weight=0.0)
    w_joint0, updates_joint0, steps_joint0 = _train_minimal(
        X_drug, y_drug, X_aux, y_aux, mode="joint", aux_weight=0.0)
    w_joint5, updates_joint5, steps_joint5 = _train_minimal(
        X_drug, y_drug, X_aux, y_aux, mode="joint", aux_weight=0.5)

    # 药敏损失参数更新次数：单任务 == 联合（含辅助权重 0 与 0.5）
    assert updates_single == expected_drug_updates
    assert updates_joint0 == expected_drug_updates
    assert updates_joint5 == expected_drug_updates

    # 联合模式存在仅辅助更新步（总步数 > 药敏更新次数），但它们不改变
    # 药敏更新计数
    assert steps_joint0 == 2 * expected_drug_updates
    assert steps_single == expected_drug_updates

    # 辅助权重 = 0：辅助贡献精确为 0 → 参数轨迹与单任务逐位一致
    assert np.array_equal(w_single, w_joint0)
    # 辅助权重 > 0：辅助任务确实影响参数（联合训练在起作用）
    assert not np.array_equal(w_single, w_joint5)


# ---------------------------------------------------------------------------
# 第 7 项：内层验证与外层测试不参与训练预处理
# ---------------------------------------------------------------------------


def test_07_preprocessing_fits_on_train_rows_only():
    """审计条目 7：预处理仅在训练折拟合，val/test 不改变拟合参数。"""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(20260922)
    n_train, n_val, n_test, n_genes = 60, 10, 10, 6
    X_train = rng.normal(size=(n_train, n_genes))
    X_val = rng.normal(size=(n_val, n_genes))
    X_test = rng.normal(size=(n_test, n_genes))
    # 极端异常的 val/test 行（若参与拟合会剧烈改变统计量）
    X_val[0] = 1e9
    X_test[0] = -1e12
    X_val[1, 2] = np.nan
    X_test[1, 4] = np.nan
    y_train = X_train @ rng.normal(size=n_genes) + rng.normal(size=n_train) * 0.1
    y_val = rng.normal(size=n_val) * 1e6   # val 标签同样极端
    y_test = rng.normal(size=n_test) * 1e6

    # 带角色划分的完整数据（train/val/test 行序拼接）
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    roles = np.array(["train"] * n_train + ["val"] * n_val
                     + ["test"] * n_test)

    def _fit_pipeline(X: np.ndarray, y: np.ndarray,
                      role_flags: np.ndarray):
        """预处理 + 下游模型只在 train 行拟合（角色过滤在管线内）。"""
        train_mask = role_flags == "train"
        scaler = FitOnTrainTransformer().fit(
            X[train_mask], folds=[0, 1, 2, 3])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[train_mask]),
                                     y[train_mask])
        return scaler, model

    # 极端 val/test 行已在数据中：拟合参数等于手算 train-only 统计
    scaler, model = _fit_pipeline(X_all, y_all, roles)
    mean_train_only = scaler.mean_.copy()
    std_train_only = scaler.std_.copy()
    assert scaler.fit_folds == frozenset({"0", "1", "2", "3"})
    hand_mean = X_train.sum(axis=0) / n_train  # train 无缺失
    assert np.array_equal(scaler.mean_, hand_mean)

    # (a) val/test 行进一步篡改为更极端的值 → 重新跑同一管线，
    #     拟合参数与下游模型逐位不变（val/test 从未进入拟合）
    X_tampered = X_all.copy()
    tamper_mask = roles != "train"
    X_tampered[tamper_mask] = X_tampered[tamper_mask] * 1e6 + 1e9
    y_tampered = y_all.copy()
    y_tampered[tamper_mask] = y_tampered[tamper_mask] * -1e6
    scaler_t, model_t = _fit_pipeline(X_tampered, y_tampered, roles)
    assert np.array_equal(scaler_t.mean_, mean_train_only)
    assert np.array_equal(scaler_t.std_, std_train_only)
    assert np.array_equal(model_t.coef_, model.coef_)
    assert np.array_equal(
        model_t.predict(scaler_t.transform(X_train)),
        model.predict(scaler.transform(X_train)))

    # (b) 错误口径对照：把 train+val+test 一并拟合 → 参数确实不同，
    #     证明极端值一旦泄漏即改变统计量；train-only 结果等于正确口径
    scaler_all = FitOnTrainTransformer().fit(X_all)
    assert not np.array_equal(scaler_all.mean_, mean_train_only)
    assert not np.array_equal(scaler_all.std_, std_train_only)
    scaler_train_ref = FitOnTrainTransformer().fit(X_train)
    assert np.array_equal(mean_train_only, scaler_train_ref.mean_)
    assert np.array_equal(std_train_only, scaler_train_ref.std_)

    # (c) 极端 val/test 行经 train 参数变换后仍是极端 z 值
    #     （若 val/test 参与了拟合，这些行会被压回 0 附近）
    Z_val = scaler.transform(X_val)
    assert float(np.abs(Z_val[0]).max()) > 1e6
    Z_test = scaler.transform(X_test)
    assert float(np.abs(Z_test[0]).max()) > 1e6


# ---------------------------------------------------------------------------
# 第 8 项：直接药敏对照对同患者不同药物产生有效预测
# ---------------------------------------------------------------------------


def test_08_direct_drug_control_valid_per_drug_predictions():
    """审计条目 8：直接药敏对照（患者等权逐药均值基线）预测有效性。

    同一监督条件（仅训练折 observed）：对每个测试患者，不同药物得到
    互不相同、有限的预测；与观测面板 1:1 配对（含双 PDO 测试患者）；
    患者内排序主指标有定义（对照 EXP-033 routeB global_mean/ctrl_only
    臂全患者 constant_predicted 的失效模式）。
    """
    frame = _build_baseline_frame_with_folds()
    out = target_drug_mean_baseline(frame)
    test_rows = out[out["evaluation_role"] == "test"]
    assert test_rows["predicted"].notna().all()  # 训练折见过全部药物

    # 同一患者不同药物：预测互不相同（成对互异）且有限
    for patient, group in test_rows.groupby("patient_id"):
        per_drug = group.groupby("drug_id")["predicted"].first()
        assert np.isfinite(per_drug.to_numpy()).all()
        assert per_drug.nunique() == len(per_drug), (
            f"patient {patient}: constant prediction across drugs")

    # 双 PDO 测试患者（fold 1 的 p0）：样本级 1:1 配对
    p0_test = test_rows[test_rows["patient_id"] == "p0"]
    assert p0_test["sample_id"].nunique() == 2
    keys = ["fold", "patient_id", "sample_id", "drug_id"]
    merged = p0_test[keys + ["observed", "predicted"]].merge(
        frame[frame["evaluation_role"] == "test"][keys + ["observed"]],
        on=keys, suffixes=("_pred", "_obs"))
    assert len(merged) == len(p0_test)
    assert bool((merged["observed_pred"] == merged["observed_obs"]).all())

    # 患者内排序主指标有定义：无 constant_predicted / 无 NaN
    #（只看测试折的患者分数——直接对照的预测有效性语义）
    test_out = out[out["evaluation_role"] == "test"]
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(
        test_out[["patient_id", "drug_id", "observed", "predicted"]])
    test_patients = set(test_rows["patient_id"])
    assert n_valid == len(test_patients)
    assert macro is not None
    reasons = dict(zip(per_patient["patient_id"], per_patient["reason"]))
    for patient in test_patients:
        assert reasons[patient] is None

    # 配对检查（项目统一比较检查器）：与同面板第二臂可配对比较
    arm_a = out.rename(columns={"observed": "y_true", "predicted": "y_pred"})
    arm_a["seed"] = 0
    arm_b = arm_a.copy()
    arm_b["y_pred"] = arm_b["y_pred"] + 0.1  # 面板与标签不变的第二臂
    assert_comparable(arm_a, arm_b)


# ---------------------------------------------------------------------------
# 第 9 项（SubTask 4.2）：正式入口最小运行留痕
# ---------------------------------------------------------------------------


def test_09a_materialize_exp030_validation_record():
    """正式入口 ``scripts/data/materialize_exp030_keyed_predictions.py``。

    该入口已于 2026-09-22 完整运行，本测试引用其落盘 validation JSON
    留痕：status=PASS、键值重建与行序重建逐位一致、三重校验（重建一致
    性 / arm_summary 逐折一致 / biobank 独立核对）全部通过。不重跑完整
    物化（10.9M 行）。
    """
    record_path = (ARTIFACTS / "exp030_dependency_transfer"
                   / "predictions_keyed_v2_validation.json")
    assert record_path.exists()
    record = json.loads(record_path.read_text())

    assert record["status"] == "PASS"
    assert record["generated_at"].startswith("2026-09-22")
    reconstruction = record["reconstruction"]
    assert reconstruction["bit_exact_vs_row_order_rebuild"] is True
    assert reconstruction["n_rows"] == 10_966_590
    assert reconstruction["n_genes"] == 13_539
    assert sorted(reconstruction["arms"]) == ["B0", "CAL", "DA", "HD", "ZS"]
    fold_check = record["fold_macro_check"]
    assert fold_check["max_abs_diff"] <= fold_check["tolerance_abs"]
    assert fold_check["paired_genes_per_fold_all_match"] is True
    external = record["observed_vs_biobank_dependency"]
    assert external["n_value_mismatch"] == 0
    assert external["n_nan_mask_mismatch"] == 0
    assert record["consistency"]["n_evaluable_genes_match"] is True
    print(f"[入口留痕] materialize_exp030_keyed_predictions: 已于 "
          f"{record['generated_at']} 完整运行，validation PASS "
          f"(n_rows={reconstruction['n_rows']}, "
          f"max_abs_diff={fold_check['max_abs_diff']:.2e} < "
          f"tol={fold_check['tolerance_abs']:.0e}, "
          f"biobank 独立核对 0 不匹配)")


def test_09b_unified_eval_entry_minimal_run():
    """统一评估入口最小运行：aggregate_seed_patient_scores +
    run_formal_comparison（canonical 链路）端到端执行并输出数字。

    最小合成数据：6 患者 × 5 药物 × 2 种子；模型臂（p0–p2 保序、
    p3–p5 扰序）vs 逐药均值基线臂。数据无随机源，数字确定；
    ``--capture=no`` 运行时留痕。
    """
    drugs = ["d0", "d1", "d2", "d3", "d4"]
    patients = ["p0", "p1", "p2", "p3", "p4", "p5"]
    observed = {
        (p, d): 10.0 * (i % 3) + 1.7 * j - ((i * j) % 4) * 0.3
        for i, p in enumerate(patients) for j, d in enumerate(drugs)
    }
    drug_mean = {
        d: sum(observed[(p, d)] for p in patients) / len(patients)
        for d in drugs
    }

    def _model_pred(patient: str, drug: str, seed: int) -> float:
        i = patients.index(patient)
        j = drugs.index(drug)
        if i < 3:
            # 前 3 患者：保序（种子仅作均匀平移，不改变药物秩）
            return observed[(patient, drug)] + 0.05 * seed
        # 后 3 患者：seed=0 完全逆序（rho=-1）；seed=1 相邻扰序（rho=0.9）
        if seed == 0:
            return observed[(patient, drugs[len(drugs) - 1 - j])]
        swapped = 1 - j if j in (0, 1) else j
        return observed[(patient, drugs[swapped])] + 0.05

    def _frame(predict_fn, seed: int) -> pd.DataFrame:
        """canonical 评估 schema：assert_comparable 需要 fold/seed/y_true/
        sample_id，主指标需要 patient_id/drug_id/observed/predicted。"""
        return pd.DataFrame([
            {"experiment_id": "e1", "contract_id": "c1", "split_id": "s1",
             "fold": 0, "seed": seed, "patient_id": p,
             "sample_id": f"m_{p}", "drug_id": d,
             "y_true": observed[(p, d)],
             "observed": observed[(p, d)], "predicted": predict_fn(p, d)}
            for p in patients for d in drugs
        ])

    baseline_fn = lambda p, d: drug_mean[d]
    model_by_seed = {0: _frame(lambda p, d: _model_pred(p, d, 0), 0),
                     1: _frame(lambda p, d: _model_pred(p, d, 1), 1)}
    baseline_by_seed = {0: _frame(baseline_fn, 0), 1: _frame(baseline_fn, 1)}

    # --- canonical 单种子比较入口：assert_comparable → 主指标 → 配对 bootstrap
    result = run_formal_comparison(
        model_by_seed[0], baseline_by_seed[0],
        comparison="model_minus_baseline", n_bootstrap=2000, seed=0)
    hand_delta = (
        result.model_patient_scores.to_numpy(dtype=float)
        - result.baseline_patient_scores.to_numpy(dtype=float))
    assert result.point_estimate == float(np.mean(hand_delta))
    assert result.low <= result.point_estimate <= result.high
    assert result.crosses_zero == (result.low <= 0.0 <= result.high)
    assert result.n_patients == len(patients)

    # --- 多种子聚合入口：逐种子患者分数 → 种子平均（主分析口径）
    model_agg = aggregate_seed_patient_scores(model_by_seed)
    baseline_agg = aggregate_seed_patient_scores(baseline_by_seed)
    # 种子平均 = 两种子分数的等权均值（全部患者双种子有定义）
    assert bool((model_agg.n_seeds_used == 2).all())
    hand_seed_mean = model_agg.per_seed_patient_scores.mean(axis=1)
    assert bool(np.array_equal(
        model_agg.seed_mean_patient_scores.to_numpy(),
        hand_seed_mean.to_numpy()))

    # --- 患者级配对 bootstrap（多种子主口径）
    ci = patient_score_paired_bootstrap(
        model_agg.seed_mean_patient_scores,
        baseline_agg.seed_mean_patient_scores, n=2000, seed=0)
    assert ci.n_units == len(patients)
    assert ci.low <= ci.point_estimate <= ci.high

    print("[入口最小运行] run_formal_comparison(model vs 逐药均值基线, "
          f"seed=0): point={result.point_estimate:.6f}, "
          f"95%CI=[{result.low:.6f}, {result.high:.6f}], "
          f"crosses_zero={result.crosses_zero}, n={result.n_patients}")
    print("[入口最小运行] aggregate_seed_patient_scores(2 seeds) → "
          f"seed_mean_scores: "
          f"{model_agg.seed_mean_patient_scores.round(4).to_dict()}")
    print("[入口最小运行] patient_score_paired_bootstrap(种子平均口径): "
          f"point={ci.point_estimate:.6f}, "
          f"95%CI=[{ci.low:.6f}, {ci.high:.6f}]")
