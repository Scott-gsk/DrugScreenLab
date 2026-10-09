#!/usr/bin/env python3
"""Fixed-source leave-one-cell-out reference selection for EXP-050 §3.2.

The nearest-source choice is committed from untreated DepMap expression before
the response ``delta`` member of the upstream NPZ is accessed. All comparisons
remain descriptive source-development diagnostics; no PDO or test-label input
is read.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_source_reference_selection_v1"
KEY_RE = re.compile(r"^[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z]$")
KEY_COLUMNS = ["inchi_key", "dose_key", "dose_unit_key", "time_key", "time_unit_key"]
# Conservative cumulative CPU upper bound from prior source-selection package
# work. Wall-only estimates are included as upper bounds, not exact CPU usage.
PREVIOUS_PACKAGE_CPU_UPPER_BOUND_SECONDS = 706.65
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261006
ZERO_TOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_tsv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, sep="\t", index=False, na_rep="", lineterminator="\n")
    os.replace(tmp, path)


def rel_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def config_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-050" or cfg.get("run_id") != RUN_ID:
        raise ValueError("配置实验或运行标识与冻结值不符")
    if int(cfg["cpu_cap_minutes"]) != 240 or float(cfg["gpu_hours"]) != 0:
        raise ValueError("本包资源上限须保持 CPU 240 分钟、GPU 0 小时")
    models = cfg["source_models"]
    if len(models) != 4 or len({x["model_id"] for x in models}) != 4:
        raise ValueError("固定来源必须是互异的四个模型")
    domain = cfg["expected_domain"]
    if domain != {"exact_InChIKeys": 108, "exposure_conditions": 1285, "measured_genes": 978, "hallmark_sets": 50}:
        raise ValueError(f"固定域配置被更改：{domain}")
    return cfg


def baseline_selection(cfg: dict[str, Any], arrays: Any, inputs: dict[str, Path], out: Path) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, list[str]]:
    """Choose nearest sources using baseline expression only; never touch delta."""
    models = cfg["source_models"]
    model_ids = [str(x["model_id"]) for x in models]
    gene_symbols = arrays["gene_symbols"].astype(str).tolist()  # NPZ metadata member only.
    npz_models = arrays["model_ids"].astype(str).tolist()
    if npz_models != model_ids:
        raise ValueError(f"响应资产模型轴与冻结顺序不一致：{npz_models}")
    if len(gene_symbols) != 978 or len(set(gene_symbols)) != 978:
        raise ValueError("响应资产978基因轴缺失或重复")

    identity = pd.read_csv(inputs["model_identity"], sep="\t", dtype=str, keep_default_na=False)
    need = {"dataset", "LINCS_base_cell_id", "DepMap_ModelID", "LegacyMolecularSubtype"}
    if not need.issubset(identity.columns):
        raise ValueError(f"身份表缺少列：{sorted(need - set(identity.columns))}")
    identity = identity.loc[identity.dataset.eq("GSE70138")].drop_duplicates("DepMap_ModelID")
    id_subtype = dict(zip(identity.DepMap_ModelID, identity.LegacyMolecularSubtype))
    for item in models:
        if id_subtype.get(item["model_id"]) != item["subtype"]:
            raise ValueError(f"来源身份/分型与冻结配置不一致：{item['model_id']}")

    model_info = pd.read_csv(inputs["depmap_models"], usecols=["ModelID", "LegacyMolecularSubtype"], dtype=str, keep_default_na=False)
    model_info = model_info.set_index("ModelID")
    for item in models:
        if item["model_id"] not in model_info.index or model_info.loc[item["model_id"], "LegacyMolecularSubtype"] != item["subtype"]:
            raise ValueError(f"DepMap 24Q2模型分型与冻结配置不一致：{item['model_id']}")

    # Bind the fixed LINCS symbols to DepMap columns through the GSE gene_info
    # Entrez IDs. This avoids silently choosing among duplicate gene symbols.
    gene_info = pd.read_csv(inputs["gse70138_gene_info"], sep="\t", compression="gzip", dtype=str, keep_default_na=False)
    if not {"pr_gene_id", "pr_gene_symbol"}.issubset(gene_info.columns):
        raise ValueError("GSE70138 gene_info缺少Entrez ID或官方symbol列")
    gene_info["pr_gene_id"] = gene_info["pr_gene_id"].astype(str).str.strip()
    gene_info["pr_gene_symbol"] = gene_info["pr_gene_symbol"].astype(str).str.strip()
    gene_info = gene_info.loc[gene_info.pr_gene_symbol.isin(gene_symbols)].copy()
    if gene_info.pr_gene_symbol.duplicated().any() or len(gene_info) != len(gene_symbols):
        raise ValueError("LINCS固定978基因轴在gene_info中不唯一或不完整")
    symbol_to_entrez = dict(zip(gene_info.pr_gene_symbol, gene_info.pr_gene_id))
    header = pd.read_csv(inputs["depmap_baseline_expression"], nrows=0).columns.astype(str).tolist()
    first_col = header[0]
    col_by_entrez: dict[str, list[str]] = {}
    for col in header[1:]:
        match = re.search(r"\((\d+)\)\s*$", col)
        if match:
            col_by_entrez.setdefault(match.group(1), []).append(col)
    selected_columns: list[str] = []
    missing_or_ambiguous: list[str] = []
    for symbol in gene_symbols:
        entrez = symbol_to_entrez.get(symbol, "")
        matches = col_by_entrez.get(entrez, [])
        if len(matches) != 1:
            missing_or_ambiguous.append(f"{symbol}:{entrez}:{len(matches)}")
        else:
            selected_columns.append(matches[0])
    if missing_or_ambiguous or len(selected_columns) != 978 or len(set(selected_columns)) != 978:
        raise ValueError(f"DepMap→LINCS 978轴存在缺失/重复映射：{missing_or_ambiguous[:12]}")

    expr = pd.read_csv(inputs["depmap_baseline_expression"], usecols=[first_col, *selected_columns], dtype={first_col: str})
    expr = expr.drop_duplicates(first_col).set_index(first_col)
    if expr.index.duplicated().any() or set(model_ids) - set(expr.index):
        raise ValueError("DepMap未处理表达缺少来源模型或ModelID重复")
    matrix = expr.loc[model_ids, selected_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.shape != (4, 978) or not np.isfinite(matrix).all():
        raise ValueError(f"DepMap基线表达轴非有限或形状错误：{matrix.shape}")

    rows: list[dict[str, Any]] = []
    chosen: dict[str, str] = {}
    for i, item in enumerate(models):
        candidates: list[tuple[int, str, float, str]] = []
        for j, candidate in enumerate(models):
            if i == j:
                continue
            corr = spearman(matrix[i], matrix[j])
            candidates.append((j, candidate["model_id"], corr, candidate["label"]))
        # Highest pre-treatment baseline Spearman; ties resolve by frozen order.
        best = sorted(candidates, key=lambda x: (-x[2], x[0]))[0]
        chosen[item["model_id"]] = best[1]
        for rank, (j, candidate_id, corr, label) in enumerate(sorted(candidates, key=lambda x: (-x[2], x[0])), 1):
            rows.append({
                "heldout_model_id": item["model_id"], "heldout_label": item["label"],
                "heldout_subtype": item["subtype"], "selection_basis": "DepMap 24Q2 untreated log2(TPM+1), frozen 978 LINCS gene axis",
                "gene_axis_count": 978, "candidate_model_id": candidate_id, "candidate_label": label,
                "candidate_subtype": models[j]["subtype"], "baseline_spearman": corr,
                "rank": rank, "selected_nearest": candidate_id == best[1],
                "selection_status": "PRETREATMENT_BASELINE_ONLY",
            })
    selection = pd.DataFrame(rows)
    sel_path = out / "selection" / "nearest_neighbor_selection.tsv"
    write_tsv(sel_path, selection)
    readback = pd.read_csv(sel_path, sep="\t", dtype={"heldout_model_id": str, "candidate_model_id": str})
    if len(readback) != 12 or readback.selected_nearest.sum() != 4 or not readback.selection_status.eq("PRETREATMENT_BASELINE_ONLY").all():
        raise ValueError("最近邻选择表读回异常")
    receipt = {
        "status": "PRETREATMENT_SELECTION_COMMITTED_BEFORE_RESPONSE_DELTA_ACCESS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "selection_table": rel_path(sel_path),
        "selection_table_sha256": sha256(sel_path),
        "selection_rows_read_back": len(readback),
        "heldout_source_count": 4,
        "candidate_count_per_target": 3,
        "gene_axis": {"count": 978, "unique": True, "binding": "GSE70138 pr_gene_id to DepMap 24Q2 parenthesized Entrez columns; exact symbol crosswalk"},
        "depmap_models_subtype_matches": True,
        "source_identity_subtype_matches": True,
        "chosen_model_by_target": chosen,
        "response_firewall": {
            "response_npz_header_and_gene_symbols_accessed": True,
            "response_npz_delta_member_accessed": False,
            "delta_access_occurs_only_after_selection_table_and_receipt_readback": True,
            "selection_inputs": [rel_path(inputs["depmap_baseline_expression"]), rel_path(inputs["depmap_models"]), rel_path(inputs["model_identity"]), rel_path(inputs["gse70138_gene_info"])],
            "forbidden": ["PDO drug labels", "outer/locked labels", "GSE216540 sample-level metadata"],
        },
    }
    receipt_path = out / "selection" / "selection_receipt.json"
    write_json(receipt_path, receipt)
    reread = json.loads(receipt_path.read_text(encoding="utf-8"))
    if reread["response_firewall"]["response_npz_delta_member_accessed"] is not False or sha256(sel_path) != reread["selection_table_sha256"]:
        raise ValueError("最近邻防火墙回执或表格哈希读回失败")
    return readback, reread, matrix, gene_symbols


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= ZERO_TOL or np.std(y) <= ZERO_TOL:
        return float("nan")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def compare_vectors(target: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(target) & np.isfinite(pred)
    x, y = target[valid].astype(float), pred[valid].astype(float)
    nz = (np.abs(x) > ZERO_TOL) & (np.abs(y) > ZERO_TOL)
    same = np.sign(x[nz]) == np.sign(y[nz])
    diff = y - x
    return {
        "gene_spearman_rho": spearman(x, y),
        "gene_nonzero_sign_agreement": float(same.mean()) if same.size else float("nan"),
        "n_genes_both_nonzero": int(nz.sum()),
        "n_genes_scored": int(valid.sum()),
        "gene_rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan"),
        "gene_mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
        "gene_signed_bias_prediction_minus_target": float(np.mean(diff)) if diff.size else float("nan"),
    }


def parse_hallmarks(path: Path, symbols: list[str]) -> dict[str, np.ndarray]:
    gene_idx = {s: i for i, s in enumerate(symbols)}
    result: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            idx = sorted({gene_idx[g] for g in fields[2:] if g in gene_idx})
            result[fields[0]] = np.asarray(idx, dtype=int)
    if len(result) != 50:
        raise ValueError(f"固定Hallmark文件应有50个集合，实际{len(result)}")
    return result


def score_direction(target: float, pred: float) -> tuple[str, bool | None]:
    tz, pz = abs(target) <= ZERO_TOL, abs(pred) <= ZERO_TOL
    if tz and pz:
        return "target_zero_and_prediction_zero", None
    if tz:
        return "target_zero_prediction_nonzero", None
    if pz:
        return "target_nonzero_prediction_zero", None
    same = bool(np.sign(target) == np.sign(pred))
    return ("same_direction" if same else "opposite_direction"), same


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--attempt-id", help="phase=full使用新的不可覆盖attempt目录，例如attempt_003")
    parser.add_argument("--manager-readback", help="phase=full时由root填写已PASS的单对象烟测读回回执路径")
    args = parser.parse_args()
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    config_file = config_path(ROOT, args.config)
    cfg = load_config(config_file)
    if args.run_id != RUN_ID:
        raise ValueError(f"--run-id必须为{RUN_ID}")
    run_root = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
    smoke_attempt = run_root / "attempt_002"
    if args.phase == "smoke":
        attempt_id = args.attempt_id or "attempt_002"
        if attempt_id != "attempt_002":
            raise ValueError("本冻结烟测只能使用已预留attempt_002")
        out = smoke_attempt
        if out.exists():
            raise FileExistsError(f"attempt_002已存在，拒绝覆盖或重跑：{out}")
        out.mkdir(parents=True)
        smoke_dir = out / "smoke"
        selection_dir = out / "selection"
    else:
        attempt_id = args.attempt_id or "attempt_003"
        if not re.fullmatch(r"attempt_[0-9]{3}", attempt_id) or attempt_id == "attempt_002":
            raise ValueError("phase=full必须指定新的attempt编号，不得写入attempt_002")
        if not smoke_attempt.is_dir():
            raise FileNotFoundError("phase=full必须从已验收的attempt_002 smoke继续")
        out = run_root / attempt_id / "full"
        if out.exists():
            raise FileExistsError(f"拒绝覆盖已有full attempt目录：{out}")
        smoke_dir = smoke_attempt / "smoke"
        selection_dir = smoke_attempt / "selection"
        smoke_receipt_path = smoke_dir / "smoke_receipt.json"
        if not smoke_receipt_path.is_file():
            raise FileNotFoundError("phase=full被门控：缺少完整单目标smoke回执")
        smoke_gate = json.loads(smoke_receipt_path.read_text(encoding="utf-8"))
        if smoke_gate.get("status") != "REAL_FULL_TARGET_SMOKE_PASS" or smoke_gate.get("fixed_conditions") != 1285 or smoke_gate.get("exact_full_InChIKeys") != 108:
            raise ValueError("phase=full被门控：烟测未通过1285条件/108键验收")
        if not args.manager_readback:
            raise ValueError("phase=full必须提供root独立读回回执路径")
        manager_receipt_path = config_path(ROOT, args.manager_readback)
        manager_receipt = json.loads(manager_receipt_path.read_text(encoding="utf-8"))
        if manager_receipt.get("status") != "ROOT_INDEPENDENT_READBACK_PASS" or manager_receipt.get("run_id") != RUN_ID or manager_receipt.get("smoke_receipt_sha256") != sha256(smoke_receipt_path):
            raise ValueError("phase=full被门控：root读回回执状态或烟测哈希不匹配")
        out.mkdir(parents=True)
    log_path = out / "run.log"
    inputs = {key: config_path(ROOT, value) for key, value in cfg["inputs"].items()}
    missing = [key for key, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"配置输入文件缺失：{missing}")
    arrays = None
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    response_archive = np.load(inputs["response_vectors"], allow_pickle=False)
    # Load only archive metadata before pre-treatment-based selection. The delta
    # member is deliberately deferred until selection receipt is written/read.
    npz_keys = response_archive["exposure_keys"].astype(str)
    npz_models = response_archive["model_ids"].astype(str).tolist()
    cfg_models = [str(x["model_id"]) for x in cfg["source_models"]]
    if npz_models != cfg_models or npz_keys.shape != (1285, 5):
        raise ValueError(f"上游响应资产固定键或模型轴异常：models={npz_models}, keys={npz_keys.shape}")
    if len({tuple(x) for x in npz_keys.tolist()}) != 1285:
        raise ValueError("上游响应资产条件键重复")

    # Domain check is label-blind: only full chemistry/dose/time identities.
    domain = pd.read_csv(inputs["parent_domain"], sep="\t", dtype=str, keep_default_na=False)
    if set(KEY_COLUMNS) - set(domain.columns):
        raise ValueError(f"父级固定域缺少键字段：{sorted(set(KEY_COLUMNS) - set(domain.columns))}")
    domain["inchi_key"] = domain.inchi_key.str.strip().str.upper()
    strict = domain.loc[domain.inchi_key.map(lambda x: bool(KEY_RE.fullmatch(x))), KEY_COLUMNS].copy()
    vector_domain = {tuple(map(str, row)) for row in npz_keys.tolist()}
    strict_domain = {tuple(map(str, row)) for row in strict.astype(str).to_numpy().tolist()}
    if len(strict) != 1285 or strict.inchi_key.nunique() != 108 or strict_domain != vector_domain:
        raise ValueError(f"严格父域与已验收响应资产不一致：rows={len(strict)}, keys={strict.inchi_key.nunique()}, vector={len(vector_domain)}")
    condition_metrics = pd.read_csv(inputs["strict_condition_metrics"], sep="\t", dtype=str, keep_default_na=False)
    cond_domain = {tuple(map(str, row)) for row in condition_metrics[KEY_COLUMNS].to_numpy().tolist()}
    if len(condition_metrics) != 1285 or cond_domain != vector_domain:
        raise ValueError("上游逐条件指标与响应轴键不一致")

    # Freeze nearest-source choice before opening response_archive["delta"].
    if args.phase == "smoke":
        selection_rb, selection_receipt, baseline_matrix, gene_symbols = baseline_selection(cfg, response_archive, inputs, out)
        selection_time_utc = datetime.now(timezone.utc).isoformat()
        write_json(out / "selection" / "selection_committed_marker.json", {
            "status": "COMMITTED_BEFORE_RESPONSE_ARRAY_ACCESS", "run_id": RUN_ID,
            "committed_at_utc": selection_time_utc,
            "selection_receipt_sha256": sha256(out / "selection" / "selection_receipt.json"),
            "nearest_selection_sha256": sha256(out / "selection" / "nearest_neighbor_selection.tsv"),
            "delta_member_accessed": False,
        })
    else:
        selection_path = selection_dir / "nearest_neighbor_selection.tsv"
        selection_receipt_path = selection_dir / "selection_receipt.json"
        marker_path = selection_dir / "selection_committed_marker.json"
        selection_rb = pd.read_csv(selection_path, sep="\t", dtype={"heldout_model_id": str, "candidate_model_id": str})
        selection_receipt = json.loads(selection_receipt_path.read_text(encoding="utf-8"))
        marker_existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker_existing.get("delta_member_accessed") is not False or marker_existing.get("nearest_selection_sha256") != sha256(selection_path) or selection_receipt.get("selection_table_sha256") != sha256(selection_path):
            raise ValueError("phase=full最近邻选择/防火墙文件读回失败")
        gene_symbols = response_archive["gene_symbols"].astype(str).tolist()
    marker_path = selection_dir / "selection_committed_marker.json"
    marker_rb = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker_rb["delta_member_accessed"] is not False:
        raise ValueError("选择冻结标记回读失败")

    # First true held-out smoke: all 1,285 exact conditions for HS578T. The
    # manager reviews this complete one-object chain before phase=full runs.
    delta = response_archive["delta"]
    if delta.shape != (1285, 4, 978) or not np.isfinite(delta).all():
        raise ValueError(f"响应delta轴错误或含非有限值：{delta.shape}")
    vector_keys = [tuple(map(str, row)) for row in npz_keys.tolist()]
    model_idx = {mid: i for i, mid in enumerate(cfg_models)}
    model_meta = {m["model_id"]: m for m in cfg["source_models"]}
    selection_map = selection_rb.loc[selection_rb.selected_nearest].set_index("heldout_model_id").candidate_model_id.to_dict()
    source_order = cfg_models
    arms = ["all_source", "exact_subtype", "pretreatment_nearest", *[f"random_n1__{mid}" for mid in source_order]]

    # Independent fixed-domain metric mapping and resource metadata.
    hallmark_sets = parse_hallmarks(inputs["hallmark_gmt"], gene_symbols)
    if args.phase == "smoke":
        smoke_i = model_idx["ACH-000148"]
        smoke_target_id = source_order[smoke_i]
        smoke_peers = [m for m in source_order if m != smoke_target_id]
        smoke_subtype = [m for m in smoke_peers if model_meta[m]["subtype"] == model_meta[smoke_target_id]["subtype"]]
        smoke_rows: list[dict[str, Any]] = []
        smoke_predictions = np.full((1285, 7, 978), np.nan, dtype=np.float32)
        smoke_arms = ["all_source", "exact_subtype", "pretreatment_nearest", *[f"random_n1__{mid}" for mid in source_order]]
        for smoke_k, exposure_key in enumerate(vector_keys):
            smoke_target = delta[smoke_k, smoke_i].astype(float)
            smoke_predictions[smoke_k, 0] = delta[smoke_k, [model_idx[m] for m in smoke_peers]].mean(axis=0)
            smoke_predictions[smoke_k, 1] = delta[smoke_k, [model_idx[m] for m in smoke_subtype]].mean(axis=0)
            smoke_predictions[smoke_k, 2] = delta[smoke_k, model_idx[selection_map[smoke_target_id]]]
            for arm_i, candidate in enumerate(source_order, start=3):
                if candidate != smoke_target_id:
                    smoke_predictions[smoke_k, arm_i] = delta[smoke_k, model_idx[candidate]]
            for arm_i, arm in enumerate(smoke_arms):
                pred = smoke_predictions[smoke_k, arm_i].astype(float)
                if not np.isfinite(pred).all():
                    continue
                if arm == "all_source":
                    refs = smoke_peers
                elif arm == "exact_subtype":
                    refs = smoke_subtype
                elif arm == "pretreatment_nearest":
                    refs = [selection_map[smoke_target_id]]
                else:
                    candidate = arm.split("__", 1)[1]
                    if candidate == smoke_target_id:
                        continue
                    refs = [candidate]
                smoke_rows.append({"heldout_model_id": smoke_target_id, "heldout_subtype": "basal_B", **dict(zip(KEY_COLUMNS, exposure_key)), "arm": arm, "reference_status": "AVAILABLE", "source_count": len(refs), "reference_model_ids": ";".join(refs), **compare_vectors(smoke_target, pred)})
        smoke_metrics = pd.DataFrame(smoke_rows)
        write_tsv(smoke_dir / "per_condition_metrics.tsv", smoke_metrics)
        if len(smoke_metrics.loc[smoke_metrics.arm.eq("all_source")]) != 1285 or len(smoke_metrics.loc[smoke_metrics.arm.eq("exact_subtype")]) != 1285:
            raise ValueError("HS578T全域smoke未覆盖1285个条件的all/subtype臂")
        drug_rows = []
        metric_names_smoke = ["gene_spearman_rho", "gene_nonzero_sign_agreement", "n_genes_both_nonzero", "gene_rmse", "gene_mae", "gene_signed_bias_prediction_minus_target"]
        for (inchi, arm), group in smoke_metrics.groupby(["inchi_key", "arm"], sort=True):
            row = {"heldout_model_id": smoke_target_id, "heldout_subtype": "basal_B", "inchi_key": inchi, "arm": arm, "n_fixed_conditions": len(group), "source_count": int(group.source_count.iloc[0]), "reference_model_ids": group.reference_model_ids.iloc[0]}
            row.update({m: pd.to_numeric(group[m], errors="coerce").mean() for m in metric_names_smoke})
            drug_rows.append(row)
        smoke_drug = pd.DataFrame(drug_rows)
        write_tsv(smoke_dir / "per_drug_metrics.tsv", smoke_drug)
        np.savez_compressed(smoke_dir / "predictions.npz", exposure_keys=np.asarray(npz_keys, dtype=str), target_model_id=np.asarray([smoke_target_id], dtype=str), arms=np.asarray(smoke_arms, dtype=str), gene_symbols=np.asarray(gene_symbols, dtype=str), target_delta=delta[:, smoke_i, :].astype(np.float32), predictions=smoke_predictions)
        cond_rb = pd.read_csv(smoke_dir / "per_condition_metrics.tsv", sep="\t", dtype=str, keep_default_na=False)
        drug_rb = pd.read_csv(smoke_dir / "per_drug_metrics.tsv", sep="\t", dtype=str, keep_default_na=False)
        pred_rb = np.load(smoke_dir / "predictions.npz", allow_pickle=False)
        if len(cond_rb.loc[cond_rb.arm.eq("all_source")]) != 1285 or len(cond_rb.loc[cond_rb.arm.eq("exact_subtype")]) != 1285 or cond_rb.inchi_key.nunique() != 108:
            raise ValueError("HS578T逐条件读回行数/药物键失败")
        if len(drug_rb.loc[drug_rb.arm.eq("all_source")]) != 108 or len(drug_rb.loc[drug_rb.arm.eq("exact_subtype")]) != 108:
            raise ValueError("HS578T逐药读回行数失败")
        if pred_rb["predictions"].shape != (1285, 7, 978) or not np.array_equal(pred_rb["exposure_keys"].astype(str), npz_keys):
            raise ValueError("HS578T全域预测NPZ读回形状/键轴失败")
        smoke_receipt = {
            "status": "REAL_FULL_TARGET_SMOKE_PASS", "run_id": RUN_ID,
            "target_model_id": smoke_target_id, "target_subtype": "basal_B",
            "fixed_conditions": 1285, "exact_full_InChIKeys": 108,
            "reference_arms": ["all_source", "exact_subtype", "pretreatment_nearest", "random_n1_enumeration"],
            "n1_random_reference_count": 3, "same_subtype_source_count": 1,
            "selection_receipt_sha256": sha256(out / "selection" / "selection_receipt.json"),
            "selection_committed_before_delta_access": True, "response_delta_loaded_after_selection_commit": True,
            "condition_metric_rows": len(cond_rb), "condition_metric_domain_key_count": int(cond_rb.inchi_key.nunique()),
            "all_source_condition_rows": int(len(cond_rb.loc[cond_rb.arm.eq("all_source")])),
            "exact_subtype_condition_rows": int(len(cond_rb.loc[cond_rb.arm.eq("exact_subtype")])),
            "all_source_drug_rows": int(len(drug_rb.loc[drug_rb.arm.eq("all_source")])),
            "exact_subtype_drug_rows": int(len(drug_rb.loc[drug_rb.arm.eq("exact_subtype")])),
            "prediction_shape": list(pred_rb["predictions"].shape), "prediction_key_match": True,
            "missing_subtype_reference_targets": ["ACH-000536", "ACH-000019"],
            "missing_subtype_handling": "NA; no imputation",
            "gpu_hours": 0.0, "full_four_target_scoring_started": False,
        }
        write_json(smoke_dir / "smoke_receipt.json", smoke_receipt)
        print(json.dumps(smoke_receipt, ensure_ascii=False, indent=2), flush=True)
        if args.phase == "smoke":
            return 0

    elif args.phase == "full":
        smoke_receipt = json.loads((smoke_dir / "smoke_receipt.json").read_text(encoding="utf-8"))
        if smoke_receipt.get("status") != "REAL_FULL_TARGET_SMOKE_PASS" or smoke_receipt.get("fixed_conditions") != 1285:
            raise ValueError("完整评价只能在单目标全域smoke经管理者验收后启动")
        smoke_rb = pd.read_csv(smoke_dir / "per_condition_metrics.tsv", sep="\t", dtype={"heldout_model_id": str, "inchi_key": str})
        if len(smoke_rb) != int(smoke_receipt.get("condition_metric_rows", -1)) or len(smoke_rb) != 7710:
            raise ValueError("已验收烟测逐条件评分读回行数与回执不一致")

    # Fixed source leave-one-cell-out predictions over all exact conditions.
    predictions = np.full((4, 1285, len(arms), 978), np.nan, dtype=np.float32)
    for ti, target_id in enumerate(source_order):
        target_meta = model_meta[target_id]
        peer_ids = [m for m in source_order if m != target_id]
        peer_idxs = [model_idx[m] for m in peer_ids]
        subtype_ids = [m for m in peer_ids if model_meta[m]["subtype"] == target_meta["subtype"]]
        predictions[ti, :, 0, :] = delta[:, peer_idxs, :].mean(axis=1)
        if subtype_ids:
            predictions[ti, :, 1, :] = delta[:, [model_idx[m] for m in subtype_ids], :].mean(axis=1)
        predictions[ti, :, 2, :] = delta[:, model_idx[selection_map[target_id]], :]
        for ai, source_id in enumerate(source_order, start=3):
            if source_id != target_id:
                predictions[ti, :, ai, :] = delta[:, model_idx[source_id], :]
    pred_path = out / "condition_predictions.npz"
    np.savez_compressed(pred_path, exposure_keys=np.asarray(npz_keys, dtype=str), target_model_ids=np.asarray(source_order, dtype=str), arms=np.asarray(arms, dtype=str), gene_symbols=np.asarray(gene_symbols, dtype=str), target_delta=np.transpose(delta, (1, 0, 2)).astype(np.float32), predictions=predictions)

    # Per-condition scores; exact-subtype is missing for singleton subtypes.
    condition_rows: list[dict[str, Any]] = []
    condition_arm_rows: list[dict[str, Any]] = []
    for ti, target_id in enumerate(source_order):
        target_meta = model_meta[target_id]
        target_idx = model_idx[target_id]
        peer_ids = [m for m in source_order if m != target_id]
        subtype_ids = [m for m in peer_ids if model_meta[m]["subtype"] == target_meta["subtype"]]
        for ci, key in enumerate(vector_keys):
            target = delta[ci, target_idx].astype(float)
            for ai, arm in enumerate(arms):
                pred = predictions[ti, ci, ai].astype(float)
                if not np.isfinite(pred).all():
                    if arm == "exact_subtype":
                        condition_rows.append({"heldout_model_id": target_id, "heldout_label": target_meta["label"], "heldout_subtype": target_meta["subtype"], **dict(zip(KEY_COLUMNS, key)), "arm": arm, "reference_status": "MISSING_NO_OTHER_FIXED_SOURCE_OF_EXACT_SUBTYPE", "source_count": 0, "reference_model_ids": "", **{k: np.nan for k in ["gene_spearman_rho", "gene_nonzero_sign_agreement", "n_genes_both_nonzero", "n_genes_scored", "gene_rmse", "gene_mae", "gene_signed_bias_prediction_minus_target"]}})
                    continue
                if arm == "all_source":
                    refs, status = peer_ids, "AVAILABLE"
                elif arm == "exact_subtype":
                    refs, status = subtype_ids, "AVAILABLE"
                elif arm == "pretreatment_nearest":
                    refs, status = [selection_map[target_id]], "AVAILABLE_BASELINE_NEAREST"
                else:
                    ref_id = arm.split("__", 1)[1]
                    if ref_id == target_id:
                        continue
                    refs, status = [ref_id], "RANDOM_SINGLE_SOURCE_CANDIDATE"
                metrics = compare_vectors(target, pred)
                row = {"heldout_model_id": target_id, "heldout_label": target_meta["label"], "heldout_subtype": target_meta["subtype"], **dict(zip(KEY_COLUMNS, key)), "arm": arm, "reference_status": status, "source_count": len(refs), "reference_model_ids": ";".join(refs), **metrics}
                condition_rows.append(row)
                condition_arm_rows.append(row)
    cond_metrics = pd.DataFrame(condition_rows)
    write_tsv(out / "per_condition_metrics.tsv", cond_metrics)

    # Drug-level metrics give each fixed dose/time condition equal weight within
    # a full InChIKey. The three random one-source candidates are retained and
    # also summarized as the exact finite random-reference mean per key.
    metric_names = ["gene_spearman_rho", "gene_nonzero_sign_agreement", "n_genes_both_nonzero", "gene_rmse", "gene_mae", "gene_signed_bias_prediction_minus_target"]
    scored = cond_metrics.loc[cond_metrics.reference_status.ne("MISSING_NO_OTHER_FIXED_SOURCE_OF_EXACT_SUBTYPE")].copy()
    for c in metric_names + ["source_count"]:
        scored[c] = pd.to_numeric(scored[c], errors="coerce")
    per_drug = scored.groupby(["heldout_model_id", "heldout_label", "heldout_subtype", "inchi_key", "arm", "reference_status", "reference_model_ids"], dropna=False, sort=True).agg(
        n_fixed_conditions=("dose_key", "size"), **{m: (m, "mean") for m in metric_names}, source_count=("source_count", "first")
    ).reset_index()
    random = per_drug.loc[per_drug.arm.str.startswith("random_n1__")].copy()
    random_ensemble = random.groupby(["heldout_model_id", "heldout_label", "heldout_subtype", "inchi_key"], sort=True).agg(
        n_fixed_conditions=("n_fixed_conditions", "max"), **{m: (m, "mean") for m in metric_names}, random_reference_count=("arm", "nunique")
    ).reset_index()
    random_ensemble["arm"] = "random_n1_equal_subset_metric_mean"
    random_ensemble["reference_status"] = "FINITE_ENUMERATION_OF_ALL_THREE_ELIGIBLE_SINGLE_SOURCES"
    random_ensemble["reference_model_ids"] = "all_three_other_fixed_sources_equally_weighted_as_reference_distribution"
    random_ensemble["source_count"] = 1
    per_drug = pd.concat([per_drug, random_ensemble], ignore_index=True, sort=False)
    write_tsv(out / "per_drug_model_arm_metrics.tsv", per_drug)

    # Hallmark directions use equal condition aggregation within each drug, then
    # average gene changes inside each fixed set; retain all four sign cases.
    hallmark_rows: list[dict[str, Any]] = []
    for ti, target_id in enumerate(source_order):
        target_idx = model_idx[target_id]
        for key_id in sorted({k[0] for k in vector_keys}):
            condition_indices = [i for i, k in enumerate(vector_keys) if k[0] == key_id]
            target_vec = delta[condition_indices, target_idx, :].mean(axis=0)
            for ai, arm in enumerate(arms):
                if arm == "exact_subtype" and not np.isfinite(predictions[ti, condition_indices, ai, :]).all():
                    continue
                if arm.startswith("random_n1__") and arm.split("__", 1)[1] == target_id:
                    continue
                pred_vec = predictions[ti, condition_indices, ai, :].mean(axis=0)
                for name, idx in hallmark_sets.items():
                    if idx.size == 0:
                        hallmark_rows.append({"heldout_model_id": target_id, "inchi_key": key_id, "arm": arm, "hallmark": name, "n_axis_genes": 0, "target_hallmark_mean_delta": np.nan, "prediction_hallmark_mean_delta": np.nan, "direction_category": "NO_OVERLAP_WITH_FIXED_978_GENE_AXIS", "direction_agreement": np.nan})
                        continue
                    tv, pv = float(target_vec[idx].mean()), float(pred_vec[idx].mean())
                    category, agreement = score_direction(tv, pv)
                    hallmark_rows.append({"heldout_model_id": target_id, "inchi_key": key_id, "arm": arm, "hallmark": name, "n_axis_genes": len(idx), "target_hallmark_mean_delta": tv, "prediction_hallmark_mean_delta": pv, "direction_category": category, "direction_agreement": agreement})
    hallmark_frame = pd.DataFrame(hallmark_rows)
    write_tsv(out / "per_drug_hallmark_directions.tsv", hallmark_frame)
    hall_summary = hallmark_frame.groupby(["heldout_model_id", "arm", "hallmark", "direction_category"], dropna=False).size().rename("n_full_InChIKeys").reset_index()
    write_tsv(out / "hallmark_direction_summary.tsv", hall_summary)
    opposite = hallmark_frame.loc[hallmark_frame.direction_category.eq("opposite_direction")].sort_values(["heldout_model_id", "arm", "hallmark", "inchi_key"])
    write_tsv(out / "hallmark_counterexamples.tsv", opposite)

    # Paired bootstrap over the same observed exact chemical keys. Report each
    # target source separately; no patient/subtype population inference.
    pair_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [("pretreatment_nearest", "all_source"), ("pretreatment_nearest", "random_n1_equal_subset_metric_mean"), ("exact_subtype", "all_source"), ("exact_subtype", "random_n1_equal_subset_metric_mean")]
    for target_id in source_order:
        target_drug = per_drug.loc[per_drug.heldout_model_id.eq(target_id)]
        for left_arm, right_arm in comparisons:
            for metric in metric_names:
                if metric == "n_genes_both_nonzero":
                    continue
                left = target_drug.loc[target_drug.arm.eq(left_arm), ["inchi_key", metric]].rename(columns={metric: "left"})
                right = target_drug.loc[target_drug.arm.eq(right_arm), ["inchi_key", metric]].rename(columns={metric: "right"})
                joined = left.merge(right, on="inchi_key", validate="one_to_one")
                valid = joined.left.notna() & joined.right.notna()
                values = (joined.loc[valid, "left"] - joined.loc[valid, "right"]).to_numpy(dtype=float)
                if not len(values):
                    continue
                estimate, low, high = bootstrap_mean_ci(values, rng, BOOTSTRAP_N)
                pair_rows.append({"heldout_model_id": target_id, "comparison": f"{left_arm}_minus_{right_arm}", "metric": metric, "n_paired_full_InChIKeys": int(len(values)), "estimate_equal_key_paired_difference": estimate})
                boot_rows.append({"heldout_model_id": target_id, "comparison": f"{left_arm}_minus_{right_arm}", "metric": metric, "bootstrap_replicates": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "paired_unit": "exact full InChIKey", "n_paired_full_InChIKeys": int(len(values)), "estimate_equal_key_paired_difference": estimate, "ci95_low": low, "ci95_high": high, "interpretation": "descriptive for observed drug keys only; not patient or subtype population inference"})
    write_tsv(out / "paired_arm_differences.tsv", pair_rows)
    write_tsv(out / "paired_bootstrap_ci.tsv", boot_rows)

    # Random-reference combinations and the n=3 degeneracy are explicit.
    combos = []
    for target_id in source_order:
        candidates = [m for m in source_order if m != target_id]
        for ref in candidates:
            combos.append({"heldout_model_id": target_id, "reference_type": "random_n1", "subset_model_ids": ref, "source_count": 1, "enumeration_probability": 1 / len(candidates), "same_as_nearest_selection": ref == selection_map[target_id]})
        combos.append({"heldout_model_id": target_id, "reference_type": "random_n3_degenerate", "subset_model_ids": ";".join(candidates), "source_count": 3, "enumeration_probability": 1.0, "same_as_all_source_mean": True, "note": "唯一n=3子集正是all_source，不作为独立随机对照"})
    write_tsv(out / "random_reference_combinations.tsv", combos)

    # Independent local readback: predictions preserve the exact frozen axes.
    pred_rb = np.load(pred_path, allow_pickle=False)
    if pred_rb["predictions"].shape != (4, 1285, len(arms), 978) or not np.array_equal(pred_rb["exposure_keys"].astype(str), npz_keys):
        raise ValueError("预测NPZ键轴或形状读回失败")
    if len(cond_metrics.loc[cond_metrics.arm.eq("all_source")]) != 4 * 1285:
        raise ValueError("每目标的all-source逐条件评分行数不等于4×1285")
    if len(per_drug.loc[per_drug.arm.eq("all_source")]) != 4 * 108:
        raise ValueError("每目标的all-source逐药行数不等于4×108")
    if len(selection_rb) != 12 or marker_rb["delta_member_accessed"] is not False:
        raise ValueError("基线选择回执或选择表读回不通过")

    # One line per assay-missing subtype arm is retained in summary.
    missing_subtype = [m["model_id"] for m in cfg["source_models"] if not any(other["model_id"] != m["model_id"] and other["subtype"] == m["subtype"] for other in cfg["source_models"])]
    # Derive an evidence-bounded fixed-source comparison statement.
    primary_summary = []
    for target_id in source_order:
        sub = per_drug.loc[per_drug.heldout_model_id.eq(target_id)]
        for arm in ["all_source", "exact_subtype", "pretreatment_nearest", "random_n1_equal_subset_metric_mean"]:
            x = sub.loc[sub.arm.eq(arm)]
            if len(x):
                primary_summary.append({"heldout_model_id": target_id, "arm": arm, "n_full_InChIKeys": int(x.inchi_key.nunique()), **{m: float(pd.to_numeric(x[m], errors="coerce").mean()) for m in metric_names if m != "n_genes_both_nonzero"}})
    write_tsv(out / "per_target_arm_summary.tsv", primary_summary)

    current_attempt_cpu_seconds = time.process_time() - start_cpu
    cpu_seconds = current_attempt_cpu_seconds + PREVIOUS_PACKAGE_CPU_UPPER_BOUND_SECONDS
    wall_seconds = time.monotonic() - start_wall
    cpu_cap = int(cfg["cpu_cap_minutes"]) * 60
    if cpu_seconds > cpu_cap:
        raise RuntimeError(f"CPU时间超上限：{cpu_seconds:.3f}s > {cpu_cap}s")
    output_files = [p for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"sha256_manifest.tsv", "run_manifest.json"}]
    manifest_rows = [{"path": rel_path(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in output_files]
    write_tsv(out / "sha256_manifest.tsv", manifest_rows)
    summary = {
        "status": "COMPLETED_FIXED_SOURCE_REFERENCE_SELECTION_DEVELOPMENT_ONLY",
        "experiment_id": "EXP-050", "run_id": RUN_ID,
        "scope": "four fixed GSE70138 source cell lines; leave-one-source-cell-out descriptive development diagnosis; not PDO or patient validation",
        "domain": {"exact_full_InChIKeys": 108, "exact_conditions": 1285, "gene_axis": 978, "hallmark_sets": 50},
        "source_models": cfg["source_models"],
        "fifth_source_gate": "ACH-000017/SKBR3 excluded: 1282/1285 qualified response plus matched vehicle conditions; fixed four-source denominator retained",
        "nearest_selection": selection_receipt,
        "missing_subtype_references": missing_subtype,
        "random_reference_note": "n=1 enumerates each of the three eligible other sources; n=3 has one combination identical to all_source and is degenerate, not an independent control",
        "resource_accounting": {"cpu_cap_seconds": cpu_cap, "cpu_seconds_including_prior_package_upper_bound": cpu_seconds, "current_attempt_cpu_seconds": current_attempt_cpu_seconds, "prior_package_cpu_upper_bound_seconds": PREVIOUS_PACKAGE_CPU_UPPER_BOUND_SECONDS, "wall_seconds": wall_seconds, "gpu_hours": 0.0, "accounting_note": "Prior root reader CPU is exact where available; smoke and aborted attempts without final counters are charged using conservative wall-based upper bounds. System-python probe failed before data access and rounded to zero."},
        "metrics": {"conditions_first_within_key_equal_weight": True, "keys_equal_weight_across_drugs": True, "paired_unit": "full exact InChIKey", "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED},
        "hallmark_summary_rows": len(hall_summary), "hallmark_opposite_direction_rows": len(opposite),
        "selection_firewall": {"nearest_selection_committed_before_delta_member_access": True, "marker": rel_path(marker_path)},
        "readback": {"selection_rows": len(selection_rb), "smoke_rows": len(smoke_rb), "condition_metric_rows": len(cond_metrics), "all_source_condition_rows": int(len(cond_metrics.loc[cond_metrics.arm.eq("all_source")])), "drug_metric_rows": len(per_drug), "all_source_drug_rows": int(len(per_drug.loc[per_drug.arm.eq("all_source")])), "prediction_shape": list(pred_rb["predictions"].shape), "prediction_keys_match": True, "gpu_hours": 0.0},
        "caveats": ["Nearest and exact-subtype comparisons use the same small, fixed four-source development set.", "GSE70138 same-cell controls can be shared across drug exposures; source-development comparison may have shared-control dependence.", "Drug-key bootstrap intervals describe only the observed 108-key domain, not patient/subtype generalization.", "Transcript-response similarity does not establish drug efficacy or PDO migration."],
    }
    write_json(out / "summary.json", summary)
    notes = [
        "# EXP-050 §3.2 固定来源参照选择比较",
        "",
        "本结果为四个固定乳腺癌细胞系的来源开发比较。108个完整化学键覆盖1285个同药、同剂量、同时间且同细胞匹配溶剂条件，响应轴978基因，Hallmark集合50个。它不构成PDO或患者验证。",
        "",
        "最近邻只按DepMap 24Q2未处理/处理前表达的冻结978基因轴选择，并在读取处理响应delta前将选择表及回执落盘、读回。相同分型没有其他来源的目标分型参照记缺失。随机n=1参照枚举其余三源；n=3唯一子集与全部来源均值完全相同，标记为退化对照。",
        "",
        "逐条件指标先在同一InChIKey内等权汇总剂量/时间条件，再对108个完整InChIKey等权。配对bootstrap以InChIKey为单位，区间仅描述本药物域。GSE70138可能跨药物条件共享同细胞溶剂控制，存在来源开发依赖；不作无偏独立验证表述。",
        "",
        "## 结果解读",
        "",
        "参见 `per_target_arm_summary.tsv`、`paired_bootstrap_ci.tsv`、`hallmark_counterexamples.tsv` 和 `summary.json`。只有当近邻/分型参照的逐键差值与区间相对all-source和随机单源参照一致时，才可描述为在该固定来源开发集中的优势；不得推广到患者或PDO总体。",
        "",
        "## 尚需的真实PDO取数合同",
        "",
        "每个独立PDO模型应有可追溯化学身份（完整InChIKey或可核验结构编号）、剂量、处理时间、处理前RNA、匹配溶剂RNA和处理后RNA；药敏需在同一PDO模型上定量测量并保留独立培养重复。RNA与药敏复孔无需逐孔一一对应，但样本来源、培养批次、化合物、剂量和时间必须能追溯。独立验证应按PDO来源拆分，报告有效分母与缺失原因。",
    ]
    (out / "analysis_result_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    manifest_final = [{"path": rel_path(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name != "sha256_manifest.tsv"]
    write_tsv(out / "sha256_manifest.tsv", manifest_final)
    run_manifest = {
        "experiment_id": "EXP-050", "run_id": RUN_ID,
        "started_at_utc": datetime.fromtimestamp(time.time() - wall_seconds, timezone.utc).isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python {rel_path(Path(__file__))} --config {rel_path(config_file)} --run-id {RUN_ID} --phase {args.phase} --attempt-id {attempt_id}" + (f" --manager-readback {rel_path(manager_receipt_path)}" if args.phase == "full" else ""),
        "working_directory": str(ROOT), "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "pyyaml": yaml.__version__ if hasattr(yaml, "__version__") else "unknown",
        "inputs": {k: {"path": rel_path(v), "sha256": input_hashes[k], "bytes": v.stat().st_size} for k, v in inputs.items()},
        "selection_commit_marker_sha256": sha256(marker_path),
        "cpu_seconds_including_prior_package_upper_bound": cpu_seconds, "current_attempt_cpu_seconds": current_attempt_cpu_seconds,
        "prior_package_cpu_upper_bound_seconds": PREVIOUS_PACKAGE_CPU_UPPER_BOUND_SECONDS, "wall_seconds": wall_seconds,
        "gpu_hours": 0.0, "cpu_cap_seconds": cpu_cap,
        "forbidden_inputs_accessed": [], "run_status": "COMPLETED",
        "outputs_hashed_before_manifest": len(manifest_final),
    }
    write_json(out / "run_manifest.json", run_manifest)
    write_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "out": rel_path(out), "cpu_seconds": cpu_seconds, "gpu_hours": 0.0, "selection_chosen": selection_receipt["chosen_model_by_target"], "all_source_condition_rows": summary["readback"]["all_source_condition_rows"], "all_source_drug_rows": summary["readback"]["all_source_drug_rows"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
