"""Strictly scoped functional-dependency learnability check, not drug transfer."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


class _MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(); self.layers = nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(), nn.Linear(32, outputs))
    def forward(self, x): return self.layers(x)


def _split(patients: np.ndarray, seed: int) -> dict[str, str]:
    unique = np.array(sorted(set(patients))); 
    if len(unique) < 5: raise ValueError("functional feasibility requires at least five patient groups")
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    n_test, n_val = max(1, round(len(unique) * .2)), max(1, round(len(unique) * .2))
    return {str(value): "test" if i < n_test else "validation" if i < n_test + n_val else "train" for i, value in enumerate(unique)}


def _metrics(observed: np.ndarray, predicted: np.ndarray, genes: list[str]) -> dict[str, Any]:
    rows = []
    for i, gene in enumerate(genes):
        valid = np.isfinite(observed[:, i]); y, p = observed[valid, i], predicted[valid, i]
        if not len(y): continue
        rows.append({"gene": gene, "n": int(len(y)), "mae": float(np.mean(np.abs(y-p))),
                     "spearman": None if len(y) < 2 or np.ptp(y) == 0 or np.ptp(p) == 0 else float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])})
    return {"per_gene": rows, "mean_gene_mae": float(np.mean([row["mae"] for row in rows])),
            "mean_gene_spearman": float(np.mean([row["spearman"] for row in rows if row["spearman"] is not None])) if any(row["spearman"] is not None for row in rows) else None}


def run_functional_feasibility(data_dir: str | Path, output_dir: str | Path, *, seed: int = 20260912,
                               epochs: int = 100, target_tissue: str = "Colorectal") -> dict[str, Any]:
    """Run one patient-held-out colorectal/minLib dependency learnability check."""
    root, output = Path(data_dir), Path(output_dir)
    files = {name: root / name for name in ("samples.parquet", "expression_author_log2tpm.parquet", "dependency_by_library.parquet")}
    if not all(path.is_file() for path in files.values()): raise ValueError("functional data directory lacks required paired parquet files")
    if output.exists() and any(output.iterdir()): raise ValueError("output must be new or empty; never overwrite a run")
    samples, expression, dependency = (pd.read_parquet(files[name]) for name in ("samples.parquet", "expression_author_log2tpm.parquet", "dependency_by_library.parquet"))
    required = {"sample_ID", "patient_id", "primary_tumour_type"}
    if not required <= set(samples) or not {"sample_ID", "gene", "library", "LFC"} <= set(dependency): raise ValueError("functional metadata columns are missing")
    if expression.index.duplicated().any() or not set(samples.sample_ID.astype(str)) <= set(expression.index.astype(str)):
        raise ValueError("expression index must contain unique sample_ID labels")
    expression.index = expression.index.astype(str)
    counts = dependency.groupby("sample_ID").library.nunique()
    # Technical subset selection depends solely on library metadata, never LFC.
    single = set(counts.loc[counts.eq(1)].index); dep = dependency.loc[dependency.sample_ID.isin(single) & dependency.library.eq("minLib")].copy()
    selected = samples.loc[samples.sample_ID.isin(set(dep.sample_ID)) & samples.primary_tumour_type.eq(target_tissue)].copy()
    if selected.empty: raise ValueError("no single-minLib samples in requested tissue")
    selected["patient_id"] = selected.patient_id.astype(str)
    if selected.patient_id.str.strip().eq("").any(): raise ValueError("selected samples require non-empty patient_id")
    values = expression.loc[selected.sample_ID.astype(str)].to_numpy(float); gene_names = np.array([str(col).removesuffix("_exp") for col in expression.columns])
    mapping = _split(selected.patient_id.to_numpy(str), seed); split = selected.patient_id.map(mapping).to_numpy()
    train, valid, test = split == "train", split == "validation", split == "test"
    if np.isinf(values[train]).any(): raise ValueError("training expression cannot contain infinity")
    observed_columns = np.isfinite(values[train]).any(axis=0)
    if not observed_columns.any(): raise ValueError("no observed training expression genes")
    values, gene_names = values[:, observed_columns], gene_names[observed_columns]
    means = np.nanmean(values[train], axis=0); values = np.where(np.isfinite(values), values, means)
    scale = values[train].std(axis=0); scale[scale == 0] = 1
    hvg = np.argsort(-np.var(values[train], axis=0), kind="mergesort")[:min(512, values.shape[1])]; x = ((values[:, hvg] - means[hvg]) / scale[hvg]).astype(np.float32)
    pivot = dep.pivot(index="sample_ID", columns="gene", values="LFC").reindex(selected.sample_ID)
    y_all, all_genes = pivot.to_numpy(float), pivot.columns.astype(str).tolist()
    coverage = np.isfinite(y_all[train]).mean(axis=0); candidate = np.flatnonzero(coverage >= .9)
    if not len(candidate): raise ValueError("no dependency targets meet 0.9 train coverage")
    target_order = candidate[np.argsort(-np.nanvar(y_all[train][:, candidate], axis=0), kind="mergesort")[:min(128, len(candidate))]]
    genes, y = [all_genes[i] for i in target_order], y_all[:, target_order]
    y_mean, y_std = np.nanmean(y[train], axis=0), np.nanstd(y[train], axis=0); y_std[y_std == 0] = 1
    y_z = (y-y_mean)/y_std; mask = np.isfinite(y_z); y_filled = np.where(mask, y_z, 0).astype(np.float32)
    # Baseline and independent per-target ridge use training observations only.
    baseline = np.tile(y_mean, (len(y), 1)); ridge = baseline.copy(); ridge_weights = []
    for j in range(len(genes)):
        observed = train & mask[:, j]; design = x[observed]; target = y_z[observed, j]
        weight = design.T @ np.linalg.solve(design @ design.T + 10 * np.eye(len(design)), target)
        ridge_weights.append(weight)
        ridge[:, j] = (x @ weight) * y_std[j] + y_mean[j]
    torch.manual_seed(seed); network = _MLP(x.shape[1], len(genes)); opt = torch.optim.Adam(network.parameters(), lr=.001)
    best, best_loss, stale = None, float("inf"), 0; tx, ty, tm = (torch.tensor(item) for item in (x, y_filled, mask.astype(np.float32)))
    for _ in range(epochs):
        network.train(); opt.zero_grad(); out = network(tx[train]); loss = (((out-ty[train])**2)*tm[train]).sum()/tm[train].sum(); loss.backward(); opt.step()
        network.eval()
        with torch.no_grad(): val = float(((((network(tx[valid])-ty[valid])**2)*tm[valid]).sum()/tm[valid].sum()))
        if val < best_loss: best_loss, stale, best = val, 0, {key: value.detach().clone() for key, value in network.state_dict().items()}
        else:
            stale += 1
            if stale >= 10: break
    network.load_state_dict(best); network.eval()
    with torch.no_grad(): mlp = (network(tx).numpy()*y_std+y_mean)
    # The diagnostic shuffles held-out expression within tissue when possible;
    # this colorectal-only run otherwise documents a patient-global permutation.
    test_x, mode = x[test].copy(), "within_tissue"
    tissue = selected.primary_tumour_type.to_numpy(str)[test]
    if len(set(tissue)) == 1: mode = "patient_global"
    np.random.default_rng(seed + 1).shuffle(test_x)
    with torch.no_grad(): shuffled = network(torch.tensor(test_x)).numpy()*y_std+y_mean
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": network.state_dict(), "input_dim": x.shape[1], "output_dim": len(genes)}, output / "mlp_weights.pt")
    np.savez_compressed(output / "preprocessing_and_targets.npz", hvg_gene_ids=gene_names[hvg], expression_mean=means[hvg], expression_scale=scale[hvg], target_gene_ids=np.asarray(genes), target_mean=y_mean, target_std=y_std)
    np.savez_compressed(output / "ridge_weights.npz", target_gene_ids=np.asarray(genes), weights=np.column_stack(ridge_weights), alpha=10.0)
    predictions = []
    for name, prediction in (("mean", baseline), ("ridge", ridge), ("mlp", mlp)):
        predictions.append(pd.DataFrame({"sample_ID": np.repeat(selected.sample_ID.to_numpy(), len(genes)), "patient_id": np.repeat(selected.patient_id.to_numpy(), len(genes)), "split": np.repeat(split, len(genes)), "gene": np.tile(genes, len(selected)), "observed": y.ravel(), "predicted": prediction.ravel(), "algorithm": name}))
    predictions.append(pd.DataFrame({"sample_ID": np.repeat(selected.sample_ID.to_numpy()[test], len(genes)), "patient_id": np.repeat(selected.patient_id.to_numpy()[test], len(genes)), "split": "test", "gene": np.tile(genes, test.sum()), "observed": y[test].ravel(), "predicted": shuffled.ravel(), "algorithm": "mlp_expression_permuted"}))
    pd.concat(predictions, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    frozen = {"seed": seed, "tissue": target_tissue, "library": "minLib", "library_selection": "single_library_metadata_only", "splits": {key: sorted(set(selected.loc[split == key, "patient_id"].tolist())) for key in ("train", "validation", "test")}, "hvg_genes": gene_names[hvg].tolist(), "dependency_genes": genes, "ridge_alpha": 10.0, "mlp": {"hidden_dim": 32, "max_epochs": epochs, "early_stopping_patience": 10}, "source_sha256": {name: _sha(path) for name, path in files.items()}}
    (output / "frozen_ids.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    report = {"status": "FUNCTIONAL_LEARNABILITY_ONLY", "drug_response_transfer_established": False, "scope": "single-minLib colorectal technical subset", "split": "patient_grouped_60_20_20", "permutation": {"algorithm": "mlp_expression_permuted", "mode": mode}, "test": {"mean": _metrics(y[test], baseline[test], genes), "ridge": _metrics(y[test], ridge[test], genes), "mlp": _metrics(y[test], mlp[test], genes), "mlp_expression_permuted": _metrics(y[test], shuffled, genes)}, "n_samples": int(len(selected)), "n_targets": len(genes)}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
