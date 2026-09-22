"""EXP-016 post-hoc cold-cell evaluation under the shared rank-gain contract.

This keeps ``cold_cell_v1`` untouched. New runs are a reproducible diagnostic:
every drug record for a test cell receives the same (possibly permuted) row.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from .bundles import IndexedBundle
from .chemistry import morgan_fingerprint
from .metrics_v2 import ndcg_at_k
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def split_sample_ids(sample_ids, seed=20260913):
    """Stable 60/20/20 biological-model split; no response is inspected."""
    ids = np.asarray(sorted(map(str, sample_ids)))
    np.random.default_rng(seed).shuffle(ids)
    train_end, validation_end = int(len(ids) * .6), int(len(ids) * .6) + int(len(ids) * .2)
    return ids[:train_end], ids[train_end:validation_end], ids[validation_end:]


def _ndcg(y, pred, k):
    """Compatibility entry point, now using project tie-aware rank gains."""
    return ndcg_at_k(y, pred, k=k)


class _DrugOnlyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 40), nn.ReLU(), nn.Linear(40, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def forward(self, drug): return self.net(drug).squeeze(1)


def _fit_drug_only(train_d, train_y, validation_d, validation_y, *, seed, epochs, device):
    """Capacity-matched structure-only baseline; validation selects checkpoint."""
    torch.manual_seed(int(seed)); model = _DrugOnlyNet().to(device)
    mean, std = float(np.mean(train_y)), float(np.std(train_y)) or 1.0
    x, y = torch.tensor(train_d, dtype=torch.float32, device=device), torch.tensor((train_y-mean)/std, dtype=torch.float32, device=device)
    vx, vy = torch.tensor(validation_d, dtype=torch.float32, device=device), torch.tensor((validation_y-mean)/std, dtype=torch.float32, device=device)
    opt, best, best_loss, history = torch.optim.Adam(model.parameters(), lr=1e-3), None, float("inf"), []
    for epoch in range(int(epochs)):
        model.train(); opt.zero_grad(); loss = F.huber_loss(model(x), y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): value = float(F.huber_loss(model(vx), vy))
        history.append({"epoch": epoch+1, "validation_loss": value})
        if value < best_loss:
            best_loss = value; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    return model, mean, std, history


def _per_drug_ridge(train_x, train_rows, test_x, test_rows):
    """Fixed-alpha, 64-HVG drug-specific expression baseline.

    The 64-HVG cap is fixed before test evaluation and keeps 219 separate
    linear fits computationally comparable to the small neural baselines.
    """
    fallback = float(train_rows.response.mean()); result = np.full(len(test_rows), fallback)
    for drug, group in train_rows.groupby("drug_id", sort=True):
        mask = test_rows.drug_id.to_numpy() == drug
        if mask.any():
            train_index = train_rows.index.get_indexer(group.index)
            test_index = np.flatnonzero(mask)
            result[test_index] = Ridge(alpha=10, solver="lsqr", tol=1e-4).fit(train_x[train_index, :64], group.response).predict(test_x[test_index, :64])
    return result


def _top_k_overlap(y, p, k=5):
    if len(y) < k or np.ptp(y) == 0: return np.nan
    return len(set(np.argsort(-y, kind="mergesort")[:k]) & set(np.argsort(-p, kind="mergesort")[:k])) / k


def _cell_metrics(frame):
    rows = []
    for sid, group in frame.groupby("sample_id", sort=True):
        y, p = group.observed.to_numpy(float), group.predicted.to_numpy(float)
        rho = spearmanr(y, p).statistic if np.ptp(y) and np.ptp(p) else np.nan
        rows.append({"sample_id": sid, "mae": float(np.abs(y-p).mean()), "spearman": rho,
                     "ndcg5": ndcg_at_k(y, p, k=5) if len(y) >= 10 else np.nan,
                     "top5_overlap": _top_k_overlap(y, p)})
    return pd.DataFrame(rows).set_index("sample_id").sort_index()


def _metrics(frame, *, bootstrap_seed, resamples):
    cells = _cell_metrics(frame); drug_rho = []
    for _, group in frame.groupby("drug_id", sort=True):
        y, p = group.observed.to_numpy(float), group.predicted.to_numpy(float)
        if len(y) >= 3 and np.ptp(y) and np.ptp(p): drug_rho.append(float(spearmanr(y, p).statistic))
    result = {"cell_macro": {key: None if not np.isfinite(value) else float(value) for key, value in cells.mean().items()},
              "per_drug_macro_spearman": float(np.mean(drug_rho)) if drug_rho else None,
              "evaluable_drugs": len(drug_rho), "n_cells": len(cells), "n_pairs": len(frame), "cell_block_bootstrap": {}}
    rng = np.random.default_rng(bootstrap_seed); draw = rng.integers(0, len(cells), size=(resamples, len(cells)))
    for column in cells:
        values = cells[column].to_numpy(float); boot = np.nanmean(values[draw], axis=1)
        result["cell_block_bootstrap"][column] = {"point": float(np.nanmean(values)), "ci95": [float(x) for x in np.nanquantile(boot, [.025, .975])], "resamples": int(resamples)}
    return result


def _paired_cell_bootstrap(left, right, *, seed, resamples):
    """Paired biological-cell bootstrap for a named, already frozen contrast."""
    a, b = _cell_metrics(left), _cell_metrics(right)
    if not a.index.equals(b.index): raise ValueError("paired comparison cells differ")
    rng = np.random.default_rng(seed); draw = rng.integers(0, len(a), size=(resamples, len(a)))
    result = {}
    for column in a:
        values = (a[column] - b[column]).to_numpy(float); boot = np.nanmean(values[draw], axis=1)
        result[column] = {"difference": float(np.nanmean(values)), "ci95": [float(x) for x in np.nanquantile(boot, [.025, .975])], "resamples": int(resamples)}
    return result


def _condition_rows(test_expression, tissues, train_mean, *, kind, seed):
    """Whole-row permutations that preserve drug-record identity within a cell."""
    out = test_expression.copy(); rng = np.random.default_rng(seed)
    if kind == "train_mean": out[:] = train_mean
    elif kind == "pan_cancer": out[:] = test_expression[rng.permutation(len(test_expression))]
    elif kind == "within_cancer":
        for tissue in sorted(set(tissues)):
            positions = np.flatnonzero(np.asarray(tissues) == tissue)
            out[positions] = test_expression[rng.permutation(positions)]
    else: raise ValueError(f"unknown condition {kind}")
    return out


def _bundle_hashes(bundle):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(bundle.expression).tobytes()); h.update("\n".join(map(str, bundle.sample_ids)).encode())
    return {"expression_and_ids_sha256": h.hexdigest(), "pairs_sha256": hashlib.sha256(bundle.pairs.to_csv(index=False).encode()).hexdigest(), "drugs_sha256": hashlib.sha256(bundle.drugs.to_csv(index=False).encode()).hexdigest()}


def _compact_summary(oof, paired, *, split_seeds, init_seeds):
    """Return a small, seed-explicit report derived only from frozen OOF rows."""
    models, contrasts, permutations = {}, {}, {}
    for split_seed in map(int, split_seeds):
        for init_seed in map(int, init_seeds):
            learned = oof[(oof.split_seed == split_seed) & (oof.init_seed == init_seed)]
            for model in ["DrugOnly", "DualEncoder"]:
                frame = learned[(learned.model == model) & (learned.condition == "original") & (learned.permutation_id == 0)]
                models[f"{split_seed}|{init_seed}|{model}"] = _metrics(
                    frame, bootstrap_seed=split_seed + init_seed, resamples=2
                )["cell_macro"]
            for model in ["B0", "PerDrugRidge"]:
                frame = oof[(oof.split_seed == split_seed) & (oof.init_seed == -1) & (oof.model == model)]
                models[f"{split_seed}|shared|{model}"] = _metrics(
                    frame, bootstrap_seed=split_seed, resamples=2
                )["cell_macro"]
            for comparator in ["B0", "DrugOnly"]:
                key = f"{split_seed}|{init_seed}|DualEncoder_minus_{comparator}"
                contrasts[key] = paired[key]
            for condition in ["within_cancer_row_permutation", "pan_cancer_row_permutation"]:
                values = []
                for _, frame in learned[(learned.model == "DualEncoder") & (learned.condition == condition)].groupby("permutation_id", sort=True):
                    original = learned[(learned.model == "DualEncoder") & (learned.condition == "original") & (learned.permutation_id == 0)]
                    value = _paired_cell_bootstrap(original, frame, seed=split_seed + init_seed, resamples=2)
                    values.append({metric: item["difference"] for metric, item in value.items()})
                table = pd.DataFrame(values)
                permutations[f"{split_seed}|{init_seed}|original_minus_{condition}"] = {
                    metric: {"mean": float(table[metric].mean()), "q025": float(table[metric].quantile(.025)),
                             "q975": float(table[metric].quantile(.975)), "n": int(len(table))}
                    for metric in ["mae", "spearman", "ndcg5", "top5_overlap"]
                }
    return {"status": "DERIVED_FROM_FROZEN_OOF", "models": models, "paired_contrasts": contrasts,
            "permutation_contrasts": permutations}


def run_cold_cell(*, bundle: IndexedBundle, output_dir: Path, split_seeds=(20260913,), init_seeds=(20260913,), epochs=50, device="cuda", resamples=2000, permutation_repeats=100):
    """Run diagnostics into a new directory only; do not overwrite historical runs."""
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError("output must be new or empty")
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {str(v): i for i, v in enumerate(bundle.sample_ids)}
    samples = bundle.samples.copy(); samples["sample_id"] = samples.sample_id.astype(str)
    if "sample_tissue" not in samples: raise ValueError("sample_tissue required for within-cancer permutation")
    tissue = samples.set_index("sample_id").sample_tissue.astype(str).to_dict()
    pairs = bundle.pairs.copy(); pairs["sample_id"] = pairs.sample_id.astype(str); pairs["drug_id"] = pairs.drug_id.astype(str)
    smiles = bundle.drugs.set_index("drug_id").canonical_smiles.astype(str).to_dict(); fps = {d: morgan_fingerprint(s) for d, s in smiles.items()}
    predictions, manifests, runtime = [], [], []
    for split_seed in map(int, split_seeds):
        train_ids, validation_ids, test_ids = split_sample_ids(bundle.sample_ids, split_seed)
        sets = {"train": set(train_ids), "validation": set(validation_ids), "test": set(test_ids)}
        if any(sets[a] & sets[b] for a, b in [("train","validation"),("train","test"),("validation","test")]): raise ValueError("split overlap")
        rows = {name: pairs[pairs.sample_id.isin(ids)].copy() for name, ids in sets.items()}
        prep = ExpressionPreprocessor.fit(bundle.expression[[index[x] for x in train_ids]], bundle.gene_ids, n_features=512)
        expression = prep.transform(bundle.expression, bundle.gene_ids)
        def arrays(frame, matrix=expression, lookup=index):
            return matrix[[lookup[x] for x in frame.sample_id]], np.stack([fps[x] for x in frame.drug_id]), frame.response.to_numpy(float), frame.sample_id.to_numpy(str)
        tr, va, te = arrays(rows["train"]), arrays(rows["validation"]), arrays(rows["test"])
        local_index = {x: i for i, x in enumerate(test_ids)}
        test_x = expression[[index[x] for x in test_ids]]; tissues = [tissue[x] for x in test_ids]; train_mean = expression[[index[x] for x in train_ids]].mean(0)
        if permutation_repeats < 1: raise ValueError("permutation_repeats must be positive")
        conditions = [("original", 0, test_x), ("train_mean_expression", 0, _condition_rows(test_x, tissues, train_mean, kind="train_mean", seed=split_seed))]
        conditions += [("within_cancer_row_permutation", repeat, _condition_rows(test_x, tissues, train_mean, kind="within_cancer", seed=split_seed + 101 + repeat)) for repeat in range(permutation_repeats)]
        conditions += [("pan_cancer_row_permutation", repeat, _condition_rows(test_x, tissues, train_mean, kind="pan_cancer", seed=split_seed + 10001 + repeat)) for repeat in range(permutation_repeats)]
        drug_mean, global_mean = rows["train"].groupby("drug_id").response.mean(), float(rows["train"].response.mean())
        b0 = rows["test"].drug_id.map(drug_mean).fillna(global_mean).to_numpy(float)
        predictions.append(pd.DataFrame({"split_seed":split_seed,"init_seed":-1,"condition":"original","permutation_id":0,"model":"B0","sample_id":te[3],"drug_id":rows["test"].drug_id,"observed":te[2],"predicted":b0}))
        ridge = _per_drug_ridge(tr[0], rows["train"], te[0], rows["test"])
        predictions.append(pd.DataFrame({"split_seed":split_seed,"init_seed":-1,"condition":"original","permutation_id":0,"model":"PerDrugRidge","sample_id":te[3],"drug_id":rows["test"].drug_id,"observed":te[2],"predicted":ridge}))
        for init_seed in map(int, init_seeds):
            start = time.perf_counter(); drug_only, mean, std, history = _fit_drug_only(tr[1],tr[2],va[1],va[2],seed=init_seed,epochs=epochs,device=device)
            with torch.no_grad(): dpred=drug_only(torch.tensor(te[1],dtype=torch.float32,device=device)).cpu().numpy()*std+mean
            predictions.append(pd.DataFrame({"split_seed":split_seed,"init_seed":init_seed,"condition":"original","permutation_id":0,"model":"DrugOnly","sample_id":te[3],"drug_id":rows["test"].drug_id,"observed":te[2],"predicted":dpred}))
            dual=DualEncoderPairModel(hidden_dim=32,epochs=epochs,ranking_weight=.2,max_rank_pairs=128,seed=init_seed,device=device,batch_groups=8); dual.fit(*tr[:3],pdo_ids=tr[3],validation_data=va)
            for condition, permutation_id, matrix in conditions:
                xp=matrix[[local_index[x] for x in te[3]]]
                predictions.append(pd.DataFrame({"split_seed":split_seed,"init_seed":init_seed,"condition":condition,"permutation_id":permutation_id,"model":"DualEncoder","sample_id":te[3],"drug_id":rows["test"].drug_id,"observed":te[2],"predicted":dual.predict(xp,te[1])}))
            run_dir=output_dir/"runs"/f"split{split_seed}_init{init_seed}";run_dir.mkdir(parents=True);prep.save(run_dir/"expression_transform.npz");dual.save(run_dir/"dual_encoder.pt");(run_dir/"drug_only_history.json").write_text(json.dumps(history,indent=2));runtime.append({"split_seed":split_seed,"init_seed":init_seed,"seconds":time.perf_counter()-start})
        manifests.append({"split_seed":split_seed,"train_samples":sorted(train_ids.tolist()),"validation_samples":sorted(validation_ids.tolist()),"test_samples":sorted(test_ids.tolist())})
    oof=pd.concat(predictions,ignore_index=True);oof.to_parquet(output_dir/"oof_predictions.parquet",index=False)
    metrics={"|".join(map(str,keys)):_metrics(frame,bootstrap_seed=int(keys[0])+max(int(keys[1]),0)+int(keys[3]),resamples=resamples) for keys,frame in oof.groupby(["split_seed","init_seed","condition","permutation_id","model"],sort=True)}
    paired = {}
    for split_seed in map(int, split_seeds):
        for init_seed in map(int, init_seeds):
            # B0 has no initialization seed, but every learned comparison
            # must be restricted to the one requested initialization.
            scope = oof[(oof.split_seed == split_seed) & ((oof.init_seed == init_seed) | (oof.init_seed == -1))]
            dual = scope[(scope.model == "DualEncoder") & (scope.condition == "original") & (scope.permutation_id == 0)]
            for comparator in ["B0", "DrugOnly"]:
                other = scope[(scope.model == comparator) & (scope.condition == "original")]
                if comparator == "B0": other = other.assign(init_seed=init_seed)
                paired[f"{split_seed}|{init_seed}|DualEncoder_minus_{comparator}"] = _paired_cell_bootstrap(dual, other, seed=split_seed + init_seed, resamples=resamples)
            for condition in ["within_cancer_row_permutation", "pan_cancer_row_permutation"]:
                for repeat in range(permutation_repeats):
                    permuted = scope[(scope.model == "DualEncoder") & (scope.condition == condition) & (scope.permutation_id == repeat)]
                    paired[f"{split_seed}|{init_seed}|original_minus_{condition}|{repeat}"] = _paired_cell_bootstrap(dual, permuted, seed=split_seed + init_seed + repeat, resamples=resamples)
    contract={"experiment":"EXP-016","registration":"post_hoc protocol correction; no preregistration claimed","scope":"same-study cold-cell and train-known drugs; not cold-drug or external-study evidence","split_protocol":"sample_id 60/20/20","split_seeds":list(map(int,split_seeds)),"init_seeds":list(map(int,init_seeds)),"models":["B0","PerDrugRidge(alpha=10; 64 training-selected HVGs, fixed)","DrugOnly(2048-40-32-1)","DualEncoder"],"conditions":["original","within_cancer_row_permutation","pan_cancer_row_permutation","train_mean_expression"],"permutation_repeats":int(permutation_repeats),"epochs":int(epochs),"metrics_protocol":"tie-aware rank-gain NDCG@5 and cell-block bootstrap; Top-K uses stable mergesort, therefore tied predictions have deterministic index-order resolution","input_hashes":_bundle_hashes(bundle),"runtime_seconds":runtime,"failures":[]}
    (output_dir/"contract.json").write_text(json.dumps(contract,indent=2));(output_dir/"splits.json").write_text(json.dumps(manifests,indent=2));(output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False));(output_dir/"paired_bootstrap.json").write_text(json.dumps(paired,indent=2,allow_nan=False));(output_dir/"compact_summary.json").write_text(json.dumps(_compact_summary(oof, paired, split_seeds=split_seeds, init_seeds=init_seeds),indent=2,allow_nan=False));(output_dir/"run_hashes.json").write_text(json.dumps({"contract_sha256":_hash_file(output_dir/"contract.json"),"predictions_sha256":_hash_file(output_dir/"oof_predictions.parquet"),"splits_sha256":_hash_file(output_dir/"splits.json"),"code_sha256":_hash_file(Path(__file__))},indent=2))
    return metrics
