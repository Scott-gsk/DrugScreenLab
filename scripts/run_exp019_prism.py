"""Run the frozen EXP-019 GDSC->PRISM cold-cell evaluation.

This runner uses the already frozen EXP-016 split/checkpoint. PRISM responses
are used only for the single locked evaluation, never for training or selection.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.chemistry import morgan_fingerprint
from drug_screen.transfer.cold_cell import _fit_drug_only, _condition_rows, _metrics, _paired_cell_bootstrap
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor

def broadcast_cell_rows(cell_matrix, cell_ids, pair_cell_ids):
    """Give every drug record from one cell line the same expression row."""
    ids=list(map(str,cell_ids)); lookup={x:i for i,x in enumerate(ids)}
    if len(lookup)!=len(ids): raise ValueError("cell-line identities must be unique")
    try: positions=np.asarray([lookup[str(x)] for x in pair_cell_ids],dtype=int)
    except KeyError as error: raise ValueError(f"unknown pair cell-line identity: {error.args[0]}") from error
    return np.asarray(cell_matrix)[positions]

def load_bundle(d, name, prefix=""):
    p = Path(d); stem = prefix or ""
    s = pd.read_parquet(p / f"{stem}samples.parquet"); dr = pd.read_parquet(p / f"{stem}drugs.parquet"); pa = pd.read_parquet(p / f"{stem}pairs.parquet")
    z = np.load(p / f"{stem}indexed.npz", allow_pickle=True)
    return IndexedBundle(name, "frozen", s, dr, pa, np.asarray(z["expression"]), np.asarray(z["sample_id"]), np.asarray(z["ordered_genes"]), "", True, "response")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="data/artifacts/EXP-019/prism_frozen_v2"); ap.add_argument("--device", default="cuda", choices=["cuda","cpu"]); ap.add_argument("--permutations", type=int, default=100); ap.add_argument("--bootstrap", type=int, default=2000); a = ap.parse_args()
    out = ROOT / a.output
    if out.exists() and any(out.iterdir()): raise SystemExit(f"refusing non-empty output: {out}")
    source = load_bundle(ROOT / "data/processed/exp014/gdsc_v3", "gdsc_v3")
    prism = load_bundle(ROOT / "data/processed/exp012", "prism_extended_primary", "prism_extended_primary_")
    run = ROOT / "data/artifacts/EXP-016/cold_cell_v3_split20260913_3seed_perm100/runs/split20260913_init20260913"
    prep = ExpressionPreprocessor.load(run / "expression_transform.npz"); dual = DualEncoderPairModel.load(run / "dual_encoder.pt")
    split = json.loads((ROOT / "data/artifacts/EXP-016/cold_cell_v3_split20260913_3seed_perm100/splits.json").read_text(encoding="utf-8"))[0]
    train_ids = set(map(str, split["train_samples"])); val_ids = set(map(str, split["validation_samples"]))
    model = pd.read_csv(ROOT / "data/raw/depmap/24q2/Model.csv", usecols=["ModelID","SangerModelID"]).dropna()
    # EXP-019 freezes the union of source-domain identities used across the
    # registered source evaluations; using all indexed GDSC identities avoids
    # letting a source test identity re-enter PRISM as an external test.
    source_ids = set(map(str, source.sample_ids))
    seen = set(model.loc[model.SangerModelID.astype(str).isin(source_ids), "ModelID"].astype(str))
    keep = prism.samples.sample_id.astype(str).map(lambda x: x not in seen)
    strict_ids = prism.samples.loc[keep, "sample_id"].astype(str).tolist()
    if len(strict_ids) not in (353, 354): raise RuntimeError(f"unexpected strict unseen cell-line count: {len(strict_ids)}")
    idx_s = {str(x): i for i,x in enumerate(prism.sample_ids)}; idx_d = {str(x): i for i,x in enumerate(prism.drugs.drug_id)}
    p = prism.pairs.copy(); p.sample_id=p.sample_id.astype(str); p.drug_id=p.drug_id.astype(str); p=p[p.sample_id.isin(strict_ids)].copy()
    source_pairs=source.pairs.copy(); source_pairs.sample_id=source_pairs.sample_id.astype(str); source_pairs.drug_id=source_pairs.drug_id.astype(str)
    src_idx={str(x):i for i,x in enumerate(source.sample_ids)}; src_dr={str(x):i for i,x in enumerate(source.drugs.drug_id)}
    train=source_pairs[source_pairs.sample_id.isin(train_ids)].copy(); val=source_pairs[source_pairs.sample_id.isin(val_ids)].copy()
    source_x=prep.transform(source.expression, source.gene_ids); prism_x=prep.transform(prism.expression, prism.gene_ids)
    fps={str(r.drug_id):morgan_fingerprint(str(r.canonical_smiles)) for r in prism.drugs.itertuples()}
    sfps={str(r.drug_id):morgan_fingerprint(str(r.canonical_smiles)) for r in source.drugs.itertuples()}
    def arr(frame, X, lookup, fp): return X[[lookup[x] for x in frame.sample_id]], np.stack([fp[x] for x in frame.drug_id]), frame.response.to_numpy(float)
    tr=arr(train,source_x,src_idx,sfps); va=arr(val,source_x,src_idx,sfps); te=arr(p,prism_x,idx_s,fps)
    # B0 uses source training responses, mapped by exact InChIKey drug_id.
    dm=train.groupby("drug_id").response.mean(); b0=p.drug_id.map(dm).fillna(float(train.response.mean())).to_numpy(float)
    drug_only, mean, std, hist = _fit_drug_only(tr[1],tr[2],va[1],va[2],seed=20260913,epochs=50,device=a.device)
    import torch
    with torch.no_grad(): do=(drug_only(torch.tensor(te[1],dtype=torch.float32,device=a.device)).cpu().numpy()*std+mean)
    dual_pred=dual.predict(te[0],te[1]); rows=[]
    base=pd.DataFrame({"sample_id":p.sample_id.to_numpy(),"drug_id":p.drug_id.to_numpy(),"observed":te[2]})
    rows += [base.assign(model="B0",condition="original",permutation_id=0,predicted=b0), base.assign(model="DrugOnly",condition="original",permutation_id=0,predicted=do), base.assign(model="DualEncoder",condition="original",permutation_id=0,predicted=dual_pred)]
    tissues=prism.samples.set_index("sample_id").sample_tissue.astype(str).to_dict()
    strict_x=prism_x[[idx_s[x] for x in strict_ids]]
    strict_tissues=[tissues[x] for x in strict_ids]
    train_mean=source_x[[src_idx[x] for x in train_ids]].mean(0)
    for kind, offset in [("within_cancer",101),("pan_cancer",10001)]:
        for r in range(a.permutations):
            # Permute unique cell-line rows, then broadcast one replacement row
            # to every drug record belonging to that cell line.
            cell_matrix=_condition_rows(strict_x,strict_tissues,train_mean,kind=kind,seed=20260913+offset+r)
            rows.append(base.assign(model="DualEncoder",condition=kind+"_row_permutation",permutation_id=r,predicted=dual.predict(broadcast_cell_rows(cell_matrix,strict_ids,p.sample_id),te[1])))
    out.mkdir(parents=True,exist_ok=True); oof=pd.concat(rows,ignore_index=True); oof.to_parquet(out/"predictions.parquet",index=False)
    metrics={"|".join(map(str,k)):_metrics(g,bootstrap_seed=20260913+int(k[2]),resamples=a.bootstrap) for k,g in oof.groupby(["model","condition","permutation_id"],sort=True)}
    original=oof[(oof.model=="DualEncoder")&(oof.condition=="original")]; contrasts={}
    for name in ["B0","DrugOnly"]:
        contrasts["DualEncoder_minus_"+name]=_paired_cell_bootstrap(original,oof[(oof.model==name)&(oof.condition=="original")],seed=20260913,resamples=a.bootstrap)
    contract={"experiment":"EXP-019","strict_unseen_cell_lines":len(strict_ids),"source_train_cell_lines":len(train_ids),"source_validation_cell_lines":len(val_ids),"prism_drugs":int(p.drug_id.nunique()),"permutation_repeats":a.permutations,"bootstrap":a.bootstrap,"checkpoint":str(run/"dual_encoder.pt"),"note":"strict unseen set derived by DepMap ModelID<-SangerModelID exclusion; PRISM labels used only for the locked final evaluation, never for training or selection"}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8"); (out/"contrasts.json").write_text(json.dumps(contrasts,indent=2,allow_nan=False),encoding="utf-8"); (out/"contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8"); print(json.dumps({"output":str(out),"contract":contract,"contrasts":contrasts},indent=2,allow_nan=False))
if __name__ == "__main__": main()
