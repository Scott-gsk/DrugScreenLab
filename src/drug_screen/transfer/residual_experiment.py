"""One fixed-parameter EXP-015 residual-adaptation diagnostic runner."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from .benchmark import _fit_b0, _predict_b0, _table
from .cohorts import load_registered_cohort, sha256
from .metrics_v2 import evaluate_predictions
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor
from .splits import assign_patient_folds
from .transfer_experiment import _fingerprints, _target_receipt, _with_comparator


def _calibrate(s, y):
    c=s-s.mean(); a=max(0., float(c@(y-y.mean())/(c@c))) if c@c else 0.; return a, float(y.mean()-a*s.mean())

def _fit_sample_space(source, x, drug, y, groups, seed):
    """CF calibration then sample-space Ridge; avoids a 2560x2560 solve."""
    ids=np.asarray(groups, str); unique=np.array(sorted(set(ids)))
    if len(unique)<6: raise ValueError("requires at least six training patients")
    rng=np.random.default_rng(seed); rng.shuffle(unique); fmap={group:i%3 for i,group in enumerate(unique)}; oof=np.empty(len(y)); audit={}
    for fold in range(3):
        held=np.array([fmap[group]==fold for group in ids]); a,b=_calibrate(source[~held],y[~held]); oof[held]=y[held]-(a*source[held]+b)
        audit[str(fold)]={"train_groups":sorted(set(ids[~held])),"heldout_groups":sorted(set(ids[held])),"a":a,"b":b}
    design=np.c_[x,drug]; mean=design.mean(0); centered=design-mean; residual=oof-oof.mean(); alpha=10.
    dual=np.linalg.solve(centered@centered.T+alpha*np.eye(len(centered)),residual); coef=centered.T@dual; intercept=float(oof.mean()-mean@coef); a,b=_calibrate(source,y)
    return {"a":a,"b":b,"coef":coef,"intercept":intercept,"audit":audit}


def _predict(state, source, x, drug, lambda_): return state["a"]*source+state["b"]+lambda_*(np.c_[x,drug]@state["coef"]+state["intercept"])


def run_residual_diagnostic(*, source_dir, data_root, registry_path, target_admission, existing_oof, output_dir, seed=20260912, loader=load_registered_cohort):
    source_dir, root, registry, admission_path, old, output=map(Path,(source_dir,data_root,registry_path,target_admission,existing_oof,output_dir))
    if output.exists() and any(output.iterdir()): raise ValueError("output must be new or empty; never overwrite a run")
    admission = _target_receipt(admission_path, "zhu_hepatobiliary_v3", registry)
    source_model=DualEncoderPairModel.load(source_dir/"source_model.pt"); prep=ExpressionPreprocessor.load(source_dir/"expression_transform.npz")
    bundle=loader("zhu_hepatobiliary_v3",root=root,registry=registry); table=_table(bundle); expression=np.asarray(bundle.expression,float); x=prep.transform(expression,np.asarray(bundle.gene_ids,str)); fp=_fingerprints(bundle,table.drug_id)
    old_oof=pd.read_parquet(old); source_drugs=set(json.loads((source_dir/"frozen_run.json").read_text())["source_drug_ids"])
    assignment=assign_patient_folds(table.patient_id.astype(str).tolist(),n_folds=5,seed=seed); rows=[]; audits=[]; started=time.perf_counter()
    for fold in range(5):
        held={p for p,v in assignment.patient_to_test_fold.items() if v==fold}; train=table.loc[~table.patient_id.isin(held)].copy(); test=table.loc[table.patient_id.isin(held)].copy()
        if set(train.patient_id)&set(test.patient_id): raise AssertionError("patient leakage")
        def arrays(frame):
            ix=frame.expression_index.to_numpy(int); d=np.vstack([fp[str(v)] for v in frame.drug_id]); return x[ix],d
        tx,td=arrays(train); vx,vd=arrays(test); source_train=source_model.predict(tx,td); source_test=source_model.predict(vx,vd)
        state=_fit_sample_space(source_train,tx,td,train.response.to_numpy(float),train.patient_id.to_numpy(str),seed+fold)
        state["fold"],state["training_seconds"],state["alpha"]=fold,time.perf_counter()-started,10.; audits.append(state)
        b0=_predict_b0(_fit_b0(train),test.drug_id)
        for lam in (0.,.1,1.):
            pred=_predict(state,source_test,vx,vd,lam); rows.append(pd.DataFrame({"patient_id":test.patient_id.astype(str),"model_id":test.model_id.astype(str),"drug_id":test.drug_id.astype(str),"fold":fold,"algorithm":f"residual_lambda_{lam:g}","observed":test.response,"predicted":pred,"baseline":b0,"source_drug_status":np.where(test.drug_id.astype(str).isin(source_drugs),"known","unknown")}))
    output.mkdir(parents=True,exist_ok=True); predictions=pd.concat(rows,ignore_index=True); predictions.to_parquet(output/"residual_oof_predictions.parquet",index=False)
    serial=[{**{k:v for k,v in item.items() if k!="coef"},"coefficients":item["coef"].tolist()} for item in audits]; (output/"fold_parameters.json").write_text(json.dumps(serial,indent=2))
    metrics={"status":"EXP015_DEVELOPMENT_RESIDUAL_DIAGNOSTIC","not_external_test":True,"note":"Additive sample terms cancel within a PDO across drugs; this does not establish individualized ranking benefit.","elapsed_seconds":time.perf_counter()-started,"metrics":{}}
    for algorithm, frame in predictions.groupby("algorithm"):
        known=frame.loc[frame.source_drug_status.eq("known")]; base=old_oof.loc[(old_oof.algorithm==f"M1_source_scaled_seed_{seed}") & old_oof.fold.isin(known.fold.unique())]
        m3=old_oof.loc[(old_oof.algorithm==f"M3_seed_{seed}") & old_oof.fold.isin(known.fold.unique())]
        def compare(ref): return evaluate_predictions(_with_comparator(known,ref).drop(columns=["algorithm","source_drug_status"]),seed=seed,resamples=2000)
        metrics["metrics"][algorithm]={"vs_m1_source_scaled":compare(base),"vs_m3":compare(m3)}
    # Existing OOF B0 is retained as a reference report, but never used to fit.
    for name in (f"B0_seed_{seed}",f"M1_source_scaled_seed_{seed}",f"M3_seed_{seed}"):
        frame=old_oof.loc[(old_oof.algorithm==name)&old_oof.source_drug_status.eq("known")]
        metrics["metrics"][name]=evaluate_predictions(frame.drop(columns=["algorithm","source_drug_status"]),seed=seed,resamples=2000)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2)); (output/"frozen_run.json").write_text(json.dumps({"seed":seed,"alpha":10.,"lambdas":[0.,.1,1.],"target_admission_sha256":sha256(admission_path),"target_admission_status":admission["status"],"source_sha256":{n:sha256(source_dir/n) for n in ("source_model.pt","expression_transform.npz","frozen_run.json")},"target_registry_sha256":sha256(registry),"existing_oof_sha256":sha256(old)},indent=2)); return metrics
