"""Correct EXP-016 v3 paired summaries from frozen OOF predictions only.

This intentionally writes new, explicitly corrected derivative files and never
modifies the original runner output or retrains any model.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from drug_screen.transfer.cold_cell import _metrics, _paired_cell_bootstrap


def audit(root: Path, *, resamples: int = 2000):
    oof=pd.read_parquet(root/"oof_predictions.parquet")
    required={"split_seed","init_seed","condition","permutation_id","model","sample_id","drug_id","observed","predicted"}
    if not required <= set(oof): raise ValueError("not an EXP-016 v3 OOF table")
    paired, compact={},{}
    for (split, init), learned in oof[oof.init_seed.ge(0)].groupby(["split_seed","init_seed"],sort=True):
        original=learned[(learned.model=="DualEncoder")&(learned.condition=="original")&(learned.permutation_id==0)]
        b0=oof[(oof.split_seed==split)&(oof.init_seed==-1)&(oof.model=="B0")]
        drug=learned[(learned.model=="DrugOnly")&(learned.condition=="original")]
        for name,other in [("B0",b0),("DrugOnly",drug)]:
            result=_paired_cell_bootstrap(original,other,seed=int(split)+int(init),resamples=resamples)
            # Paired cell mean difference must exactly agree with the two
            # individual macro points for the same split/init/condition.
            left=_metrics(original,bootstrap_seed=int(split)+int(init),resamples=2)["cell_macro"]
            right=_metrics(other,bootstrap_seed=int(split)+int(init),resamples=2)["cell_macro"]
            for key in result:
                if not np.isclose(result[key]["difference"], left[key]-right[key], equal_nan=True): raise AssertionError(f"pair mismatch {split}|{init}|{name}|{key}")
            paired[f"{split}|{init}|DualEncoder_minus_{name}"]=result
        for condition in ["within_cancer_row_permutation","pan_cancer_row_permutation"]:
            values=[]
            for repeat,other in learned[(learned.model=="DualEncoder")&(learned.condition==condition)].groupby("permutation_id",sort=True):
                result=_paired_cell_bootstrap(original,other,seed=int(split)+int(init)+int(repeat),resamples=resamples)
                paired[f"{split}|{init}|original_minus_{condition}|{repeat}"]=result
                values.append({"permutation_id":int(repeat),**{metric:record["difference"] for metric,record in result.items()}})
            compact[f"{split}|{init}|original_minus_{condition}"]={metric:{"mean":float(pd.DataFrame(values)[metric].mean()),"q025":float(pd.DataFrame(values)[metric].quantile(.025)),"q975":float(pd.DataFrame(values)[metric].quantile(.975)),"n":len(values)} for metric in ["mae","spearman","ndcg5","top5_overlap"]}
    meta={"status":"CORRECTED_DERIVATIVE_FROM_FROZEN_OOF","original_files_unchanged":True,"resamples":resamples,"paired":paired}
    (root/"paired_bootstrap_corrected.json").write_text(json.dumps(meta,indent=2,allow_nan=False))
    (root/"3seed_100permutation_compact_summary.json").write_text(json.dumps({"status":"CORRECTED_DERIVATIVE_FROM_FROZEN_OOF","original_files_unchanged":True,"summary":compact},indent=2,allow_nan=False))
    return meta


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--resamples",type=int,default=2000)
    args=parser.parse_args();print(json.dumps(audit(args.root,resamples=args.resamples),indent=2,allow_nan=False))
