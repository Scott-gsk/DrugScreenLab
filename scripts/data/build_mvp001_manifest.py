"""Build a bounded MVP-001 cache manifest from metadata, never expression values."""
from __future__ import annotations
import argparse, gzip, json
from hashlib import sha256
from pathlib import Path
import h5py, pandas as pd

# Four candidates have an exact, unambiguous PRISM primary-screen structure
# mapping and are present in the bounded LINCS treatment universe.  Keeping
# this list explicit prevents a response-dependent post-hoc cohort expansion.
DRUGS = {"trametinib", "BMS-299897", "BMS-777607", "PD-0325901"}

def digest(path: Path) -> str:
    h=sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def canonical_dose(value: object) -> str:
    """Return a stable decimal label for LINCS float32 dose metadata.

    The source contains values such as ``11.1`` and ``11.1000003815`` for
    the same catalog dose.  Eight significant digits preserve all distinct
    screen levels in the MVP subset while removing float32 display noise.
    """
    try:
        return format(float(value), ".8g")
    except (TypeError, ValueError):
        return str(value)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("data")); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    raw=a.root/"raw/lincs/GSE92742"; cache=a.root/"processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy"
    inst=pd.read_csv(raw/"GSE92742_Broad_LINCS_inst_info.txt.gz",sep="\t",low_memory=False)
    t=inst[inst.pert_type.eq("trt_cp")].copy()
    t=t[t.pert_iname.isin(DRUGS)].copy(); key=["rna_plate","cell_id","pert_time","pert_time_unit"]
    ctl=inst[inst.pert_type.eq("ctl_vehicle")][key+["inst_id"]]
    t=t.merge(ctl.groupby(key,dropna=False).inst_id.first().rename("control_inst_id"),on=key,how="inner")
    with h5py.File(a.root/"interim/lincs/GSE92742/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx","r") as h:
        ids=[x.decode() if isinstance(x,bytes) else str(x) for x in h["0/META/COL/id"][:]]
    rows={v:i for i,v in enumerate(ids)}; t["treatment_cache_row"]=t.inst_id.astype(str).map(rows); t["control_cache_row"]=t.control_inst_id.astype(str).map(rows); t=t.dropna(subset=["treatment_cache_row","control_cache_row"])
    # Canonicalize dose before deriving group/DRT IDs.  Without this step the
    # float32 rendering of one catalog dose can become a fake unseen dose.
    t["canonical_dose"] = t["pert_dose"].map(canonical_dose)
    t["group"] = t.pert_iname.astype(str)+"|"+t.cell_id.astype(str)+"|"+t.canonical_dose.astype(str)+"|"+t.pert_dose_unit.astype(str)+"|"+t.pert_time.astype(str)+"|"+t.pert_time_unit.astype(str)
    t["drt"] = t.pert_iname.astype(str)+"|"+t.canonical_dose.astype(str)+"|"+t.pert_dose_unit.astype(str)+"|"+t.pert_time.astype(str)+"|"+t.pert_time_unit.astype(str)
    # Assign splits at the treatment-group level, but enforce the DRT support
    # invariant before expanding groups into records.  A hash-only rule can
    # (rarely) put every group of one drug/dose/time (DRT) in test, which makes
    # the holdout an unseen-Dose/Time case rather than a within-DRT holdout.
    # Such a DRT is either kept entirely in train (one group) or has at least
    # one deterministic train group retained (multiple groups).
    group_split={}
    for drt,drt_rows in t.groupby("drt",sort=True):
        groups=sorted(drt_rows["group"].unique())
        candidates={group for group in groups if int(sha256(group.encode()).hexdigest(),16)%5==0}
        if len(groups) < 2:
            candidates=set()
        elif len(candidates) == len(groups):
            candidates.remove(groups[0])
        for group in groups:
            group_split[group]="test" if group in candidates else "train"
    records=[]
    for group,g in t.groupby("group",sort=True):
        split=group_split[group]
        for r in g.itertuples(index=False): records.append({"sample_id":str(r.inst_id),"treatment_group_id":group,"drug_id":str(r.pert_iname),"dose_id":str(r.canonical_dose)+str(r.pert_dose_unit),"time_id":str(r.pert_time)+str(r.pert_time_unit),"split":split,"treatment_cache_row":int(r.treatment_cache_row),"control_cache_row":int(r.control_cache_row)})
    # Interleave split labels so bounded Tiny/Small subsets retain both partitions.
    records=sorted(records,key=lambda r:(r["drug_id"], r["split"], r["treatment_group_id"], r["sample_id"]))
    train=[r for r in records if r["split"]=="train"]; test=[r for r in records if r["split"]=="test"]; records=[x for pair in zip(train,test) for x in pair]+train[len(test):]+test[len(train):]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"format":"mvp001_compact_cache_manifest_v1","cache":{"relative_path":"processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy","sha256":digest(cache),"shape":[1319138,978]},"records":records},indent=2)+"\n")
    print(json.dumps({"records":len(records),"train":len(train),"test":len(test),"drugs":sorted(set(r["drug_id"] for r in records))}))
if __name__=="__main__": main()
