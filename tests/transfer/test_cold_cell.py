import numpy as np, pandas as pd
from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.cold_cell import split_sample_ids, _ndcg, _paired_cell_bootstrap

def test_split_sample_ids_is_group_disjoint_and_frozen():
    ids=np.array([f's{i}' for i in range(20)])
    a,b,c=split_sample_ids(ids,7)
    assert len(a)==12 and len(b)==4 and len(c)==4
    assert not (set(a)&set(b)|set(a)&set(c)|set(b)&set(c))
    assert np.array_equal(a,split_sample_ids(ids,7)[0])

def test_ndcg_perfect_and_reverse():
    y=np.array([3.,2.,1.,0.])
    assert _ndcg(y,y,3)==1.0
    assert _ndcg(y,y[::-1],3)<1.0


def test_paired_bootstrap_keeps_initialization_rows_isolated():
    def frame(seed, prediction):
        return pd.DataFrame({"sample_id":["a","a","b","b"],"drug_id":["x","y","x","y"],
                             "observed":[0.,1.,0.,1.],"predicted":prediction,"init_seed":seed})
    # The caller must pass one seed's rows.  A different seed has a deliberately
    # opposite result and would change the paired point if it leaked in.
    good, b0 = frame(1,[0.,1.,0.,1.]), frame(-1,[1.,0.,1.,0.])
    bad = frame(2,[1.,0.,1.,0.])
    isolated=_paired_cell_bootstrap(good,b0,seed=1,resamples=10)
    mixed=_paired_cell_bootstrap(pd.concat([good,bad]),pd.concat([b0,b0]),seed=1,resamples=10)
    assert isolated["spearman"]["difference"] > mixed["spearman"]["difference"]
