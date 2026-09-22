"""Run a frozen M3 fold only on its held-out PDOs; produce a usable demo."""
from pathlib import Path
import numpy as np
import pandas as pd
from drug_screen.transfer.inference import predict_transfer


def main():
    run = Path('data/artifacts/EXP-015/zhu_transfer_v1')
    p = pd.read_parquet(run/'outer_oof_predictions.parquet')
    held = set(p.loc[p.algorithm.eq('M3_seed_20260912') & p.fold.eq(0),'model_id'])
    with np.load('data/processed/transfer/zhu_v2/indexed.npz',allow_pickle=False) as z:
        mask = np.array([key in held for key in z['sample_id']])
        np.savez_compressed(run/'heldout_demo_input.npz',sample_id=z['sample_id'][mask],
                            ordered_genes=z['ordered_genes'],expression=z['expression'][mask])
    predict_transfer(run/'seed_20260912_fold_0_m3.pt',run/'seed_20260912_fold_0_source_preprocessor.npz',
        run/'heldout_demo_input.npz',Path('data/processed/transfer/zhu_v2/drugs.parquet'),
        run/'heldout_demo',explain_top_k=3)


if __name__=='__main__':main()
