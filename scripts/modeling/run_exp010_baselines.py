"""Run real-data M0/M1 on the fixed EXP-010 patient folds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-input-dir", type=Path)
    args = parser.parse_args()
    root = args.data_root / "processed/exp010"
    expr = pd.read_parquet(root / "mcrc_gene_contract.parquet")
    rows = pd.read_parquet(root / "mcrc_monotherapy_dss.parquet")
    input_dir = args.model_input_dir or root / "model_inputs"
    supported_drugs = set(json.loads((input_dir / "source_supported_mcrc_drugs.json").read_text()))
    rows["source_supported"] = rows.drug_name.isin(supported_drugs)
    manifest = json.loads((input_dir / f"fold{args.fold}_manifest.json").read_text())
    partition = {x["sample_id"]: x["partition"] for x in manifest}
    rows["partition"] = rows.sample_id.astype(str).map(partition)
    train, test = rows[rows.partition == "train"].copy(), rows[rows.partition == "test"].copy()
    medians = train.groupby("drug_name").dss.median()
    global_median = float(train.dss.median())
    test["M0"] = test.drug_name.map(medians).fillna(global_median)
    genes = sorted(expr.loc[sorted(train.sample_id.unique())].var(ddof=0).items(), key=lambda x: (-x[1], x[0]))[:512]
    genes = [x[0] for x in genes]
    def design(frame: pd.DataFrame) -> pd.DataFrame:
        values = expr.loc[frame.sample_id.astype(str), genes].reset_index(drop=True)
        values["drug_name"] = frame.drug_name.astype(str).reset_index(drop=True)
        return values
    transformer = ColumnTransformer([("gene", StandardScaler(), genes),
                                     ("drug", OneHotEncoder(handle_unknown="ignore"), ["drug_name"])])
    ridge = make_pipeline(transformer, Ridge(alpha=1.0)).fit(design(train), train.dss)
    forest = make_pipeline(transformer, RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                           n_jobs=-1, random_state=args.seed)).fit(design(train), train.dss)
    test["M1_ridge"] = ridge.predict(design(test))
    test["M1_rf"] = forest.predict(design(test))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    test[["patient_id", "sample_id", "drug_name", "dss", "source_supported", "M0", "M1_ridge", "M1_rf"]].to_parquet(args.output, index=False)
    print(json.dumps({"fold": args.fold, "train_rows": len(train), "test_rows": len(test),
                      "common_support_test_rows": int(test.source_supported.sum()),
                      "all_drug_diagnostic_test_rows": len(test),
                      "train_only_hvg": len(genes), "output": str(args.output)}))

if __name__ == "__main__":
    main()
