"""EXP-036 A1：训练侧 PCA 加逐药低秩 Ridge 交互及患者表达置换。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from drug_screen.evaluation.metrics import spearman_rho


ALPHAS = (0.1, 1.0, 10.0, 100.0)
PERMUTATION_SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patient_scores(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, group in frame.groupby("patient_id", sort=True):
        if group["drug_id"].nunique() < 3:
            continue
        rows.append({"patient_id": patient,
                     "spearman": spearman_rho(group["observed"], group["predicted"]),
                     "n_drugs": int(group["drug_id"].nunique())})
    return pd.DataFrame(rows)


class TrainExpressionProjection:
    def __init__(self, maximum_genes: int = 500, maximum_components: int = 32):
        self.maximum_genes = maximum_genes
        self.maximum_components = maximum_components

    def fit(self, values: np.ndarray) -> "TrainExpressionProjection":
        values = np.asarray(values, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0, ddof=0)
        variance = np.nanvar(values, axis=0)
        self.columns = np.argsort(variance, kind="stable")[-self.maximum_genes:]
        selected_scale = self.scale[self.columns].copy()
        selected_scale[selected_scale <= 1e-8] = 1.0
        self.selected_scale = selected_scale
        normalized = (values[:, self.columns] - self.mean[self.columns]) / selected_scale
        components = min(self.maximum_components, len(values) - 1, normalized.shape[1])
        if components < 1:
            raise ValueError("at least two training patients are required")
        self.pca = PCA(n_components=components, svd_solver="full").fit(normalized)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        normalized = (values[:, self.columns] - self.mean[self.columns]) / self.selected_scale
        return self.pca.transform(normalized)


def design(rows: pd.DataFrame, patient_z: dict[str, np.ndarray], drug_position: dict[str, int],
           *, interaction: bool) -> np.ndarray:
    n_drugs = len(drug_position)
    onehot = np.zeros((len(rows), n_drugs), dtype=np.float64)
    positions = np.array([drug_position[x] for x in rows["drug_id"]], dtype=int)
    onehot[np.arange(len(rows)), positions] = 1.0
    if not interaction:
        return onehot
    z = np.stack([patient_z[x] for x in rows["patient_id"]])
    return np.concatenate([onehot, (z[:, :, None] * onehot[:, None, :]).reshape(len(rows), -1)], axis=1)


def mean_score(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    frame = rows[["patient_id", "drug_id", "response"]].rename(columns={"response": "observed"})
    frame["predicted"] = prediction
    scores = patient_scores(frame)
    return float(scores["spearman"].mean()) if len(scores) else float("nan")


def bootstrap(difference: np.ndarray, seed: int = 20260922) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                      for _ in range(2000)])
    observed = abs(float(difference.mean()))
    perm = np.array([abs(float(np.mean(difference * rng.choice([-1, 1], len(difference)))))
                     for _ in range(10000)])
    return {"estimate": float(difference.mean()), "low": float(np.quantile(draws, .025)),
            "high": float(np.quantile(draws, .975)),
            "p_two_sided": float((1 + (perm >= observed).sum()) / 10001),
            "n_patients": len(difference)}


def run(contract_dir: Path, split_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    response = pd.read_parquet(contract_dir / "target_pairs.parquet").merge(
        samples[["sample_id", "patient_id"]], on="sample_id", validate="many_to_one")
    response = response.groupby(["patient_id", "drug_id"], as_index=False)["response"].mean()
    expression = np.load(contract_dir / "target_expression.npy")
    sample_expression = {sample: expression[i] for i, sample in enumerate(samples["sample_id"])}
    patient_expression = {patient: np.stack([sample_expression[x] for x in group["sample_id"]]).mean(axis=0)
                          for patient, group in samples.groupby("patient_id", sort=True)}
    drugs = pd.read_parquet(contract_dir / "drugs.parquet")["drug_id"].tolist()
    drug_position = {drug: i for i, drug in enumerate(drugs)}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    predictions, scores, selections = [], [], []
    for fold_index, fold in enumerate(split["target_folds"]):
        test = response.loc[response["patient_id"].isin(fold["outer_test_ids"])]
        for fraction_data in fold["label_fractions"]:
            fraction = float(fraction_data["fraction"])
            candidate_scores = {
                model: {alpha: [] for alpha in ALPHAS}
                for model in ("drug_only_ridge", "expression_drug_interaction")
            }
            for inner in fraction_data["inner_folds"]:
                inner_train_ids = inner["train_ids"]
                inner_val_ids = inner["validation_ids"]
                projection = TrainExpressionProjection().fit(
                    np.stack([patient_expression[x] for x in inner_train_ids]))
                z = {patient: projection.transform(patient_expression[patient][None])[0]
                     for patient in set(inner_train_ids) | set(inner_val_ids)}
                train = response.loc[response["patient_id"].isin(inner_train_ids)]
                val = response.loc[response["patient_id"].isin(inner_val_ids)]
                seen = set(train["drug_id"])
                val = val.loc[val["drug_id"].isin(seen)]
                for model_name, interaction in (("drug_only_ridge", False),
                                                ("expression_drug_interaction", True)):
                    x_train = design(train, z, drug_position, interaction=interaction)
                    x_val = design(val, z, drug_position, interaction=interaction)
                    for alpha in ALPHAS:
                        model = Ridge(alpha=alpha).fit(x_train, train["response"])
                        candidate_scores[model_name][alpha].append(
                            mean_score(val, model.predict(x_val)))
            mean_candidates = {
                model: {alpha: float(np.nanmean(value)) for alpha, value in by_alpha.items()}
                for model, by_alpha in candidate_scores.items()
            }
            selected_alpha = {
                model: min(ALPHAS, key=lambda value: (-by_alpha[value], value))
                for model, by_alpha in mean_candidates.items()
            }
            train_ids = fraction_data["train_ids"]
            projection = TrainExpressionProjection().fit(
                np.stack([patient_expression[x] for x in train_ids]))
            all_ids = set(train_ids) | set(fold["outer_test_ids"])
            z = {patient: projection.transform(patient_expression[patient][None])[0] for patient in all_ids}
            train = response.loc[response["patient_id"].isin(train_ids)]
            seen = set(train["drug_id"])
            eligible_test = test.loc[test["drug_id"].isin(seen)].copy()
            for model_name, interaction in (("drug_only_ridge", False), ("expression_drug_interaction", True)):
                model = Ridge(alpha=selected_alpha[model_name]).fit(
                    design(train, z, drug_position, interaction=interaction), train["response"])
                frame = eligible_test[["patient_id", "drug_id", "response"]].rename(
                    columns={"response": "observed"})
                frame["predicted"] = model.predict(
                    design(eligible_test, z, drug_position, interaction=interaction))
                frame.insert(0, "fold", fold_index); frame.insert(1, "fraction", fraction)
                frame.insert(2, "model", model_name); frame.insert(3, "seed", -1)
                predictions.append(frame)
                unit = patient_scores(frame)
                unit.insert(0, "fold", fold_index); unit.insert(1, "fraction", fraction)
                unit.insert(2, "model", model_name); unit.insert(3, "seed", -1)
                scores.append(unit)
            for seed in PERMUTATION_SEEDS:
                rng = np.random.default_rng(seed + fold_index)
                donors = list(train_ids); shuffled = donors.copy(); rng.shuffle(shuffled)
                permuted_z = dict(z)
                for patient, donor in zip(donors, shuffled, strict=True):
                    permuted_z[patient] = z[donor]
                model = Ridge(alpha=selected_alpha["expression_drug_interaction"]).fit(
                    design(train, permuted_z, drug_position, interaction=True), train["response"])
                frame = eligible_test[["patient_id", "drug_id", "response"]].rename(
                    columns={"response": "observed"})
                frame["predicted"] = model.predict(
                    design(eligible_test, z, drug_position, interaction=True))
                frame.insert(0, "fold", fold_index); frame.insert(1, "fraction", fraction)
                frame.insert(2, "model", "patient_expression_permuted"); frame.insert(3, "seed", seed)
                predictions.append(frame)
                unit = patient_scores(frame)
                unit.insert(0, "fold", fold_index); unit.insert(1, "fraction", fraction)
                unit.insert(2, "model", "patient_expression_permuted"); unit.insert(3, "seed", seed)
                scores.append(unit)
            selections.append({"fold": fold_index, "fraction": fraction,
                               "selected_alpha": selected_alpha,
                               "inner_mean_spearman": mean_candidates,
                               "n_train_patients": len(train_ids),
                               "n_test_patients": len(set(eligible_test["patient_id"])),
                               "n_seen_drugs": len(seen),
                               "pca_components": projection.pca.n_components_})
        print(json.dumps({"completed_fold": fold_index}), flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    scores = pd.concat(scores, ignore_index=True)
    prediction_path = output_dir / "predictions.parquet"
    score_path = output_dir / "patient_scores.parquet"
    predictions.to_parquet(prediction_path, index=False); scores.to_parquet(score_path, index=False)
    comparisons = []
    for fraction in (0.25, 0.5, 1.0):
        subset = scores.loc[scores["fraction"].eq(fraction)]
        interaction = subset.loc[subset["model"].eq("expression_drug_interaction")].set_index("patient_id")
        drug_only = subset.loc[subset["model"].eq("drug_only_ridge")].set_index("patient_id")
        permuted = subset.loc[subset["model"].eq("patient_expression_permuted")].groupby("patient_id")["spearman"].mean()
        common = interaction.index.intersection(drug_only.index).intersection(permuted.index)
        comparisons.append({"fraction": fraction,
                            "interaction_minus_drug_only": bootstrap(
                                (interaction.loc[common, "spearman"] - drug_only.loc[common, "spearman"]).to_numpy()),
                            "interaction_minus_permuted": bootstrap(
                                (interaction.loc[common, "spearman"] - permuted.loc[common]).to_numpy())})
    report = {"experiment_id": "EXP-036", "run_id": output_dir.name,
              "status": "RIDGE_INTERACTION_COMPLETE_PENDING_REVIEW",
              "feature_protocol": "train-only top-500 variance genes, standardized, PCA <=32; drug one-hot interactions",
              "alpha_candidates": list(ALPHAS), "permutation_seeds": list(PERMUTATION_SEEDS),
              "selections": selections, "comparisons": comparisons,
              "data_manifest_sha256": sha256(contract_dir / "manifest.json"),
              "split_sha256": sha256(split_path),
              "files_sha256": {"predictions.parquet": sha256(prediction_path),
                               "patient_scores.parquet": sha256(score_path)}}
    temporary = output_dir / "run_summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "run_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract_dir, args.split, args.output_dir)
    print(json.dumps({"status": result["status"], "comparisons": result["comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
