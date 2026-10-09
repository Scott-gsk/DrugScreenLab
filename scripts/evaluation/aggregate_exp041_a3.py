"""独立复核并汇总 EXP-041 A3 的五折、患者级适配器比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.exp031_eval import (
    aggregate_seed_patient_scores,
    patient_score_paired_bootstrap,
)
from drug_screen.evaluation.permutation import permutation_p_value


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("random_full", "source_head", "source_full", "source_adapter")
SEEDS = (20260922, 20260923, 20260924)
FRACTION = 0.5
MIN_DRUGS = 3
BOOTSTRAPS = 2_000
SIGN_FLIPS = 10_000
STAT_SEED = 20260922


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _ledger_event(events: list[dict], attempt_id: str, kind: str) -> dict:
    matches = [
        event for event in events
        if event.get("attempt_id") == attempt_id and event.get("event") == kind
    ]
    if len(matches) != 1:
        raise ValueError(f"{attempt_id}: expected one {kind} ledger event, found {len(matches)}")
    return matches[0]


def _normalize_training_command(command: list[str]) -> list[str]:
    normalized = list(command)
    for flag, placeholder in (("--fold", "<FOLD>"), ("--output-dir", "<FOLD_OUTPUT>")):
        if flag not in normalized:
            raise ValueError(f"training command lacks required argument {flag}")
        normalized[normalized.index(flag) + 1] = placeholder
    return normalized


def verify_run_provenance(folds_dir: Path) -> dict:
    """Verify the immutable parent manifest and ledger-backed per-fold addendum."""
    path = folds_dir / "run_provenance.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("status") != "RUN_PROVENANCE_CAPTURED":
        raise ValueError("EXP-036 shared run lacks a captured provenance receipt")

    supplement_path = folds_dir / "run_provenance_supplement_v1.json"
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    parent_sha = sha256(path)
    if (supplement.get("status") != "PROVENANCE_SUPPLEMENT_VERIFIED_FROM_APPEND_ONLY_LEDGER"
            or supplement.get("parent_provenance_sha256") != parent_sha
            or supplement.get("parent_provenance_path") != str(path.relative_to(ROOT))):
        raise ValueError("versioned provenance supplement is missing, stale, or bound to another parent")
    if (supplement.get("parent_run_id") != provenance.get("parent_run_id")
            or supplement.get("queue_id") != provenance.get("queue_id")):
        raise ValueError("provenance supplement does not match the captured parent run")

    override_path = folds_dir / "source_snapshot_overrides_v1.json"
    source_overrides = {}
    if override_path.is_file():
        override_receipt = json.loads(override_path.read_text(encoding="utf-8"))
        if (override_receipt.get("status") != "HISTORICAL_SOURCE_SNAPSHOT_OVERRIDES"
                or override_receipt.get("parent_provenance_sha256") != parent_sha
                or override_receipt.get("parent_supplement_sha256") != sha256(supplement_path)):
            raise ValueError("historical source snapshot overrides are stale or bind another run")
        source_overrides = override_receipt.get("source_snapshots", {})
        for relative, item in source_overrides.items():
            snapshot = ROOT / item["path"]
            if not snapshot.is_file() or sha256(snapshot) != item.get("sha256"):
                raise ValueError(f"historical source snapshot is missing or changed: {relative}")

    queue_runner = "scripts/modeling/run_exp036_approved_incremental_queue.py"
    aggregate_script = "scripts/evaluation/aggregate_exp041_a3.py"
    wrapper_scope = supplement.get("queue_wrapper_change_scope", {})
    queue_snapshot_sha = source_overrides.get(queue_runner, {}).get(
        "sha256", sha256(ROOT / queue_runner))
    if (wrapper_scope.get("fold_2_sha256") != provenance["code_sha256"][queue_runner]
            or wrapper_scope.get("fold_3_4_sha256") != queue_snapshot_sha):
        raise ValueError("queue-wrapper per-fold revision hashes do not match parent/current sources")
    training_script = "scripts/modeling/exp036_pharmaformer_fold.py"
    if (training_script in source_overrides
            and source_overrides[training_script].get("sha256")
            != provenance["code_sha256"].get(training_script)):
        raise ValueError("historical trainer snapshot differs from the accepted run provenance")

    aggregate_snapshot = supplement.get("original_aggregate_script_snapshot", {})
    snapshot_path = ROOT / aggregate_snapshot.get("path", "")
    if (not snapshot_path.is_file()
            or sha256(snapshot_path) != provenance["code_sha256"].get(aggregate_script)
            or aggregate_snapshot.get("sha256") != provenance["code_sha256"].get(aggregate_script)):
        raise ValueError("original A3 aggregator source snapshot does not match parent provenance")

    for relative, expected in provenance.get("code_sha256", {}).items():
        if relative in {queue_runner, aggregate_script}:
            continue
        source = ROOT / source_overrides[relative]["path"] if relative in source_overrides else ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"provenance source is missing: {relative}")
        if sha256(source) != expected:
            raise ValueError(f"provenance hash mismatch (code_sha256): {relative}")
    for relative, expected in provenance.get("input_sha256", {}).items():
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"provenance input is missing: {relative}")
        if sha256(source) != expected:
            raise ValueError(f"provenance hash mismatch (input_sha256): {relative}")

    ledger_path = ROOT / supplement["ledger_path"]
    if not ledger_path.is_file():
        raise FileNotFoundError("append-only shared queue ledger is missing")
    events = [
        json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts = supplement.get("new_fold_attempts", {})
    if set(attempts) != {"2", "3", "4"}:
        raise ValueError("provenance supplement must bind exactly new folds 2-4")
    normalized_commands = []
    common_codes = None
    common_inputs = None
    for fold in (2, 3, 4):
        entry = attempts[str(fold)]
        started = entry["attempt_started"]
        attempt_id = started.get("attempt_id")
        if not attempt_id:
            raise ValueError(f"fold {fold}: provenance supplement lacks attempt identity")
        actual_started = _ledger_event(events, attempt_id, "attempt_started")
        actual_finished = _ledger_event(events, attempt_id, "attempt_finished")
        actual_promoted = _ledger_event(events, attempt_id, "fold_promoted")
        if (actual_started != started
                or actual_finished != entry.get("attempt_finished")
                or actual_promoted != entry.get("fold_promoted")):
            raise ValueError(f"fold {fold}: versioned provenance events differ from the append-only ledger")
        if (actual_started.get("fold") != fold
                or actual_finished.get("fold") != fold
                or actual_finished.get("exit_code") != 0
                or actual_finished.get("outcome") != "EXIT_0"
                or actual_promoted.get("fold") != fold):
            raise ValueError(f"fold {fold}: run completion or promotion identity is invalid")
        expected_canonical = (
            f"artifacts/experiments/EXP-036/20260926_pharmaformer_transfer_50pct/fold_{fold}"
        )
        if actual_promoted.get("canonical_dir") != expected_canonical:
            raise ValueError(f"fold {fold}: promotion target differs from the frozen EXP-036 output")

        hashes = actual_started.get("input_and_code_hashes", {})
        attempt_inputs = {name: hashes.get(name) for name in provenance["input_sha256"]}
        attempt_codes = {name: hashes.get(name) for name in provenance["code_sha256"]}
        if attempt_inputs != provenance["input_sha256"]:
            raise ValueError(f"fold {fold}: attempt input hashes differ from parent provenance")
        if attempt_codes.get(queue_runner) != entry.get("queue_runner_sha256"):
            raise ValueError(f"fold {fold}: queue-wrapper hash differs from its attempt ledger")
        if fold == 2 and attempt_codes[queue_runner] != provenance["code_sha256"][queue_runner]:
            raise ValueError("fold 2 queue-wrapper hash differs from parent provenance")
        if fold in (3, 4) and attempt_codes[queue_runner] != queue_snapshot_sha:
            raise ValueError(f"fold {fold}: queue-wrapper hash differs from its verified source snapshot")
        non_wrapper_codes = {k: v for k, v in attempt_codes.items() if k != queue_runner}
        expected_non_wrapper = {
            k: v for k, v in provenance["code_sha256"].items() if k != queue_runner
        }
        if non_wrapper_codes != expected_non_wrapper:
            raise ValueError(f"fold {fold}: non-wrapper source hashes differ from parent provenance")
        if common_codes is not None and non_wrapper_codes != common_codes:
            raise ValueError("common model/training/evaluation source hashes vary between new folds")
        if common_inputs is not None and attempt_inputs != common_inputs:
            raise ValueError("frozen input hashes vary between new folds")
        common_codes = non_wrapper_codes
        common_inputs = attempt_inputs

        normalized = _normalize_training_command(actual_started.get("command", []))
        if normalized != supplement.get("normalized_training_commands_by_fold", {}).get(str(fold)):
            raise ValueError(f"fold {fold}: child training command differs from the supplement")
        normalized_commands.append(normalized)

        fold_dir = folds_dir / f"fold_{fold}"
        summary_path = fold_dir / "fold_summary.json"
        verification_path = fold_dir / "verification.json"
        if (_tree_sha256(fold_dir) != actual_promoted.get("output_tree_sha256")
                or sha256(summary_path) != actual_promoted.get("summary_sha256")
                or sha256(verification_path) != actual_promoted.get("verification_sha256")):
            raise ValueError(f"fold {fold}: promoted tree, summary, or verification hash differs from ledger")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if verification.get("status") != "PASS" or verification.get("fold") != fold:
            raise ValueError(f"fold {fold}: promoted independent verification is not PASS")
    if any(command != normalized_commands[0] for command in normalized_commands[1:]):
        raise ValueError("new-fold training commands differ beyond fold and output directory")
    if (supplement.get("common_code_sha256_for_new_folds") != common_codes
            or supplement.get("common_input_sha256_for_new_folds") != common_inputs):
        raise ValueError("supplement's cross-fold common hash summary is stale")

    if provenance.get("reused_verified_folds") != [0, 1]:
        raise ValueError("captured provenance no longer declares reused folds 0 and 1")
    reused = supplement.get("reused_verified_folds", {})
    for fold in (0, 1):
        source = (folds_dir / f"fold_{fold}").resolve(strict=True)
        original = (folds_dir.parent / provenance["parent_run_id"] / f"fold_{fold}").resolve(strict=True)
        entry = reused.get(str(fold), {})
        verification_path = source / "verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if (source != original
                or entry.get("original_source") != str(source.relative_to(ROOT))
                or entry.get("verification_sha256") != sha256(verification_path)
                or verification.get("status") != "PASS"
                or verification.get("fold") != fold):
            raise ValueError(f"fold {fold}: reused original source verification failed")

    return {
        "parent": provenance,
        "parent_sha256": parent_sha,
        "supplement": supplement,
        "supplement_sha256": sha256(supplement_path),
        "source_snapshot_overrides": source_overrides,
        "source_snapshot_overrides_sha256": sha256(override_path) if override_path.is_file() else None,
        "aggregate_script_snapshot_sha256": aggregate_snapshot["sha256"],
        "current_aggregate_script_sha256": sha256(Path(__file__).resolve()),
        "queue_runner_sha256_by_fold": {
            str(fold): attempts[str(fold)]["queue_runner_sha256"] for fold in (2, 3, 4)
        },
    }


def validate_prediction_keys(
    predictions: pd.DataFrame,
    expected_pairs: pd.DataFrame,
    *,
    fold: int,
) -> None:
    required = {
        "fold", "fraction", "arm", "seed", "patient_id", "drug_id",
        "observed", "predicted",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction file lacks columns: {sorted(missing)}")
    if predictions.empty or not predictions["fold"].eq(fold).all():
        raise ValueError(f"fold {fold}: empty predictions or wrong fold identity")
    if not np.allclose(predictions["fraction"].to_numpy(float), FRACTION):
        raise ValueError("predictions are not from the frozen 50% label tier")
    if set(predictions["arm"]) != set(ARMS):
        raise ValueError("prediction arms differ from the registered four-arm matrix")
    if set(predictions["seed"].astype(int)) != set(SEEDS):
        raise ValueError("prediction seeds differ from the three frozen initializations")
    identity = ["fold", "arm", "seed", "patient_id", "drug_id"]
    if predictions[identity].isna().any().any() or predictions.duplicated(identity).any():
        raise ValueError("prediction keys contain missing or duplicate identities")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy(float)).all():
        raise ValueError("prediction or observed response contains non-finite values")

    expected = expected_pairs[["patient_id", "drug_id", "response"]].copy()
    expected["patient_id"] = expected["patient_id"].astype(str)
    expected["drug_id"] = expected["drug_id"].astype(str)
    if expected.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError(f"fold {fold}: frozen input has duplicate patient/drug labels")
    expected_keys = set(map(tuple, expected[["patient_id", "drug_id"]].to_numpy()))
    expected_y = {
        (patient, drug): float(value)
        for patient, drug, value in expected.itertuples(index=False, name=None)
    }
    for arm in ARMS:
        for seed in SEEDS:
            part = predictions.loc[
                predictions["arm"].eq(arm) & predictions["seed"].eq(seed)
            ].copy()
            part["patient_id"] = part["patient_id"].astype(str)
            part["drug_id"] = part["drug_id"].astype(str)
            keys = set(map(tuple, part[["patient_id", "drug_id"]].to_numpy()))
            if keys != expected_keys:
                raise ValueError(
                    f"fold {fold}, {arm}, seed {seed}: frozen test keys differ "
                    f"(missing={len(expected_keys - keys)}, extra={len(keys - expected_keys)})"
                )
            for patient, drug, value in part[
                ["patient_id", "drug_id", "observed"]
            ].itertuples(index=False, name=None):
                if not np.isclose(float(value), expected_y[(patient, drug)], rtol=0, atol=1e-10):
                    raise ValueError(
                        f"fold {fold}, {arm}, seed {seed}: observed label mismatch"
                    )


def _verify_saved_patient_scores(
    predictions: pd.DataFrame, saved_scores: pd.DataFrame, *, fold: int
) -> float:
    required = {"fold", "fraction", "arm", "seed", "patient_id", "spearman", "n_drugs"}
    missing = required - set(saved_scores.columns)
    if missing:
        raise ValueError(f"patient score file lacks columns: {sorted(missing)}")
    score_key = ["fold", "arm", "seed", "patient_id"]
    if saved_scores.duplicated(score_key).any() or not saved_scores["fold"].eq(fold).all():
        raise ValueError(f"fold {fold}: patient score keys are invalid")
    maximum_error = 0.0
    indexed = saved_scores.set_index(score_key).sort_index()
    for arm in ARMS:
        for seed in SEEDS:
            part = predictions.loc[
                predictions["arm"].eq(arm) & predictions["seed"].eq(seed)
            ]
            saved = indexed.loc[(fold, arm, seed)]
            if set(saved.index.astype(str)) != set(part["patient_id"].astype(str)):
                raise ValueError(f"fold {fold}, {arm}, seed {seed}: patient set differs")
            for patient, group in part.groupby("patient_id", sort=True):
                key = (fold, arm, seed, patient)
                row = indexed.loc[key]
                observed = group["observed"].to_numpy(float)
                predicted = group["predicted"].to_numpy(float)
                if len(group) < MIN_DRUGS or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
                    score = float("nan")
                else:
                    score = float(spearmanr(observed, predicted).statistic)
                if int(row["n_drugs"]) != len(group):
                    raise ValueError(f"fold {fold}, {arm}, seed {seed}: drug denominator differs")
                stored = float(row["spearman"])
                if np.isfinite(score) != np.isfinite(stored):
                    raise ValueError(f"fold {fold}, {arm}, seed {seed}: score validity differs")
                if np.isfinite(score):
                    error = abs(score - stored)
                    maximum_error = max(maximum_error, error)
                    if error > 1e-12:
                        raise ValueError(f"fold {fold}, {arm}, seed {seed}: score mismatch")
    return maximum_error


def _verify_fold(folds_dir: Path, input_dir: Path, split_sha: str, manifest_sha: str, fold: int):
    fold_dir = folds_dir / f"fold_{fold}"
    summary_path = fold_dir / "fold_summary.json"
    verification_path = fold_dir / "verification.json"
    prediction_path = fold_dir / "predictions.parquet"
    scores_path = fold_dir / "patient_scores.parquet"
    inner_path = fold_dir / "inner_selection.parquet"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if summary.get("status") != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW":
        raise ValueError(f"fold {fold}: not a full formal run (or smoke output)")
    if verification.get("status") != "PASS" or verification.get("fold") != fold:
        raise ValueError(f"fold {fold}: independent verification is missing or failed")
    if verification.get("n_checkpoints_verified") != len(ARMS) * len(SEEDS):
        raise ValueError(f"fold {fold}: not all arm/seed checkpoints were verified")
    if (summary.get("fold") != fold or summary.get("fraction") != FRACTION
            or set(summary.get("arms", [])) != set(ARMS)
            or tuple(summary.get("seeds", [])) != SEEDS
            or summary.get("split_sha256") != split_sha
            or summary.get("input_manifest_sha256") != manifest_sha):
        raise ValueError(f"fold {fold}: frozen experiment inputs or axes differ")
    if summary.get("candidate_count_per_arm") != {arm: 18 for arm in ARMS}:
        raise ValueError(f"fold {fold}: candidate count differs from the registered search")
    for name, path in (("predictions.parquet", prediction_path),
                       ("patient_scores.parquet", scores_path),
                       ("inner_selection.parquet", inner_path)):
        expected_hash = summary.get("files_sha256", {}).get(name)
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"fold {fold}: artifact hash mismatch for {name}")
    if (verification.get("predictions_sha256") != sha256(prediction_path)
            or verification.get("patient_scores_sha256") != sha256(scores_path)):
        raise ValueError(f"fold {fold}: independent verification hashes are stale")
    predictions = pd.read_parquet(prediction_path)
    scores = pd.read_parquet(scores_path)
    inner_selection = pd.read_parquet(inner_path)
    if len(predictions) != verification.get("n_predictions"):
        raise ValueError(f"fold {fold}: prediction row count differs from verification")
    required_inner = {"arm", "candidate", "inner_fold", "elapsed_seconds"}
    if required_inner - set(inner_selection.columns):
        raise ValueError(f"fold {fold}: inner candidate timing fields are missing")
    expected_fits = len(ARMS) * 18 * 3
    fit_counts = inner_selection.groupby("arm").size().to_dict()
    fit_seconds = inner_selection.groupby("arm")["elapsed_seconds"].sum().to_dict()
    candidate_fit_counts = inner_selection.groupby(
        ["arm", "candidate"], sort=True,
    ).size()
    if (len(inner_selection) != expected_fits
            or fit_counts != {arm: 54 for arm in ARMS}
            or any(
                inner_selection.loc[inner_selection["arm"].eq(arm), "candidate"].nunique() != 18
                for arm in ARMS
            )
            or not candidate_fit_counts.eq(3).all()
            or set(inner_selection["inner_fold"].astype(int)) != {0, 1, 2}
            or not np.isfinite(inner_selection["elapsed_seconds"].to_numpy(float)).all()
            or (inner_selection["elapsed_seconds"].to_numpy(float) < 0).any()):
        raise ValueError(f"fold {fold}: candidate fit counts or timing values are invalid")
    final_receipts = summary.get("final_receipts", [])
    final_keys = {(row.get("arm"), int(row.get("seed", -1))) for row in final_receipts}
    if (len(final_receipts) != len(ARMS) * len(SEEDS)
            or any(not np.isfinite(float(row.get("elapsed_seconds", np.nan)))
                   or float(row["elapsed_seconds"]) < 0
                   or int(row.get("n_trainable_parameters", 0)) < 1
                   for row in final_receipts)
            or {row["arm"] for row in final_receipts} != set(ARMS)
            or {int(row["seed"]) for row in final_receipts} != set(SEEDS)
            or len(final_keys) != len(ARMS) * len(SEEDS)):
        raise ValueError(f"fold {fold}: final refit timing or parameter receipts are invalid")
    score_error = _verify_saved_patient_scores(predictions, scores, fold=fold)
    source_receipts = {
        "fold": fold,
        "summary_sha256": sha256(summary_path),
        "verification_sha256": sha256(verification_path),
        "predictions_sha256": sha256(prediction_path),
        "patient_scores_sha256": sha256(scores_path),
        "inner_selection_sha256": sha256(inner_path),
        "n_predictions": len(predictions),
        "n_patient_scores": len(scores),
        "independent_score_max_error": score_error,
        "inner_candidate_fit_count_by_arm": {
            arm: int(fit_counts[arm]) for arm in ARMS
        },
        "inner_candidate_fit_seconds_by_arm": {
            arm: float(fit_seconds[arm]) for arm in ARMS
        },
        "final_refit_seconds_by_arm": {
            arm: float(sum(float(row["elapsed_seconds"])
                           for row in final_receipts if row["arm"] == arm))
            for arm in ARMS
        },
        "trainable_parameter_counts_by_arm": {
            arm: sorted({
                int(row["n_trainable_parameters"])
                for row in final_receipts if row["arm"] == arm
            })
            for arm in ARMS
        },
    }
    return predictions, scores, summary, source_receipts


def _paired_statistics(scores_a: pd.Series, scores_b: pd.Series) -> dict:
    if not scores_a.index.equals(scores_b.index):
        raise ValueError("paired patient score indexes differ")
    delta = scores_a.to_numpy(float) - scores_b.to_numpy(float)
    if len(delta) < MIN_DRUGS or not np.isfinite(delta).all():
        raise ValueError("too few finite paired patient scores")
    ci = patient_score_paired_bootstrap(
        scores_a, scores_b, n=BOOTSTRAPS, seed=STAT_SEED,
    )
    rng = np.random.default_rng(STAT_SEED)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(SIGN_FLIPS, len(delta)))
    null = np.abs((signs * delta).mean(axis=1))
    p_value = permutation_p_value(abs(float(delta.mean())), null, alternative="greater")
    return {
        "gain": ci.point_estimate,
        "ci95": [ci.low, ci.high],
        "paired_sign_flip_p_two_sided_finite_corrected": p_value,
        "n_valid_patients": ci.n_units,
        "bootstrap_replicates": BOOTSTRAPS,
        "sign_flip_replicates": SIGN_FLIPS,
        "random_seed": STAT_SEED,
        "unit": "patient_id",
    }


def analyze_predictions(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compute the frozen A3 contrast and descriptive fixed-seed stability."""
    summaries = {}
    for arm in ARMS:
        selected = predictions.loc[predictions["arm"].eq(arm)].copy()
        summaries[arm] = aggregate_seed_patient_scores(selected, min_drugs=MIN_DRUGS)
    adapter = summaries["source_adapter"]
    full = summaries["source_full"]
    all_patients = sorted(set(adapter.seed_mean_patient_scores.index)
                          | set(full.seed_mean_patient_scores.index))
    eligible = (
        adapter.seed_mean_patient_scores.reindex(all_patients).notna()
        & full.seed_mean_patient_scores.reindex(all_patients).notna()
        & adapter.n_seeds_used.reindex(all_patients, fill_value=0).eq(len(SEEDS))
        & full.n_seeds_used.reindex(all_patients, fill_value=0).eq(len(SEEDS))
    )
    included = [patient for patient, valid in zip(all_patients, eligible) if valid]
    excluded = [patient for patient, valid in zip(all_patients, eligible) if not valid]
    if len(included) < MIN_DRUGS:
        raise ValueError("fewer than three patients have complete defined scores in both A3 arms")
    adapter_scores = adapter.seed_mean_patient_scores.reindex(included).astype(float)
    full_scores = full.seed_mean_patient_scores.reindex(included).astype(float)
    main = _paired_statistics(adapter_scores, full_scores)

    per_seed = []
    for seed in SEEDS:
        adapter_column = str(seed)
        full_column = str(seed)
        a = adapter.per_seed_patient_scores.reindex(included)[adapter_column]
        b = full.per_seed_patient_scores.reindex(included)[full_column]
        paired = a.notna() & b.notna()
        if paired.any():
            per_seed.append({
                "seed": seed,
                "mean_gain": float((a[paired] - b[paired]).mean()),
                "n_valid_patients": int(paired.sum()),
            })
        else:
            per_seed.append({"seed": seed, "mean_gain": None, "n_valid_patients": 0})

    complete = [
        patient for patient in included
        if adapter.per_seed_patient_scores.loc[patient].notna().all()
        and full.per_seed_patient_scores.loc[patient].notna().all()
    ]
    stability = {"interpretation": "descriptive for the three frozen seeds; not an inference over all initializations"}
    patient_rows: list[dict] = []
    for patient in all_patients:
        row = {
            "patient_id": patient,
            "source_adapter_seed_mean_spearman": _finite_or_none(
                adapter.seed_mean_patient_scores.get(patient, np.nan)),
            "source_full_seed_mean_spearman": _finite_or_none(
                full.seed_mean_patient_scores.get(patient, np.nan)),
            "n_seeds_adapter": int(adapter.n_seeds_used.get(patient, 0)),
            "n_seeds_full": int(full.n_seeds_used.get(patient, 0)),
            "eligible_primary": patient in included,
        }
        row["gain_adapter_minus_full"] = (
            row["source_adapter_seed_mean_spearman"] - row["source_full_seed_mean_spearman"]
            if patient in included else None
        )
        for seed in SEEDS:
            row[f"source_adapter_seed_{seed}"] = _finite_or_none(
                adapter.per_seed_patient_scores.loc[patient, str(seed)]
                if patient in adapter.per_seed_patient_scores.index else np.nan
            )
            row[f"source_full_seed_{seed}"] = _finite_or_none(
                full.per_seed_patient_scores.loc[patient, str(seed)]
                if patient in full.per_seed_patient_scores.index else np.nan
            )
        patient_rows.append(row)

    if complete:
        adapter_sd = adapter.per_seed_patient_scores.loc[complete].to_numpy(float).std(axis=1, ddof=1)
        full_sd = full.per_seed_patient_scores.loc[complete].to_numpy(float).std(axis=1, ddof=1)
        sd_delta = adapter_sd - full_sd
        rng = np.random.default_rng(STAT_SEED)
        indexes = rng.integers(0, len(sd_delta), size=(BOOTSTRAPS, len(sd_delta)))
        boot = sd_delta[indexes].mean(axis=1)
        stability.update({
            "n_patients_all_seeds_defined_in_both_arms": len(complete),
            "mean_patient_seed_sd_adapter": float(adapter_sd.mean()),
            "mean_patient_seed_sd_full_finetune": float(full_sd.mean()),
            "mean_sd_difference_adapter_minus_full": float(sd_delta.mean()),
            "descriptive_patient_bootstrap_ci95_for_mean_sd_difference": [
                float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            ],
        })
    else:
        stability.update({
            "n_patients_all_seeds_defined_in_both_arms": 0,
            "mean_patient_seed_sd_adapter": None,
            "mean_patient_seed_sd_full_finetune": None,
            "mean_sd_difference_adapter_minus_full": None,
            "descriptive_patient_bootstrap_ci95_for_mean_sd_difference": None,
        })
    patient_frame = pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)
    report = {
        "status": "A3_PRIMARY_CONTRAST_READY_LEARNING_CURVE_AND_ROUTE_A_HOLM_PENDING",
        "primary_comparison": "source_adapter_minus_source_full",
        "primary_metric": "patient-within-drug Spearman; patient score averaged across the three frozen seeds",
        "main": main,
        "n_patients_with_complete_defined_scores": len(included),
        "n_patients_excluded_for_undefined_or_incomplete_scores": len(excluded),
        "excluded_patient_ids": excluded,
        "per_seed_descriptive_gains": per_seed,
        "seed_stability": stability,
        "holm_adjustment": None,
        "holm_note": "Unadjusted A3 p-value must be combined with every preregistered A1/A2 primary p-value in the route-A family.",
    }
    return report, patient_frame


def _finite_or_none(value) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def aggregate(folds_dir: Path, input_dir: Path, split_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"EXP-041 output exists: {output_dir}")
    provenance = verify_run_provenance(folds_dir)
    manifest_path = input_dir / "manifest.json"
    pairs_path = input_dir / "patient_pairs.parquet"
    if not manifest_path.is_file() or not pairs_path.is_file():
        raise FileNotFoundError("frozen EXP-036 patient-pair input is incomplete")
    split_sha = sha256(split_path)
    manifest_sha = sha256(manifest_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    folds = split.get("target_folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("frozen target split must contain exactly five outer folds")
    patient_pairs = pd.read_parquet(pairs_path)
    required_pairs = {"patient_id", "drug_id", "response"}
    if required_pairs - set(patient_pairs.columns):
        raise ValueError("patient-pair input lacks required identities or response")
    if patient_pairs.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("frozen patient/drug labels are not unique")
    if not np.isfinite(patient_pairs["response"].to_numpy(float)).all():
        raise ValueError("frozen patient response contains non-finite values")

    all_test_ids: set[str] = set()
    prediction_frames = []
    source_receipts = []
    for fold in range(5):
        fold_info = folds[fold]
        if fold_info.get("fold") != fold:
            raise ValueError("outer-fold ordering differs from its registered fold index")
        test_ids = set(map(str, fold_info["outer_test_ids"]))
        if all_test_ids & test_ids:
            raise ValueError("a patient appears in more than one outer test fold")
        all_test_ids.update(test_ids)
        expected = patient_pairs.loc[
            patient_pairs["patient_id"].astype(str).isin(test_ids),
            ["patient_id", "drug_id", "response"],
        ]
        if set(expected["patient_id"].astype(str)) != test_ids:
            raise ValueError(f"fold {fold}: some frozen test patients have no patient/drug labels")
        predictions, scores, summary, receipt = _verify_fold(
            folds_dir, input_dir, split_sha, manifest_sha, fold,
        )
        validate_prediction_keys(predictions, expected, fold=fold)
        if len(expected) != summary.get("n_test_patient_drug_pairs"):
            raise ValueError(f"fold {fold}: frozen test-pair denominator differs from run summary")
        prediction_frames.append(predictions)
        source_receipts.append(receipt)
    if all_test_ids != set(patient_pairs["patient_id"].astype(str).unique()):
        raise ValueError("outer test folds do not cover all registered target patients exactly once")

    combined = pd.concat(prediction_frames, ignore_index=True)
    if combined.duplicated(["fold", "arm", "seed", "patient_id", "drug_id"]).any():
        raise ValueError("cross-fold prediction keys are duplicated")
    report, patient_scores = analyze_predictions(combined)
    report.update({
        "experiment_id": "EXP-041",
        "parent_experiment_id": "EXP-036",
        "parent_run_id": folds_dir.name,
        "fraction": FRACTION,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "n_test_patients_total": int(len(all_test_ids)),
        "n_patient_drug_predictions": int(len(combined)),
        "source_run_provenance_sha256": provenance["parent_sha256"],
        "source_run_provenance_supplement_sha256": provenance["supplement_sha256"],
        "source_aggregate_script_snapshot_sha256": provenance["aggregate_script_snapshot_sha256"],
        "aggregation_script_current_sha256": provenance["current_aggregate_script_sha256"],
        "queue_runner_sha256_by_fold": provenance["queue_runner_sha256_by_fold"],
        "input_manifest_sha256": manifest_sha,
        "split_sha256": split_sha,
        "source_folds": source_receipts,
        "provenance_hashes_rechecked": True,
    })
    cost_by_arm = {}
    for arm in ARMS:
        parameter_counts = [
            count
            for receipt in source_receipts
            for count in receipt["trainable_parameter_counts_by_arm"][arm]
        ]
        if not parameter_counts:
            raise ValueError(f"no parameter count recorded for arm {arm}")
        candidate_seconds = sum(
            receipt["inner_candidate_fit_seconds_by_arm"][arm]
            for receipt in source_receipts
        )
        refit_seconds = sum(
            receipt["final_refit_seconds_by_arm"][arm]
            for receipt in source_receipts
        )
        cost_by_arm[arm] = {
            "inner_candidate_fits": sum(
                receipt["inner_candidate_fit_count_by_arm"][arm]
                for receipt in source_receipts
            ),
            "inner_candidate_fit_seconds": candidate_seconds,
            "final_refit_seconds": refit_seconds,
            "measured_fit_seconds_total": candidate_seconds + refit_seconds,
            "trainable_parameter_count_min": min(parameter_counts),
            "trainable_parameter_count_median": float(np.median(parameter_counts)),
            "trainable_parameter_count_max": max(parameter_counts),
            "parameter_count_scope": "trainable parameters recorded in final-fold receipts",
        }
    report["training_cost"] = {
        "unit": "seconds summed over the five independent fold runs",
        "timing_scope": "model fit time for inner candidates and final refits; excludes data loading and aggregation",
        "gpu_scope": "single RTX A5000 per fold run",
        "by_arm": cost_by_arm,
        "adapter_to_full_trainable_parameter_ratio_median": float(
            cost_by_arm["source_adapter"]["trainable_parameter_count_median"]
            / cost_by_arm["source_full"]["trainable_parameter_count_median"]
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        score_path = staging / "patient_scores.parquet"
        patient_scores.to_parquet(score_path, index=False)
        report["patient_scores_sha256"] = sha256(score_path)
        summary_path = staging / "a3_summary.json"
        summary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "experiment_id": "EXP-041",
            "status": report["status"],
            "outputs": {
                "a3_summary.json": sha256(summary_path),
                "patient_scores.parquet": sha256(score_path),
            },
            "source_run_provenance_sha256": report["source_run_provenance_sha256"],
            "source_run_provenance_supplement_sha256": report[
                "source_run_provenance_supplement_sha256"
            ],
            "source_aggregate_script_snapshot_sha256": report[
                "source_aggregate_script_snapshot_sha256"
            ],
            "aggregation_script_current_sha256": report["aggregation_script_current_sha256"],
            "queue_runner_sha256_by_fold": report["queue_runner_sha256_by_fold"],
            "source_folds": source_receipts,
            "input_manifest_sha256": manifest_sha,
            "split_sha256": split_sha,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.folds_dir, args.input_dir, args.split, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "status": result["status"],
                      "main": result["main"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
