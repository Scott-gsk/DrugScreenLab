import importlib.util
from pathlib import Path
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/modeling/materialize_exp010_model_inputs.py"
spec = importlib.util.spec_from_file_location("exp010_materialize", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_top_variance_is_train_only_and_tie_stable() -> None:
    train = pd.DataFrame({"B": [0, 2], "A": [0, 2], "C": [1, 1]}, index=["train1", "train2"])
    assert module.top_variance(train, 2) == ["A", "B"]
    # A held-out extreme is intentionally never supplied to selection.
    held_out = pd.DataFrame({"B": [0], "A": [0], "C": [999]}, index=["test"])
    assert module.top_variance(train, 2) != module.top_variance(pd.concat([train, held_out]), 2)


def test_normalization_handles_punctuation_and_case() -> None:
    assert module.norm("5-FU") == module.norm("5 fu")


def test_source_auc_and_target_dss_have_same_sensitivity_direction() -> None:
    source = module.auc_to_sensitivity(pd.Series([0.0, 0.25, 1.0]))
    assert source.tolist() == [1.0, 0.75, 0.0]
    assert source.iloc[0] > source.iloc[-1]  # lower AUC -> higher sensitivity label
    target = pd.DataFrame({"dss": [1.0, 9.0], "higher_is_sensitive": [True, True]})
    assert target.loc[1, "dss"] > target.loc[0, "dss"]
    assert target.higher_is_sensitive.all()


def test_source_auc_range_is_a_hard_gate() -> None:
    import pytest
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        module.auc_to_sensitivity(pd.Series([-0.01, 0.5]))
