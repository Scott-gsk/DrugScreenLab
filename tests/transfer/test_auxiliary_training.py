import numpy as np
import os
import subprocess
import sys
from pathlib import Path

from drug_screen.transfer.auxiliary_training import (
    AuxiliaryInputs, AuxiliaryTransferModel, permute_dependency_rows,
    select_dependency_targets,
)
from drug_screen.transfer.models import DualEncoderPairModel


def _drug_model():
    rng = np.random.default_rng(4); x = rng.normal(size=(12, 4)); d = rng.normal(size=(12, 8)); y = rng.normal(size=12)
    model = DualEncoderPairModel(hidden_dim=32, epochs=1, seed=2)
    model.fit(x, d, y, pdo_ids=np.repeat(["a", "b", "c"], 4))
    return model, x, d, y


def test_dependency_target_selection_and_stratified_whole_row_permutation():
    values = np.array([[1., 2.], [3., np.nan], [5., 6.], [7., 8.]])
    chosen, genes = select_dependency_targets(values, [True, True, True, False], ["a", "b"], max_targets=2, min_coverage=.6)
    assert chosen.shape == (4, 2) and set(genes) == {"a", "b"}
    clean = np.nan_to_num(values, nan=-99.)
    permuted = permute_dependency_rows(clean, ["x", "x", "y", "y"], ["L", "L", "L", "L"], seed=3)
    assert {tuple(row) for row in permuted[:2]} == {tuple(row) for row in clean[:2]}
    assert {tuple(row) for row in permuted[2:]} == {tuple(row) for row in clean[2:]}


def test_n3_save_load_preserves_expression_smiles_inference(tmp_path):
    model, x, d, y = _drug_model(); rng = np.random.default_rng(9)
    aux = AuxiliaryInputs(rng.normal(size=(5, 4)), rng.normal(size=(5, 3)), np.array(["CRC"] * 5), np.array(["minLib"] * 5), np.array([f"s{i}" for i in range(5)]), np.array(["g1", "g2", "g3"]))
    fitted = AuxiliaryTransferModel(model, dependency_dim=3).fit(x, d, y, np.repeat(["a", "b", "c"], 4), aux, variant="N3", auxiliary_weight=.1, steps=2, seed=1)
    before = fitted.predict(x[:2], d[:2]); path = tmp_path / "model.pt"; fitted.save(path)
    loaded = AuxiliaryTransferModel.load(path)
    assert np.allclose(before, loaded.predict(x[:2], d[:2]))
    assert len(loaded.history) == 2


def test_runner_exposes_bounded_protocol_control_flags():
    root = Path(__file__).resolve().parents[2]
    env = os.environ | {"PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, str(root / "scripts" / "run_nature_aux_transfer.py"), "--help"],
                            capture_output=True, text=True, check=True, env=env)
    for flag in ("--steps", "--auxiliary-weight", "--permutation-seed", "--disable-dependency", "--aggregate-dir"):
        assert flag in result.stdout
