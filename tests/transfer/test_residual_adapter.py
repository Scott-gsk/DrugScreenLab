import numpy as np
from drug_screen.transfer.residual_adapter import ResidualAdapter


def test_crossfit_audits_patient_holds_lambda_zero_and_restore(tmp_path):
    rng = np.random.default_rng(2); n = 24; groups = np.repeat([f"p{i}" for i in range(6)], 4)
    score = rng.normal(size=n); x, drug = rng.normal(size=(n, 3)), rng.normal(size=(n, 2)); y = 2*score + x[:, 0] + rng.normal(scale=.01, size=n)
    adapter = ResidualAdapter(lambda_=0, seed=4).fit(score, x, drug, y, groups)
    expected = adapter.a*score+adapter.b
    assert np.allclose(adapter.predict(score, x, drug), expected)
    for fold in adapter.crossfit_audit["folds"].values(): assert not set(fold["train_groups"]) & set(fold["heldout_groups"])
    path = tmp_path / "adapter.pkl"; adapter.save(path)
    assert np.allclose(ResidualAdapter.load(path).predict(score, x, drug), expected)
