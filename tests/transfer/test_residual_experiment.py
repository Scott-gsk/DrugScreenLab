import json
import numpy as np
import pytest
from drug_screen.transfer.residual_experiment import _fit_sample_space,_predict

def test_sample_space_residual_crossfit_and_lambdas_share_weights():
 rng=np.random.default_rng(1); n=24; s=rng.normal(size=n); x=rng.normal(size=(n,3)); d=rng.normal(size=(n,5)); y=s+x[:,0]; groups=np.repeat([f"p{i}" for i in range(6)],4)
 state=_fit_sample_space(s,x,d,y,groups,2)
 for fold in state["audit"].values(): assert not set(fold["train_groups"])&set(fold["heldout_groups"])
 assert np.allclose(_predict(state,s,x,d,0),state["a"]*s+state["b"])


def test_runner_requires_hash_bound_target_admission(tmp_path):
 from drug_screen.transfer.residual_experiment import run_residual_diagnostic
 registry = tmp_path / "registry.json"; registry.write_text("{}")
 admission = tmp_path / "admission.json"; admission.write_text(json.dumps({"cohort_id":"zhu_hepatobiliary_v3","status":"EXPLORATORY_ADMITTED","registry_sha256":"wrong"}))
 with pytest.raises(ValueError, match="registry hash mismatch"):
  run_residual_diagnostic(source_dir=tmp_path, data_root=tmp_path, registry_path=registry, target_admission=admission, existing_oof=tmp_path / "oof.parquet", output_dir=tmp_path / "output")
