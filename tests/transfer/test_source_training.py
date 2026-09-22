import json
import numpy as np
import pandas as pd
import pytest
from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.cohorts import sha256
from drug_screen.transfer.source_training import train_source


def test_source_run_groups_and_frozen_reload(tmp_path):
    ids = np.array([f's{i}' for i in range(12)])
    samples = pd.DataFrame({'sample_id': ids, 'bio_group': [f'g{i//2}' for i in range(12)]})
    # Protocol minimum is ten independent groups, not ten observations.
    registry = tmp_path/'registry.json'; registry.write_text('{}')
    admission = tmp_path/'admission.json'
    admission.write_text(json.dumps({'status':'SOURCE_ADMITTED','cohort_id':'synthetic',
        'registry_sha256':sha256(registry),'split_group_column':'bio_group','smiles_column':'smiles'}))
    drugs = pd.DataFrame({'drug_id':['d1','d2'],'smiles':['CCO','CCN']})
    pairs = pd.DataFrame([{'sample_id': s,'drug_id': d,'response':float(i+j)}
                         for i,s in enumerate(ids) for j,d in enumerate(drugs.drug_id)])
    bundle = IndexedBundle('synthetic','SOURCE',samples,drugs,pairs,
                           np.random.default_rng(8).normal(size=(12,4)),ids,
                           np.array(['a','b','c','d']),'log2_TPM',True,'sensitivity')
    config = {'cohort_id':'synthetic','seed':2,'n_features':3,
              'model':{'hidden_dim':4,'epochs':2,'seed':2}}
    kwargs = dict(root=tmp_path,registry=registry,admission=admission,
                  output=tmp_path/'out',loader=lambda *args,**kwargs: bundle)
    with pytest.raises(ValueError, match='ten biological'):
        train_source(config,**kwargs)
    samples['bio_group'] = ids
    result = train_source(config,**kwargs)
    assert result['pdo_transfer_established'] is False
    frozen = json.loads((tmp_path/'out/frozen_run.json').read_text())
    assert not set(frozen['training_groups']) & set(frozen['validation_groups'])
    from drug_screen.transfer.models import DualEncoderPairModel
    loaded = DualEncoderPairModel.load(tmp_path/'out/source_model.pt')
    assert loaded.network is not None
    with pytest.raises(ValueError,match='never overwrite'):
        train_source(config,**kwargs)
