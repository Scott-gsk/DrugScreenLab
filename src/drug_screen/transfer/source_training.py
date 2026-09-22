"""Hash-admitted cell-line pretraining; does not establish PDO transfer value."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .chemistry import morgan_fingerprint
from .cohorts import load_registered_cohort, sha256
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor


def train_source(config: dict, *, root: Path, registry: Path, admission: Path,
                 output: Path, loader=load_registered_cohort) -> dict:
    receipt = json.loads(admission.read_text(encoding='utf-8'))
    if (receipt.get('status') != 'SOURCE_ADMITTED'
            or receipt.get('cohort_id') != config['cohort_id']
            or receipt.get('registry_sha256') != sha256(registry)):
        raise ValueError('source admission or registry hash mismatch')
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty; never overwrite a run')
    bundle = loader(config['cohort_id'], root=root, registry=registry)
    samples = bundle.samples.set_index('sample_id')
    # Explicit biological grouping is a mandatory part of admission.
    group_field = receipt.get('split_group_column')
    if group_field not in samples.columns:
        raise ValueError('admitted biological split column missing')
    groups = samples[group_field]
    if groups.isna().any() or groups.astype(str).str.strip().eq('').any():
        raise ValueError('missing biological split identity')
    unique = np.asarray(sorted(groups.astype(str).unique()))
    if len(unique) < 10:
        raise ValueError('source protocol requires at least ten biological groups')
    rng = np.random.default_rng(config['seed']); rng.shuffle(unique)
    # Fixed 80/20 source development split; no source test claim is made.
    valid = set(unique[:max(2, int(np.ceil(len(unique) * .2)))])
    group_for = groups.astype(str).to_dict()
    indices = {str(key): i for i, key in enumerate(bundle.sample_ids)}
    pairs = bundle.pairs.copy()
    pairs['group'] = pairs.sample_id.map(group_for)
    if pairs.group.isna().any():
        raise ValueError('response group missing')
    tr = pairs.loc[~pairs.group.isin(valid)].copy()
    va = pairs.loc[pairs.group.isin(valid)].copy()
    if tr.empty or va.empty:
        raise ValueError('empty source split')
    prep = ExpressionPreprocessor.fit(
        bundle.expression[[indices[k] for k in sorted(tr.sample_id.unique())]],
        bundle.gene_ids, n_features=config['n_features'])
    expression = prep.transform(bundle.expression, bundle.gene_ids)
    smiles_field = receipt.get('smiles_column')
    if smiles_field not in bundle.drugs:
        raise ValueError('admitted SMILES column missing')
    chemical = bundle.drugs.set_index('drug_id')[smiles_field].to_dict()
    fingerprints = {d: morgan_fingerprint(chemical[d], radius=2, n_bits=2048)
                    for d in pairs.drug_id.unique()}
    def arrays(rows):
        return (expression[[indices[k] for k in rows.sample_id]],
                np.stack([fingerprints[d] for d in rows.drug_id]),
                rows.response.to_numpy(float), rows.sample_id.to_numpy(str))
    x, d, y, ids = arrays(tr)
    model = DualEncoderPairModel(**config['model'])
    start = time.perf_counter()
    model.fit(x, d, y, pdo_ids=ids, validation_data=arrays(va))
    elapsed = time.perf_counter() - start
    vx, vd, vy, _ = arrays(va)
    predicted = model.predict(vx, vd)
    output.mkdir(parents=True, exist_ok=True)
    prep.save(output / 'expression_transform.npz'); model.save(output / 'source_model.pt')
    (output / 'training_history.json').write_text(json.dumps(model.training_history, indent=2))
    va.assign(predicted=predicted).to_parquet(output / 'source_validation_predictions.parquet', index=False)
    frozen = {'config': config, 'admission': receipt, 'registry_sha256': sha256(registry),
              'training_groups': sorted(set(tr.group)), 'validation_groups': sorted(valid),
              'source_drug_ids': sorted(set(tr.drug_id)),
              'validation_used_for_early_stopping': True}
    (output / 'frozen_run.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    summary = {'status': 'SOURCE_DEVELOPMENT_ONLY', 'pdo_transfer_established': False,
               'training_pairs': len(tr), 'validation_pairs': len(va),
               'validation_mae_original_endpoint': float(np.mean(np.abs(vy-predicted))),
               'training_seconds': elapsed,
               'parameter_count': sum(p.numel() for p in model.network.parameters()),
               'note': 'Early-stopping validation is not an independent test result.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
