"""Registry-driven, hash-checked loading of immutable indexed cohorts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundles import IndexedBundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'asset escapes data root: {relative}')
    return path


def load_registered_cohort(cohort_id: str, *, root: Path,
                           registry: Path | None = None) -> IndexedBundle:
    registry = registry or root / 'registry' / 'transfer_cohorts.json'
    entries = json.loads(registry.read_text(encoding='utf-8'))['cohorts']
    matches = [entry for entry in entries if entry['cohort_id'] == cohort_id]
    if len(matches) != 1:
        raise ValueError(f'expected one registered cohort: {cohort_id}')
    entry = matches[0]
    paths = {}
    for name in ('samples', 'drugs', 'pairs', 'expression'):
        asset = entry['assets'][name]
        path = confined(root, asset['path'])
        if sha256(path) != asset['sha256']:
            raise ValueError(f'checksum mismatch: {name} {path}')
        paths[name] = path
    samples, drugs, pairs = [pd.read_parquet(paths[key]) for key in ('samples', 'drugs', 'pairs')]
    # The legacy NPZ contains object ID arrays. It is local, explicitly registered,
    # and hash-checked above; arbitrary downloaded pickle/NPZ is never accepted.
    with np.load(paths['expression'], allow_pickle=True) as archive:
        expression = np.asarray(archive['expression'], dtype=float)
        sample_ids = archive['sample_id'].astype(str)
        gene_ids = archive['ordered_genes'].astype(str)
    bundle = IndexedBundle(cohort_id, entry['role'], samples, drugs, pairs,
                           expression, sample_ids, gene_ids,
                           entry['expression_unit'], entry['higher_is_sensitive'],
                           entry['response_name'])
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: IndexedBundle) -> None:
    for table, key in ((bundle.samples, 'sample_id'), (bundle.drugs, 'drug_id')):
        if key not in table or table[key].isna().any() or table[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'missing identity: {key}')
        if table[key].duplicated().any():
            raise ValueError(f'duplicate identity: {key}')
    if bundle.expression.shape != (len(bundle.sample_ids), len(bundle.gene_ids)):
        raise ValueError('expression dimensions do not align with identity arrays')
    if len(set(bundle.sample_ids)) != len(bundle.sample_ids) or len(set(bundle.gene_ids)) != len(bundle.gene_ids):
        raise ValueError('duplicate expression identity')
    if set(bundle.samples.sample_id.astype(str)) != set(bundle.sample_ids):
        raise ValueError('sample table and expression identities differ')
    for key, allowed in [('sample_id', set(bundle.sample_ids)), ('drug_id', set(bundle.drugs.drug_id.astype(str)))]:
        if key not in bundle.pairs or bundle.pairs[key].isna().any() or not set(bundle.pairs[key].astype(str)) <= allowed:
            raise ValueError(f'orphan response: {key}')
    if not np.isfinite(bundle.pairs.response.to_numpy(float)).all():
        raise ValueError('nonfinite response')
    if not bundle.higher_is_sensitive or not bundle.pairs.higher_is_sensitive.eq(True).all():
        raise ValueError('unresolved response direction')
    if set(bundle.pairs.response_name) != {bundle.response_name}:
        raise ValueError('mixed endpoint contract')
    if bundle.pairs.duplicated(['sample_id', 'drug_id']).any():
        raise ValueError('unresolved replicate response; explicit aggregation required')
    if np.isinf(bundle.expression).any():
        raise ValueError('infinite expression')
    if 'PDO' in bundle.role:
        if 'patient_id' not in bundle.samples or bundle.samples.patient_id.isna().any():
            raise ValueError('PDO missing patient identity')


def audit_bundle(bundle: IndexedBundle) -> dict:
    """Structural QC, not proof of baseline timing or original assay provenance."""
    validate_bundle(bundle)
    finite = np.isfinite(bundle.expression)
    panels = bundle.pairs.groupby('sample_id').drug_id.nunique()
    patients = bundle.samples.get('patient_id', pd.Series(dtype=str)).dropna().astype(str)
    patients = patients[patients.str.strip().ne('')]
    values = bundle.expression[finite]
    return {
        'cohort_id': bundle.cohort_id, 'role': bundle.role,
        'structural_validation': 'PASS', 'scientific_admission': 'SOURCE_AUDIT_REQUIRED',
        'n_samples': len(bundle.samples), 'n_patients': int(patients.nunique()),
        'n_drugs': len(bundle.drugs), 'n_response_pairs': len(bundle.pairs),
        'n_genes': len(bundle.gene_ids), 'expression_unit': bundle.expression_unit,
        'expression_missing_fraction': float(1 - finite.mean()),
        'all_missing_genes': int((~finite.any(axis=0)).sum()),
        'all_missing_samples': int((~finite.any(axis=1)).sum()),
        'expression_quantiles': np.quantile(values, [0, .01, .5, .99, 1]).tolist() if len(values) else [],
        'panels_ndcg5': int((panels >= 10).sum()),
        'panels_ndcg3': int(((panels >= 5) & (panels < 10)).sum()),
        'panels_ineligible': int((panels < 5).sum()),
        'metadata_columns': list(bundle.samples.columns),
        'missing_metadata': {key: int(bundle.samples[key].isna().sum()) if key in bundle.samples else len(bundle.samples)
                             for key in ('patient_id', 'pdo_id', 'sample_tissue', 'batch', 'culture_condition')},
        'limitations': ['Indexed-bundle integrity is distinct from original-source verification.',
                       'Previously used cohorts are development resources, not untouched external tests.',
                       'Dose window, timing and technical-repeat QC require original-source metadata.'],
    }
