"""Build a library-preserving paired functional index from official tables."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
from drug_screen.transfer.cohorts import sha256


def main():
    raw = Path('data/raw/transfer_public_assets/organoid_biobank')
    output = Path('data/processed/transfer/biobank_functional_v1')
    if output.exists() and any(output.iterdir()):
        raise ValueError('immutable derived version already exists')
    expression_path = raw/'data_RNAseq_ALL-organoids.csv'
    expression = pd.read_csv(expression_path, index_col=0)
    zip_path = raw/'41586_2026_10830_MOESM3_ESM.zip'
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open('supplementary_table_2_revision.xlsx') as handle:
            samples = pd.read_excel(handle, sheet_name='data_availability')
        with archive.open('supplementary_table_6_revision.csv') as handle:
            dependency = pd.read_csv(handle, sep=r'\s+')
    if samples.sample_ID.duplicated().any():
        raise ValueError('ambiguous sample metadata')
    required = ['sample_ID','gene','library','LFC']
    if dependency[required].isna().any().any():
        raise ValueError('missing dependency identity or score')
    if dependency.duplicated(['sample_ID','gene','library']).any():
        raise ValueError('unresolved within-library repeats')
    shared = sorted(set(expression.index) & set(dependency.sample_ID) & set(samples.sample_ID))
    selected = samples.set_index('sample_ID').loc[shared].reset_index()
    if selected.individual_ID.isna().any():
        raise ValueError('paired functional models lack patient mapping')
    selected['patient_id'] = selected.individual_ID.astype(str)
    dep = dependency.loc[dependency.sample_ID.isin(shared)]
    if not np.isfinite(dep.LFC.to_numpy(float)).all():
        raise ValueError('nonfinite functional score')
    output.mkdir(parents=True)
    dep.to_parquet(output/'dependency_by_library.parquet', index=False)
    selected.to_parquet(output/'samples.parquet', index=False)
    expression.loc[shared].to_parquet(output/'expression_author_log2tpm.parquet')
    report = {'status':'PAIRED_RESOURCE_NOT_YET_TRAINED', 'resource':'Nature organoid biobank',
              'paired_models':len(shared),'paired_patients':selected.patient_id.nunique(),
              'dependency_rows':len(dep),'dependency_genes':dep.gene.nunique(),
              'expression_genes':expression.shape[1],
              'libraries':dep.groupby('library').sample_ID.nunique().to_dict(),
              'models_with_multiple_libraries':int((dep.groupby('sample_ID').library.nunique()>1).sum()),
              'tissues':selected.primary_tumour_type.value_counts().to_dict(),
              'source_hashes':{'supplement_zip':sha256(zip_path),'expression':sha256(expression_path)},
              'output_hashes':{p.name:sha256(p) for p in output.iterdir()},
              'remaining':'Apply author library/QC rules; remove cross-resource overlap by patient; split before training. Dependencies do not replace drug response.'}
    dest = Path('data/artifacts/EXP-015/nature_dependency_pairing.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
