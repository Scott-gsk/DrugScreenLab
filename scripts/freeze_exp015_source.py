"""Freeze manager-reviewed exploratory source admission and registry."""
from pathlib import Path
import json
from drug_screen.transfer.cohorts import load_registered_cohort, sha256


def main():
    root = Path('data')
    report_path = root/'artifacts/EXP-014/gdsc_v3_build.json'
    report = json.loads(report_path.read_text())
    assets = {}
    for kind, filename in [('samples','samples.parquet'),('drugs','drugs.parquet'),
                           ('pairs','pairs.parquet'),('expression','indexed.npz')]:
        path = root/'processed/exp014/gdsc_v3'/filename
        if sha256(path) != report['output_sha256'][filename]:
            raise ValueError('source build checksum mismatch')
        assets[kind] = {'path':path.relative_to(root).as_posix(),'sha256':sha256(path)}
    registry = root/'registry/exp015_source.json'
    entry = {'cohort_id':'gdsc2_v2','role':'SOURCE_DEVELOPMENT',
             'expression_unit':'log2(CMP TPM+1)','response_name':'1_minus_auc',
             'higher_is_sensitive':True,'assets':assets}
    registry.write_text(json.dumps({'schema_version':1,'cohorts':[entry]},indent=2))
    bundle = load_registered_cohort('gdsc2_v2',root=root,registry=registry)
    receipt = {'cohort_id':'gdsc2_v2','status':'SOURCE_ADMITTED',
               'scope':'EXP015 exploratory source training; not PDO validation',
               'registry_sha256':sha256(registry),'split_group_column':'patient_id',
               'smiles_column':'canonical_smiles','source_audit':str(report_path),
               'source_audit_sha256':sha256(report_path),
               'limitations':['Legacy source structure cache may omit stereochemistry; no recovery of unspecified stereochemistry is claimed.',
                             'Unavailable structures and nonunique original response keys excluded before training.',
                             'Source early-stopping validation is not an independent test.',
                             'Batch/culture metadata unavailable; no batch-specific claims.']}
    # In this source bundle patient_id explicitly aliases the SANGER model ID.
    if not bundle.samples.patient_id.astype(str).equals(bundle.samples.sample_id.astype(str)):
        raise ValueError('source biological group differs from SANGER model ID')
    output = root/'artifacts/EXP-015/source_admission.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'status':receipt['status'],'samples':len(bundle.samples),
                      'drugs':len(bundle.drugs),'pairs':len(bundle.pairs)}))


if __name__ == '__main__':
    main()
