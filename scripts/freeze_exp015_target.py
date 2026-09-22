"""Freeze the source-rebuilt Zhu exploratory target with explicit limitations."""
from pathlib import Path
import json
import argparse
from drug_screen.transfer.cohorts import load_registered_cohort, sha256


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--version', choices=['v2','v3'],default='v2')
    version = parser.parse_args().version
    suffix = '' if version == 'v2' else '_v3'
    root = Path('data'); report_path = root/f'artifacts/EXP-015/{"zhu_source_audit.json" if version == "v2" else "zhu_v3_source_audit.json"}'
    report = json.loads(report_path.read_text())
    assets = {}
    for kind, filename in [('samples','samples.parquet'),('drugs','drugs.parquet'),
                           ('pairs','pairs.parquet'),('expression','indexed.npz')]:
        path = root/f'processed/transfer/zhu_{version}'/filename
        if sha256(path) != report['output_sha256'][filename]:
            raise ValueError('target build checksum mismatch')
        assets[kind] = {'path':path.relative_to(root).as_posix(),'sha256':sha256(path)}
    registry = root/f'registry/exp015_target{suffix}.json'
    registry.write_text(json.dumps({'schema_version':1,'cohorts':[{
        'cohort_id':f'zhu_hepatobiliary_{version}','role':'PDO_EXPLORATORY_DEVELOPMENT',
        'expression_unit':'author FPKM unlogged','response_name':'1_minus_auc',
        'higher_is_sensitive':True,'assets':assets}]},indent=2))
    bundle = load_registered_cohort(f'zhu_hepatobiliary_{version}',root=root,registry=registry)
    receipt = {'cohort_id':f'zhu_hepatobiliary_{version}','status':'EXPLORATORY_ADMITTED',
        'registry_sha256':sha256(registry),'source_audit_sha256':sha256(report_path),
        'limitations':['Patient grouping uses author tumour number parsed from PDO ID; a separate clinical patient map remains unavailable.',
                      'Only supported structures exactly linked to author AUC included; panel coverage is reduced.',
                      'Author FPKM retained; source log-TPM mismatch is a domain shift, not silently corrected.',
                      'Previously used development cohort; never external validation.'],
        'approved_scope':'EXP015 fixed-parameter exploratory feasibility; no independent or clinical claims'}
    (root/f'artifacts/EXP-015/target_admission{suffix}.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'models':len(bundle.samples),'drugs':len(bundle.drugs),'pairs':len(bundle.pairs)}))


if __name__ == '__main__':
    main()
