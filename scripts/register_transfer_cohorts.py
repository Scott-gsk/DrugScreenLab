"""Migrate selected existing indexed cohorts into an explicit verified registry."""
import argparse
import json
from pathlib import Path

from drug_screen.transfer.cohorts import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--index', default='processed/exp012/cohort_index_manifest.json')
    args = parser.parse_args()
    index_path = args.root / args.index
    index = json.loads(index_path.read_text())
    selected = {'mcrc_pdo', 'zhu_hepatobiliary', 'gdsc2_cell_line'}
    entries = []
    hash_keys = {'samples': 'output_samples_parquet', 'drugs': 'output_drugs_parquet',
                 'pairs': 'output_pairs_parquet', 'raw_npz': 'output_indexed_npz'}
    for row in index['cohorts']:
        if row['cohort_id'] not in selected:
            continue
        assets = {}
        for key, hash_key in hash_keys.items():
            path = index_path.parent / row['bundle'][key]
            actual = sha256(path)
            if actual != row['hashes'][hash_key]:
                raise ValueError(f'legacy manifest checksum mismatch: {path}')
            assets['expression' if key == 'raw_npz' else key] = {
                'path': path.relative_to(args.root).as_posix(), 'sha256': actual,
            }
        entries.append({
            'cohort_id': row['cohort_id'], 'role': row['role'],
            'expression_unit': row['expression_unit'], 'response_name': row['response_name'],
            'higher_is_sensitive': row['higher_is_sensitive'], 'assets': assets,
            'source_index': args.index, 'source_index_sha256': sha256(index_path),
            'original_source_audit': 'required', 'external_test_eligible': False,
            'reason': 'Legacy development data; hash verification is not new independent validation.',
        })
    value = {'schema_version': 1, 'cohorts': entries}
    output = args.root / 'registry/transfer_cohorts.json'
    if output.exists() and json.loads(output.read_text()) != value:
        raise ValueError('refusing to replace a changed registry; review migration explicitly')
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print(f'registered {len(entries)} hash-verified legacy cohorts: {output}')


if __name__ == '__main__':
    main()
