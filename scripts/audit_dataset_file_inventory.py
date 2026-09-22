"""Check registered local assets; file completeness is not scientific admission."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-limit-mb', type=int, default=200)
    args = parser.parse_args()
    rows = []
    for registry in sorted((args.root / 'registry').glob('datasets*.json')):
        entries = json.loads(registry.read_text(encoding='utf-8'))
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            path = args.root / entry['path']['relative']
            checksum = entry.get('checksum', {})
            files = entry.get('files') or checksum.get('files') or {}
            if not files and checksum.get('file'):
                files = {checksum['file']: {checksum.get('algorithm', 'sha256'): checksum.get('value', '')}}
            if not files and path.is_file():
                value = checksum.get('value', '')
                files = {path.name: {checksum.get('algorithm', 'sha256'): value,
                                     'bytes': entry.get('bytes', checksum.get('bytes'))}}
                base = path.parent
            else:
                base = path.parent if path.is_file() else path
            checked = []
            for name, spec in files.items():
                if isinstance(spec, str):
                    spec = {'sha256': spec}
                target = base / name
                result = {'name': name, 'exists': target.is_file()}
                if target.is_file():
                    size = target.stat().st_size
                    result['bytes'] = size
                    result['size_match'] = size == spec['bytes'] if spec.get('bytes') is not None else None
                    for algorithm in ('sha256', 'sha512'):
                        expected = spec.get(algorithm, '')
                        if len(expected) != hashlib.new(algorithm).digest_size * 2:
                            continue
                        if size > args.hash_limit_mb * 1024 * 1024:
                            result['hash_check'] = 'skipped_size_limit'
                        else:
                            digest = hashlib.new(algorithm)
                            with target.open('rb') as stream:
                                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                                    digest.update(chunk)
                            result['hash_check'] = 'match' if digest.hexdigest() == expected.lower() else 'MISMATCH'
                checked.append(result)
            rows.append({'id': entry['id'], 'registry': str(registry), 'path': str(path),
                         'path_exists': path.exists(), 'declared_files': checked,
                         'scope': 'registered files only; no claim of whole-study or paired completeness'})
    failures = [r['id'] for r in rows if not r['path_exists'] or any(
        not f['exists'] or f.get('size_match') is False or f.get('hash_check') == 'MISMATCH'
        for f in r['declared_files'])]
    result = {'entries': rows, 'failed_entries': failures,
              'hash_limit_mb': args.hash_limit_mb, 'all_registered_paths_present': all(r['path_exists'] for r in rows)}
    legacy_path = args.root / 'registry' / 'legacy_runtime_registry.jsonl'
    legacy_rows = []
    if legacy_path.exists():
        for line in legacy_path.read_text(encoding='utf-8').splitlines():
            entry = json.loads(line)
            missing, caches = [], []
            for item in entry.get('file_hashes', []):
                relative = item['path']
                target = args.root / relative if relative.startswith('external/') else args.root / 'raw' / relative
                if not target.is_file():
                    (caches if relative.endswith('.pyc') else missing).append(relative)
            legacy_rows.append({'id': entry['study_id'], 'declared_files': len(entry.get('file_hashes', [])),
                                'missing_data_files': missing, 'missing_python_cache_files': caches,
                                'verification': 'existence_only; migrated external paths resolved under data/external'})
    result['legacy_entries'] = legacy_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'entries': len(rows), 'failed_entries': failures,
                      'declared_files': sum(len(r['declared_files']) for r in rows)}))


if __name__ == '__main__':
    main()
