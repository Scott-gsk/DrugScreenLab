"""Validate acquisition manifests and identify redundant copies by content hash."""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('manifests', nargs='+', type=Path)
    args = parser.parse_args()
    checks = []
    for manifest_path in args.manifests:
        value = json.loads(manifest_path.read_text(encoding='utf-8'))
        files = value if isinstance(value, list) else value.get('files', [])
        for item in files:
            path = args.data_root / item['relative_path']
            digest = hashlib.sha256()
            if path.is_file():
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
            checks.append({
                'manifest': str(manifest_path), 'dataset_id': item['dataset_id'],
                'path': str(path), 'exists': path.is_file(),
                'size_match': path.is_file() and path.stat().st_size == item['bytes'],
                'hash_match': path.is_file() and digest.hexdigest() == item['sha256'].lower(),
            })
    by_hash = {}
    for check in checks:
        if check['hash_match']:
            by_hash.setdefault(Path(check['path']).resolve(), set())
    # Manifest aliases to one canonical file are allowed; only distinct physical paths are redundant.
    content_paths = {}
    for check in checks:
        if check['hash_match']:
            path = Path(check['path'])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            content_paths.setdefault(digest, set()).add(str(path.resolve()))
    duplicates = {digest: sorted(paths) for digest, paths in content_paths.items() if len(paths) > 1}
    failures = [check for check in checks if not all(check[key] for key in ('exists', 'size_match', 'hash_match'))]
    report = {'files_checked': len(checks), 'failures': failures,
              'duplicate_content_on_distinct_paths': duplicates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'files_checked': len(checks), 'failures': len(failures),
                      'distinct_path_duplicate_groups': len(duplicates)}))
    return 1 if failures or duplicates else 0


if __name__ == '__main__':
    raise SystemExit(main())
