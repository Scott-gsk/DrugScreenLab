import hashlib
import json
from pathlib import Path
import runpy
import sys


def test_inventory_resolves_file_and_directory_checksums(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    (root / 'registry').mkdir(parents=True)
    (root / 'raw').mkdir()
    payload = root / 'raw' / 'data.txt'
    payload.write_bytes(b'example')
    digest = hashlib.sha256(b'example').hexdigest()
    rows = [
        {'id': 'file', 'path': {'relative': 'raw/data.txt'},
         'checksum': {'algorithm': 'sha256', 'value': digest, 'file': 'data.txt'}},
        {'id': 'directory', 'path': {'relative': 'raw'},
         'checksum': {'files': {'data.txt': digest}}},
        {'id': 'stale', 'path': {'relative': 'raw/data.txt'},
         'checksum': {'algorithm': 'sha256', 'value': '0' * 64}},
    ]
    (root / 'registry' / 'datasets.json').write_text(json.dumps(rows))
    output = tmp_path / 'audit.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--root', str(root), '--output', str(output)])
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'audit_dataset_file_inventory.py'
    runpy.run_path(str(script), run_name='__main__')
    report = json.loads(output.read_text())
    assert report['failed_entries'] == ['stale']
    assert all(row['declared_files'][0]['exists'] for row in report['entries'])
    assert report['entries'][0]['declared_files'][0]['hash_check'] == 'match'
    assert report['entries'][1]['declared_files'][0]['hash_check'] == 'match'
