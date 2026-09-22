"""Audit existing registered cohorts without modifying source assets."""
import argparse
import json
from pathlib import Path

from drug_screen.transfer.cohorts import audit_bundle, load_registered_cohort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('data/artifacts/EXP-014/cohort_audit.json'))
    args = parser.parse_args()
    entries = json.loads((args.root / 'registry/transfer_cohorts.json').read_text())['cohorts']
    result = []
    for entry in entries:
        try:
            result.append(audit_bundle(load_registered_cohort(entry['cohort_id'], root=args.root)))
        except Exception as exc:
            result.append({'cohort_id': entry['cohort_id'], 'structural_validation': 'FAIL', 'reason': str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(row['structural_validation'] != 'PASS' for row in result))


if __name__ == '__main__':
    raise SystemExit(main())
