"""Inspect official supplementary schema without extracting raw assets."""
from pathlib import Path
import json
import zipfile
import pandas as pd


def main():
    root = Path('data/raw/transfer_public_assets/organoid_biobank')
    report = {}
    with zipfile.ZipFile(root/'41586_2026_10830_MOESM3_ESM.zip') as archive:
        for name in ['supplementary_table_5_revision.csv','supplementary_table_6_revision.csv']:
            with archive.open(name) as stream:
                frame = pd.read_csv(stream, sep=r'\s+', nrows=4)
            report[name] = {'columns': frame.columns.tolist(), 'preview': frame.to_dict('records')}
        for name in ['supplementary_table_1_revision.xlsx','supplementary_table_2_revision.xlsx']:
            with archive.open(name) as stream:
                sheets = pd.read_excel(stream, sheet_name=None, nrows=3)
            report[name] = {key: {'columns': frame.columns.tolist(), 'preview':frame.to_dict('records')}
                            for key, frame in sheets.items()}
    destination = Path('data/artifacts/EXP-015/nature_dependency_schema.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
