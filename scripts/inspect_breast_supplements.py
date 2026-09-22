"""Inspect public breast PDO supplementary Excel workbooks without altering them."""
from pathlib import Path
import json
import pandas as pd


def main():
    for study, folder in (
        ("xu_2026", Path("data/raw/breast_pdo_xu_2026/v1")),
        ("chen_2021", Path("data/raw/breast_pdo_chen_2021/v1")),
    ):
        print(study)
        for path in sorted(folder.glob("*.xlsx")):
            book = pd.ExcelFile(path)
            details = []
            for sheet in book.sheet_names:
                frame = pd.read_excel(path, sheet_name=sheet)
                details.append({
                    "sheet": sheet,
                    "shape": list(frame.shape),
                    "columns": [str(c) for c in frame.columns[:12]],
                    "first_rows": frame.head(2).astype(str).iloc[:, :12].to_dict("records"),
                })
            print(json.dumps({"file": path.name, "sheets": details}, ensure_ascii=False))
    clinical = pd.read_excel("data/raw/breast_pdo_chen_2021/v1/ADVS-8-2101176-s001.xlsx", header=1)
    drugs = pd.read_excel("data/raw/breast_pdo_chen_2021/v1/ADVS-8-2101176-s003.xlsx", header=None)
    print("chen_identity_columns", list(clinical.columns[:12]))
    print("chen_identity_flags", clinical.iloc[:, 6:10].astype(str).apply(lambda col: col.value_counts().head(5).to_dict()).to_dict())
    print("chen_drug_rows", drugs.iloc[2:55, 0].dropna().astype(str).tolist())
    print("chen_pdo_columns", drugs.iloc[1, 3:].dropna().astype(str).tolist())


if __name__ == "__main__":
    main()
