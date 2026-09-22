"""Run the reproducible original mCRC source reconstruction audit."""
import argparse
import json
from pathlib import Path
from drug_screen.transfer.source_audit import audit_mcrc_source

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('data'))
parser.add_argument('--output',type=Path,default=Path('data/artifacts/EXP-014/mcrc_source_audit.json'))
if __name__=='__main__':
    args=parser.parse_args()
    report=audit_mcrc_source(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='drug_audit'},indent=2,ensure_ascii=False))
