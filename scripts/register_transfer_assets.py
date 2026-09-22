"""Freeze acquired assets in a separate registry without admitting them to training."""
import argparse
import json
from pathlib import Path
from drug_screen.data.public_assets import load_manifest, _validate_final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data'))
    parser.add_argument('--manifest',type=Path,default=Path('configs/data/transfer_public_assets.json'))
    args=parser.parse_args()
    manifest=load_manifest(args.manifest)
    target=args.root/'registry/datasets_transfer.json'
    existing=json.loads(target.read_text()) if target.exists() else []
    entries={entry['id']:entry for entry in existing}
    for asset in manifest['assets']:
        if asset.get('status') != 'ready': continue
        path=args.root/asset['relative_path']
        if not path.exists(): continue
        size,digest=_validate_final(path,asset)
        key='transfer_'+asset['key']
        value={'id':key,'version':1,'asset_type':'raw','path':{'relative':asset['relative_path']},
               'immutable':True,'source':{'url':asset['url'],'source_page':asset.get('source_page')},
               'checksum':{'algorithm':'sha256','value':digest},'bytes':size,
               'intended_role':'SOURCE_AUDIT_ONLY','scientific_admission':'PENDING'}
        if key in entries and entries[key]['checksum']['value'] != digest:
            raise ValueError(f'registered bytes changed: {key}')
        entries[key]=value
    target.write_text(json.dumps(list(entries.values()),indent=2,ensure_ascii=False)+'\n')
    print(f'registered {len(entries)} verified files; none automatically admitted for training')


if __name__=='__main__':main()
