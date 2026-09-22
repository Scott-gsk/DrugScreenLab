"""Pretrain the admitted EXP-014 cell-line source model."""
import argparse
import json
from pathlib import Path
from drug_screen.transfer.source_training import train_source


def main():
    parser = argparse.ArgumentParser()
    for key in ('config', 'registry', 'admission', 'output', 'root'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train_source(json.loads(args.config.read_text(encoding='utf-8')),
          root=args.root, registry=args.registry, admission=args.admission, output=args.output), indent=2))


if __name__ == '__main__':
    main()
