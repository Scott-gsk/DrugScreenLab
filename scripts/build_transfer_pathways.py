"""Build separate human pathway mappings using explicit source gene IDs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from drug_screen.transfer.cohorts import sha256
from drug_screen.transfer.pathways import PathwayResource, parse_gmt, score_pathways


def build(root: Path, output: Path):
    output.mkdir(parents=True,exist_ok=True)
    source=root/'raw/mcrc_pdo_2026/v3/Data S2.xlsx'
    annotation=pd.read_excel(source,usecols=['entrezgene_id','hgnc_symbol']).dropna()
    mapping=defaultdict(set)
    for row in annotation.itertuples():
        mapping[str(int(row.entrezgene_id))].add(str(row.hgnc_symbol).upper())
    ambiguous={key:sorted(value) for key,value in mapping.items() if len(value)!=1}
    one={key:next(iter(value)) for key,value in mapping.items() if len(value)==1}
    raw=root/'raw/transfer_public_assets'
    resources=[]
    kegg=defaultdict(set)
    for line in (raw/'kegg/kegg_hsa_gene_pathway.tsv').read_text().splitlines():
        gene,pathway=line.split('\t')
        gene=gene.removeprefix('hsa:')
        kegg[pathway].add(one.get(gene, 'UNMAPPED_ENTREZ:'+gene))
    resources.append((PathwayResource('KEGG','download_'+sha256(raw/'kegg/kegg_hsa_gene_pathway.tsv')[:12],
                      {k:tuple(sorted(v)) for k,v in sorted(kegg.items())}),raw/'kegg/kegg_hsa_gene_pathway.tsv'))
    reactome=defaultdict(set)
    with (raw/'reactome/NCBI2Reactome_All_Levels.txt').open() as handle:
        for line in handle:
            fields=line.rstrip('\n').split('\t')
            if len(fields)>=6 and fields[-1]=='Homo sapiens':
                reactome[fields[1]].add(one.get(fields[0], 'UNMAPPED_ENTREZ:'+fields[0]))
    resources.append((PathwayResource('Reactome','download_'+sha256(raw/'reactome/NCBI2Reactome_All_Levels.txt')[:12],
                      {k:tuple(sorted(v)) for k,v in sorted(reactome.items())}),raw/'reactome/NCBI2Reactome_All_Levels.txt'))
    hallmark=root/'raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt'
    resources.append((parse_gmt(hallmark,resource_id='Hallmark',version='2023.2.Hs'),hallmark))
    report=[]
    for resource,path in resources:
        value={'resource_id':resource.resource_id,'version':resource.version,'pathways':resource.pathways,
               'source_sha256':sha256(path),'annotation_sha256':sha256(source),
               'annotation_scope':'mCRC author Entrez/HGNC table; not a complete genome crosswalk',
               'unmapped_policy':'retain UNMAPPED_ENTREZ tokens in coverage denominator; never impute them as observed genes',
               'mapping_ambiguous_entrez':ambiguous,'usage':'separate resource; annotation mapping only, no response-based selection'}
        target=output/(resource.resource_id.lower()+'.json')
        serialized=json.dumps(value,indent=2,sort_keys=True)+'\n'
        if target.exists() and target.read_text()!=serialized:raise ValueError(f'refusing changed pathway resource {target}')
        target.write_text(serialized)
        all_genes=set().union(*map(set,resource.pathways.values()))
        report.append({'resource':resource.resource_id,'n_pathways':len(resource.pathways),'n_genes':len(all_genes),
                       'unmapped_entrez_tokens':sum(g.startswith('UNMAPPED_ENTREZ:') for g in all_genes),
                       'path':str(target),'sha256':sha256(target),'ambiguous_entrez_excluded':len(ambiguous)})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('data/processed/transfer/pathways_v2'))
    args=parser.parse_args();report=build(args.root,args.output)
    dest=args.root/'artifacts/EXP-014/pathway_build.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
