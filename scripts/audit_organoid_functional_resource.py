from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import pyreadr
from drug_screen.transfer.cohorts import sha256
root=Path('data');base=root/'raw/transfer_public_assets/organoid_biobank'
x=pd.read_csv(base/'data_RNAseq_ALL-organoids.csv',index_col=0)
c=pd.read_csv(base/'data_covariates_ALL-organoids.csv',index_col=0)
d=next(iter(pyreadr.read_r(str(base/'07_EssMatrix_qnorm_corrected_logFCs.RData')).values()))
shared=sorted(set(x.index)&set(d.columns));ids=sorted(set(x.index)&set(c.index))
report={'resource':'figshare.28339340.v1','expression_shape':list(x.shape),'dependency_shape_gene_by_model':list(d.shape),
 'expression_dependency_direct_id_overlap':len(shared),'expression_covariates_overlap':len(ids),
 'dependency_id_prefix_counts':pd.Series([re.match(r'[A-Za-z]+', str(v)).group() if re.match(r'[A-Za-z]+', str(v)) else 'OTHER' for v in d.columns]).value_counts().to_dict(),
 'expression_id_examples':x.index[:4].tolist(),'dependency_id_examples':d.columns[:4].tolist(),
 'expression_gene_suffix':'_exp','expression_unit':'AUTHOR_LOG2_TPM_OFFSET_UNVERIFIED_DO_NOT_RELOG',
 'expression_unit_evidence':'https://github.com/Garnett-Lab/SangerOrganoidBiobank/blob/main/scripts/CRISPR_differential_dependency_analysis.R (RNAseq gather variable log2TPM)',
 'organoid_dependency_required_source':'supplementary_table_6.csv (LFC); supplementary_table_5.csv (binary BAGEL2 calls)',
 'dependency_role':'NOT_ADMITTED_AS_ORGANOID_DEPENDENCY: ACH/SIDM identifiers and zero direct organoid expression overlap; comparator identity requires confirmation',
 'dependency_algorithm':'qnorm_corrected_logFCs_file_label_requires_paper_protocol_verification',
 'expression_nonfinite':int((~np.isfinite(x.to_numpy(float))).sum()),
 'dependency_nonfinite':int((~np.isfinite(d.to_numpy(float))).sum()),
 'tissue_counts':c.tissue.value_counts().to_dict(),'patient_identity':'NOT_YET_RESOLVED',
 'admission':'PENDING_IDENTITY_AND_UNITS_AUDIT',
 'source_sha256':{f.name:sha256(f) for f in [base/'data_RNAseq_ALL-organoids.csv',base/'data_covariates_ALL-organoids.csv',base/'07_EssMatrix_qnorm_corrected_logFCs.RData']}}
(root/'artifacts/EXP-014/biobank_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
