# MVP-001 validation log

Execution date: 2026-08-12 (WSL2, `drugscreening-gpu`). This is a compact
checkpoint record, not a substitute for the large local cache or checkpoint.

## Governance and registry

Command:

```text
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python -m pytest --capture=no
```

Result: `51 passed in 7.24s`.

Command:

```text
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python -m drug_screen.data.registry --root data
```

Result: `PASS`.

## Core artifact checks

| command | result |
| --- | --- |
| `python mvp/core_data/build_crc_signature_gse74602.py` | PASS; GSE74602 source/platform checksums, GCTX exact-978 order, 947 rows, formal direction gate |
| `python mvp/core_data/build_prism_compact.py` | PASS; 135 finite rows, 35 colorectal `depmap_id` lines, four frozen candidates |
| `python mvp/core_eval/compute_observed_oracle.py` | PASS; 695 matched instances, 218 groups, four finite rankings |
| `python mvp/core_eval/compute_predicted_reversal.py` | PASS; 144 held-out rows, 42 groups, same four-candidate cohort |
| `python mvp/core_eval/evaluate_prism.py` | PASS; 33 eligible lines, `PROMISING`, `CORE_MVP_FEASIBILITY_PROMISING` |

The observed ranking SHA256 is
`3b283ee85d4b86cd1181974a0a3089794d7f1170929e92a88147172792f3f0e8`; the
predicted ranking SHA256 is
`e9d05c4af643ac2b7b25c6f1d28b88c57449ac43a49399d17ec79ce5c4311870`; the
compact PRISM response SHA256 is
`5114e789299a794792bc44c818af98e679e8f00d6f6f3fb246781e6df2e960f1`.

## Runtime and asset policy

Tiny and Small were completed before the bounded core evaluation. Both retain
complete treatment groups, validate cache/config/manifest binding, write a
checkpoint and summary, and do not scan the full 65 GB Level-3 source matrix.
Raw source downloads, the large exact-978 cache, full PRISM matrix, temporary
logs, and model checkpoint bodies remain local/ignored. Their IDs, relative
paths, checksums, generator revisions and commands are tracked in the audits,
`MVP-001_core_eval_evidence.json`, and `MVP-001_model_provenance.json`.

The final MVP label is a feasibility label only. No formal scientific claim,
new EXP-ID, EXP-003 execution, or automatic next experiment was initiated.
