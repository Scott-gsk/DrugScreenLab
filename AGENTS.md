# DrugScreenLab Agent Map

## Mission and state
Develop, evaluate, and iteratively improve AI models for drug screening through reproducible experiments. The current state is recorded in `PROJECT_STATE.yaml`.

## Repository map
- `data/`: registered datasets and versioned splits; raw data is immutable.
- `experiments/`: EXP-ID registry and records.
- `artifacts/`: experiment outputs, always bound to an EXP-ID.
- `src/`: small reusable package code.
- `docs/`: detailed policy, design, and research notes.

## Research policy
Human approval is required before implementation. Research Manager defines the question and EXP-ID; Engineer implements approved work; Reviewer independently returns `VALID`, `INVALID`, or `INCONCLUSIVE`.

## Forbidden operations
Do not modify raw data, migrate old model/prediction/report/log artifacts, use model names as dataset identities, or run destructive Git cleanup without explicit confirmation. Do not add dependencies on MCPIRE_PDO or TriPerturb.

## Validation
Run `python -m pytest` and `python -m drug_screen.data.registry --root data`. Keep heavy data and artifacts out of Git.
