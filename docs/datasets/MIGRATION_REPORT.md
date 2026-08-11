# Dataset Migration Report

Migration source: `D:/Code/Drug_model-MCPIRE_PDO/MCPIRE_PDO/runtime/`.

Migrated assets are registered in `data/registry/datasets.json` and audited in `data/registry/migration_manifest.json`. File counts and aggregate byte totals were checked after the move. The raw collection carried source `SHA256SUMS.local` files; the external collection has a SHA-512 content digest recorded in the registry.

Excluded by policy: `runtime/models/`, `runtime/predictions/`, `runtime/reports/`, `runtime/logs/`, `runtime/splits/triperturb_v2/`, and `runtime/processed/triperturb_v2/`. These are model, experiment, cache, or obsolete implementation artifacts rather than reusable dataset assets.

The old runtime registry is preserved as `data/registry/legacy_runtime_registry.jsonl` for provenance reference only; it does not define the new dataset identities.
