# Data

`raw/` is immutable source data. `external/` contains third-party processed assets. `interim/` contains regenerable intermediate data. `processed/` contains experiment-ready datasets. `splits/` contains versioned, metadata-backed dataset splits.

Large assets are intentionally ignored by Git. Every dataset used by an experiment must have a JSON entry under `registry/` with identity, version, source, relative path, checksum, and provenance.
