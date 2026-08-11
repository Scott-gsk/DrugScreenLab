# Architecture

DrugScreenLab separates dataset identity/version from model and experiment code. Data is resolved through `DRUGSCREEN_DATA_ROOT` (default: repository `data/`). Experiments use immutable EXP-IDs, and artifacts live under the matching EXP-ID. Accepted releases are created only after independent review and final validation.

No model, training, or evaluation framework is part of the bootstrap; those components are introduced by approved experiments.
