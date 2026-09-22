# EXP-014 pair-model implementation review

`RidgePairModel` and `DualEncoderPairModel` are small model primitives for
already-preprocessed sample features and Morgan drug fingerprints.  They do
not load cohorts, perform splits, or claim biological mechanism.

## Interfaces and training modes

Both models expose `fit`, `predict`, `explain`, `save`, and `load`.
`RidgePairModel` is a ridge baseline with additive sample and drug blocks plus
a capped elementwise interaction block.  Its explanation is exact: reference,
sample, drug, and interaction contributions add to the prediction.

`DualEncoderPairModel` has explicit hyperparameters and seed.  A new instance
is M1 random initialization.  `load` restores a saved source model for M2
source initialization.  Calling `fit` on that loaded model is M3 fine-tuning.
Its default `reset_label_scaler=True` calibrates only on the supplied target
training labels and algebraically adjusts the final linear head so raw
predictions are unchanged at the start of tuning.  `False` retains the loaded
scale and is only suitable when the caller has established that both endpoints
already share that scale; it must not be presented as endpoint conversion.
M5 is deliberately outside this implementation.

The neural objective is Huber regression plus an optional softplus ranking loss.
Ranking pairs are created only when two unequal labels share the same PDO ID;
ties and cross-PDO comparisons are excluded.  Ranking loss is averaged within
PDO before PDOs are averaged, and `max_rank_pairs` bounds pairs per PDO.  This
small CPU implementation is appropriate for synthetic tests and small cohorts,
not unconstrained source-scale training. Optional validation data is used
only to choose the early-stop checkpoint, never for fitting the scaler or model.

`explain` uses zero-input integrated gradients, returned in original-label
units.  Its reference plus sample and drug attributions reconstructs the
prediction to numerical integration tolerance.  These values are model
attributions, not mechanism evidence.

## Verification

`tests/transfer/test_models.py` covers synthetic interaction recovery, ranking
masking (including a different PDO), save/load prediction equality, M3
fine-tuning scaler isolation, and neural attribution completeness.  It must be
run through the required WSL2 `drugscreening-gpu` Conda environment.
