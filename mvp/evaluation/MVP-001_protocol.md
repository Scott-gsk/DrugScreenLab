# MVP-001 Evaluation Protocol

## Scope and freeze

This document is the retained legacy `GSE117548`/Betge branch of the approved
`MVP-001` design.  Its execution status is `DEFERRED_PDO_LEG`: it was not used
in the Cell-Line Core result and no final PDO label comparison was run.  The
protocol remains a frozen reference for a later, separately approved milestone.
The Cell-Line Core MVP loop is documented under `mvp/core_eval/`.

This protocol applies only to the approved `MVP-001` Tiny/Small/MVP integration
loop.  It is an end-to-end feasibility check, not a Formal EXP: no bootstrap CI,
multi-seed claim, hyperparameter search, Reviewer verdict, biological claim, or
clinical efficacy claim is permitted.  `GSE117548` activity values are external
read-only evidence.  They must not be used to select features, mappings, model
settings, signatures, cohorts, thresholds, or a preferred result.

The Data Steward's MVP-001 data contract remains the authority for file identity,
GPL570 annotation, exact-978 ordering, LINCS compound identity, and `auroc`
direction semantics.  This protocol becomes executable only once those contract
fields are available.

## Units and GSE117548 line aggregation

The raw GSE117548 series matrix contains 25 untreated CRC PDO samples labelled
by 16 sample IDs (`D004` through `D046`).  The **external biological unit** is
the PDO line/sample ID, never a microarray sample, factor view, drug-row, gene,
or technical replicate.  Multiple samples from the same line are repeated
profiles, not independent external observations.

For each eligible line `i` and mapped landmark gene `g`:

1. Map GPL570 probes to the contract's unambiguous gene identifier; if several
   retained probes map to one gene in a sample, take their median.
2. Aggregate all 25 sample-level values within each line coordinate-wise by
   median: `x[i,g] = median_s(x[s,g])`.  A one-sample line is retained and
   explicitly flagged as `n_samples=1`; it does not receive an invented
   replicate variance.
3. Construct its leave-one-line-out state-deviation proxy
   `state[i,g] = x[i,g] - median_{k != i}(x[k,g])`.

The reference uses one already-aggregated vector per *other line*, so a line
with two array samples cannot receive twice the weight of a one-sample line.
This operational cross-PDO contrast is neither a tumour-versus-normal contrast
nor evidence that a drug reverses disease biology.

## Eligibility and mapping gates

Before loading external `auroc` values for scoring, write and freeze a mapping
manifest containing source ID, GSE117548 line ID, source sample IDs, sample
count, retained exact-978 IDs in contract order, and candidate drug identities.
All of these conditions are required.

| Gate | Requirement | Failure action |
| --- | --- | --- |
| PDO identity | Every retained expression line has one canonical `sample_id`; an activity `id` maps one-to-one to that line, and its `line` suffix is recorded. | Exclude only the unmappable line with a reason; zero eligible lines is `BROKEN`. |
| Expression state | Untreated CRC PDO sample status, platform annotation revision, transformation state, and no missing retained landmark values are attested by the data contract. | `BROKEN`. |
| Landmark overlap | At least 700 unique, unambiguous exact-978 genes survive in every eligible line; ordering equals the exact-978 contract. | `BROKEN`; do not fill, infer, or use non-landmark genes. |
| Drug identity | Each candidate has a deterministic canonical compound ID linking activity `drug_name_normalized` to LINCS metadata.  Aliases require a recorded primary mapping/structure basis. | Drop candidate; fewer than 3 retained drugs is `BROKEN`. |
| Activity endpoint | `auroc` is finite and the contract states whether larger or smaller values mean greater desired activity. | `BROKEN`; direction must not be guessed from the number. |
| Factor duplicate | Activity may contain multiple factor views.  Collapse only after asserting all rows with a common `(id, drug_name_normalized)` have one identical `auroc`; retain one row and record all factors. | Mismatch is `BROKEN`, not an opportunity to select a factor. |

The frozen external cohort is the intersection of expression-eligible lines and
activity-eligible `(line, drug)` pairs after these gates.  It cannot be changed
after an activity label has been read.  All dropped lines, drugs, and reasons
must be reported.

## Held-out perturbation diagnostic

This is an internal model diagnostic, not OOD generalization.  From the frozen
GSE92742 MVP subset, assign each chemical treatment group a deterministic
within-drug holdout using the run-config hash of canonical treatment-group ID;
20% per canonical drug (with at least one train group and, where possible, one
test group) is test.  A treatment group, its technical replicates, and its
paired-control construction remain in one side.  Split assignment is derived
without GSE117548 expression or activity values.

The diagnostic unit is one held-out treatment group after the contract-defined
within-group replicate aggregation, not 978 genes.  For each group, report
Pearson, Spearman, MAE, and RMSE over its exact-978 vector.  The primary
diagnostic summary is the macro mean of group MAE and Spearman, first within
canonical drug and then equally across drugs; include every drug's count and
metric so a pooled gene result cannot conceal a failed drug.

Two predeclared train-only baselines use exactly the same test groups and gene
ordering:

* **Constant baseline:** the coordinate-wise mean training `Delta978` over all
  training treatment groups.
* **Drug-mean baseline:** the coordinate-wise mean training `Delta978` only for
  the held-out group's canonical drug.  A drug without a train mean is not
  silently replaced; record the group as baseline-ineligible and apply the same
  eligibility accounting to all methods.

The learned model's primary diagnostic is positive only when it has strictly
lower macro MAE **and** strictly higher macro Spearman than both baselines on
their common eligible test groups.  Equal or mixed results are `NO_SIGNAL`,
not a tuned rerun.  Report undefined correlations rather than coercing them to
zero.  There is one MVP seed/configuration; selecting the best seed is forbidden.

## Signature and external direction score

Before activity-label access, the run config must freeze one supported LINCS
condition set per mapped candidate drug using training metadata only.  Generate
the model prediction for every frozen supported `(cell, dose, time)` condition
in that set, then define the drug signature `delta[d,g]` as its coordinate-wise
median.  No activity label can choose a condition or aggregation.

For every eligible `(line i, drug d)`, calculate the primary model score
`reversal[i,d] = -Spearman(state[i,*], delta[d,*])` over the same retained,
ordered genes.  Higher values mean a stronger *operational* anti-correlation;
they do not mean greater biological efficacy.  Tied values use average ranks;
an undefined correlation is retained as an invalid score and reported.

Within each line, rank all its eligible drugs by descending `reversal` and by
the contract-declared desirable direction of `auroc`.  Require at least three
drugs with finite scores per line.  Report for every eligible line:

* Spearman correlation of the two within-line ranks (`rho_i`);
* pairwise direction agreement: concordant non-tied drug pairs divided by all
  non-tied pairs (`agreement_i`);
* candidate count, sample count, retained-gene count, and any undefined/tied
  scores.

The primary external summaries are the **median across lines** of `rho_i` and
of `agreement_i`, plus the fraction of lines with `rho_i > 0` and
`agreement_i > 0.5`.  A pooled line-drug correlation is descriptive only and
may never override a per-line failure pattern.  There is no Top-K enrichment
claim with five candidates; emit each line's ordered candidate list solely as a
traceable ranking digest.

External directional signal is present only if at least three lines are eligible
and both median summaries are positive (`median rho > 0`, `median agreement >
0.5`) **and** a strict majority of eligible lines meets the same respective
direction.  This fixed rule deliberately prevents a small number of lines from
masking context-specific failure.

## Decision mapping and required compact output

| Decision | Required evidence |
| --- | --- |
| `PROMISING` | All gates pass; learned held-out diagnostic is positive versus both baselines; at least three mapped drugs and three eligible lines; external directional-signal rule passes. |
| `NO_SIGNAL` | All required inputs and computations complete, but either the held-out rule or external directional-signal rule fails.  This is not permission to alter mapping, select a factor, tune on test labels, or rerun seeds until favourable. |
| `BROKEN` | Any required schema/identity/endpoint-direction/landmark-overlap/artifact gate fails, fewer than three candidates or three eligible lines remain, or required scores cannot be calculated. |

The single MVP summary must include data and manifest digests, GSE117548
line-to-sample counts, the activity factor-collapse assertion, split/config
digest, baseline eligibility and metrics by drug, signature condition-set digest,
per-line ranking metrics, all exclusions, and the decision.  No confidence
interval, bootstrap, p-value, or biological-performance language is to be
reported for this MVP loop.
