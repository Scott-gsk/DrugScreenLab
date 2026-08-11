# EXP-002 DATA CONTRACT

## Frozen Inputs And Target

- Source: `data/raw/lincs/GSE92742/` metadata and the Level-3 GCTX file under `data/interim/`.
- Gene universe: only `gene_info.pr_is_lm=1`, exactly 978 genes. Inferred non-landmark columns
  are forbidden in `Delta978`.
- A target row is a `trt_cp` Level-3 instance minus a `ctl_vehicle` instance matched on
  `rna_plate`, `cell_id`, `pert_time`, and `pert_time_unit`.
- `canonical_compound_id`: `inchi_key` when present, otherwise `LINCS:<pert_id>`; the fallback
  is explicit rather than discarding source records.
- `canonical_context_id`: `base_cell_id` when present, otherwise `cell_id`.
- `replicate_family_id`: canonical compound, canonical context, dose/unit, and time/unit.

## Strict Leakage Boundary

No raw vehicle instance, treatment-control family, or `rna_plate` component may occur across
splits. This is stricter than XPert's published same-plate random DMSO pairing. The XPert
primary source establishes its preprocessing choice; it does not authorize relaxing this
experiment's frozen no-cross-split contract.

Cold-drug components union canonical compound, raw vehicle instance, and plate. Cold-context
components union canonical context, raw vehicle instance, and plate. A split is feasible only
when each required cold protocol has non-degenerate train/validation/test components.

## PRIMARY_SOURCE_VERIFICATION

[GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) defines Level-3
as normalized direct landmark plus inferred genes. [XPert Methods](https://www.nature.com/articles/s42256-025-01165-w)
states that treatment/control Level-3 profiles are matched to a randomly selected same-plate
DMSO. The latter is source evidence for pairing, not for cross-split raw-control reuse.

## Materialization Gate

Materialize only after the strict feasibility audit reports `DATA_READY`. On `DATA_BLOCKED`,
write only small audit evidence and do not create a training dataset or split manifest.

If the gate is later satisfied under a newly approved split policy, each derived manifest row
must contain `treatment_instance_id`, `control_instance_id`, `replicate_family_id`,
`canonical_compound_id`, `canonical_context_id`, `rna_plate`, `split`, and `delta_row`; the
associated target array must have shape `(n_pairs, 978)` and `float32` values. This is an
output schema, not an assertion that such an array exists in this blocked run.
