"""Measured perturbation + matched-control loader skeleton (DRAFT).

Route-2 phase-1 scaffolding only. Does **not** train models and does **not**
load Level-3/5 expression matrices into memory by default.

Responsibilities:
- Resolve registry / on-disk paths for LINCS GSE92742 and ChemPert.
- Read lightweight metadata (inst_info, ChemPert response CSV).
- Propose matched-control pairs for Level-3 delta supervision candidates.
- Expose leakage-group keys for ChemPert∩LINCS (duplicate-ID style).

Hard rules (研究实施方案 ch.6):
- chemical vs genetic kept separate;
- level3_delta vs level5_signature never mixed silently;
- no reliable control ⇒ not a delta-supervision label;
- ChemPert∩LINCS duplicates must not cross train/test (groups only here).

Raw under data/raw is read-only. No GPU / no Ridge / no XPert calls.
"""
from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from drug_screen.data.expression_contracts import PerturbationGeneRecord
from drug_screen.data.p0 import PerturbationRecord as P0PerturbationRecord
from drug_screen.data.p0 import assert_matched_controls
from drug_screen.data.root import data_root, repository_root

# --- relative paths (project root) -------------------------------------------------

LINCS_RAW_REL = Path("data/raw/lincs/GSE92742")
LINCS_INTERIM_REL = Path("data/interim/lincs/GSE92742")
LINCS_PROCESSED_REL = Path("data/processed/lincs/GSE92742")
LINCS_EXACT978_REL = Path("data/processed/lincs/GSE92742/exact978_cache_v1")
CHEMPERT_RAW_REL = Path("data/raw/resource_acquisition_20260916/improve_chempert")
REGISTRY_REL = Path("data/registry/datasets.json")

LINCS_INST_INFO = "GSE92742_Broad_LINCS_inst_info.txt.gz"
LINCS_PERT_INFO = "GSE92742_Broad_LINCS_pert_info.txt.gz"
LINCS_CELL_INFO = "GSE92742_Broad_LINCS_cell_info.txt.gz"
CHEMPERT_RESP_CSV = "chempert_information_for_transcriptional_responses.csv"
CHEMPERT_PERT_CSV = "chempert_information_for_perturbagens.csv"

# LINCS pert_type values treated as controls for Level-3 matching.
_LINCS_CONTROL_TYPES = frozenset({"ctl_vehicle", "ctl_untrt", "ctl_vector"})
_LINCS_CHEM_TREAT = frozenset({"trt_cp"})
_LINCS_GENETIC_TREAT = frozenset({"trt_sh", "trt_oe", "trt_xpr", "trt_sh.cgs"})


@dataclass(frozen=True)
class MeasuredPerturbationPaths:
    root: Path
    lincs_raw: Path
    lincs_interim: Path
    lincs_processed: Path
    lincs_exact978: Path
    chempert_raw: Path
    registry: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "MeasuredPerturbationPaths":
        r = Path(root) if root is not None else repository_root()
        return cls(
            root=r,
            lincs_raw=r / LINCS_RAW_REL,
            lincs_interim=r / LINCS_INTERIM_REL,
            lincs_processed=r / LINCS_PROCESSED_REL,
            lincs_exact978=r / LINCS_EXACT978_REL,
            chempert_raw=r / CHEMPERT_RAW_REL,
            registry=r / REGISTRY_REL,
        )


@dataclass(frozen=True)
class AssetStatus:
    name: str
    path: Path
    present: bool
    detail: str = ""


@dataclass
class ControlPairCandidate:
    """One treatment instance with a proposed plate-matched control."""

    treatment_inst_id: str
    control_inst_id: str
    cell_id: str
    rna_plate: str
    pert_id: str
    pert_iname: str
    pert_type: str
    pert_time: str
    perturbation_class: str  # chemical | genetic
    signature_level: str = "level3_delta"
    source: str = "LINCS_GSE92742"


@dataclass
class MeasuredAssetReport:
    paths: MeasuredPerturbationPaths
    assets: list[AssetStatus] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "root": str(self.paths.root),
            "assets": [
                {
                    "name": a.name,
                    "path": str(a.path),
                    "present": a.present,
                    "detail": a.detail,
                }
                for a in self.assets
            ],
            "notes": list(self.notes),
        }


def describe_assets(root: Path | str | None = None) -> MeasuredAssetReport:
    """Inventory on-disk assets needed for measured-control loading (no matrix I/O)."""
    paths = MeasuredPerturbationPaths.from_root(root)
    report = MeasuredAssetReport(paths=paths)
    checks = [
        ("lincs_inst_info", paths.lincs_raw / LINCS_INST_INFO),
        ("lincs_pert_info", paths.lincs_raw / LINCS_PERT_INFO),
        ("lincs_cell_info", paths.lincs_raw / LINCS_CELL_INFO),
        ("lincs_raw_dir", paths.lincs_raw),
        ("lincs_interim_dir", paths.lincs_interim),
        ("lincs_processed_dir", paths.lincs_processed),
        ("lincs_exact978_cache", paths.lincs_exact978),
        ("chempert_responses_csv", paths.chempert_raw / CHEMPERT_RESP_CSV),
        ("chempert_perturbagens_csv", paths.chempert_raw / CHEMPERT_PERT_CSV),
        ("chempert_raw_dir", paths.chempert_raw),
        ("registry_datasets_json", paths.registry),
    ]
    for name, path in checks:
        present = path.exists()
        detail = ""
        if present and path.is_file():
            detail = f"size_bytes={path.stat().st_size}"
        elif present and path.is_dir():
            detail = "directory"
        report.assets.append(AssetStatus(name=name, path=path, present=present, detail=detail))
    report.notes.append(
        "Skeleton never opens Level3/4/5 .gctx(.gz) by default; use metadata APIs only."
    )
    report.notes.append(
        "ChemPert expression RDS materialization is out of scope here; CSV metadata only."
    )
    return report


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("rt", encoding="utf-8", errors="replace", newline="")


def iter_lincs_inst_info(
    root: Path | str | None = None,
    *,
    limit: Optional[int] = None,
) -> Iterator[dict[str, str]]:
    """Yield LINCS instance metadata rows (vehicle/controls + treatments)."""
    paths = MeasuredPerturbationPaths.from_root(root)
    path = paths.lincs_raw / LINCS_INST_INFO
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing LINCS inst_info (raw read-only expected): {path}"
        )
    with _open_text(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        n = 0
        for row in reader:
            yield {k: (v or "") for k, v in row.items()}
            n += 1
            if limit is not None and n >= limit:
                return


def _class_for_pert_type(pert_type: str) -> Optional[str]:
    if pert_type in _LINCS_CHEM_TREAT:
        return "chemical"
    if pert_type in _LINCS_GENETIC_TREAT:
        return "genetic"
    return None


def propose_lincs_control_pairs(
    root: Path | str | None = None,
    *,
    perturbation_class: str = "chemical",
    limit_treatments: Optional[int] = 5000,
    max_scan_rows: Optional[int] = 200_000,
) -> list[ControlPairCandidate]:
    """Propose plate+cell+time matched controls for LINCS treatments (metadata only).

    Matching key: (rna_plate, cell_id, pert_time) → first control inst on that plate.
    Treatments without a same-plate control are skipped (not delta labels).
    """
    if perturbation_class not in {"chemical", "genetic"}:
        raise ValueError("perturbation_class must be 'chemical' or 'genetic'")

    controls_by_key: dict[tuple[str, str, str], str] = {}
    pending: list[dict[str, str]] = []
    scanned = 0
    for row in iter_lincs_inst_info(root, limit=max_scan_rows):
        scanned += 1
        pert_type = row.get("pert_type", "")
        key = (row.get("rna_plate", ""), row.get("cell_id", ""), row.get("pert_time", ""))
        if pert_type in _LINCS_CONTROL_TYPES:
            # Prefer first control on the plate; do not overwrite.
            controls_by_key.setdefault(key, row["inst_id"])
            continue
        cls = _class_for_pert_type(pert_type)
        if cls == perturbation_class:
            pending.append(row)

    pairs: list[ControlPairCandidate] = []
    for row in pending:
        if limit_treatments is not None and len(pairs) >= limit_treatments:
            break
        key = (row.get("rna_plate", ""), row.get("cell_id", ""), row.get("pert_time", ""))
        ctl = controls_by_key.get(key)
        if not ctl:
            continue
        pairs.append(
            ControlPairCandidate(
                treatment_inst_id=row["inst_id"],
                control_inst_id=ctl,
                cell_id=row.get("cell_id", ""),
                rna_plate=row.get("rna_plate", ""),
                pert_id=row.get("pert_id", ""),
                pert_iname=row.get("pert_iname", ""),
                pert_type=row.get("pert_type", ""),
                pert_time=row.get("pert_time", ""),
                perturbation_class=perturbation_class,
            )
        )
    return pairs


def pairs_to_p0_records(pairs: Sequence[ControlPairCandidate]) -> list[P0PerturbationRecord]:
    """Convert control-pair candidates into p0.PerturbationRecord for assert_matched_controls."""
    records: list[P0PerturbationRecord] = []
    seen_ctl: set[str] = set()
    for p in pairs:
        if p.control_inst_id not in seen_ctl:
            records.append(
                P0PerturbationRecord(
                    sample_id=p.control_inst_id,
                    compound_id=None,
                    gene_id=None,
                    context_id=p.cell_id,
                    donor_id=None,
                    control_id=None,
                    replicate_family_id=p.rna_plate,
                    plate_id=p.rna_plate,
                    timepoint=str(p.pert_time),
                    is_control=True,
                )
            )
            seen_ctl.add(p.control_inst_id)
        records.append(
            P0PerturbationRecord(
                sample_id=p.treatment_inst_id,
                compound_id=p.pert_id if p.perturbation_class == "chemical" else None,
                gene_id=p.pert_id if p.perturbation_class == "genetic" else None,
                context_id=p.cell_id,
                donor_id=None,
                control_id=p.control_inst_id,
                replicate_family_id=p.pert_id,
                plate_id=p.rna_plate,
                timepoint=str(p.pert_time),
                is_control=False,
            )
        )
    assert_matched_controls(records)
    return records


def load_chempert_response_metadata(
    root: Path | str | None = None,
    *,
    limit: Optional[int] = None,
) -> list[dict[str, str]]:
    """Load ChemPert transcriptional-response information CSV (no RDS matrices)."""
    paths = MeasuredPerturbationPaths.from_root(root)
    path = paths.chempert_raw / CHEMPERT_RESP_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Missing ChemPert response metadata CSV: {path}")
    rows: list[dict[str, str]] = []
    with path.open("rt", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: (v or "") for k, v in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def chempert_lincs_overlap_keys(
    chempert_rows: Sequence[Mapping[str, str]],
) -> dict[str, list[str]]:
    """Build leakage-group hints: Data_Accession → Response_IDs (GSE92742 / GSE70138 / other).

    Formal freeze tables remain with DSL-数据管家; this only exposes duplicate-source keys.
    """
    groups: dict[str, list[str]] = {}
    for row in chempert_rows:
        acc = (row.get("Data_Accession") or "UNKNOWN").strip()
        rid = (row.get("Response_ID") or "").strip()
        if not rid:
            continue
        groups.setdefault(acc, []).append(rid)
    return groups


class MeasuredPerturbationControlLoader:
    """DRAFT loader implementing metadata / control-matching side of the contract.

    ``iter_records`` yields metadata-backed ``PerturbationGeneRecord`` shells with
    empty ``values`` — full Δexpression materialization is intentionally deferred
    (exact978 / gctx readers land in a later non-blocking PR after data contract freeze).
    """

    name = "measured_perturbation_controls"
    version = "draft-0.1-measured-control-skeleton"

    def __init__(self, root: Path | str | None = None) -> None:
        self.paths = MeasuredPerturbationPaths.from_root(root)

    def list_sources(self) -> Sequence[str]:
        return ("LINCS_GSE92742", "ChemPert")

    def describe(self) -> dict:
        return describe_assets(self.paths.root).as_dict()

    def iter_records(
        self,
        *,
        source: str,
        signature_level: str,
        compound_ids: Optional[Sequence[str]] = None,
        cell_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = 1000,
    ) -> Iterable[PerturbationGeneRecord]:
        if signature_level not in {"level3_delta", "level5_signature"}:
            raise ValueError(
                f"signature_level must be level3_delta or level5_signature, got {signature_level!r}"
            )
        want_compounds = set(compound_ids) if compound_ids else None
        want_cells = set(cell_ids) if cell_ids else None

        if source == "LINCS_GSE92742":
            if signature_level != "level3_delta":
                raise NotImplementedError(
                    "Skeleton ships Level-3 control pairing only; Level-5 signature "
                    "iteration waits for frozen contract (do not mix with L3 deltas)."
                )
            pairs = propose_lincs_control_pairs(
                self.paths.root,
                perturbation_class="chemical",
                limit_treatments=limit,
            )
            n = 0
            for p in pairs:
                if want_compounds is not None and p.pert_id not in want_compounds:
                    continue
                if want_cells is not None and p.cell_id not in want_cells:
                    continue
                yield PerturbationGeneRecord(
                    record_id=p.treatment_inst_id,
                    compound_id=p.pert_id,
                    cell_or_model_id=p.cell_id,
                    dose=None,
                    dose_unit=None,
                    time_hr=float(p.pert_time) if p.pert_time not in {"", None} else None,
                    signature_level="level3_delta",
                    control_id=p.control_inst_id,
                    gene_ids=(),
                    values=None,
                    gene_order_contract_id="lincs_landmark978_pending",
                    perturbation_class="chemical",
                    source="LINCS_GSE92742",
                    metadata={
                        "rna_plate": p.rna_plate,
                        "pert_iname": p.pert_iname,
                        "pert_type": p.pert_type,
                        "values_status": "deferred_exact978_or_gctx",
                    },
                )
                n += 1
                if limit is not None and n >= limit:
                    return
            return

        if source == "ChemPert":
            rows = load_chempert_response_metadata(self.paths.root, limit=limit)
            for row in rows:
                cid = row.get("Chemical_ID") or row.get("Chemical_Compound") or ""
                cell = row.get("Cell_Source") or ""
                if want_compounds is not None and cid not in want_compounds:
                    continue
                if want_cells is not None and cell not in want_cells:
                    continue
                # ChemPert CSV does not always expose a matched vehicle id → no delta label.
                yield PerturbationGeneRecord(
                    record_id=row.get("Response_ID") or "",
                    compound_id=cid,
                    cell_or_model_id=cell,
                    dose=None,
                    dose_unit=None,
                    time_hr=None,
                    signature_level=signature_level,
                    control_id=None,
                    gene_ids=(),
                    values=None,
                    gene_order_contract_id="chempert_pending",
                    perturbation_class="chemical",
                    source="ChemPert",
                    metadata={
                        "data_accession": row.get("Data_Accession"),
                        "dataset_name": row.get("Dataset_Name"),
                        "expression_id": row.get("Expression_ID"),
                        "control_status": "unresolved_in_skeleton",
                        "delta_supervision": False,
                    },
                )
            return

        raise ValueError(f"Unknown source {source!r}; expected LINCS_GSE92742 or ChemPert")

    def leakage_groups(
        self, records: Sequence[PerturbationGeneRecord]
    ) -> Mapping[str, Sequence[str]]:
        """Group by (source, compound_id) and by ChemPert Data_Accession when present."""
        groups: dict[str, list[str]] = {}
        for rec in records:
            key = f"compound::{rec.source}::{rec.compound_id}"
            groups.setdefault(key, []).append(rec.record_id)
            acc = None
            if isinstance(rec.metadata, Mapping):
                acc = rec.metadata.get("data_accession")
            if acc:
                g = f"accession::{acc}"
                groups.setdefault(g, []).append(rec.record_id)
        return groups


def smoke_measured_control_loader(
    root: Path | str | None = None,
    *,
    pair_limit: int = 200,
) -> dict:
    """Small CPU-only smoke: asset describe + limited control pairing + p0 assert."""
    loader = MeasuredPerturbationControlLoader(root)
    report = loader.describe()
    out: dict = {"describe": report, "ok": True, "errors": []}
    try:
        pairs = propose_lincs_control_pairs(
            loader.paths.root,
            perturbation_class="chemical",
            limit_treatments=pair_limit,
            max_scan_rows=80_000,
        )
        out["n_control_pairs_sample"] = len(pairs)
        if pairs:
            records = pairs_to_p0_records(pairs[: min(50, len(pairs))])
            out["n_p0_records_checked"] = len(records)
            out["assert_matched_controls"] = "passed"
        chem = load_chempert_response_metadata(loader.paths.root, limit=500)
        overlap = chempert_lincs_overlap_keys(chem)
        out["chempert_n_rows_sample"] = len(chem)
        out["chempert_accession_group_sizes"] = {
            k: len(v) for k, v in list(overlap.items())[:12]
        }
    except Exception as exc:  # noqa: BLE001 — smoke surfaces blockers
        out["ok"] = False
        out["errors"].append(repr(exc))
    return out


__all__ = [
    "AssetStatus",
    "ControlPairCandidate",
    "MeasuredAssetReport",
    "MeasuredPerturbationControlLoader",
    "MeasuredPerturbationPaths",
    "chempert_lincs_overlap_keys",
    "describe_assets",
    "iter_lincs_inst_info",
    "load_chempert_response_metadata",
    "pairs_to_p0_records",
    "propose_lincs_control_pairs",
    "smoke_measured_control_loader",
]