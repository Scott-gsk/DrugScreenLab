"""Immutable public-data freezer and integrity gates for EXP-010.

The module deliberately never extracts into or edits ``data/raw``.  Downloads
land below ``data/external/exp010`` and are accepted only after a byte count,
SHA-256 digest, and a lightweight format/schema check have been recorded.
"""

from __future__ import annotations

import hashlib
import gzip
import csv
import json
import shutil
import tempfile
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PublicAsset:
    dataset: str
    filename: str
    url: str
    version: str
    license: str
    required: bool = True
    archive: str | None = None


@dataclass(frozen=True)
class SampleContext:
    sample_id: str
    patient_id: str
    model_id: str
    tissue: str
    region_id: str | None = None
    dose: float | None = None
    dose_unit: str | None = None
    time_hours: float | None = None
    control_id: str | None = None


@dataclass(frozen=True)
class DrugRecord:
    drug_id: str
    primary_name: str
    canonical_smiles: str | None
    pubchem_cid: str | None
    inchikey: str | None
    is_combination: bool = False


@dataclass(frozen=True)
class ResponseLabel:
    sample_id: str
    drug_id: str
    value: float
    metric: str
    higher_is_sensitive: bool


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    drug_id: str
    fold: int
    model: str
    prediction: float


ASSETS: tuple[PublicAsset, ...] = (
    PublicAsset("GDSC2", "GDSC2_fitted_dose_response_27Oct23.xlsx", "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx", "release-8.5/2023-10-27", "GDSC academic-use terms"),
    PublicAsset("GDSC2", "GDSC2_public_raw_data_27Oct23.csv", "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_public_raw_data_27Oct23.csv", "release-8.5/2023-10-27", "GDSC academic-use terms"),
    PublicAsset("GDSC2", "screened_compounds_rel_8.5.csv", "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv", "release-8.5", "GDSC academic-use terms"),
    PublicAsset("GDSC2", "Cell_Lines_Details.xlsx", "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx", "release-8.5", "GDSC academic-use terms"),
    PublicAsset("CMP", "rnaseq_all_20220624.zip", "https://cog.sanger.ac.uk/cmp/download/rnaseq_all_20220624.zip", "2022-06-24", "Cell Model Passports terms", archive="zip"),
    PublicAsset("GSE303509", "GSE303509_RAW.tar", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303509/suppl/GSE303509_RAW.tar", "GEO snapshot 2026-07-09", "NCBI GEO public data", archive="tar"),
    PublicAsset("GSE287429", "GSE287429_normalized_counts_allsample.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE287nnn/GSE287429/suppl/GSE287429_normalized_counts_allsample.txt.gz", "GEO snapshot 2025-01-19", "NCBI GEO public data", archive="gzip"),
    PublicAsset("GSE276609", "GSE276609_RAW.tar", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE276nnn/GSE276609/suppl/GSE276609_RAW.tar", "GEO snapshot 2024-10-03", "NCBI GEO public data", archive="tar"),
    PublicAsset("GSE276609", "GSE276609_GSE161529_reanalysis.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE276nnn/GSE276609/suppl/GSE276609_GSE161529_reanalysis.tar.gz", "GEO snapshot 2024-09-07", "NCBI GEO public data", archive="tar"),
    PublicAsset("GSE276609", "GSE276609_GSE235168_reanalysis.tar.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE276nnn/GSE276609/suppl/GSE276609_GSE235168_reanalysis.tar.gz", "GEO snapshot 2024-09-07", "NCBI GEO public data", archive="tar"),
    PublicAsset("GSE276609", "GSE276609_sample_file_association_for_reanalysis.txt.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE276nnn/GSE276609/suppl/GSE276609_sample_file_association_for_reanalysis.txt.gz", "GEO snapshot 2024-09-07", "NCBI GEO public data", archive="gzip"),
    PublicAsset("Reactome", "ReactomePathways.gmt.zip", "https://reactome.org/download/current/ReactomePathways.gmt.zip", "current-at-freeze", "Reactome Content Service terms", archive="zip"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s004.pdf", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s004.pdf", "10.1002/advs.76541", "publisher supplementary-data terms"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s001.xlsx", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s001.xlsx", "10.1002/advs.76541/Table-S1", "publisher supplementary-data terms"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s003.xlsx", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s003.xlsx", "10.1002/advs.76541/Table-S2", "publisher supplementary-data terms"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s002.xlsx", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s002.xlsx", "10.1002/advs.76541/Table-S3", "publisher supplementary-data terms"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s005.xlsx", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s005.xlsx", "10.1002/advs.76541/Table-S4", "publisher supplementary-data terms"),
    PublicAsset("MultiRegionalBreastPDO", "ADVS-9999-e76541-s006.xlsx", "https://pmc.ncbi.nlm.nih.gov/articles/instance/13384037/bin/ADVS-9999-e76541-s006.xlsx", "10.1002/advs.76541/Table-S5", "publisher supplementary-data terms"),
)


FOLDERS = {
    "GDSC2": "gdsc2_release8.5", "CMP": "cmp_20220624", "GSE303509": "geo/gse303509",
    "GSE287429": "geo/gse287429", "GSE276609": "geo/gse276609", "Reactome": "reactome",
    "MultiRegionalBreastPDO": "multiregional_breast_pdo",
}


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_gene_universe(*gene_sets: Iterable[str], minimum: int = 15_000) -> tuple[list[str], str]:
    """Return a deterministic common gene universe and its contract hash."""
    normalized = [{str(g).strip().upper() for g in genes if str(g).strip()} for genes in gene_sets]
    common = sorted(set.intersection(*normalized)) if normalized else []
    if len(common) < minimum:
        raise ValueError(f"common gene universe {len(common)} < required {minimum}")
    digest = hashlib.sha256(("\n".join(common) + "\n").encode()).hexdigest()
    return common, digest


def patient_split_gate(contexts: Sequence[SampleContext], folds: Mapping[str, int]) -> dict[str, object]:
    """Reject missing identities or any model/region assigned across patient folds."""
    if any(not c.patient_id or not c.model_id or not c.sample_id for c in contexts):
        raise ValueError("patient_id, model_id and sample_id are mandatory")
    patient_folds: dict[str, set[int]] = {}
    model_folds: dict[str, set[int]] = {}
    region_folds: dict[str, set[int]] = {}
    for context in contexts:
        if context.sample_id not in folds:
            raise ValueError(f"sample missing fold: {context.sample_id}")
        fold = int(folds[context.sample_id])
        patient_folds.setdefault(context.patient_id, set()).add(fold)
        model_folds.setdefault(context.model_id, set()).add(fold)
        if context.region_id:
            region_folds.setdefault(context.region_id, set()).add(fold)
    leaked = {
        "patients": sorted(k for k, v in patient_folds.items() if len(v) != 1),
        "models": sorted(k for k, v in model_folds.items() if len(v) != 1),
        "regions": sorted(k for k, v in region_folds.items() if len(v) != 1),
    }
    if any(leaked.values()):
        raise ValueError(f"identity leakage: {leaked}")
    return {"patients": len(patient_folds), "models": len(model_folds), "regions": len(region_folds), "leakage": False}


def target_path(root: Path, asset: PublicAsset) -> Path:
    return root / FOLDERS[asset.dataset] / asset.filename


def download(asset: PublicAsset, root: Path) -> Path:
    """Resume-safe download with fail-closed Range and quarantine semantics."""
    target = target_path(root, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    remote_bytes: int | None = None
    try:
        with urllib.request.urlopen(urllib.request.Request(asset.url, method="HEAD")) as head:
            value = head.headers.get("Content-Length")
            remote_bytes = int(value) if value else None
    except Exception:
        pass
    if target.exists():
        complete = target.stat().st_size > 0 and (remote_bytes is None or target.stat().st_size == remote_bytes)
        try:
            complete = complete and bool(archive_gate(target, asset.archive)["archive_ok"])
        except Exception:
            complete = False
        if complete:
            return target
        quarantine = target.with_name(target.name + ".quarantine")
        if quarantine.exists():
            raise RuntimeError(f"invalid target and quarantine already exists: {target}")
        shutil.move(target, quarantine)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(asset.url)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    response = urllib.request.urlopen(request)
    status = getattr(response, "status", response.getcode())
    mode = "wb"
    if offset and status == 206:
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {offset}-"):
            response.close()
            raise RuntimeError(f"invalid Content-Range for {target}: {content_range!r}")
        mode = "ab"
    elif offset and status != 200:
        response.close()
        raise RuntimeError(f"resume expected HTTP 206 (or safe full 200), got {status}")
    # A server ignoring Range with HTTP 200 is safe only by replacing .part.
    with response, partial.open(mode) as out:
        while chunk := response.read(8 << 20):
            out.write(chunk)
    if remote_bytes is not None and partial.stat().st_size != remote_bytes:
        raise RuntimeError(f"byte mismatch for {target}: {partial.stat().st_size} != {remote_bytes}")
    if not archive_gate(partial, asset.archive)["archive_ok"]:
        raise RuntimeError(f"format gate failed: {partial}")
    partial.replace(target)
    return target


def archive_gate(path: Path, kind: str | None) -> dict[str, object]:
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            return {"archive_ok": bad is None, "members": len(archive.infolist()), "bad_member": bad}
    if kind == "tar":
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            corrupt: list[str] = []
            checked = 0
            for member in members:
                if not member.isfile():
                    continue
                checked += 1
                extracted = archive.extractfile(member)
                if extracted is None:
                    corrupt.append(member.name + ":unreadable")
                    continue
                try:
                    if member.name.endswith(".tar.gz"):
                        with tempfile.SpooledTemporaryFile(max_size=64 << 20) as spool:
                            shutil.copyfileobj(extracted, spool, length=8 << 20)
                            spool.seek(0)
                            with gzip.GzipFile(fileobj=spool, mode="rb") as compressed:
                                while compressed.read(8 << 20):
                                    pass
                            spool.seek(0)
                            nested = tarfile.open(fileobj=spool, mode="r|gz")
                            for nested_member in nested:
                                if nested_member.isfile():
                                    nested_file = nested.extractfile(nested_member)
                                    if nested_file:
                                        while nested_file.read(8 << 20):
                                            pass
                            nested.close()
                    elif member.name.endswith(".gz"):
                        with gzip.GzipFile(fileobj=extracted) as compressed:
                            while compressed.read(8 << 20):
                                pass
                    else:
                        while extracted.read(8 << 20):
                            pass
                except (EOFError, OSError, tarfile.TarError) as error:
                    corrupt.append(f"{member.name}:{type(error).__name__}:{error}")
            return {"archive_ok": not corrupt, "members": len(members), "analysis_members_checked": checked, "corrupt_members": corrupt}
    if kind == "gzip":
        with gzip.open(path, "rb") as handle:
            head = handle.read(1024)
        return {"archive_ok": bool(head), "decompressed_head_bytes": len(head)}
    head = path.read_bytes()[:8]
    if path.suffix.lower() == ".xlsx":
        return {"archive_ok": zipfile.is_zipfile(path), "format": "xlsx"}
    if path.suffix.lower() == ".pdf":
        return {"archive_ok": head.startswith(b"%PDF-"), "format": "pdf"}
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        return {"archive_ok": len(header) > 1, "format": "csv", "columns": header}
    return {"archive_ok": path.stat().st_size > 0}


def freeze_manifest(root: Path, output: Path, assets: Iterable[PublicAsset] = ASSETS) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for asset in assets:
        path = target_path(root, asset)
        record = asdict(asset) | {"relative_path": str(path.relative_to(root)), "present": path.is_file()}
        if path.is_file():
            record |= {"bytes": path.stat().st_size, "sha256": sha256(path)}
            record |= archive_gate(path, asset.archive)
        records.append(record)
    missing = [r["filename"] for r in records if r["required"] and not r["present"]]
    corrupt = [r["filename"] for r in records if r["present"] and r.get("archive_ok") is False]
    payload = {
        "experiment_id": "EXP-010",
        "root": str(root),
        "forbidden": ["CLOUPE files", "test-derived feature/threshold selection"],
        "data_status": "DATA_READY" if not missing and not corrupt else "DATA_PARTIAL",
        "missing_required": missing,
        "corrupt_assets": corrupt,
        "assets": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
