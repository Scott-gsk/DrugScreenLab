"""Bounded, immutable downloader for small public research assets.

This module deliberately does not admit assets to the data registry.  It writes
only previously absent files below a caller supplied data root and leaves an
auditable JSON receipt for a later human data-intake decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping


USER_AGENT = "DrugScreenLab/public-assets (reproducible research fetch)"
CHUNK_BYTES = 1024 * 1024


class AssetError(RuntimeError):
    """An asset could not be safely fetched or verified."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(data_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AssetError(f"unsafe relative path: {relative!r}")
    root = data_root.resolve()
    destination = (root / candidate).resolve()
    if os.path.commonpath((str(root), str(destination))) != str(root):
        raise AssetError(f"path escapes data root: {relative!r}")
    return destination


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("assets"), list):
        raise AssetError("manifest must be an object with an assets list")
    keys: set[str] = set()
    for row in value["assets"]:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise AssetError("every asset needs a string key")
        if row["key"] in keys:
            raise AssetError(f"duplicate asset key: {row['key']}")
        keys.add(row["key"])
        if row.get("status", "ready") == "ready":
            if not isinstance(row.get("url"), str) or not isinstance(row.get("relative_path"), str):
                raise AssetError(f"ready asset {row['key']} needs url and relative_path")
            _safe_destination(Path("."), row["relative_path"])
            if not isinstance(row.get("max_bytes"), int) or row["max_bytes"] <= 0:
                raise AssetError(f"ready asset {row['key']} needs a positive max_bytes")
    return value


def _headers(response: Any) -> Mapping[str, str]:
    return {str(k).lower(): str(v) for k, v in response.headers.items()}


def _validator(headers: Mapping[str, str]) -> dict[str, str]:
    result = {key: headers[key] for key in ("etag", "last-modified") if headers.get(key)}
    return result


def _read_part_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"invalid partial metadata: {path}") from exc
    if not isinstance(value, dict):
        raise AssetError(f"invalid partial metadata: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate_final(destination: Path, asset: Mapping[str, Any]) -> tuple[int, str]:
    size = destination.stat().st_size
    if size == 0:
        raise AssetError('empty asset')
    if size > int(asset["max_bytes"]):
        raise AssetError(f"existing file exceeds max_bytes ({size} > {asset['max_bytes']})")
    actual = sha256_file(destination)
    expected = asset.get("sha256")
    if expected and actual.lower() != str(expected).lower():
        raise AssetError(f"existing file sha256 mismatch for {destination}")
    expected_size = asset.get("bytes")
    if asset.get('md5'):
        digest = hashlib.md5()
        with destination.open('rb') as handle:
            for block in iter(lambda: handle.read(CHUNK_BYTES), b''):
                digest.update(block)
        if digest.hexdigest() != asset['md5']:
            raise AssetError('official md5 mismatch')
    if expected_size is not None and size != expected_size:
        raise AssetError(f"existing file size mismatch for {destination}")
    return size, actual


def fetch_asset(
    asset: Mapping[str, Any], data_root: Path, *, opener: Callable[..., Any] = urllib.request.urlopen, timeout: int = 120
) -> dict[str, Any]:
    """Fetch one asset once, resuming only validator-protected byte ranges."""
    destination = _safe_destination(data_root, str(asset["relative_path"]))
    receipt: dict[str, Any] = {"key": asset["key"], "url": asset["url"], "relative_path": asset["relative_path"]}
    if destination.exists():
        size, digest = _validate_final(destination, asset)
        return receipt | {"status": "exists", "bytes": size, "sha256": digest, "path": str(destination)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    part_meta = part.with_name(part.name + ".json")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    partial_meta: dict[str, Any] | None = None
    if offset:
        if not part_meta.exists():
            raise AssetError(f"refusing untracked partial file: {part}")
        partial_meta = _read_part_metadata(part_meta)
        validators = partial_meta.get("validators")
        if partial_meta.get("url") != asset["url"] or not isinstance(validators, dict) or not validators:
            raise AssetError(f"refusing partial file without matching URL and validator: {part}")
        headers["Range"] = f"bytes={offset}-"
        headers["If-Range"] = validators.get("etag") or validators.get("last-modified")

    request = urllib.request.Request(str(asset["url"]), headers=headers)
    try:
        response = opener(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise AssetError(f"HTTP {exc.code} for {asset['url']}") from exc
    except OSError as exc:
        raise AssetError(f"network error for {asset['url']}: {exc}") from exc
    with response:
        response_headers = _headers(response)
        status = getattr(response, "status", response.getcode())
        final_url = response.geturl()
        if offset:
            expected_range = f"bytes {offset}-"
            if status != 206 or not response_headers.get("content-range", "").startswith(expected_range):
                raise AssetError("server did not honor a valid Content-Range resume")
            if not _validator(response_headers):
                raise AssetError("resumed response lacks a validator")
            previous = partial_meta['validators']
            chosen = 'etag' if previous.get('etag') else 'last-modified'
            if response_headers.get(chosen) != previous[chosen]:
                raise AssetError('resume validator changed')
            mode = "ab"
        else:
            if status != 200:
                raise AssetError(f"unexpected HTTP status {status}")
            mode = "wb"
        content_length = response_headers.get("content-length")
        if content_length and offset + int(content_length) > int(asset["max_bytes"]):
            raise AssetError("declared response exceeds max_bytes")
        validators = _validator(response_headers)
        total_size = None
        if offset:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response_headers.get('content-range', ''))
            if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                raise AssetError('invalid or incomplete Content-Range')
            total_size = int(match[3])
        elif content_length:
            total_size = int(content_length)
        _write_json_atomic(part_meta, {'url': asset['url'], 'validators': validators, 'updated_at': utc_now()})
        # Validators are mandatory for resuming an existing partial.  Some
        # official APIs (including KEGG REST) do not emit them, but a complete
        # first transfer can still be integrity-checked and atomically sealed.
        with part.open(mode) as handle:
            written = offset
            for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                written += len(chunk)
                if written > int(asset["max_bytes"]):
                    raise AssetError("streamed response exceeds max_bytes")
                handle.write(chunk)
        if total_size is not None and written != total_size:
            raise AssetError('truncated response: actual length differs from declared total')
    _write_json_atomic(part_meta, {"url": asset["url"], "validators": validators, "updated_at": utc_now()})
    size, digest = _validate_final(part, asset)
    os.replace(part, destination)
    part_meta.unlink(missing_ok=True)
    return receipt | {
        "status": "downloaded", "bytes": size, "sha256": digest, "path": str(destination),
        "final_url": final_url, "validators": validators,
    }


def run_manifest(manifest: Mapping[str, Any], data_root: Path, mode: str) -> list[dict[str, Any]]:
    if mode not in {"dry-run", "fetch", "validate"}:
        raise AssetError(f"unknown mode: {mode}")
    results: list[dict[str, Any]] = []
    for asset in manifest["assets"]:
        result: dict[str, Any] = {"key": asset["key"], "source_page": asset.get("source_page"), "status": asset.get("status", "ready")}
        if asset.get("status", "ready") != "ready":
            result["reason"] = asset.get("reason", "not publicly resolved")
        elif mode == "dry-run":
            destination = _safe_destination(data_root, asset["relative_path"])
            result.update({"status": "would_fetch", "url": asset["url"], "path": str(destination)})
        else:
            try:
                if mode == "validate":
                    destination = _safe_destination(data_root, asset["relative_path"])
                    if not destination.exists():
                        raise AssetError("final file is absent")
                    size, digest = _validate_final(destination, asset)
                    result.update({"status": "valid", "bytes": size, "sha256": digest, "path": str(destination)})
                else:
                    result = fetch_asset(asset, data_root)
            except AssetError as exc:
                result.update({"status": "error", "error": str(exc)})
        results.append(result)
    return results


def write_receipt(path: Path, manifest_path: Path, mode: str, results: list[dict[str, Any]]) -> None:
    _write_json_atomic(path, {"created_at": utc_now(), "manifest": str(manifest_path), "mode": mode, "results": results})
