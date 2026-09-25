"""Processed-manifest reading and release provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from osm_polygon_wikidata_only.hf._stats_release.models import StatsReleaseError
from osm_polygon_wikidata_only.io.hashing import sha256_file

_MANIFEST_RELATIVE_PATH = Path("manifests/processed_pbfs.json")


def _manifest_stem(key: str) -> str:
    return key.removesuffix(".pbf").removesuffix(".osm")


def _manifest_entries(raw: object, manifest_path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise StatsReleaseError(f"processed manifest is not an object: {manifest_path}")
    entries = raw.get("regions", raw)
    if not isinstance(entries, dict) or not entries:
        raise StatsReleaseError(f"processed manifest has no region entries: {manifest_path}")
    result: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        result.append(_manifest_entry(key, value, manifest_path))
    return sorted(result)


def _manifest_entry(
    key: object,
    value: object,
    manifest_path: Path,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise StatsReleaseError(f"processed manifest entry is not an object: {manifest_path}")
    return str(key), cast(dict[str, Any], value)


def _manifest_row_count(entry: Mapping[str, Any], manifest_path: Path) -> int:
    row_counts = entry.get("row_counts")
    value = (
        row_counts.get("polygons") if isinstance(row_counts, dict) else entry.get("polygon_count")
    )
    if isinstance(value, bool) or not isinstance(value, int):
        raise StatsReleaseError(f"processed manifest lacks polygon row count: {manifest_path}")
    return value


def _manifest_polygon_metadata(
    key: str,
    entry: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[str, str, int]:
    polygons_path = entry.get("polygons_path")
    expected_path = f"polygons/{_manifest_stem(key)}.parquet"
    if polygons_path != expected_path:
        raise StatsReleaseError(
            f"processed manifest entry {key!r} must point to {expected_path!r}: {manifest_path}"
        )
    source_pbf = entry.get("source_pbf") or key
    if not isinstance(source_pbf, str) or not source_pbf:
        raise StatsReleaseError(f"processed manifest entry lacks source_pbf: {manifest_path}")
    return expected_path, source_pbf, _manifest_row_count(entry, manifest_path)


def _read_manifest(processed_dir: Path) -> tuple[Path, bytes, object]:
    manifest_path = processed_dir / _MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise StatsReleaseError(f"complete processed manifest is required: {manifest_path}")
    try:
        raw_bytes = manifest_path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StatsReleaseError(
            f"cannot read processed manifest {manifest_path}: {error}"
        ) from error
    return manifest_path, raw_bytes, raw


def _manifest_revision(raw: object, *names: str) -> str | None:
    if not isinstance(raw, dict):
        return None
    raw_mapping = cast(dict[str, Any], raw)
    for name in names:
        revision = _manifest_revision_for_name(raw_mapping, name)
        if revision:
            return revision
    return None


def _manifest_revision_for_name(raw: Mapping[str, Any], name: str) -> str | None:
    direct = _non_empty_string(raw.get(name))
    return direct or _nested_manifest_revision(raw, name)


def _nested_manifest_revision(raw: Mapping[str, Any], name: str) -> str | None:
    nested = raw.get(name.removesuffix("_revision"))
    if not isinstance(nested, dict):
        return None
    return _non_empty_string(nested.get("revision"))


def _non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _file_row_count(path: Path) -> int:
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception as error:
        raise StatsReleaseError(f"cannot read Parquet metadata for {path}: {error}") from error


def _inventory_rows(
    processed_dir: Path,
    manifest_path: Path,
    entries: Sequence[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    manifest_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for key, entry in entries:
        relative_path, row = _inventory_row(processed_dir, manifest_path, key, entry)
        manifest_paths.add(relative_path)
        rows.append(row)
    actual_paths = {
        str(path.relative_to(processed_dir))
        for path in sorted((processed_dir / "polygons").glob("*.parquet"))
    }
    unlisted = sorted(actual_paths - manifest_paths)
    missing = sorted(manifest_paths - actual_paths)
    if unlisted:
        raise StatsReleaseError("polygon files absent from the manifest: " + ", ".join(unlisted))
    if missing:
        raise StatsReleaseError("manifest-listed polygon files are missing: " + ", ".join(missing))
    return rows


def _inventory_row(
    processed_dir: Path,
    manifest_path: Path,
    key: str,
    entry: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    relative_path, source_pbf, declared_rows = _manifest_polygon_metadata(key, entry, manifest_path)
    path = processed_dir / relative_path
    if not path.is_file():
        raise StatsReleaseError(f"manifest-listed polygon file is missing: {relative_path}")
    actual_rows = _file_row_count(path)
    if actual_rows != declared_rows:
        raise StatsReleaseError(
            f"polygon row count mismatch for {relative_path}: "
            f"manifest={declared_rows}, file={actual_rows}"
        )
    return relative_path, {
        "path": relative_path,
        "row_count": actual_rows,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_pbf": source_pbf,
    }


def build_provenance(
    processed_dir: Path,
    *,
    source_revision: str | None,
    data_revision: str | None,
) -> dict[str, Any]:
    manifest_path, raw_bytes, raw = _read_manifest(processed_dir)
    entries = _manifest_entries(raw, manifest_path)
    polygons = _inventory_rows(processed_dir, manifest_path, entries)
    source_pbf_files = _source_pbf_files(polygons)
    resolved_source_revision, resolved_data_revision = _provenance_revisions(
        raw,
        source_revision=source_revision,
        data_revision=data_revision,
    )
    return {
        "contract_version": _contract_version(entries),
        "data": {"revision": resolved_data_revision},
        "data_revision": resolved_data_revision,
        "manifest": _manifest_provenance(raw_bytes, len(entries)),
        "polygons": polygons,
        "source": {"pbf_files": source_pbf_files, "revision": resolved_source_revision},
        "source_pbf_files": source_pbf_files,
        "source_revision": resolved_source_revision,
    }


def _source_pbf_files(polygons: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(item["source_pbf"]) for item in polygons})


def _provenance_revisions(
    raw: object,
    *,
    source_revision: str | None,
    data_revision: str | None,
) -> tuple[str | None, str | None]:
    return (
        source_revision or _manifest_revision(raw, "source_revision", "source"),
        data_revision or _manifest_revision(raw, "data_revision", "data"),
    )


def _contract_version(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str | list[str]:
    versions = sorted(
        {str(entry["contract_version"]) for _, entry in entries if entry.get("contract_version")}
    )
    return versions[0] if len(versions) == 1 else versions


def _manifest_provenance(raw_bytes: bytes, entry_count: int) -> dict[str, Any]:
    return {
        "entry_count": entry_count,
        "path": str(_MANIFEST_RELATIVE_PATH),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }
