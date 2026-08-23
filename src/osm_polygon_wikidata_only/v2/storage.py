"""Atomic, deterministic storage for one V2 region."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.io.hashing import sha256_file
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION
from osm_polygon_wikidata_only.v2.deduplication import (
    deduplicate_documents,
    deduplicate_links,
)
from osm_polygon_wikidata_only.v2.schema import (
    polygon_document_link_v2_schema,
    polygon_v2_schema,
    wikipedia_document_v2_schema,
)


@dataclass(frozen=True, slots=True)
class V2RegionArtifacts:
    """Final local paths and content hashes for a persisted V2 region."""

    polygons_path: Path
    documents_path: Path
    sections_path: Path
    links_path: Path
    manifest_path: Path
    manifest_entry: dict[str, Any]
    file_hashes: dict[str, str]


def _write_table(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    normalized = [{field.name: row.get(field.name) for field in schema} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")  # type: ignore[no-untyped-call]


def _stage_path(final: Path) -> Path:
    fd, raw = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    os.close(fd)
    return Path(raw)


def _replace_transaction(staged: dict[Path, Path]) -> None:
    backups = _backup_existing(staged)
    replaced: list[Path] = []
    try:
        _replace_staged(staged, replaced)
    except BaseException:
        _restore_backups(backups, replaced)
        raise
    finally:
        _cleanup_transaction(staged, backups)


def _backup_existing(staged: dict[Path, Path]) -> dict[Path, Path]:
    backups: dict[Path, Path] = {}
    for final in staged:
        if final.exists():
            backup = _stage_path(final)
            os.replace(final, backup)
            backups[final] = backup
    return backups


def _replace_staged(staged: dict[Path, Path], replaced: list[Path]) -> None:
    for final, temporary in staged.items():
        os.replace(temporary, final)
        replaced.append(final)


def _restore_backups(backups: dict[Path, Path], replaced: list[Path]) -> None:
    for final in replaced:
        final.unlink(missing_ok=True)
    for final, backup in backups.items():
        if backup.exists():
            os.replace(backup, final)


def _cleanup_transaction(staged: dict[Path, Path], backups: dict[Path, Path]) -> None:
    for temporary in staged.values():
        temporary.unlink(missing_ok=True)
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def load_v2_manifest(processed_v2: Path) -> dict[str, dict[str, Any]]:
    """Load the V2 manifest, returning an empty mapping when absent."""
    path = processed_v2 / "manifests" / "processed_pbfs.json"
    if not path.exists():
        return {}
    raw = _read_manifest(path)
    entries = _manifest_entries(raw, path)
    return _coerce_manifest_entries(entries)


def _read_manifest(path: Path) -> object:
    try:
        return json_loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"Malformed V2 manifest: {exc}") from exc


def _manifest_entries(raw: object, path: Path) -> dict[object, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"V2 manifest is not an object: {path}")
    raw_dict = cast(dict[object, object], raw)
    entries = raw_dict.get("regions", raw_dict)
    if not isinstance(entries, dict):
        raise ValueError(f"V2 manifest regions is not an object: {path}")
    return cast(dict[object, object], entries)


def _coerce_manifest_entries(entries: dict[object, object]) -> dict[str, dict[str, Any]]:
    return {
        str(key): cast(dict[str, Any], dict(value))
        for key, value in entries.items()
        if isinstance(value, dict)
    }


def write_v2_region(
    processed_v2: Path,
    stem: str,
    *,
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    links: list[dict[str, Any]],
    sections: list[dict[str, Any]] | None = None,
    v1_index_reconciled: bool = True,
) -> V2RegionArtifacts:
    """Atomically persist V2 core tables and update the manifest last.

    Wikipedia sections use the exact V1 section schema.  An empty section
    table is still written so every V2 region has the same required artifact
    set, including regions with no Wikipedia documents.
    """
    _validate_stem(stem)
    documents = deduplicate_documents(documents)
    links = deduplicate_links(links)
    final_paths = _region_paths(processed_v2, stem)
    manifest_path = processed_v2 / "manifests" / "processed_pbfs.json"
    schemas = (
        polygon_v2_schema(),
        wikipedia_document_v2_schema(),
        section_schema(),
        polygon_document_link_v2_schema(),
    )
    rows = (polygons, documents, sections or [], links)
    _stage_region_tables(final_paths, rows, schemas)

    file_hashes = _hash_region_paths(processed_v2, final_paths)
    entry = _region_manifest_entry(
        processed_v2,
        stem,
        final_paths,
        polygons,
        documents,
        sections or [],
        links,
        file_hashes,
        v1_index_reconciled,
    )
    _write_region_manifest(processed_v2, manifest_path, stem, entry)
    polygons_path, documents_path, sections_path, links_path = final_paths
    return V2RegionArtifacts(
        polygons_path=polygons_path,
        documents_path=documents_path,
        sections_path=sections_path,
        links_path=links_path,
        manifest_path=manifest_path,
        manifest_entry=entry,
        file_hashes=file_hashes,
    )


def _validate_stem(stem: str) -> None:
    if not stem or "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise ValueError(f"Invalid V2 stem: {stem!r}")


def _region_paths(processed_v2: Path, stem: str) -> tuple[Path, Path, Path, Path]:
    return (
        processed_v2 / "polygons" / f"{stem}.parquet",
        processed_v2 / "wikipedia" / "documents" / f"{stem}.parquet",
        processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet",
        processed_v2 / "polygon_document_links" / f"{stem}.parquet",
    )


def _stage_region_tables(
    final_paths: tuple[Path, ...],
    rows: tuple[list[dict[str, Any]], ...],
    schemas: tuple[pa.Schema, ...],
) -> None:
    staged: dict[Path, Path] = {}
    try:
        for final, region_rows, schema in zip(final_paths, rows, schemas, strict=True):
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = _stage_path(final)
            staged[final] = temporary
            _write_table(temporary, region_rows, schema)
            _validate_written_table(temporary, final, schema)
        _replace_transaction(staged)
    finally:
        _cleanup_staged_paths(staged)


def _validate_written_table(temporary: Path, final: Path, schema: pa.Schema) -> None:
    with pq.ParquetFile(temporary) as parquet_file:
        written = parquet_file.read()
    if not written.schema.equals(schema, check_metadata=True):
        raise ValueError(f"V2 table schema validation failed: {final}")


def _cleanup_staged_paths(staged: dict[Path, Path]) -> None:
    for temporary in staged.values():
        temporary.unlink(missing_ok=True)


def _hash_region_paths(processed_v2: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.relative_to(processed_v2)): sha256_file(path) for path in paths}


def _region_manifest_entry(
    processed_v2: Path,
    stem: str,
    paths: tuple[Path, ...],
    polygons: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    links: list[dict[str, Any]],
    file_hashes: dict[str, str],
    v1_index_reconciled: bool,
) -> dict[str, Any]:
    polygons_path, documents_path, sections_path, links_path = paths
    return {
        "contract_version": V2_CONTRACT_VERSION,
        "source_pbf": f"{stem}.osm.pbf",
        "region": stem.removesuffix("-latest"),
        "polygons_path": str(polygons_path.relative_to(processed_v2)),
        "documents_path": str(documents_path.relative_to(processed_v2)),
        "sections_path": str(sections_path.relative_to(processed_v2)),
        "links_path": str(links_path.relative_to(processed_v2)),
        "v1_index_reconciled": v1_index_reconciled,
        "row_counts": {
            "polygons": len(polygons),
            "documents": len(documents),
            "sections": len(sections),
            "links": len(links),
        },
        "file_hashes": dict(sorted(file_hashes.items())),
    }


def _write_region_manifest(
    processed_v2: Path,
    manifest_path: Path,
    stem: str,
    entry: dict[str, Any],
) -> None:
    entries = load_v2_manifest(processed_v2)
    entries[stem] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_path,
        json_dumps(
            {"contract_version": V2_CONTRACT_VERSION, "regions": dict(sorted(entries.items()))}
        )
        + "\n",
    )


__all__ = ["V2RegionArtifacts", "load_v2_manifest", "write_v2_region"]
