"""Atomic, deterministic storage for one V2 region."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_wikidata_only.augmentation.schema import section_schema
from osm_polygon_wikidata_only.io.atomic import atomic_write_text
from osm_polygon_wikidata_only.utils.json import dumps as json_dumps
from osm_polygon_wikidata_only.utils.json import loads as json_loads
from osm_polygon_wikidata_only.v2.config import V2_CONTRACT_VERSION
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for final in staged:
            if final.exists():
                backup = _stage_path(final)
                os.replace(final, backup)
                backups[final] = backup
        for final, temporary in staged.items():
            os.replace(temporary, final)
            replaced.append(final)
    except BaseException:
        for final in replaced:
            final.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists():
                os.replace(backup, final)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def load_v2_manifest(processed_v2: Path) -> dict[str, dict[str, Any]]:
    """Load the V2 manifest, returning an empty mapping when absent."""
    path = processed_v2 / "manifests" / "processed_pbfs.json"
    if not path.exists():
        return {}
    raw = json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"V2 manifest is not an object: {path}")
    entries = raw.get("regions", raw)
    if not isinstance(entries, dict):
        raise ValueError(f"V2 manifest regions is not an object: {path}")
    return {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}


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
    if not stem or "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise ValueError(f"Invalid V2 stem: {stem!r}")
    polygons_path = processed_v2 / "polygons" / f"{stem}.parquet"
    documents_path = processed_v2 / "wikipedia" / "documents" / f"{stem}.parquet"
    sections_path = processed_v2 / "wikipedia" / "sections" / f"{stem}.parquet"
    links_path = processed_v2 / "polygon_document_links" / f"{stem}.parquet"
    manifest_path = processed_v2 / "manifests" / "processed_pbfs.json"
    final_paths = (polygons_path, documents_path, sections_path, links_path)
    schemas = (
        polygon_v2_schema(),
        wikipedia_document_v2_schema(),
        section_schema(),
        polygon_document_link_v2_schema(),
    )
    rows = (polygons, documents, sections or [], links)
    staged: dict[Path, Path] = {}
    try:
        for final, region_rows, schema in zip(final_paths, rows, schemas, strict=True):
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = _stage_path(final)
            staged[final] = temporary
            _write_table(temporary, region_rows, schema)
            with pq.ParquetFile(temporary) as parquet_file:
                written = parquet_file.read()
            if not written.schema.equals(schema, check_metadata=True):
                raise ValueError(f"V2 table schema validation failed: {final}")
        _replace_transaction(staged)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

    file_hashes = {str(path.relative_to(processed_v2)): _sha256(path) for path in final_paths}
    entry = {
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
            "sections": len(sections or []),
            "links": len(links),
        },
        "file_hashes": dict(sorted(file_hashes.items())),
    }
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
    return V2RegionArtifacts(
        polygons_path=polygons_path,
        documents_path=documents_path,
        sections_path=sections_path,
        links_path=links_path,
        manifest_path=manifest_path,
        manifest_entry=entry,
        file_hashes=file_hashes,
    )


__all__ = ["V2RegionArtifacts", "load_v2_manifest", "write_v2_region"]
