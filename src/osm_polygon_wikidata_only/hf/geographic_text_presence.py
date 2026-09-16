"""Factual polygon coverage by non-empty Wikipedia or Wikivoyage text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ._geographic.models import CoverageMapError, RenderResult
from ._geographic.parquet_inputs import read_required_columns, require_directory, sorted_parquets
from ._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonIndex,
    PolygonRecord,
    load_unique_polygon_records,
)
from ._links.reader import DocumentLink, is_canonical_link_schema, read_document_links
from .coverage_map import generate_coverage_map


@dataclass(frozen=True, slots=True)
class CoveredPoint:
    polygon_id: str
    wikidata: str
    lon: float
    lat: float
    identity: PolygonIdentity = ("legacy", "")


@dataclass(frozen=True, slots=True)
class TextPresenceSnapshot:
    polygon_count: int
    wikipedia_covered_polygon_ids: frozenset[str]
    combined_covered_polygon_ids: frozenset[str]
    wikipedia_document_ids: frozenset[str]
    wikivoyage_document_ids: frozenset[str]
    covered_points: tuple[CoveredPoint, ...]
    wikipedia_polygon_identities: frozenset[PolygonIdentity] = frozenset()
    combined_polygon_identities: frozenset[PolygonIdentity] = frozenset()


def _non_blank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _document_identity_column(names: set[str] | list[str] | tuple[str, ...]) -> str:
    return "document_id" if "document_id" in names else "article_id"


def load_text_presence(
    processed_root: Path,
    *,
    links_dir: Path | None = None,
) -> TextPresenceSnapshot:
    """Load exact Wikipedia and combined text coverage from canonical tables.

    V1 callers use the default ``polygon_articles`` directory.  Isolated
    contracts may provide their own unified link directory.
    """
    polygons_dir, wikipedia_dir, source_links_dir, wikivoyage_dir = _presence_directories(
        processed_root, links_dir
    )

    polygon_index = load_unique_polygon_records(sorted_parquets(polygons_dir))
    document_ids, legacy_wikivoyage_qids = _text_document_ids(wikipedia_dir, wikivoyage_dir)
    links = read_document_links(processed_root, links_dir=source_links_dir)
    wikipedia_polygons, combined_ids, unresolved = _linked_text_polygon_ids(
        links,
        document_ids,
        polygon_index,
    )
    _merge_legacy_wikivoyage_ids(
        combined_ids,
        polygon_index,
        legacy_wikivoyage_qids,
        source_links_dir,
    )
    _validate_link_polygon_ids(unresolved, polygon_index.by_polygon_id)
    return _presence_snapshot(
        polygon_index,
        wikipedia_polygons,
        combined_ids,
        document_ids,
    )


def _presence_directories(
    processed_root: Path,
    links_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    wikipedia_dir = require_directory(
        _wikipedia_documents_directory(processed_root),
        label="wikipedia/documents",
    )
    source_links_dir = links_dir or processed_root / "polygon_articles"
    require_directory(source_links_dir, label="polygon links")
    return (
        polygons_dir,
        wikipedia_dir,
        source_links_dir,
        processed_root / "wikivoyage" / "documents",
    )


def _wikipedia_documents_directory(processed_root: Path) -> Path:
    canonical = processed_root / "wikipedia" / "documents"
    return canonical if canonical.exists() else processed_root / "articles"


def _text_document_ids(
    wikipedia_dir: Path,
    wikivoyage_dir: Path,
) -> tuple[dict[str, set[str]], set[str]]:
    wikipedia_ids = _wikipedia_text_ids(wikipedia_dir)
    wikivoyage_ids, legacy_wikivoyage_qids = _wikivoyage_text_ids(wikivoyage_dir)
    return {
        "wikipedia": wikipedia_ids,
        "wikivoyage": wikivoyage_ids,
    }, legacy_wikivoyage_qids


def _merge_legacy_wikivoyage_ids(
    combined_ids: set[PolygonIdentity],
    polygon_index: PolygonIndex,
    legacy_wikivoyage_qids: set[str],
    source_links_dir: Path,
) -> None:
    if _has_canonical_links(source_links_dir):
        return
    combined_ids.update(
        identity
        for identity, record in polygon_index.records.items()
        if record.wikidata in legacy_wikivoyage_qids
    )


def _presence_snapshot(
    polygon_index: PolygonIndex,
    wikipedia_polygons: set[PolygonIdentity],
    combined_ids: set[PolygonIdentity],
    document_ids: dict[str, set[str]],
) -> TextPresenceSnapshot:
    points = tuple(
        _covered_point_from_record(polygon_index.records[identity])
        for identity in sorted(combined_ids, key=_identity_sort_key)
    )
    return TextPresenceSnapshot(
        polygon_count=len(polygon_index.records),
        wikipedia_covered_polygon_ids=frozenset(
            polygon_index.records[identity].polygon_id for identity in wikipedia_polygons
        ),
        combined_covered_polygon_ids=frozenset(
            polygon_index.records[identity].polygon_id for identity in combined_ids
        ),
        wikipedia_document_ids=frozenset(document_ids["wikipedia"]),
        wikivoyage_document_ids=frozenset(document_ids["wikivoyage"]),
        covered_points=points,
        wikipedia_polygon_identities=frozenset(wikipedia_polygons),
        combined_polygon_identities=frozenset(combined_ids),
    )


def _identity_sort_key(identity: PolygonIdentity) -> tuple[str, str]:
    return identity[0], str(identity[1])


def _successful_text_row(
    identity: object,
    full_text: object,
    fetch_status: object,
    *,
    has_fetch_status: bool,
) -> bool:
    return bool(
        identity and _non_blank(full_text) and (not has_fetch_status or fetch_status == "ok")
    )


def _document_columns(path: Path, identifier_column: str) -> tuple[str, ...]:
    names = set(pq.read_schema(path).names)
    columns = [identifier_column, "full_text"]
    if "fetch_status" in names:
        columns.append("fetch_status")
    return tuple(columns)


def _successful_document_row(
    row: dict[str, Any],
    identifier_column: str,
    *,
    has_fetch_status: bool,
) -> bool:
    return _successful_text_row(
        row.get(identifier_column),
        row.get("full_text"),
        row.get("fetch_status"),
        has_fetch_status=has_fetch_status,
    )


def _wikipedia_text_ids(wikipedia_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in sorted_parquets(wikipedia_dir):
        values.update(_wikipedia_file_text_ids(path))
    return values


def _wikipedia_file_text_ids(path: Path) -> set[str]:
    names = set(pq.read_schema(path).names)
    identifier_column = _document_identity_column(names)
    has_fetch_status = "fetch_status" in names
    values: set[str] = set()
    for row in read_required_columns(
        path,
        _document_columns(path, identifier_column),
        label="wikipedia",
    ):
        if _successful_document_row(
            row,
            identifier_column,
            has_fetch_status=has_fetch_status,
        ):
            values.add(str(row[identifier_column]))
    return values


def _wikivoyage_text_ids(directory: Path) -> tuple[set[str], set[str]]:
    document_ids: set[str] = set()
    qids: set[str] = set()
    for path in sorted_parquets(directory):
        file_ids, file_qids = _wikivoyage_file_text_ids(path)
        document_ids.update(file_ids)
        qids.update(file_qids)
    return document_ids, qids


def _wikivoyage_file_text_ids(path: Path) -> tuple[set[str], set[str]]:
    names = set(pq.read_schema(path).names)
    has_fetch_status = "fetch_status" in names
    document_ids: set[str] = set()
    qids: set[str] = set()
    columns = _wikivoyage_columns(path, names)
    for row in read_required_columns(path, columns, label="wikivoyage"):
        document_id, wikidata = _wikivoyage_row_ids(row, has_fetch_status=has_fetch_status)
        if document_id:
            document_ids.add(document_id)
        if wikidata:
            qids.add(wikidata)
    return document_ids, qids


def _wikivoyage_columns(path: Path, names: set[str]) -> tuple[str, ...]:
    columns = _document_columns(path, "document_id")
    return (*columns, "wikidata") if "wikidata" in names else columns


def _wikivoyage_row_ids(
    row: dict[str, Any],
    *,
    has_fetch_status: bool,
) -> tuple[str | None, str | None]:
    if not _successful_document_row(row, "document_id", has_fetch_status=has_fetch_status):
        return None, None
    document_id = str(row["document_id"]) if row.get("document_id") else None
    wikidata = str(row["wikidata"]) if row.get("wikidata") else None
    return document_id, wikidata


def _has_canonical_links(source_links_dir: Path) -> bool:
    return any(
        is_canonical_link_schema(pq.read_schema(path)) for path in sorted_parquets(source_links_dir)
    )


def _linked_text_polygon_ids(
    links: tuple[DocumentLink, ...],
    document_ids: dict[str, set[str]],
    polygon_index: PolygonIndex,
) -> tuple[set[PolygonIdentity], set[PolygonIdentity], set[str]]:
    wikipedia_ids: set[PolygonIdentity] = set()
    combined_ids: set[PolygonIdentity] = set()
    unresolved: set[str] = set()
    for link in links:
        if link.document_id not in document_ids.get(link.project, set()):
            continue
        identity = polygon_index.by_polygon_id.get(link.polygon_id)
        if identity is None:
            unresolved.add(link.polygon_id)
            continue
        combined_ids.add(identity)
        if link.project == "wikipedia":
            wikipedia_ids.add(identity)
    return wikipedia_ids, combined_ids, unresolved


def _covered_point_from_record(record: PolygonRecord) -> CoveredPoint:
    if record.lon is None or record.lat is None:
        raise CoverageMapError(
            f"polygons parquet {record.source_path} has invalid lat/lon coordinates "
            f"for {record.polygon_id}"
        )
    return CoveredPoint(
        record.polygon_id,
        record.wikidata,
        record.lon,
        record.lat,
        record.identity,
    )


def _validate_link_polygon_ids(
    unresolved: set[str],
    polygon_ids: dict[str, PolygonIdentity],
) -> None:
    missing = unresolved - polygon_ids.keys()
    if missing:
        raise CoverageMapError(
            f"polygon_articles contains {len(missing)} polygon id(s) absent from polygons"
        )


def generate_geographic_text_presence(
    processed_root: Path,
    output_path: Path,
    *,
    land_geojson_path: Path | None = None,
    snapshot: TextPresenceSnapshot | None = None,
) -> RenderResult:
    """Render one point for every polygon with Wikipedia or Wikivoyage text."""
    snapshot = snapshot or load_text_presence(processed_root)
    points = snapshot.covered_points
    generate_coverage_map(
        [point.lon for point in points],
        [point.lat for point in points],
        output_path,
        land_geojson_path=land_geojson_path,
        title="Polygons with Wikipedia or Wikivoyage text",
        point_color="#2563EB",
        point_edge="#1E40AF",
    )
    rate = len(points) / snapshot.polygon_count if snapshot.polygon_count else 0.0
    caption = (
        f"{len(points):,} of {snapshot.polygon_count:,} unique (osm_type, osm_id) "
        f"polygon identities ({rate:.1%}) have successfully extracted "
        "(`fetch_status=ok`) non-empty Wikipedia or Wikivoyage text."
    )
    return RenderResult(output_path=output_path, caption=caption)


__all__ = [
    "CoveredPoint",
    "TextPresenceSnapshot",
    "generate_geographic_text_presence",
    "load_text_presence",
]
