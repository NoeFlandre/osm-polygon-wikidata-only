"""Factual polygon coverage by non-empty Wikipedia or Wikivoyage text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ._geographic.models import CoverageMapError, RenderResult
from ._geographic.parquet_inputs import read_required_columns, require_directory, sorted_parquets
from ._links.reader import DocumentLink, is_canonical_link_schema, read_document_links
from .coverage_map import generate_coverage_map


@dataclass(frozen=True, slots=True)
class CoveredPoint:
    polygon_id: str
    wikidata: str
    lon: float
    lat: float


@dataclass(frozen=True, slots=True)
class TextPresenceSnapshot:
    polygon_count: int
    wikipedia_covered_polygon_ids: frozenset[str]
    combined_covered_polygon_ids: frozenset[str]
    wikipedia_document_ids: frozenset[str]
    wikivoyage_document_ids: frozenset[str]
    covered_points: tuple[CoveredPoint, ...]


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
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    canonical_wikipedia = processed_root / "wikipedia" / "documents"
    wikipedia_dir = require_directory(
        canonical_wikipedia if canonical_wikipedia.exists() else processed_root / "articles",
        label="wikipedia/documents",
    )
    source_links_dir = links_dir or processed_root / "polygon_articles"
    require_directory(source_links_dir, label="polygon links")
    wikivoyage_dir = processed_root / "wikivoyage" / "documents"

    wikipedia_ids = _wikipedia_text_ids(wikipedia_dir)
    wikivoyage_ids, legacy_wikivoyage_qids = _wikivoyage_text_ids(wikivoyage_dir)
    document_ids = {
        "wikipedia": wikipedia_ids,
        "wikivoyage": wikivoyage_ids,
    }
    links = read_document_links(processed_root, links_dir=source_links_dir)
    wikipedia_polygons, combined_ids = _linked_text_polygon_ids(links, document_ids)
    has_canonical_links = _has_canonical_links(source_links_dir)
    all_polygon_ids, points_by_id = _polygon_points(
        polygons_dir,
        combined_ids,
        legacy_wikivoyage_qids,
        has_canonical_links,
    )
    _validate_link_polygon_ids(combined_ids, all_polygon_ids)
    points = tuple(points_by_id[polygon_id] for polygon_id in sorted(points_by_id))
    return TextPresenceSnapshot(
        polygon_count=len(all_polygon_ids),
        wikipedia_covered_polygon_ids=frozenset(wikipedia_polygons),
        combined_covered_polygon_ids=frozenset(combined_ids),
        wikipedia_document_ids=frozenset(wikipedia_ids),
        wikivoyage_document_ids=frozenset(wikivoyage_ids),
        covered_points=points,
    )


def _wikipedia_text_ids(wikipedia_dir: Path) -> set[str]:
    values: set[str] = set()
    for path in sorted_parquets(wikipedia_dir):
        values.update(_wikipedia_file_text_ids(path))
    return values


def _wikipedia_file_text_ids(path: Path) -> set[str]:
    identifier_column = _document_identity_column(set(pq.read_schema(path).names))  # type: ignore[no-untyped-call]
    values: set[str] = set()
    for row in read_required_columns(path, (identifier_column, "full_text"), label="wikipedia"):
        if row.get(identifier_column) and _non_blank(row.get("full_text")):
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
    document_ids: set[str] = set()
    qids: set[str] = set()
    for row in read_required_columns(
        path, ("document_id", "wikidata", "full_text"), label="wikivoyage"
    ):
        if not _non_blank(row.get("full_text")):
            continue
        if row.get("document_id"):
            document_ids.add(str(row["document_id"]))
        if row.get("wikidata"):
            qids.add(str(row["wikidata"]))
    return document_ids, qids


def _linked_text_polygon_ids(
    links: tuple[DocumentLink, ...],
    document_ids: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    wikipedia_ids: set[str] = set()
    combined_ids: set[str] = set()
    for link in links:
        if link.document_id not in document_ids.get(link.project, set()):
            continue
        combined_ids.add(link.polygon_id)
        if link.project == "wikipedia":
            wikipedia_ids.add(link.polygon_id)
    return wikipedia_ids, combined_ids


def _has_canonical_links(source_links_dir: Path) -> bool:
    return any(
        is_canonical_link_schema(pq.read_schema(path))  # type: ignore[no-untyped-call]
        for path in sorted_parquets(source_links_dir)
    )


def _polygon_points(
    polygons_dir: Path,
    combined_ids: set[str],
    legacy_wikivoyage_qids: set[str],
    has_canonical_links: bool,
) -> tuple[set[str], dict[str, CoveredPoint]]:
    all_polygon_ids: set[str] = set()
    points_by_id: dict[str, CoveredPoint] = {}
    for path in sorted_parquets(polygons_dir):
        file_ids, file_points = _polygon_file_points(
            path,
            combined_ids,
            legacy_wikivoyage_qids,
            has_canonical_links,
        )
        all_polygon_ids.update(file_ids)
        points_by_id.update(file_points)
    return all_polygon_ids, points_by_id


def _polygon_file_points(
    path: Path,
    combined_ids: set[str],
    legacy_wikivoyage_qids: set[str],
    has_canonical_links: bool,
) -> tuple[set[str], dict[str, CoveredPoint]]:
    all_polygon_ids: set[str] = set()
    points_by_id: dict[str, CoveredPoint] = {}
    for row in read_required_columns(
        path, ("polygon_id", "wikidata", "lon", "lat"), label="polygons"
    ):
        _record_polygon_row(
            path,
            row,
            all_polygon_ids,
            points_by_id,
            combined_ids,
            legacy_wikivoyage_qids,
            has_canonical_links,
        )
    return all_polygon_ids, points_by_id


def _record_polygon_row(
    path: Path,
    row: dict[str, Any],
    all_polygon_ids: set[str],
    points_by_id: dict[str, CoveredPoint],
    combined_ids: set[str],
    legacy_wikivoyage_qids: set[str],
    has_canonical_links: bool,
) -> None:
    polygon_id = _row_text(row, "polygon_id")
    qid = _row_text(row, "wikidata")
    if not polygon_id:
        raise CoverageMapError(f"polygons parquet {path} contains an empty polygon_id")
    all_polygon_ids.add(polygon_id)
    if not has_canonical_links and qid in legacy_wikivoyage_qids:
        combined_ids.add(polygon_id)
    if polygon_id in combined_ids:
        points_by_id.setdefault(polygon_id, _covered_point(path, polygon_id, qid, row))


def _row_text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return str(value) if value else ""


def _covered_point(
    path: Path,
    polygon_id: str,
    qid: str,
    row: dict[str, Any],
) -> CoveredPoint:
    try:
        return CoveredPoint(polygon_id, qid, float(row["lon"]), float(row["lat"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CoverageMapError(
            f"polygons parquet {path} has invalid coordinates for {polygon_id}"
        ) from error


def _validate_link_polygon_ids(combined_ids: set[str], all_polygon_ids: set[str]) -> None:
    unresolved = combined_ids - all_polygon_ids
    if unresolved:
        raise CoverageMapError(
            f"polygon_articles contains {len(unresolved)} polygon id(s) absent from polygons"
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
        f"{len(points):,} of {snapshot.polygon_count:,} dataset polygons "
        f"({rate:.1%}) have non-empty Wikipedia or Wikivoyage text."
    )
    return RenderResult(output_path=output_path, caption=caption)


__all__ = [
    "CoveredPoint",
    "TextPresenceSnapshot",
    "generate_geographic_text_presence",
    "load_text_presence",
]
