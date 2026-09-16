"""Deterministic continent assignment and public Markdown rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MatplotlibPath

from ._geographic.models import CoverageMapError
from ._geographic.parquet_inputs import sorted_parquets
from ._geographic.polygon_identities import (
    PolygonIdentity,
    PolygonRecord,
    load_unique_polygon_records,
)
from ._links.reader import read_document_links
from .geographic_text_presence import load_text_presence


def assign_continents(
    points: Sequence[tuple[float, float]], features: Sequence[dict[str, Any]]
) -> list[str]:
    """Assign ``(lon, lat)`` points to Natural Earth continent polygons."""
    assignments = np.full(len(points), "Unassigned", dtype=object)
    point_array = np.asarray(points, dtype=float)
    for feature in features:
        continent, polygons = _feature_polygons(feature)
        for polygon in polygons:
            _assign_polygon(assignments, point_array, polygon, continent)
    return [str(value) for value in assignments]


def _feature_polygons(feature: dict[str, Any]) -> tuple[str, Sequence[Any]]:
    continent = str(feature.get("properties", {}).get("CONTINENT") or "Unassigned")
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates") or []
    polygons = coordinates if geometry.get("type") == "MultiPolygon" else [coordinates]
    return continent, polygons


def _assign_polygon(
    assignments: np.ndarray,
    point_array: np.ndarray,
    polygon: Any,
    continent: str,
) -> None:
    if not polygon:
        return
    outer = np.asarray(polygon[0], dtype=float)
    if outer.size == 0:
        return
    indexes = _candidate_indexes(assignments, point_array, outer)
    if indexes.size == 0:
        return
    inside = _inside_polygon(point_array[indexes], polygon)
    assignments[indexes[inside]] = continent


def _candidate_indexes(
    assignments: np.ndarray,
    point_array: np.ndarray,
    outer: np.ndarray,
) -> np.ndarray:
    unassigned = assignments == "Unassigned"
    candidates = (
        unassigned
        & (point_array[:, 0] >= outer[:, 0].min())
        & (point_array[:, 0] <= outer[:, 0].max())
        & (point_array[:, 1] >= outer[:, 1].min())
        & (point_array[:, 1] <= outer[:, 1].max())
    )
    return np.flatnonzero(candidates)


def _inside_polygon(point_array: np.ndarray, polygon: Any) -> np.ndarray:
    inside = MatplotlibPath(np.asarray(polygon[0], dtype=float)).contains_points(
        point_array, radius=1e-9
    )
    for hole in polygon[1:]:
        inside &= ~MatplotlibPath(np.asarray(hole, dtype=float)).contains_points(
            point_array, radius=-1e-9
        )
    return inside


def render_continent_stats(rows: Sequence[tuple[str, int, int, int, int, int]]) -> str:
    """Render public per-continent statistics in deterministic order."""
    lines = [
        "## Geographic distribution by continent",
        "",
        "This table is recomputed from the finalized Parquet tables before each dataset-card "
        "publication. Each polygon's WGS84 centroid is spatially matched to the bundled "
        "Natural Earth 1:110m Admin-0 country boundaries, then assigned the country's "
        "continent. Because this is a coarse global reference, offshore points and centroids "
        "outside its country polygons remain `Unassigned` rather than being guessed.",
        "",
        "| Continent | Polygons | Wikipedia documents | Wikivoyage documents | "
        "Polygons with Wikipedia text | Polygons with Wikipedia or Wikivoyage text | Text coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for continent, polygons, wikipedia_docs, voyage_docs, wiki_polygons, combined in sorted(rows):
        rate = combined / polygons if polygons else 0.0
        lines.append(
            f"| {continent} | {polygons:,} | {wikipedia_docs:,} | {voyage_docs:,} | "
            f"{wiki_polygons:,} | {combined:,} | {rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "**Metric definitions:**",
            "",
            "- `Polygons`: globally unique `(osm_type, osm_id)` identities whose deterministic "
            "representative centroid is assigned to the continent. Every identity appears in "
            "exactly one continent row, including `Unassigned`; regional polygon rows remain "
            "separate source records.",
            "- `Wikipedia documents`: distinct successfully extracted Wikipedia documents "
            "with trimmed non-empty `full_text`, connected to those identities through "
            "`polygon_articles`. A document is counted once within a "
            "continent, but may appear in more than one continent when linked polygons span "
            "more than one continent.",
            "- `Wikivoyage documents`: distinct successfully extracted Wikivoyage documents "
            "with trimmed non-empty `full_text` whose Wikidata entity is shared by an identity "
            "in the continent. The same cross-continent counting rule applies.",
            "- `Polygons with Wikipedia text`: unique identities linked to at least one "
            "successfully extracted (`fetch_status=ok`) Wikipedia document with trimmed "
            "non-empty `full_text`.",
            "- `Polygons with Wikipedia or Wikivoyage text`: unique identities satisfying the "
            "Wikipedia condition or linked to a successfully extracted Wikivoyage document "
            "with trimmed non-empty `full_text`. Each identity is counted once.",
            "- `Text coverage`: `combined text-covered identities / all unique polygon "
            "identities` in the continent row.",
        ]
    )
    return "\n".join(lines) + "\n"


def compute_continent_stats(
    processed_root: Path, country_geojson_path: Path
) -> list[tuple[str, int, int, int, int, int]]:
    """Compute deterministic document and polygon coverage by continent."""
    features = _load_continent_features(country_geojson_path)
    polygon_index = load_unique_polygon_records(sorted_parquets(processed_root / "polygons"))
    polygon_rows = [
        polygon_index.records[identity]
        for identity in sorted(polygon_index.records, key=_identity_sort_key)
    ]
    assignments = _assign_polygon_rows(polygon_rows, features)
    polygon_continent = _polygon_continents(polygon_rows, assignments)
    polygon_counts = _count_assignments(assignments)
    presence = load_text_presence(processed_root)
    wikipedia_docs, wikivoyage_docs = _document_counts(
        read_document_links(processed_root),
        polygon_continent,
        presence,
        polygon_index=polygon_index.by_polygon_id,
    )
    wiki_polygon_counts = _covered_polygon_counts(
        presence.wikipedia_polygon_identities, polygon_continent
    )
    combined_counts = _covered_polygon_counts(
        presence.combined_polygon_identities, polygon_continent
    )
    return [
        (
            continent,
            polygon_counts[continent],
            len(wikipedia_docs[continent]),
            len(wikivoyage_docs[continent]),
            wiki_polygon_counts[continent],
            combined_counts[continent],
        )
        for continent in sorted(polygon_counts)
    ]


def _load_continent_features(country_geojson_path: Path) -> list[dict[str, Any]]:
    data = json.loads(country_geojson_path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Natural Earth file has no feature list: {country_geojson_path}")
    return features


def _assign_polygon_rows(
    polygon_rows: Sequence[PolygonRecord], features: Sequence[dict[str, Any]]
) -> list[str]:
    points = [_record_point(row) for row in polygon_rows]
    return assign_continents(points, features)


def _polygon_continents(
    polygon_rows: Sequence[PolygonRecord], assignments: Sequence[str]
) -> dict[PolygonIdentity, str]:
    return {
        row.identity: continent for row, continent in zip(polygon_rows, assignments, strict=True)
    }


def _record_point(record: PolygonRecord) -> tuple[float, float]:
    if record.lon is None or record.lat is None:
        raise CoverageMapError(
            f"polygons parquet {record.source_path} has invalid lat/lon coordinates "
            f"for {record.polygon_id}"
        )
    return record.lon, record.lat


def _identity_sort_key(identity: PolygonIdentity) -> tuple[str, str]:
    return identity[0], str(identity[1])


def _count_assignments(assignments: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for continent in assignments:
        counts[continent] += 1
    return counts


def _document_counts(
    links: Sequence[Any],
    polygon_continent: Mapping[PolygonIdentity, str],
    presence: Any,
    *,
    polygon_index: Mapping[str, PolygonIdentity],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    wikipedia_docs: dict[str, set[str]] = defaultdict(set)
    wikivoyage_docs: dict[str, set[str]] = defaultdict(set)
    document_ids = {
        "wikipedia": presence.wikipedia_document_ids,
        "wikivoyage": presence.wikivoyage_document_ids,
    }
    target_sets = {"wikipedia": wikipedia_docs, "wikivoyage": wikivoyage_docs}
    for link in links:
        identity = polygon_index.get(link.polygon_id)
        continent = polygon_continent.get(identity) if identity is not None else None
        if continent is None or link.document_id not in document_ids.get(link.project, set()):
            continue
        target_sets[link.project][continent].add(link.document_id)
    return wikipedia_docs, wikivoyage_docs


def _covered_polygon_counts(
    polygon_ids: Collection[PolygonIdentity],
    polygon_continent: Mapping[PolygonIdentity, str],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for polygon_id in polygon_ids:
        continent = polygon_continent.get(polygon_id)
        if continent is not None:
            counts[continent] += 1
    return counts


__all__ = ["assign_continents", "compute_continent_stats", "render_continent_stats"]
