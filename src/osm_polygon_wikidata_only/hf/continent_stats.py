"""Deterministic continent assignment and public Markdown rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MatplotlibPath

from ._geographic.parquet_inputs import read_required_columns, sorted_parquets
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
            "- `Polygons`: dataset polygons whose centroid is assigned to the continent. "
            "Every polygon appears in exactly one continent row, including `Unassigned`.",
            "- `Wikipedia documents`: distinct non-empty Wikipedia documents connected to "
            "those polygons through `polygon_articles`. A document is counted once within a "
            "continent, but may appear in more than one continent when linked polygons span "
            "more than one continent.",
            "- `Wikivoyage documents`: distinct non-empty Wikivoyage documents whose Wikidata "
            "entity is shared by a polygon in the continent. The same cross-continent counting "
            "rule applies.",
            "- `Polygons with Wikipedia text`: polygons linked to at least one non-empty "
            "Wikipedia document.",
            "- `Polygons with Wikipedia or Wikivoyage text`: polygons satisfying the Wikipedia "
            "condition or sharing a Wikidata entity with at least one non-empty Wikivoyage "
            "document. Each polygon is counted once.",
            "- `Text coverage`: `combined text-covered polygons / all dataset polygons` in the "
            "continent row.",
        ]
    )
    return "\n".join(lines) + "\n"


def compute_continent_stats(
    processed_root: Path, country_geojson_path: Path
) -> list[tuple[str, int, int, int, int, int]]:
    """Compute deterministic document and polygon coverage by continent."""
    features = _load_continent_features(country_geojson_path)
    polygon_rows = _load_polygon_rows(processed_root)
    assignments = _assign_polygon_rows(polygon_rows, features)
    polygon_continent = _polygon_continents(polygon_rows, assignments)
    polygon_counts = _count_assignments(assignments)
    presence = load_text_presence(processed_root)
    wikipedia_docs, wikivoyage_docs = _document_counts(
        read_document_links(processed_root), polygon_continent, presence
    )
    wiki_polygon_counts = _covered_polygon_counts(
        presence.wikipedia_covered_polygon_ids, polygon_continent
    )
    combined_counts = _covered_polygon_counts(
        presence.combined_covered_polygon_ids, polygon_continent
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


def _load_polygon_rows(processed_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted_parquets(processed_root / "polygons"):
        rows.extend(
            read_required_columns(path, ("polygon_id", "wikidata", "lon", "lat"), label="polygons")
        )
    return rows


def _assign_polygon_rows(
    polygon_rows: Sequence[dict[str, Any]], features: Sequence[dict[str, Any]]
) -> list[str]:
    points = [(float(row["lon"]), float(row["lat"])) for row in polygon_rows]
    return assign_continents(points, features)


def _polygon_continents(
    polygon_rows: Sequence[dict[str, Any]], assignments: Sequence[str]
) -> dict[str, str]:
    return {
        str(row["polygon_id"]): continent
        for row, continent in zip(polygon_rows, assignments, strict=True)
    }


def _count_assignments(assignments: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for continent in assignments:
        counts[continent] += 1
    return counts


def _document_counts(
    links: Sequence[Any], polygon_continent: dict[str, str], presence: Any
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    wikipedia_docs: dict[str, set[str]] = defaultdict(set)
    wikivoyage_docs: dict[str, set[str]] = defaultdict(set)
    document_ids = {
        "wikipedia": presence.wikipedia_document_ids,
        "wikivoyage": presence.wikivoyage_document_ids,
    }
    target_sets = {"wikipedia": wikipedia_docs, "wikivoyage": wikivoyage_docs}
    for link in links:
        continent = polygon_continent.get(link.polygon_id)
        if continent is None or link.document_id not in document_ids.get(link.project, set()):
            continue
        target_sets[link.project][continent].add(link.document_id)
    return wikipedia_docs, wikivoyage_docs


def _covered_polygon_counts(
    polygon_ids: Collection[str], polygon_continent: dict[str, str]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for polygon_id in polygon_ids:
        counts[polygon_continent[polygon_id]] += 1
    return counts


__all__ = ["assign_continents", "compute_continent_stats", "render_continent_stats"]
