"""Deterministic continent assignment and public Markdown rendering."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MatplotlibPath

from ._geographic.parquet_inputs import read_required_columns, sorted_parquets
from .geographic_text_presence import load_text_presence


def assign_continents(
    points: Sequence[tuple[float, float]], features: Sequence[dict[str, Any]]
) -> list[str]:
    """Assign ``(lon, lat)`` points to Natural Earth continent polygons."""
    assignments = np.full(len(points), "Unassigned", dtype=object)
    point_array = np.asarray(points, dtype=float)
    for feature in features:
        continent = str(feature.get("properties", {}).get("CONTINENT") or "Unassigned")
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates") or []
        polygons = coordinates if geometry.get("type") == "MultiPolygon" else [coordinates]
        for polygon in polygons:
            if not polygon:
                continue
            outer = np.asarray(polygon[0], dtype=float)
            if outer.size == 0:
                continue
            unassigned = assignments == "Unassigned"
            candidates = (
                unassigned
                & (point_array[:, 0] >= outer[:, 0].min())
                & (point_array[:, 0] <= outer[:, 0].max())
                & (point_array[:, 1] >= outer[:, 1].min())
                & (point_array[:, 1] <= outer[:, 1].max())
            )
            indexes = np.flatnonzero(candidates)
            if indexes.size == 0:
                continue
            inside = MatplotlibPath(outer).contains_points(point_array[indexes], radius=1e-9)
            for hole in polygon[1:]:
                inside &= ~MatplotlibPath(np.asarray(hole, dtype=float)).contains_points(
                    point_array[indexes], radius=-1e-9
                )
            assignments[indexes[inside]] = continent
    return [str(value) for value in assignments]


def render_continent_stats(rows: Sequence[tuple[str, int, int, int, int, int]]) -> str:
    """Render public per-continent statistics in deterministic order."""
    lines = [
        "## Geographic distribution by continent",
        "",
        "Polygon centroids are assigned to Natural Earth countries and continents. "
        "Document counts are distinct within each continent; polygons are counted once.",
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
    return "\n".join(lines) + "\n"


def compute_continent_stats(
    processed_root: Path, country_geojson_path: Path
) -> list[tuple[str, int, int, int, int, int]]:
    """Compute deterministic document and polygon coverage by continent."""
    data = json.loads(country_geojson_path.read_text(encoding="utf-8"))
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Natural Earth file has no feature list: {country_geojson_path}")

    polygon_rows: list[dict[str, Any]] = []
    for path in sorted_parquets(processed_root / "polygons"):
        polygon_rows.extend(
            read_required_columns(path, ("polygon_id", "wikidata", "lon", "lat"), label="polygons")
        )
    assignments = assign_continents(
        [(float(row["lon"]), float(row["lat"])) for row in polygon_rows], features
    )
    polygon_continent = {
        str(row["polygon_id"]): continent
        for row, continent in zip(polygon_rows, assignments, strict=True)
    }
    qid_continents: dict[str, set[str]] = defaultdict(set)
    polygon_counts: dict[str, int] = defaultdict(int)
    for row, continent in zip(polygon_rows, assignments, strict=True):
        polygon_counts[continent] += 1
        qid_continents[str(row["wikidata"])].add(continent)

    presence = load_text_presence(processed_root)
    wikipedia_docs: dict[str, set[str]] = defaultdict(set)
    for path in sorted_parquets(processed_root / "polygon_articles"):
        for row in read_required_columns(path, ("polygon_id", "article_id"), label="links"):
            article_id = str(row.get("article_id") or "")
            link_continent = polygon_continent.get(str(row.get("polygon_id") or ""))
            if article_id in presence.wikipedia_document_ids and link_continent:
                wikipedia_docs[link_continent].add(article_id)

    wikivoyage_docs: dict[str, set[str]] = defaultdict(set)
    for path in sorted_parquets(processed_root / "wikivoyage" / "documents"):
        for row in read_required_columns(
            path, ("document_id", "wikidata", "full_text"), label="wikivoyage"
        ):
            text = row.get("full_text")
            if not isinstance(text, str) or not text.strip():
                continue
            for continent in qid_continents.get(str(row.get("wikidata") or ""), set()):
                wikivoyage_docs[continent].add(str(row["document_id"]))

    wiki_polygon_counts: dict[str, int] = defaultdict(int)
    combined_counts: dict[str, int] = defaultdict(int)
    for polygon_id in presence.wikipedia_covered_polygon_ids:
        wiki_polygon_counts[polygon_continent[polygon_id]] += 1
    for polygon_id in presence.combined_covered_polygon_ids:
        combined_counts[polygon_continent[polygon_id]] += 1
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


__all__ = ["assign_continents", "compute_continent_stats", "render_continent_stats"]
