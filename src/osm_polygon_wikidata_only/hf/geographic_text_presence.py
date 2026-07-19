"""Factual polygon coverage by non-empty Wikipedia or Wikivoyage text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._geographic.models import CoverageMapError, RenderResult
from ._geographic.parquet_inputs import read_required_columns, require_directory, sorted_parquets
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


def load_text_presence(processed_root: Path) -> TextPresenceSnapshot:
    """Load exact Wikipedia and combined text coverage from canonical tables."""
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    canonical_wikipedia = processed_root / "wikipedia" / "documents"
    wikipedia_dir = require_directory(
        canonical_wikipedia if canonical_wikipedia.exists() else processed_root / "articles",
        label="wikipedia/documents",
    )
    links_dir = require_directory(processed_root / "polygon_articles", label="polygon_articles")
    wikivoyage_dir = processed_root / "wikivoyage" / "documents"

    wikipedia_ids: set[str] = set()
    for path in sorted_parquets(wikipedia_dir):
        for row in read_required_columns(path, ("article_id", "full_text"), label="wikipedia"):
            if row.get("article_id") and _non_blank(row.get("full_text")):
                wikipedia_ids.add(str(row["article_id"]))

    wikipedia_polygons: set[str] = set()
    for path in sorted_parquets(links_dir):
        for row in read_required_columns(path, ("polygon_id", "article_id"), label="links"):
            if str(row.get("article_id")) in wikipedia_ids and row.get("polygon_id"):
                wikipedia_polygons.add(str(row["polygon_id"]))

    wikivoyage_ids: set[str] = set()
    wikivoyage_qids: set[str] = set()
    for path in sorted_parquets(wikivoyage_dir):
        for row in read_required_columns(
            path, ("document_id", "wikidata", "full_text"), label="wikivoyage"
        ):
            if not _non_blank(row.get("full_text")):
                continue
            if row.get("document_id"):
                wikivoyage_ids.add(str(row["document_id"]))
            if row.get("wikidata"):
                wikivoyage_qids.add(str(row["wikidata"]))

    all_polygon_ids: set[str] = set()
    combined_ids: set[str] = set(wikipedia_polygons)
    points: list[CoveredPoint] = []
    for path in sorted_parquets(polygons_dir):
        for row in read_required_columns(
            path, ("polygon_id", "wikidata", "lon", "lat"), label="polygons"
        ):
            polygon_id = str(row.get("polygon_id") or "")
            qid = str(row.get("wikidata") or "")
            if not polygon_id:
                raise CoverageMapError(f"polygons parquet {path} contains an empty polygon_id")
            all_polygon_ids.add(polygon_id)
            if qid in wikivoyage_qids:
                combined_ids.add(polygon_id)
            if polygon_id in combined_ids:
                try:
                    points.append(
                        CoveredPoint(polygon_id, qid, float(row["lon"]), float(row["lat"]))
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise CoverageMapError(
                        f"polygons parquet {path} has invalid coordinates for {polygon_id}"
                    ) from error

    unresolved = combined_ids - all_polygon_ids
    if unresolved:
        raise CoverageMapError(
            f"polygon_articles contains {len(unresolved)} polygon id(s) absent from polygons"
        )
    points.sort(key=lambda point: point.polygon_id)
    return TextPresenceSnapshot(
        polygon_count=len(all_polygon_ids),
        wikipedia_covered_polygon_ids=frozenset(wikipedia_polygons),
        combined_covered_polygon_ids=frozenset(combined_ids),
        wikipedia_document_ids=frozenset(wikipedia_ids),
        wikivoyage_document_ids=frozenset(wikivoyage_ids),
        covered_points=tuple(points),
    )


def generate_geographic_text_presence(
    processed_root: Path,
    output_path: Path,
    *,
    land_geojson_path: Path | None = None,
) -> RenderResult:
    """Render one point for every polygon with Wikipedia or Wikivoyage text."""
    snapshot = load_text_presence(processed_root)
    points = snapshot.covered_points
    generate_coverage_map(
        [point.lon for point in points],
        [point.lat for point in points],
        output_path,
        land_geojson_path=land_geojson_path,
        title="Polygons with Wikipedia or Wikivoyage text",
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
