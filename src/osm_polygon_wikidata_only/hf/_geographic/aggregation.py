"""Deterministic pure aggregation for the geographic visualizations.

The aggregators only read parquet inputs, count rows, and produce
deterministic :class:`CoverageCell` / :class:`PolygonCountCell`
sequences. They perform no rendering or external I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .h3_geometry import DEFAULT_H3_RESOLUTION, DEFAULT_MIN_POLYGONS_PER_CELL
from .models import CoverageCell, CoverageMapError, PolygonCountCell
from .parquet_inputs import (
    load_covered_polygon_ids,
    load_polygon_cells,
    load_qualifying_article_ids,
    require_directory,
)

LOGGER = logging.getLogger(__name__)


def aggregate_geographic_text_coverage(
    processed_root: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> list[CoverageCell]:
    """Aggregate Wikipedia text coverage statistics per H3 cell.

    The denominator counts every polygon row in
    ``processed/polygons/*.parquet`` exactly once; the numerator
    counts unique polygons (never polygon-article links) linked to at
    least one article with non-empty full text. Both inputs are
    inherited from the upstream schema where polygons must already
    carry an OSM ``wikidata=*`` tag.
    """
    if min_polygons_per_cell < 1:
        raise CoverageMapError(f"min_polygons_per_cell must be >= 1; got {min_polygons_per_cell}")
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    canonical_documents_dir = processed_root / "wikipedia" / "documents"
    legacy_articles_dir = processed_root / "articles"
    articles_dir = require_directory(
        canonical_documents_dir if canonical_documents_dir.exists() else legacy_articles_dir,
        label=("wikipedia/documents" if canonical_documents_dir.exists() else "articles"),
    )
    links_dir = require_directory(processed_root / "polygon_articles", label="polygon_articles")

    qualifying_article_ids = load_qualifying_article_ids(articles_dir)
    covered_polygon_ids = load_covered_polygon_ids(links_dir, qualifying_article_ids)
    polygon_cells = load_polygon_cells(polygons_dir, h3_resolution=h3_resolution)

    counts: dict[str, int] = {}
    covered_counts: dict[str, int] = {}
    for polygon_id, cell in polygon_cells:
        counts[cell] = counts.get(cell, 0) + 1
        if polygon_id in covered_polygon_ids:
            covered_counts[cell] = covered_counts.get(cell, 0) + 1

    cells: list[CoverageCell] = []
    for h3_cell in sorted(counts):
        polygon_count = counts[h3_cell]
        covered_polygon_count = covered_counts.get(h3_cell, 0)
        coverage_rate = covered_polygon_count / polygon_count if polygon_count else 0.0
        cells.append(
            CoverageCell(
                h3_cell=h3_cell,
                polygon_count=polygon_count,
                covered_polygon_count=covered_polygon_count,
                coverage_rate=coverage_rate,
                is_low_sample=polygon_count < min_polygons_per_cell,
            )
        )
    LOGGER.info(
        "Aggregated %d H3 cell(s); %d covered polygon(s) of %d total.",
        len(cells),
        sum(c.covered_polygon_count for c in cells),
        sum(c.polygon_count for c in cells),
    )
    return cells


def aggregate_geographic_polygon_count(
    processed_root: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> list[PolygonCountCell]:
    """Aggregate raw polygon counts per H3 cell.

    Every dataset polygon is counted exactly once. Polygons are
    conditional on the upstream OSM ``wikidata=*`` filter; the count
    ignores article text and link membership because the metric is the
    raw count itself. Cells with fewer than ``min_polygons_per_cell``
    polygons are flagged as low-sample but remain visible on the map.
    """
    if min_polygons_per_cell < 1:
        raise CoverageMapError(f"min_polygons_per_cell must be >= 1; got {min_polygons_per_cell}")
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    polygon_cells = load_polygon_cells(polygons_dir, h3_resolution=h3_resolution)

    counts: dict[str, int] = {}
    for _, cell in polygon_cells:
        counts[cell] = counts.get(cell, 0) + 1

    cells: list[PolygonCountCell] = []
    for h3_cell in sorted(counts):
        polygon_count = counts[h3_cell]
        cells.append(
            PolygonCountCell(
                h3_cell=h3_cell,
                polygon_count=polygon_count,
                is_low_sample=polygon_count < min_polygons_per_cell,
            )
        )
    LOGGER.info(
        "Aggregated polygon counts for %d H3 cell(s); %d polygon(s) total.",
        len(cells),
        sum(c.polygon_count for c in cells),
    )
    return cells
