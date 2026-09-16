"""Deterministic pure aggregation for the geographic visualizations.

The aggregators only read parquet inputs, count rows, and produce
deterministic :class:`CoverageCell` / :class:`PolygonCountCell`
sequences. They perform no rendering or external I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osm_polygon_wikidata_only.hf.geographic_text_presence import TextPresenceSnapshot

from .h3_geometry import DEFAULT_H3_RESOLUTION, DEFAULT_MIN_POLYGONS_PER_CELL
from .models import CoverageCell, CoverageMapError, PolygonCountCell
from .parquet_inputs import (
    load_polygon_cells,
    require_directory,
    sorted_parquets,
)
from .polygon_identities import PolygonIdentity, load_unique_polygon_records

LOGGER = logging.getLogger(__name__)


def aggregate_geographic_text_coverage(
    processed_root: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
) -> list[CoverageCell]:
    """Aggregate Wikipedia text coverage statistics per H3 cell.

    The denominator counts each unique ``(osm_type, osm_id)`` polygon
    identity once; the numerator counts the subset linked to at least
    one successfully extracted Wikipedia document with trimmed,
    non-empty ``full_text``. Regional polygon rows remain available to
    row-based reporting. Both inputs are inherited from the upstream
    schema where polygons must already carry an OSM ``wikidata=*`` tag.
    """
    if min_polygons_per_cell < 1:
        raise CoverageMapError(f"min_polygons_per_cell must be >= 1; got {min_polygons_per_cell}")
    polygons_dir, _articles_dir, _links_dir = _coverage_input_dirs(processed_root)
    from osm_polygon_wikidata_only.hf.geographic_text_presence import load_text_presence

    presence: TextPresenceSnapshot = load_text_presence(processed_root)
    polygon_index = load_unique_polygon_records(sorted_parquets(polygons_dir))
    polygon_cells = [
        (polygon_index.by_polygon_id[polygon_id], cell)
        for polygon_id, cell in load_polygon_cells(
            polygons_dir,
            h3_resolution=h3_resolution,
        )
    ]
    cells = _build_coverage_cells(
        polygon_cells,
        set(presence.wikipedia_polygon_identities),
        min_polygons_per_cell,
    )
    LOGGER.info(
        "Aggregated %d H3 cell(s); %d covered polygon(s) of %d total.",
        len(cells),
        sum(c.covered_polygon_count for c in cells),
        sum(c.polygon_count for c in cells),
    )
    return cells


def _coverage_input_dirs(processed_root: Path) -> tuple[Path, Path, Path]:
    polygons_dir = require_directory(processed_root / "polygons", label="polygons")
    canonical_documents_dir = processed_root / "wikipedia" / "documents"
    legacy_articles_dir = processed_root / "articles"
    articles_dir = require_directory(
        canonical_documents_dir if canonical_documents_dir.exists() else legacy_articles_dir,
        label=("wikipedia/documents" if canonical_documents_dir.exists() else "articles"),
    )
    links_dir = require_directory(processed_root / "polygon_articles", label="polygon_articles")
    return polygons_dir, articles_dir, links_dir


def _build_coverage_cells(
    polygon_cells: list[tuple[PolygonIdentity, str]],
    covered_polygon_ids: set[PolygonIdentity],
    min_polygons_per_cell: int,
) -> list[CoverageCell]:
    counts: dict[str, int] = {}
    covered_counts: dict[str, int] = {}
    for polygon_id, cell in polygon_cells:
        counts[cell] = counts.get(cell, 0) + 1
        if polygon_id in covered_polygon_ids:
            covered_counts[cell] = covered_counts.get(cell, 0) + 1
    return [
        _coverage_cell(h3_cell, counts[h3_cell], covered_counts, min_polygons_per_cell)
        for h3_cell in sorted(counts)
    ]


def _coverage_cell(
    h3_cell: str,
    polygon_count: int,
    covered_counts: dict[str, int],
    min_polygons_per_cell: int,
) -> CoverageCell:
    covered_polygon_count = covered_counts.get(h3_cell, 0)
    return CoverageCell(
        h3_cell=h3_cell,
        polygon_count=polygon_count,
        covered_polygon_count=covered_polygon_count,
        coverage_rate=covered_polygon_count / polygon_count if polygon_count else 0.0,
        is_low_sample=polygon_count < min_polygons_per_cell,
    )


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
