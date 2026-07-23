"""H3 density of polygons with non-empty Wikipedia or Wikivoyage text."""

from __future__ import annotations

import logging
from pathlib import Path

from ._geographic.h3_geometry import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MIN_POLYGONS_PER_CELL,
    assign_h3_cell,
)
from ._geographic.models import PolygonCountCell, RenderResult
from ._geographic.polygon_count import render_count_map
from .geographic_text_presence import TextPresenceSnapshot, load_text_presence

LOGGER = logging.getLogger(__name__)


def aggregate_geographic_text_density(
    processed_root: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
    snapshot: TextPresenceSnapshot | None = None,
) -> list[PolygonCountCell]:
    """Count unique text-covered polygon centroids in deterministic H3 cells."""
    snapshot = snapshot or load_text_presence(processed_root)
    counts: dict[str, int] = {}
    for point in snapshot.covered_points:
        cell = assign_h3_cell(point.lat, point.lon, resolution=h3_resolution)
        counts[cell] = counts.get(cell, 0) + 1
    return [
        PolygonCountCell(
            h3_cell=cell,
            polygon_count=counts[cell],
            is_low_sample=counts[cell] < min_polygons_per_cell,
        )
        for cell in sorted(counts)
    ]


def generate_geographic_text_density(
    processed_root: Path,
    output_path: Path,
    *,
    h3_resolution: int = DEFAULT_H3_RESOLUTION,
    min_polygons_per_cell: int = DEFAULT_MIN_POLYGONS_PER_CELL,
    land_cache_dir: Path | None = None,
    snapshot: TextPresenceSnapshot | None = None,
) -> RenderResult:
    """Aggregate and render raw combined-text polygon density."""
    cells = aggregate_geographic_text_density(
        processed_root,
        h3_resolution=h3_resolution,
        min_polygons_per_cell=min_polygons_per_cell,
        snapshot=snapshot,
    )
    from ._geographic.basemap import load_land_basemap

    land_features = load_land_basemap(land_cache_dir) if land_cache_dir else None
    total = sum(cell.polygon_count for cell in cells)
    caption = (
        "Geographic Wikipedia + Wikivoyage Text Density. Colour encodes the raw "
        "number of unique dataset polygons with non-empty text per H3 cell on a "
        f"logarithmic scale. {total:,} polygons across {len(cells):,} H3 cells."
    )
    return render_count_map(
        cells,
        output_path,
        title="Geographic Wikipedia + Wikivoyage Text Density",
        caption=caption,
        colorbar_label="Text-covered polygons per H3 cell",
        land_features=land_features,
        allow_empty=True,
    )


__all__ = ["aggregate_geographic_text_density", "generate_geographic_text_density"]
