"""Generate deterministic H3-aggregated geographic visualizations.

Two world maps are produced from the same local Parquet inputs:

1. ``assets/geographic_wikipedia_text_coverage.png`` -- for each H3 cell
   at the configured resolution, the fraction of dataset polygons linked
   to at least one Wikipedia article with non-empty ``full_text``. The
   denominator is every polygon row in ``processed/polygons/*.parquet``
   (already conditional on the upstream ``wikidata=*`` filter). Cell
   colour encodes coverage from 0% to 100%; polygon count is **not**
   encoded as opacity. Grey cells are reserved for low-sample cells
   below the configured threshold.

2. ``assets/geographic_polygon_count.png`` -- the same H3 layout, but
   colour encodes the raw polygon count per cell using a logarithmic
   normalization because counts are highly skewed across the world. Low
   counts are the metric and are not greyed out; opacity is not used as
   a second data encoding.

Both maps share a basemap, world extent, deterministic cell ordering,
atomic output via a temporary file, and publication-quality styling.
Outputs are sorted by H3 cell id and the figure layout is fully
deterministic. The module is independent of CLI parsing and the network:
it reads local Parquet files and writes PNGs without external HTTP.

This module is a thin compatibility facade. The implementation lives
in :mod:`osm_polygon_wikidata_only.hf._geographic` and every public
name below is re-exported unchanged.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from ._geographic.aggregation import (
    aggregate_geographic_polygon_count,
    aggregate_geographic_text_coverage,
)
from ._geographic.coverage import (
    generate_geographic_text_coverage,
    render_geographic_text_coverage,
)
from ._geographic.h3_geometry import (
    DEFAULT_H3_RESOLUTION,
    DEFAULT_MIN_POLYGONS_PER_CELL,
    assign_h3_cell,
)
from ._geographic.models import (
    CoverageCell,
    CoverageMapError,
    PolygonCountCell,
    RenderResult,
)
from ._geographic.polygon_count import (
    generate_geographic_polygon_count,
    render_geographic_polygon_count,
)

REMOTE_TEXT_COVERAGE_ASSET_PATH: str = "assets/geographic_wikipedia_text_coverage.png"
LOCAL_TEXT_COVERAGE_ASSET_PATH: str = REMOTE_TEXT_COVERAGE_ASSET_PATH
REMOTE_POLYGON_COUNT_ASSET_PATH: str = "assets/geographic_polygon_count.png"
LOCAL_POLYGON_COUNT_ASSET_PATH: str = REMOTE_POLYGON_COUNT_ASSET_PATH

# Backwards-compatible aliases for the historical single-asset naming.
LOCAL_ASSET_PATH: str = LOCAL_TEXT_COVERAGE_ASSET_PATH
REMOTE_ASSET_PATH: str = REMOTE_TEXT_COVERAGE_ASSET_PATH


__all__ = [
    "DEFAULT_H3_RESOLUTION",
    "DEFAULT_MIN_POLYGONS_PER_CELL",
    "LOCAL_ASSET_PATH",
    "LOCAL_POLYGON_COUNT_ASSET_PATH",
    "LOCAL_TEXT_COVERAGE_ASSET_PATH",
    "REMOTE_ASSET_PATH",
    "REMOTE_POLYGON_COUNT_ASSET_PATH",
    "REMOTE_TEXT_COVERAGE_ASSET_PATH",
    "CoverageCell",
    "CoverageMapError",
    "PolygonCountCell",
    "RenderResult",
    "aggregate_geographic_polygon_count",
    "aggregate_geographic_text_coverage",
    "assign_h3_cell",
    "generate_geographic_polygon_count",
    "generate_geographic_text_coverage",
    "render_geographic_polygon_count",
    "render_geographic_text_coverage",
]
