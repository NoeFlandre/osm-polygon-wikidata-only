"""Dataclasses for the geographic visualization domain.

These classes are re-exported by the
:mod:`osm_polygon_wikidata_only.hf.geographic_text_coverage` facade
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class CoverageMapError(RuntimeError):
    """Raised for invalid inputs or missing required dataset artifacts."""


@dataclass(frozen=True, slots=True)
class CoverageCell:
    """One H3 cell's aggregated Wikipedia text coverage statistics.

    Polygon counts are retained so the renderer can include them in the
    caption summary, but they are **not** encoded as opacity. Coverage
    rate is the sole colour channel for eligible cells.
    """

    h3_cell: str
    polygon_count: int
    covered_polygon_count: int
    coverage_rate: float
    is_low_sample: bool


@dataclass(frozen=True, slots=True)
class PolygonCountCell:
    """One H3 cell's aggregated polygon count.

    Polygons are conditional on the upstream ``wikidata=*`` filter from
    the dataset schema. Low-sample cells remain visible on this map
    because low counts are the metric and must not be greyed out.
    """

    h3_cell: str
    polygon_count: int
    is_low_sample: bool


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Outcome of a render function.

    The PNG is written to ``output_path`` and the exact caption text
    rendered onto the figure is exposed here so callers and tests can
    introspect it without parsing the rasterized image.
    """

    output_path: Path
    caption: str
