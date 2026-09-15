"""Frozen containers for the polygon surface and geometry statistics.

Every value is derived from the published ``polygons/<stem>.parquet``
table; nothing here is hand written or fetched from an external
service. Field order, the frozen flag, and ``slots=True`` are part of
the documented contract; do not reorder.

Units
-----
* Areas are square metres, exactly as recorded in the polygon table's
  ``area_m2`` column (WGS84 equirectangular approximation).
* Extent values are WGS84 decimal degrees, plus the equirectangular
  metre conversion used by the extraction code.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bumped whenever the payload shape or a counting rule changes.
STATS_CONTRACT_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class Distribution:
    """Order statistics of one per-polygon quantity.

    All six values are ``0.0`` when no polygon contributed a sample.
    ``median`` and the percentiles use linear interpolation between the
    two closest ranks, which is deterministic for a given input set.
    """

    minimum: float = 0.0
    maximum: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


@dataclass(frozen=True, slots=True)
class AreaSummary:
    """Surface statistics over the polygon table's ``area_m2`` column.

    * ``total_m2`` — sum of every recorded area.
    * ``p1`` … ``p99`` — percentiles with the same interpolation rule
      as :class:`Distribution`.
    * ``non_positive_count`` — rows whose recorded area is ``<= 0``,
      i.e. degenerate geometry.
    * ``below_one_m2_count`` — rows with ``0 < area_m2 < 1``, i.e.
      suspiciously small but not degenerate.
    * ``null_count`` — rows with no recorded area at all.
    """

    total_m2: float = 0.0
    minimum_m2: float = 0.0
    maximum_m2: float = 0.0
    mean_m2: float = 0.0
    median_m2: float = 0.0
    p1_m2: float = 0.0
    p5_m2: float = 0.0
    p25_m2: float = 0.0
    p75_m2: float = 0.0
    p95_m2: float = 0.0
    p99_m2: float = 0.0
    non_positive_count: int = 0
    below_one_m2_count: int = 0
    null_count: int = 0


@dataclass(frozen=True, slots=True)
class AreaHistogramBucket:
    """One half-open log-scale area bucket ``[lower_m2, upper_m2)``.

    ``upper_m2`` is ``None`` for the final unbounded bucket, and
    ``lower_m2`` is ``None`` for the leading bucket that collects
    degenerate ``area_m2 <= 0`` rows.
    """

    label: str
    lower_m2: float | None
    upper_m2: float | None
    count: int


@dataclass(frozen=True, slots=True)
class ShapeSummary:
    """Ring and vertex structure decoded from the ``geometry`` column.

    * ``polygon_count`` / ``multipolygon_count`` — GeoJSON ``type``.
    * ``unreadable_count`` — rows whose geometry is missing, not valid
      JSON, or not a Polygon/MultiPolygon. They contribute no ring,
      vertex, or component sample.
    * ``with_holes_count`` — rows carrying at least one inner ring.
    * ``total_rings`` / ``total_holes`` / ``total_vertices`` — sums over
      every readable row. A vertex is one coordinate pair; the repeated
      closing coordinate of each ring is not counted twice.
    * ``components`` — MultiPolygon member counts; Polygons contribute
      a sample of ``1``.
    """

    polygon_count: int = 0
    multipolygon_count: int = 0
    unreadable_count: int = 0
    with_holes_count: int = 0
    total_rings: int = 0
    total_holes: int = 0
    total_vertices: int = 0
    vertices: Distribution = Distribution()
    rings: Distribution = Distribution()
    components: Distribution = Distribution()


@dataclass(frozen=True, slots=True)
class ExtentSummary:
    """Dataset bounding box and per-polygon bbox size distributions.

    The four ``dataset_*`` values are the envelope of every readable
    per-row ``bbox``. ``width_m`` and ``height_m`` convert the degree
    spans with the same equirectangular rule the extraction code uses
    for ``area_m2``, evaluated at each bbox's own mean latitude.

    * ``wider_than_180_deg_count`` — bboxes spanning more than half the
      globe, the signature of an antimeridian-crossing polygon.
    * ``pole_touching_count`` — bboxes reaching ``|lat| >= 90``.
    * ``unreadable_count`` — rows whose ``bbox`` is missing or is not a
      JSON list of four finite numbers.
    """

    dataset_min_lon: float = 0.0
    dataset_min_lat: float = 0.0
    dataset_max_lon: float = 0.0
    dataset_max_lat: float = 0.0
    width_deg: Distribution = Distribution()
    height_deg: Distribution = Distribution()
    width_m: Distribution = Distribution()
    height_m: Distribution = Distribution()
    wider_than_180_deg_count: int = 0
    pole_touching_count: int = 0
    unreadable_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceAreaSummary:
    """Per ``source_pbf`` surface breakdown, sorted by source name."""

    source_pbf: str
    polygon_count: int
    total_area_m2: float
    median_area_m2: float
    maximum_area_m2: float


@dataclass(frozen=True, slots=True)
class PolygonGeometryStats:
    """Complete deterministic snapshot of the polygon table's geometry."""

    contract_version: str = STATS_CONTRACT_VERSION
    file_count: int = 0
    polygon_count: int = 0
    area: AreaSummary = AreaSummary()
    area_histogram: tuple[AreaHistogramBucket, ...] = ()
    shape: ShapeSummary = ShapeSummary()
    extent: ExtentSummary = ExtentSummary()
    per_source: tuple[SourceAreaSummary, ...] = ()


__all__ = [
    "STATS_CONTRACT_VERSION",
    "AreaHistogramBucket",
    "AreaSummary",
    "Distribution",
    "ExtentSummary",
    "PolygonGeometryStats",
    "ShapeSummary",
    "SourceAreaSummary",
]
